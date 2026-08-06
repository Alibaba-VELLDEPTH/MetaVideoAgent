"""Codex authoring for five-module MetaVideoAgent bundles."""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
from runtime_paths import (
    require_metavideoagent_runtime,
    runtime_dir,
    runtime_output_root,
    use_runtime_path,
)

RUNTIME_DIR = use_runtime_path()
from bundle_contract import (
    COMBO_SLOT as BUNDLE_COMBO_SLOT,
)
from bundle_contract import (
    MODULE_TYPES as BUNDLE_MODULE_TYPES,
)
from bundle_contract import (
    bundle_fingerprint,
    make_candidate_bundle,
    validate_candidate_bundle,
)
from bundle_validation import (
    MODULE_SOURCE_FILES,
    get_full_module_code,
)
from candidate_forensics import (
    forensic_prompt_block,
    materialize_supervised_forensic_workspace,
    scan_candidate_source_for_specialization,
    validate_codex_forensic_report,
)
from capability_registry import PROFILE_MANUAL_PATH
from codex_runtime_settings import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    codex_exec_model_args,
    resolve_codex_runtime_settings,
)
from coding_agent_runtime import coding_agent_environment, resolve_coding_agent_executable
from config import DEFAULT_WORKSPACE
from diagnosis_execution_brief import (
    validate_bundle_machine_evaluation_contract,
    validate_bundle_execution_brief,
)
from distribution_spec import load_distribution_spec, resolve_distribution_manifest_path
from module_protocol import parse_json_object
from target_selection import canonical_task_ref

MAX_BUNDLE_CODEX_CONTEXT_CHARS = 900_000



def _abs(path: str) -> str:
    """Resolve a path relative to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _load_diagnosis(path: str) -> dict:
    with open(_abs(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_optional(path: str) -> dict:
    if not path or not os.path.exists(_abs(path)):
        return {}
    with open(_abs(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_required(path: str, label: str) -> dict:
    abs_path = _abs(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Required {label} not found: {path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _reference_from_report(path: str) -> str:
    report_path = _abs(path)
    report = _load_json_optional(report_path)
    if not report:
        return ""
    result = (
        report.get("results_path")
        or report.get("full_eval_results")
        or report.get("source_full_eval_results")
        or report.get("sandbox_dir")
        or ""
    )
    if result and not os.path.isabs(result):
        joined = os.path.normpath(os.path.join(os.path.dirname(report_path), result))
        if os.path.exists(joined):
            return joined
    return result



def _resolve_report_relative(path: str, report_path: str) -> str:
    if not path:
        return ""
    candidates = [path] if os.path.isabs(path) else [
        os.path.normpath(os.path.join(os.path.dirname(_abs(report_path)), path)),
        _abs(path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return _abs(candidate)
    return ""


def _looks_like_initial_bundle(path: str) -> bool:
    """Return True only for full five-module bundle records, not candidate modules."""
    if not path or not os.path.exists(path):
        return False
    payload = _load_json_optional(path)
    if not isinstance(payload, dict):
        return False
    modules = payload.get("modules")
    combo = payload.get("combo")
    module_names = payload.get("module_names")
    if isinstance(modules, dict) and isinstance(combo, dict):
        return True
    if isinstance(module_names, dict) and isinstance(combo, dict):
        return True
    required_module_types = {"video_structuring", "localization", "perception", "memory", "thinking"}
    if isinstance(modules, list):
        found = {
            item.get("module_type")
            for item in modules
            if isinstance(item, dict) and item.get("module_type")
        }
        return required_module_types.issubset(found) and isinstance(combo, dict)
    return False


def _reference_bundle_from_report(ref_report: dict, report_path: str) -> str:
    """Resolve the complete five-module bundle backing a reference report.

    Candidate bundles are deliberately excluded because they are not immutable
    reference bundles for smoke replay.
    """
    combo_policy = ref_report.get("combo_policy") or {}
    candidates = [
        combo_policy.get("effective_bundle_path", ""),
        combo_policy.get("current_best_bundle_path", ""),
        ref_report.get("effective_initial_baseline_bundle", ""),
        ref_report.get("initial_baseline_bundle", ""),
        combo_policy.get("base_bundle_path", ""),
    ]
    for raw in candidates:
        resolved = _resolve_report_relative(raw, report_path)
        if _looks_like_initial_bundle(resolved):
            return resolved
    local = os.path.join(os.path.dirname(_abs(report_path)), "initial_baseline_bundle.json")
    if _looks_like_initial_bundle(local):
        return local
    return ""


def _reference_combo_from_report(ref_report: dict) -> tuple:
    """Return the explicit current-best combo and its provenance."""
    report_combo = ref_report.get("combo") or ref_report.get("agent_combo") or {}
    combo_policy = ref_report.get("combo_policy") or {}
    base_combo = combo_policy.get("base_combo") or {}
    if isinstance(report_combo, dict) and report_combo:
        return dict(report_combo), {"source": "reference_report.combo"}
    if isinstance(base_combo, dict) and base_combo:
        return dict(base_combo), {"source": "reference_report.combo_policy.base_combo"}
    return {}, {"source": "none"}





def _resolve_rollout_path(diagnosis_path: str, explicit: str = "") -> str:
    if explicit:
        return _abs(explicit)
    abs_diag = _abs(diagnosis_path)
    base = os.path.basename(abs_diag)
    if base.startswith("diagnosis_"):
        candidate = os.path.join(
            os.path.dirname(abs_diag),
            base.replace("diagnosis_", "diagnosis_rollout_", 1),
        )
        if os.path.exists(candidate):
            return candidate
    return ""



def _snapshot_files(files: List[str]) -> Dict[str, str]:
    originals = {}
    for rel_path in files:
        abs_path = _abs(rel_path)
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                originals[rel_path] = f.read()
    return originals


def _restore_files(originals: Dict[str, str]) -> None:
    for rel_path, content in originals.items():
        abs_path = _abs(rel_path)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)


def _copy_source_snapshot(files: List[str], dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for rel_path in files:
        src = _abs(rel_path)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _write_source_diff(originals: Dict[str, str], diff_path: str) -> None:
    chunks = []
    for rel_path, before in originals.items():
        abs_path = _abs(rel_path)
        after = ""
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                after = f.read()
        chunks.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"source_before/{rel_path}",
                tofile=f"source_after/{rel_path}",
            )
        )
    with open(diff_path, "w", encoding="utf-8") as f:
        f.writelines(chunks)



def _trim(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]


TEACHER_ONLY_TOOL_MAP = {
    "DIAG_VIDEO_AUDIT": "Video review capability",
    "DIAG_DETAIL_VIDEO_AUDIT": "Detailed video review capability",
    "AUDIT_VIDEO": "Video review capability",
    "AUDIT_DETAIL_VIDEO": "Detailed video review capability",
    "AUDIT_ENHANCED_VIDEO": "Enhance video review capabilities",
    "AUDIT_STRUCTURE": "Structure library review capability",
    "AUDIT_TRANSCRIPT": "Subtitle/audio text review capabilities",
    "watch_video": "Video review capability",
    "video_review": "Video review capability",
    "teacher_video_verification": "Video review capability",
    "detailed_watch": "Detailed video review capability",
    "detail_video_review": "Detailed video review capability",
    "fine_video_review": "Detailed video review capability",
    "teacher_detailed_video_verification": "Detailed video review capability",
    "enhanced_watch": "Enhance video review capabilities",
    "enhanced_video_review": "Enhance video review capabilities",
    "teacher_enhanced_video_verification": "Enhance video review capabilities",
    "browse_struct_db": "Structure library review capability",
    "structure_review": "Structure library review capability",
    "teacher_structure_db_audit": "Structure library review capability",
    "read_transcript": "Subtitle/audio text review capabilities",
    "transcript_review": "Subtitle/audio text review capabilities",
    "teacher_transcript_audit": "Subtitle/audio text review capabilities",
    "listen_audio": "Subtitle/audio text review capabilities",
}

DISALLOWED_RUNTIME_TOOL_MAP = {
    "frame_inspect": "visual observation ability",
    "single_frame_inspect": "visual observation ability",
    "inspect_frames": "visual observation ability",
    "runtime_visual_observation_tool": "visual observation ability",
    "runtime_visual_observation_capability": "visual observation ability",
    "search_evidence": "Positioning/structure search capabilities",
    "clip_search": "Positioning/structure search capabilities",
    "frame_clip_search": "Positioning/structure search capabilities",
    "runtime_localization_or_structure_retrieval_tool": "Positioning/structure search capabilities",
    "runtime_localization_or_structure_retrieval_capability": "Positioning/structure search capabilities",
    "runtime_evidence_retrieval_capability": "Positioning/structure search capabilities",
    "global_browse": "Global structure browsing capabilities",
    "runtime_broad_structure_retrieval_tool": "Global structure browsing capabilities",
    "runtime_broad_structure_retrieval_capability": "Global structure browsing capabilities",
    "audio_analysis": "Audio/subtitle understanding ability",
    "runtime_audio_or_transcript_evidence_tool": "Audio/subtitle understanding ability",
    "runtime_audio_or_transcript_capability": "Audio/subtitle understanding ability",
}




def _tool_methods_from_code(code: str, class_name: str = "") -> list:
    tools = []
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return tools
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if class_name and node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name.startswith("tool_"):
                tools.append(item.name[5:])
    return sorted(set(tools))


def _literal_default(node) -> str:
    if node is None:
        return ""
    try:
        return repr(ast.literal_eval(node))
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "..."


def _annotation_text(node) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _tool_contracts_from_code(code: str, class_name: str = "") -> dict:
    """Extract runtime tool signatures for prompt-visible contracts.

    The execution layer validates parameters against the actual `tool_*`
    method signatures.  Giving Codex only tool names is insufficient for
    thinking-module evolution because missing required `tool_params` cause
    circuit breaks at runtime.
    """
    contracts = {}
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return contracts
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if class_name and node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or not item.name.startswith("tool_"):
                continue
            args = list(item.args.args or [])
            defaults = list(item.args.defaults or [])
            default_offset = len(args) - len(defaults)
            params = []
            required = []
            for idx, arg in enumerate(args):
                if arg.arg == "self":
                    continue
                default_node = defaults[idx - default_offset] if idx >= default_offset else None
                has_default = default_node is not None
                record = {
                    "name": arg.arg,
                    "annotation": _annotation_text(arg.annotation),
                    "required": not has_default,
                }
                if has_default:
                    record["default"] = _literal_default(default_node)
                else:
                    required.append(arg.arg)
                params.append(record)
            contracts[item.name[5:]] = {
                "signature": (
                    f"{item.name[5:]}("
                    + ", ".join(
                        p["name"] + (f": {p['annotation']}" if p.get("annotation") else "")
                        + (f" = {p['default']}" if "default" in p else "")
                        for p in params
                    )
                    + ")"
                ),
                "required_tool_params": required,
                "parameters": params,
                "docstring": _trim(ast.get_docstring(item) or "", 500),
            }
    return contracts


def run_codex(prompt: str, codex_cli: str, timeout: int, *, settings: dict | None = None,
              workdir: str = "") -> dict:
    settings = settings or resolve_codex_runtime_settings()
    if not str(workdir or "").strip():
        raise ValueError("Codex authoring requires an explicit isolated working directory")
    workdir = os.path.abspath(workdir)
    cmd = [
        codex_cli,
        "exec",
        *codex_exec_model_args(settings),
        "--sandbox",
        "workspace-write",
        "-C",
        workdir,
        "--skip-git-repo-check",
        # Codex accepts `-` as stdin. Passing long diagnosis/repair prompts as
        # one argv item hits Linux MAX_ARG_STRLEN (~128 KiB) before Codex starts.
        "-",
    ]
    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=prompt,
            env=coding_agent_environment(),
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "elapsed_sec": round(time.time() - started, 1),
            "runtime_environment_error": _is_codex_runtime_environment_error(result.stderr),
            "runtime_settings": settings,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "returncode": None,
            "stdout": e.stdout or "",
            "stderr": f"Codex timed out after {timeout}s",
            "elapsed_sec": round(time.time() - started, 1),
            "runtime_environment_error": False,
            "runtime_settings": settings,
        }
    except OSError as e:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(e),
            "elapsed_sec": round(time.time() - started, 1),
            "runtime_environment_error": True,
            "runtime_settings": settings,
        }


def _is_codex_runtime_environment_error(stderr: str) -> bool:
    text = stderr or ""
    markers = (
        "failed to open state db",
        "attempt to write a readonly database",
        "failed to initialize in-process app-server client",
        "Read-only file system",
    )
    return any(marker in text for marker in markers)




def _iter_jsonl_rows(path: str) -> list:
    rows = []
    paths = _sandbox_jsonl_files(path)
    for item in paths:
        try:
            with open(item, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        obj["_source_path"] = item
                        rows.append(obj)
        except OSError:
            continue
    return rows


def _sandbox_jsonl_files(path: str) -> list[str]:
    """Return every raw sandbox trajectory file in deterministic order.

    This is intentionally a file-level helper: feedback handoff must preserve
    the original JSONL bytes, not reserialize selected rows through an audit.
    """
    if not path or not os.path.exists(path):
        return []
    if os.path.isfile(path):
        return [path] if path.endswith(".jsonl") else []
    return sorted(
        glob_path
        for glob_path in __import__("glob").glob(os.path.join(path, "*_sandbox.jsonl"))
        if os.path.isfile(glob_path)
    )


def _tool_counts(row: dict) -> dict:
    """Count actual executed tool steps, never mentions inside observations."""
    counts = {}
    for key in ("trajectory", "steps", "tool_trace", "react_trace"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for step in value:
            if not isinstance(step, dict):
                continue
            if step.get("step_type") != "tool_execution":
                continue
            tool = step.get("tool") or step.get("tool_name") or step.get("action")
            if isinstance(tool, str) and tool.strip():
                tool = tool.strip()
                counts[tool] = counts.get(tool, 0) + 1
    return counts


def _capability_events(row: dict) -> list[dict]:
    return [item for item in (row.get("capability_events") or []) if isinstance(item, dict)]


def _structured_runtime_contract_errors(row: dict) -> list[str]:
    """Read execution failures only from structured runtime trajectory fields."""
    errors = []
    for step in row.get("trajectory") or []:
        if not isinstance(step, dict) or step.get("step_type") != "tool_execution":
            continue
        status = str(step.get("execution_status") or "ok")
        if status != "ok":
            errors.append(str(step.get("error_code") or status))
    if row.get("error"):
        errors.append("sandbox_exception")
    return sorted(set(errors))


def _trajectory_len(row: dict) -> int:
    for key in ("trajectory", "steps", "tool_trace", "react_trace"):
        value = row.get(key)
        if isinstance(value, list):
            return len(value)
    counts = _tool_counts(row)
    return sum(counts.values())


def _frames_viewed(row: dict) -> int:
    return sum(
        1 for item in _capability_events(row)
        if item.get("event") == "frame_selection" and item.get("status") == "ok"
    )


def _llm_calls(row: dict) -> int:
    return sum(
        1 for item in _capability_events(row)
        if item.get("event") in {"inference", "transcription"}
        and item.get("status") in {"ok", "no_content", "no_speech"}
    )


def _answer_shape_issue(answer: str) -> bool:
    value = (answer or "").strip()
    if not value:
        return True
    lower = value.lower()
    if lower.startswith(("the question asks", "need to ", "we need to ", "no prior observations")):
        return True
    if len(value) > 240 and not re.search(r"\([a-z]\)|^[a-z]\.|^\d+\b", value, re.I):
        return True
    return False


def build_probe_audit(verification: dict) -> dict:
    """Build an end-to-end, non-overfit audit from probe report + sandbox rows.

    The audit intentionally abstracts away concrete answers/video timestamps.
    It gives Codex repair feedback about behavior classes: wasteful loops,
    weak evidence handoff, modality mismatch, answer-contract issues, etc.
    """
    verification = verification or {}
    sandbox_dir = verification.get("sandbox_dir", "")
    probe_plan = verification.get("probe_plan") if isinstance(verification.get("probe_plan"), dict) else {}
    selected_refs = list(probe_plan.get("selected_refs") or [])
    selected_repair_refs = list(probe_plan.get("selected_repair_refs") or [])
    selected_repair_opportunity_refs = list(
        probe_plan.get("selected_repair_opportunity_refs") or []
    )
    selected_guard_refs = list(probe_plan.get("selected_guard_refs") or [])
    reference_incorrect_repair_refs = list(
        probe_plan.get("reference_incorrect_repair_refs") or []
    )
    rows = _iter_jsonl_rows(sandbox_dir)
    valid_rows = [r for r in rows if r.get("question") or r.get("task_id")]
    implementation_spec = verification.get("implementation_spec", {}) or {}
    declared_capability_contract = (
        implementation_spec.get("capability_contract")
        or implementation_spec.get("runtime_capabilities")
        or []
    )
    expected_capabilities = {
        str(item.get("capability") or "")
        for item in declared_capability_contract
        if isinstance(item, dict) and item.get("capability")
    }
    expected_profiles = {
        str(item.get("capability") or ""): str(item.get("profile_id") or "")
        for item in declared_capability_contract
        if isinstance(item, dict) and item.get("capability") and item.get("profile_id")
    }
    failure_contract = (
        (verification.get("implementation_spec", {}) or {}).get("failure_contract", {})
    )
    if not isinstance(failure_contract, dict):
        failure_contract = {}
    selected_witness_refs = {
        canonical_task_ref(ref) for ref in (probe_plan.get("selected_diagnosis_hypothesis_refs") or [])
        if canonical_task_ref(ref)
    }
    mfp_witness_states = {}
    for item in (probe_plan.get("diagnosis_hypothesis_witnesses") or []):
        if not isinstance(item, dict):
            continue
        hypothesis_id = str(item.get("hypothesis_id") or "")
        refs = {
            canonical_task_ref(ref)
            for ref in (item.get("available_refs") or item.get("refs") or [])
            if canonical_task_ref(ref)
        }
        if not hypothesis_id:
            continue
        mfp_witness_states[hypothesis_id] = {
            "hypothesis_id": hypothesis_id,
            "selected": bool(refs & selected_witness_refs),
            "available_count": len(refs),
            "observed_rows": 0,
            "candidate_correct_rows": 0,
            "candidate_incorrect_rows": 0,
        }

    def row_refs(row: dict) -> set[str]:
        """Return machine-only aliases for matching a probe row to its witness."""
        refs = {
            canonical_task_ref(row.get("task_id") or "", video_id=str(row.get("video_id") or "")),
            canonical_task_ref(row.get("time_reference") or "", video_id=str(row.get("video_id") or "")),
        }
        video_id = str(row.get("video_id") or "")
        time_reference = str(row.get("time_reference") or "")
        if video_id and time_reference:
            refs.add(canonical_task_ref(f"{video_id}::{time_reference}"))
        return {ref for ref in refs if ref}
    categories = {}

    def add_issue(code: str, severity: str, rationale: str, repair_goal: str) -> None:
        item = categories.setdefault(code, {
            "code": code,
            "severity": severity,
            "count": 0,
            "rationales": [],
            "repair_goal": repair_goal,
        })
        item["count"] += 1
        if rationale not in item["rationales"][:5]:
            item["rationales"].append(rationale)

    row_summaries = []
    # Probe is a training-stage evolution signal.  Preserve every failed
    # training trajectory (including its judged answer) for Codex analysis,
    # while source validation forbids copying any of these literals into the
    # generated module.  This lets Codex derive general mechanisms from real
    # failures instead of guessing from an empty aggregate label.
    train_feedback_candidates = []
    mechanism_decision_rows = 0
    mechanism_fallback_rows = 0
    runtime_contract_error_rows = 0
    capability_summary = {"declared_or_observed": {}, "successful_inference_rows": 0, "event_rows": 0}
    observed_capabilities_all = set()
    observed_profiles_by_capability = {}
    for row in valid_rows:
        all_tool_steps = [
            step for step in (row.get("trajectory") or [])
            if isinstance(step, dict) and step.get("step_type") == "tool_execution"
        ]
        target_steps = all_tool_steps
        counts = _tool_counts({"trajectory": target_steps})
        runtime_contract_errors = _structured_runtime_contract_errors(row)
        steps = _trajectory_len(row)
        frames = _frames_viewed(row)
        llm_calls = _llm_calls(row)
        answer = str(row.get("answer") or row.get("final_agent_answer") or "")
        is_correct = row.get("is_correct")
        decision_signal = {}
        for step in reversed(row.get("trajectory") or []):
            if not isinstance(step, dict):
                continue
            payload = step.get("action_input") if isinstance(step.get("action_input"), dict) else {}
            summary = str(payload.get("evidence_summary") or "")
            if not summary:
                continue
            try:
                parsed = json.loads(summary)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict) and parsed.get("verification"):
                decision_signal = {
                    "mode": str(parsed.get("verification") or ""),
                    "rankings": parsed.get("rankings") if isinstance(parsed.get("rankings"), list) else [],
                }
                break
        if decision_signal:
            mechanism_decision_rows += 1
            if "fallback" in decision_signal.get("mode", "") or "low_support" in decision_signal.get("mode", ""):
                mechanism_fallback_rows += 1
        matched_witnesses = [
            state for state in mfp_witness_states.values()
            if set(row_refs(row)) & {
                canonical_task_ref(ref) for item in (probe_plan.get("diagnosis_hypothesis_witnesses") or [])
                if isinstance(item, dict) and str(item.get("hypothesis_id") or "") == state["hypothesis_id"]
                for ref in (item.get("available_refs") or item.get("refs") or [])
            }
        ]
        for state in matched_witnesses:
            state["observed_rows"] += 1
            if is_correct is True:
                state["candidate_correct_rows"] += 1
            else:
                state["candidate_incorrect_rows"] += 1
        tool_total = sum(counts.values())
        dominant_tool = max(counts, key=lambda k: counts[k]) if counts else ""
        events = _capability_events(row)
        observed_capabilities = {str(item.get("capability") or "") for item in events}
        observed_capabilities_all.update(observed_capabilities)
        for event in events:
            capability = str(event.get("capability") or "")
            profile_id = str(event.get("profile_id") or "")
            if capability and profile_id and event.get("status") in {"ok", "no_content", "no_speech"}:
                observed_profiles_by_capability.setdefault(capability, set()).add(profile_id)
        if events:
            capability_summary["event_rows"] += 1
        successful_inference = False
        for event in events:
            capability = str(event.get("capability") or "unknown")
            bucket = capability_summary["declared_or_observed"].setdefault(
                capability, {"events": 0, "ok": 0, "errors": 0},
            )
            bucket["events"] += 1
            if event.get("status") == "ok":
                bucket["ok"] += 1
                if event.get("event") in {"inference", "transcription"}:
                    successful_inference = True
            elif event.get("status") not in {"no_content", "no_speech"}:
                bucket["errors"] += 1
        capability_summary["successful_inference_rows"] += int(successful_inference)
        provider_failures = [
            item for item in events
            if item.get("status") in {"api_error", "asset_error"}
            or (
                item.get("status") == "unavailable"
                and item.get("reason") not in {"", "no_active_time_windows"}
            )
        ]
        if provider_failures:
            add_issue(
                "capability_provider_failure", "high",
                "A structured runtime capability event reported unavailable, asset, or provider failure.",
                "Repair the execution environment or provider availability and rerun the same probe; do not ask Codex to alter a working module for this failure.",
            )
        active_window_contract_failures = [
            item for item in events
            if item.get("status") == "unavailable"
            and item.get("reason") == "no_active_time_windows"
        ]
        if active_window_contract_failures:
            add_issue(
                "capability_active_window_contract_failure", "critical",
                "A declared bounded capability was invoked without an active-media window.",
                "Keep the linked bundle on the existing runtime path and call the runtime adapter only after it receives a valid bounded window from the preserved pipeline.",
            )
        if runtime_contract_errors:
            runtime_contract_error_rows += 1
            add_issue(
                "runtime_tool_contract_failure",
                "critical",
                "A probe trajectory hit runtime tool/API contract errors instead of executing the intended evidence operation.",
                "Repair the candidate against the actual execution-layer tool names and parameter schemas before judging accuracy.",
            )
        if frames >= 120 or llm_calls >= 20:
            add_issue(
                "cost_explosion",
                "high",
                "A probe answer consumed many VLM frames/LLM calls relative to one question.",
                "Reduce redundant evidence gathering; expose compact answer-ready evidence before escalating to expensive inspection.",
            )
        consumer_events = [
            item for item in events
            if item.get("event") == "consumer_consumed"
            and item.get("status") == "ok"
            and item.get("contract_valid") is True
        ]
        requires_consumer_event = bool(
            ((verification.get("implementation_spec", {}) or {}).get("consumer_contract", {}) or {}).get(
                "requires_runtime_consumption_event"
            )
        )
        if requires_consumer_event and not consumer_events:
            add_issue(
                "output_not_consumed", "critical",
                "A bundle module produced a runtime result but no consumer event recorded that the downstream module received it.",
                "Return through the existing execute/router path so the agent stores the observation and the next thinking turn consumes it.",
            )
        if any(
            str(step.get("execution_status") or "ok") == "ok"
            and not str(step.get("observation") or "").strip()
            for step in target_steps
        ):
            add_issue(
                "empty_evidence", "high",
                "A successful target-tool execution yielded no observation payload for downstream reasoning.",
                "Return a bounded structured evidence object or an explicit unavailable/no-speech fallback.",
            )
        if ((dominant_tool and counts.get(dominant_tool, 0) >= 4) or (tool_total >= 8 and dominant_tool)) and not consumer_events:
            add_issue(
                "non_convergent_tool_loop",
                "high",
                "The agent repeated similar tool calls without a clear state transition to verification or final answer.",
                "Make retrieval outputs actionable: include confidence, interval ranking, why this evidence matters, and next-step guidance.",
            )
        if len([tool for tool, count in counts.items() if count >= 2]) >= 2 and is_correct is False:
            add_issue(
                "weak_evidence_handoff",
                "high",
                "The structure/retrieval layer returned evidence, but the downstream reasoning loop still failed to make the right decision.",
                "Return fewer, more discriminative evidence cards and explicitly separate visual, audio, OCR, and uncertainty fields.",
            )
        if _answer_shape_issue(answer):
            add_issue(
                "poor_answer_contract",
                "medium",
                "A final answer looks like planning/reasoning text or is not shaped as a direct answer.",
                "Ensure the module supports downstream answer formation instead of encouraging another search plan at termination.",
            )
        if frames >= 250:
            add_issue(
                "over_dense_indexing_or_verification",
                "medium",
                "Dense structure did not prevent repeated downstream visual inspection.",
                "Do not solve uncertainty by blindly increasing frame density; improve segment summaries and retrieval ranking.",
            )

        if provider_failures:
            failure_stage = "provider_or_asset_failure"
        elif active_window_contract_failures:
            failure_stage = "active_window_contract_failure"
        elif runtime_contract_errors:
            failure_stage = "interface_contract_failure"
        elif requires_consumer_event and not consumer_events:
            failure_stage = "output_not_consumed"
        elif any(
            str(step.get("execution_status") or "ok") == "ok"
            and not str(step.get("observation") or "").strip()
            for step in target_steps
        ):
            failure_stage = "empty_evidence"
        elif (dominant_tool and counts.get(dominant_tool, 0) >= 4) or (tool_total >= 8 and dominant_tool):
            failure_stage = "true_tool_loop"
        elif is_correct is False:
            failure_stage = "behavior_no_gain"
        else:
            failure_stage = "none"
        row_summaries.append({
            "task_id_hash": str(row.get("task_id", ""))[-12:],
            "is_correct": bool(is_correct),
            "steps": steps,
            "tool_counts": counts,
            "target_tool_step_count": len(target_steps),
            "runtime_contract_errors": runtime_contract_errors,
            "frames_viewed": frames,
            "llm_calls": llm_calls,
            "answer_shape_issue": _answer_shape_issue(answer),
            "capability_events": events[:16],
            "consumer_events": consumer_events[:4],
            "successful_capability_inference": successful_inference,
            "failure_stage": failure_stage,
        })
        if is_correct is False:
            evidence_excerpts = []
            for step in (row.get("trajectory") or []):
                if not isinstance(step, dict) or step.get("step_type") != "tool_execution":
                    continue
                observation = str(step.get("observation") or "").strip()
                if observation:
                    evidence_excerpts.append({
                        "producer_module": str(step.get("producer_module") or ""),
                        "action": str(step.get("action") or ""),
                    "observation": observation,
                    })
            reasoning_after_evidence = False
            seen_evidence = False
            for step in (row.get("trajectory") or []):
                if not isinstance(step, dict):
                    continue
                if step.get("step_type") == "tool_execution" and str(step.get("observation") or "").strip():
                    seen_evidence = True
                elif seen_evidence and step.get("step_type") in {"reasoning", "planning"}:
                    reasoning_after_evidence = True
            gaps = []
            if evidence_excerpts:
                gaps.append("evidence_observed_but_final_decision_incorrect")
            else:
                gaps.append("finalized_without_observed_evidence")
            if not reasoning_after_evidence:
                gaps.append("no_post_evidence_reasoning_or_verification_step")
            if _answer_shape_issue(answer):
                gaps.append("final_answer_contract_not_direct")
            if decision_signal.get("mode") in {"verification_low_support", "verification_skipped_no_candidates"}:
                gaps.append("declared_decision_mechanism_fell_back")
            if any("asr_status\\\": \\\"not_supplied" in str(item.get("observation") or "") for item in evidence_excerpts):
                gaps.append("requested_audio_channel_had_no_transcript")
            if any(any(token in str(item.get("observation") or "").lower() for token in ("cannot determine", "no visible action", "sequence")) for item in evidence_excerpts):
                gaps.append("available_evidence_is_insufficient_for_temporal_or_relational_decision")
            train_feedback_candidates.append({
                "machine_ref": str(row.get("task_id") or ""),
                "hypothesis_ids": sorted({
                    str(state.get("hypothesis_id") or "") for state in matched_witnesses
                    if str(state.get("hypothesis_id") or "")
                }),
                "witness_relation": (
                    "diagnosis_hypothesis_mfp_witness"
                    if matched_witnesses else "selected_train_failure_fallback"
                ),
                "question": str(row.get("question") or ""),
                "candidate_answer": answer,
                "gold_answer": str(row.get("gt_answer") or row.get("ground_truth") or ""),
                "active_media_windows": row.get("runtime_time_windows") or [],
                "trajectory_shape": {
                    "total_steps": steps,
                    "target_tool_steps": len(target_steps),
                    "post_evidence_reasoning": reasoning_after_evidence,
                    "evidence_operation_count": len(evidence_excerpts),
                },
                "observed_gap_classes": gaps,
                "evidence_excerpts": evidence_excerpts,
                "decision_signal": decision_signal,
                "training_only": True,
                "gold_answer_omitted": False,
            })

    missing_capabilities = expected_capabilities - observed_capabilities_all
    if missing_capabilities:
        add_issue(
            "capability_not_executed", "critical",
            "A diagnosis-declared capability left no structured runtime event across the complete probe set.",
            "Use the declared runtime_evidence adapter under its bounded trigger before retrying Codex.",
        )
    profile_mismatches = {
        capability: {"expected": expected, "observed": sorted(observed_profiles_by_capability.get(capability, set()))}
        for capability, expected in expected_profiles.items()
        if observed_profiles_by_capability.get(capability)
        and expected not in observed_profiles_by_capability[capability]
    }
    if profile_mismatches:
        add_issue(
            "capability_profile_mismatch", "critical",
            "A declared capability executed through a different runtime profile than the validated implementation specification: "
            + json.dumps(profile_mismatches, ensure_ascii=False, sort_keys=True),
            "Keep the selected profile_id in the linked bundle's runtime_evidence adapter or update the diagnosis brief before a new candidate is generated.",
        )

    if verification.get("regressions"):
        add_issue(
            "current_best_regression",
            "critical",
            "Probe contains at least one regression against the explicit reference/current-best combo.",
            "Treat this as a risk signal for Codex repair or expanded probing; do not conclude full-eval ineffectiveness from this alone.",
        )
    for runtime_issue in (
        list(verification.get("runtime_issues") or [])
        + list(verification.get("runtime_engineering_issues") or [])
    ):
        if not isinstance(runtime_issue, dict):
            continue
        code = str(runtime_issue.get("issue_type") or "sandbox_runtime_failure")
        add_issue(
            code,
            str(runtime_issue.get("severity") or "critical"),
            "Probe setup or routing aborted before a complete candidate behavior result.",
            str(runtime_issue.get("fix_hint") or "Repair the structured sandbox failure before judging behavior."),
        )
    repair_opportunity_refs = (
        selected_repair_opportunity_refs
        or reference_incorrect_repair_refs
        or selected_repair_refs
    )
    if not repair_opportunity_refs:
        add_issue(
            "guard_only_probe",
            "medium",
            "The selected probe contains no repair-opportunity rows where the reference/current-best was wrong.",
            "Use this result only as a regression guard; expand or reselect probe with repair targets before judging benefit.",
        )
    elif not verification.get("corrections"):
        add_issue(
            "no_observed_probe_benefit",
            "high",
            "Probe selected repair-opportunity rows but found no correction on those targets.",
            "Use bounded same-direction Codex feedback with aggregate failure classes and safe training witnesses; do not fit an individual probe answer.",
        )

    probe_execution_complete = bool(verification.get("probe_execution_complete", True))
    if not probe_execution_complete:
        add_issue(
            "incomplete_probe_execution", "critical",
            "One or more selected probe rows were missing, duplicated, or could not be aligned to the reference.",
            "Repair the probe execution/reference alignment and rerun the same candidate; do not rewrite candidate behavior from an incomplete comparison.",
        )
    issues = sorted(
        categories.values(),
        key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["severity"], 9),
    )
    provider_failure_rows = sum(
        item.get("failure_stage") == "provider_or_asset_failure" for item in row_summaries
    )
    engineering_invalid = bool(verification.get("engineering_invalid")) or runtime_contract_error_rows > 0
    probe_status = verification.get("probe_status")
    if engineering_invalid:
        probe_status = "invalid"
    mfp_observations = []
    for state in mfp_witness_states.values():
        if not state["selected"]:
            outcome = "unobserved_budget_or_selection"
        elif not state["observed_rows"]:
            outcome = "unobserved_execution"
        elif state["candidate_correct_rows"]:
            outcome = "end_to_end_improved"
        else:
            outcome = "end_to_end_not_improved"
        mfp_observations.append({
            **state,
            "outcome": outcome,
        })
    witness_outcome = (
        "end_to_end_improved" if any(item["outcome"] == "end_to_end_improved" for item in mfp_observations)
        else "end_to_end_not_improved" if any(item["outcome"] == "end_to_end_not_improved" for item in mfp_observations)
        else "unobserved" if mfp_observations else "not_configured"
    )

    return {
        "schema_version": 1,
        "audit_type": "probe_behavior_audit",
        "probe_status": probe_status,
        "total_questions": verification.get("total_questions"),
        "accuracy_delta": verification.get("accuracy_delta"),
        "corrections": len(verification.get("corrections", []) or []),
        "regressions": len(verification.get("regressions", []) or []),
        "engineering_invalid": engineering_invalid,
        "engineering_invalid_reason": (
            "probe trajectories contain runtime tool/API contract errors"
            if runtime_contract_error_rows else verification.get("engineering_invalid_reason", "")
        ),
        "execution_layer_rerun_required": provider_failure_rows > 0 or not probe_execution_complete,
        "probe_execution_complete": probe_execution_complete,
        "probe_execution_accounting": verification.get("probe_execution_accounting", {}),
        "runtime_contract_error_rows": runtime_contract_error_rows,
        "failure_contract": failure_contract,
        "mechanism_witness": {
            "configured": bool(mfp_witness_states),
            "outcome": witness_outcome,
            "observations": mfp_observations[:20],
            "note": "MFP observations are cross-module end-to-end evidence, not a hard probe gate.",
        },
        "train_feedback_witnesses": sorted(
            train_feedback_candidates,
            key=lambda item: (
                item.get("witness_relation") != "diagnosis_hypothesis_mfp_witness",
                -int(item.get("trajectory_shape", {}).get("evidence_operation_count") or 0),
            ),
        ),
        "mechanism_execution": {
            "decision_signal_rows": mechanism_decision_rows,
            "fallback_decision_rows": mechanism_fallback_rows,
            "all_observed_decisions_fell_back": bool(mechanism_decision_rows)
            and mechanism_decision_rows == mechanism_fallback_rows,
        },
        "capability_summary": capability_summary,
        "expected_profiles": expected_profiles,
        "observed_profiles": {key: sorted(value) for key, value in observed_profiles_by_capability.items()},
        "consumer_contract_summary": {
            "rows_with_tool_execution": sum(
                any(
                    isinstance(step, dict)
                    and step.get("step_type") == "tool_execution"
                    for step in (row.get("trajectory") or [])
                ) for row in valid_rows
            ),
            "rows_with_consumer_consumption": sum(
                any(
                    item.get("event") == "consumer_consumed"
                    and item.get("status") == "ok"
                    and item.get("contract_valid") is True
                    for item in _capability_events(row)
                )
                for row in valid_rows
            ),
            "note": "Tool execution and consumer consumption are counted only from structured runtime events/trajectory fields.",
        },
        "failure_stage_counts": {
            stage: sum(item.get("failure_stage") == stage for item in row_summaries)
            for stage in sorted({item.get("failure_stage") for item in row_summaries})
        },
        "selected_refs": selected_refs[:20],
        "selected_repair_refs": selected_repair_refs[:20],
        "selected_repair_opportunity_refs": selected_repair_opportunity_refs[:20],
        "selected_guard_refs": selected_guard_refs[:20],
        "reference_incorrect_repair_refs": reference_incorrect_repair_refs[:20],
        "probe_selection_note": probe_plan.get("selection_note", ""),
        "issues": issues,
        "row_summaries": row_summaries[:20],
        "repair_policy": {
            "do_not_hardcode": True,
            "do_not_patch_specific_question": True,
            "repair_for_behavior_class": True,
            "module_scope_locked": True,
            "retargeting_out_of_scope": True,
            "success_requirement": (
                "Codex probe feedback handles implementation/runtime defects and "
                "every runnable same-direction candidate that has not earned full "
                "evaluation. Net-positive probe candidates should advance to full "
                "evaluation with regression risk recorded."
            ),
        },
    }


def build_bundle_probe_audit(verification: dict, bundle: dict) -> dict:
    """Audit an adaptive candidate as one five-module dataflow graph.

    The base audit provides end-to-end accounting. This layer adds per-module
    runtime facts and declared handoff checks for linked-bundle feedback.
    """
    verification = dict(verification or {})
    bundle = bundle if isinstance(bundle, dict) else {}
    verification["implementation_spec"] = dict(bundle.get("implementation_spec") or {})
    verification["implementation_spec"].setdefault(
        "handoff_contracts", list(bundle.get("handoff_contracts") or [])
    )
    base = build_probe_audit(verification)
    rows = [
        row for row in _iter_jsonl_rows(verification.get("sandbox_dir", ""))
        if isinstance(row, dict) and (row.get("question") or row.get("task_id"))
    ]
    per_module = {
        module: {
            "tool_execution_rows": 0,
            "tool_execution_count": 0,
            "empty_observation_count": 0,
            "result_statuses": {},
            "consumer_consumption_rows": 0,
        }
        for module in BUNDLE_MODULE_TYPES
    }
    handoff_checks = []
    contracts = list(bundle.get("handoff_contracts") or [])
    for row in rows:
        trajectory = row.get("trajectory") or []
        tool_steps = [
            step for step in trajectory
            if isinstance(step, dict) and step.get("step_type") == "tool_execution"
        ]
        events = _capability_events(row)
        for module in BUNDLE_MODULE_TYPES:
            steps = [step for step in tool_steps if str(step.get("producer_module") or "") == module]
            if steps:
                per_module[module]["tool_execution_rows"] += 1
                per_module[module]["tool_execution_count"] += len(steps)
            for step in steps:
                if not str(step.get("observation") or "").strip():
                    per_module[module]["empty_observation_count"] += 1
                parsed = parse_json_object(step.get("observation"))
                status = str(parsed.get("status") or step.get("execution_status") or "unknown")
                counts = per_module[module]["result_statuses"]
                counts[status] = int(counts.get(status) or 0) + 1
            if any(
                isinstance(event, dict) and event.get("event") == "consumer_consumed"
                and event.get("status") == "ok" and event.get("contract_valid") is True
                and str(event.get("producer_module") or "") == module
                for event in events
            ):
                per_module[module]["consumer_consumption_rows"] += 1
        for contract in contracts:
            if not isinstance(contract, dict):
                continue
            producer = str(contract.get("producer") or "")
            consumer = str(contract.get("consumer") or "")
            fields = [str(value) for value in (contract.get("required_fields") or []) if str(value)]
            producer_index = None
            producer_payload = {}
            producer_text = ""
            for index, step in enumerate(tool_steps):
                if str(step.get("producer_module") or "") != producer:
                    continue
                producer_index = index
                producer_text = str(step.get("observation") or "")
                producer_payload = parse_json_object(producer_text)
                break
            # Module bases wrap their domain payload under ``evidence``.  In
            # addition, the persisted trajectory deliberately truncates very
            # large observations, so an unparseable audit copy is not proof
            # that the runtime payload lacked the declared fields.  Unwrap
            # parseable envelopes; otherwise defer to the structured
            # consumer-consumed event below instead of manufacturing a
            # critical handoff failure from logging truncation.
            payload_parseable = bool(producer_payload)
            if isinstance(producer_payload.get("evidence"), dict):
                producer_payload = {
                    **producer_payload,
                    **dict(producer_payload.get("evidence") or {}),
                }
            missing = (
                [field for field in fields if field not in producer_payload]
                if payload_parseable else []
            )
            consumer_seen = any(
                str(step.get("producer_module") or "") == consumer
                for step in tool_steps[(producer_index + 1 if producer_index is not None else 0):]
            )
            consumed = any(
                isinstance(event, dict) and event.get("event") == "consumer_consumed"
                and event.get("status") == "ok" and event.get("contract_valid") is True
                and str(event.get("producer_module") or "") == producer
                # Runtime instrumentation emits ``consumer``.  Keep the old
                # spelling readable for historical probe artifacts, but never
                # mark a real current producer→consumer handoff as missing.
                and str(event.get("consumer") or event.get("consumer_module") or "") == consumer
                for event in events
            )
            handoff_checks.append({
                "producer": producer, "consumer": consumer,
                "required_fields": fields,
                "producer_observed": producer_index is not None,
                "missing_fields": missing,
                "payload_parseable": payload_parseable,
                "payload_truncated_or_unparseable": bool(producer_text) and not payload_parseable,
                "consumer_observed": consumer_seen or consumed,
                "consumption_event": consumed,
                "row_task_id": str(row.get("task_id") or ""),
            })
    joint_summary = []
    issues = list(base.get("issues") or [])
    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        producer, consumer = str(contract.get("producer") or ""), str(contract.get("consumer") or "")
        matched = [
            item for item in handoff_checks
            if item["producer"] == producer and item["consumer"] == consumer
        ]
        observed = [item for item in matched if item["producer_observed"]]
        valid = [
            item for item in observed
            if not item["missing_fields"] and item["consumer_observed"]
        ]
        summary = {
            "producer": producer, "consumer": consumer,
            "rows": len(matched), "producer_observed_rows": len(observed),
            "valid_consumed_rows": len(valid),
            "conditional_unobserved": not observed,
        }
        joint_summary.append(summary)
        # A conditional path may simply not be selected on this batch.  Only
        # diagnose a malformed handoff after its producer actually ran.
        if observed and not valid:
            issues.append({
                "code": "bundle_handoff_not_consumable",
                "severity": "critical",
                "count": len(observed),
                "rationales": [f"Triggered {producer}→{consumer} handoff lacked required fields or later consumption."],
                "repair_goal": "Return a parseable declared handoff and let the downstream module consume it through the existing runtime path.",
            })
    base.update({
        "schema_version": 1,
        "audit_type": "bundle_probe_behavior_audit",
        "target_modules": list(bundle.get("changed_modules") or []),
        "changed_modules": list(bundle.get("changed_modules") or []),
        "handoff_contracts": contracts,
        "per_module_runtime": per_module,
        "joint_handoff_summary": joint_summary,
        "joint_handoff_checks": handoff_checks,
        "issues": issues,
        "repair_policy": {
            **dict(base.get("repair_policy") or {}),
            "module_scope_locked": False,
            "module_scope": "Codex may change any related module within the five-module bundle.",
        },
    })
    return base


def _metric_count(value) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def probe_result_requires_codex_repair(verification: dict, probe_audit: dict = None) -> bool:
    """Return True only when probe exposed code/runtime breakage.

    Probe is a lightweight candidate screen. A stable no-gain or behavior-level
    risk is a round-level evolution signal, not evidence that Codex should keep
    rewriting the same candidate. Codex repair is reserved for implementation
    defects: invalid execution, runtime/API/tool-contract errors, or explicit
    engineering-invalid reports.
    """
    verification = verification or {}
    probe_audit = probe_audit or {}
    if probe_audit.get("execution_layer_rerun_required"):
        return False
    if verification.get("engineering_invalid") or probe_audit.get("engineering_invalid"):
        return True
    status = str(verification.get("probe_status") or probe_audit.get("probe_status") or "")
    if status == "invalid":
        return True
    if int(probe_audit.get("runtime_contract_error_rows") or 0) > 0:
        return True
    issue_codes = {
        str(item.get("code") or "")
        for item in (probe_audit.get("issues") or [])
        if isinstance(item, dict)
    }
    # A repeated target-tool loop is an executable dataflow/serialization
    # defect. It is not a no-gain signal for a new diagnosis: Codex must first
    # repair the same linked bundle so downstream modules can consume
    # its output.
    if "non_convergent_tool_loop" in issue_codes:
        return True
    if issue_codes & {
        "capability_not_executed", "interface_contract_failure",
        "runtime_tool_contract_failure", "output_not_consumed",
        "empty_evidence", "perception_not_reached",
        "capability_active_window_contract_failure", "capability_profile_mismatch",
    }:
        return True
    return False


def probe_result_requires_execution_rerun(verification: dict, probe_audit: dict = None) -> bool:
    """Provider/asset failures are infrastructure failures, never Codex feedback."""
    verification = verification or {}
    probe_audit = probe_audit or {}
    return bool(
        verification.get("execution_layer_rerun_required")
        or probe_audit.get("execution_layer_rerun_required")
    )


def probe_result_qualifies_for_full_eval(verification: dict,
                                         probe_audit: dict = None) -> bool:
    """Return whether a runnable probe has enough benefit for full eval.

    Probe is a training-stage quality gate: it requires a strict majority on
    the fixed batch as well as positive net benefit. The helper is shared by
    the router and same-direction feedback predicate so their decisions cannot
    drift.
    """
    verification = verification or {}
    probe_audit = probe_audit or {}
    delta = verification.get("accuracy_delta", probe_audit.get("accuracy_delta", 0))
    try:
        delta = float(delta or 0)
    except Exception:
        delta = 0.0
    corrections = _metric_count(verification.get("corrections"))
    if corrections == 0:
        corrections = _metric_count(probe_audit.get("corrections"))
    regressions = _metric_count(verification.get("regressions"))
    if regressions == 0:
        regressions = _metric_count(probe_audit.get("regressions"))
    # A numerical gain caused solely by the candidate's declared decision
    # mechanism falling back is not evidence that the mechanism works.  This
    # is not an MFP hard gate: it applies only when the candidate itself emits
    # a structured decision signal on every observed row and every one reports
    # fallback.  A mechanism that actually makes non-fallback decisions may
    # still advance even when its representative MFP is unresolved.
    mechanism_execution = probe_audit.get("mechanism_execution") or {}
    mechanism_never_activated = bool(
        isinstance(mechanism_execution, dict)
        and mechanism_execution.get("all_observed_decisions_fell_back")
    )
    replay = verification.get("reference_probe_replay") or {}
    repeatability = replay.get("repeatability") if isinstance(replay, dict) else {}
    repeatable = bool(isinstance(repeatability, dict) and repeatability.get("passed"))
    total = int(verification.get("total_questions") or probe_audit.get("total_questions") or 0)
    candidate_correct = int(
        verification.get("candidate_correct") or 0
    )
    # Probe is a training-stage quality gate, not merely a direction signal:
    # full evaluation is worthwhile only after the candidate answers a strict
    # majority of the fixed probe batch correctly (4/7 for the current setup).
    majority_correct = total > 0 and candidate_correct > (total / 2.0)
    return repeatable and (not mechanism_never_activated) and majority_correct and delta > 0 and corrections >= 1 and (
        regressions == 0 or corrections > regressions
    )


def probe_result_requires_same_direction_behavior_iteration(verification: dict,
                                                            probe_audit: dict = None) -> bool:
    """Return True for runnable probes that need another Codex iteration.

    This route is deliberately based on aggregate behavior, not individual
    MFP rows. It keeps the same diagnosis, research handoff and linked direction
    fixed, while giving Codex the complete candidate/reference training
    trajectories, source snapshots, capability events and gold labels inside
    an immutable supervised forensic workspace.  This prevents a negative or
    no-gain probe from needlessly losing a useful concrete design in a fresh
    diagnosis round.
    """
    verification = verification or {}
    probe_audit = probe_audit or {}
    if not verification and not probe_audit:
        return False
    if not (
        verification.get("probe_status")
        or probe_audit.get("probe_status")
        or "accuracy_delta" in verification
        or "accuracy_delta" in probe_audit
    ):
        # An absent/incomplete audit is not behavioral no-gain evidence.  It
        # must be diagnosed as an artifact-routing problem rather than causing
        # Codex to rewrite a runnable candidate without a real observation.
        return False
    return not (
        probe_result_requires_execution_rerun(verification, probe_audit)
        or probe_result_requires_codex_repair(verification, probe_audit)
        or probe_result_qualifies_for_full_eval(verification, probe_audit)
    )



def classify_probe_next_action(verification: dict, probe_audit: dict = None) -> dict:
    """Classify the next step after a probe without pretending probe is final eval."""
    verification = verification or {}
    probe_audit = probe_audit or {}
    if probe_result_requires_execution_rerun(verification, probe_audit):
        return {
            "action": "execution_layer_rerun",
            "reason": "probe encountered provider/asset availability failure; repair execution environment and rerun the same candidate",
            "codex_feedback_allowed": False,
        }
    if probe_result_requires_codex_repair(verification, probe_audit):
        return {
            "action": "codex_engineering_repair",
            "reason": "probe exposed runtime/interface/API/tool-contract invalidity",
            "codex_feedback_allowed": True,
        }
    if probe_result_qualifies_for_full_eval(verification, probe_audit):
        regressions = _metric_count(verification.get("regressions"))
        if regressions == 0:
            regressions = _metric_count(probe_audit.get("regressions"))
        return {
            "action": "full_eval_candidate",
            "reason": (
                "probe found net-positive behavior; full eval must decide final "
                "validity" if regressions == 0 else
                "probe is net-positive despite regression risk; full eval must decide final validity"
            ),
            "codex_feedback_allowed": False,
            "regression_risk": bool(regressions),
            "requires_full_eval": True,
        }
    if probe_result_requires_same_direction_behavior_iteration(verification, probe_audit):
        return {
            "action": "codex_same_direction_behavior_iteration",
            "reason": (
                "probe was runnable but did not show enough aggregate benefit; "
                "return complete supervised forensic evidence to Codex for a bounded "
                "same-direction bundle iteration"
            ),
            "codex_feedback_allowed": True,
            "regression_risk": _metric_count(verification.get("regressions")) > 0,
        }
    return {
        "action": "round_level_rediagnosis",
        "reason": "probe routing could not be classified; require a new diagnosis audit",
        "codex_feedback_allowed": False,
    }


def _find_upward_file(start_dir: str, filename: str, max_levels: int = 6) -> str:
    cur = _abs(start_dir)
    for _ in range(max_levels + 1):
        candidate = os.path.join(cur, filename)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return ""


def _load_probe_verification_from_run(run_dir: str) -> dict:
    verification = _load_json_optional(os.path.join(run_dir, "probe_verification.json"))
    if verification:
        return verification
    sandbox = _load_json_optional(os.path.join(run_dir, "sandbox_report.json"))
    if not sandbox:
        return {}
    return (
        sandbox.get("verification")
        or sandbox.get("sandbox")
        or (sandbox.get("conclusion") or {}).get("verification")
        or {}
    )


def _sandbox_dir_from_probe_run(run_dir: str, verification: dict) -> str:
    candidates = [
        verification.get("sandbox_dir", "") if isinstance(verification, dict) else "",
        os.path.join(run_dir, "sandbox"),
        os.path.join(run_dir, "sandbox"),
    ]
    sandbox_report = _load_json_optional(os.path.join(run_dir, "sandbox_report.json"))
    if sandbox_report:
        nested = sandbox_report.get("verification") or sandbox_report.get("sandbox") or {}
        if isinstance(nested, dict):
            candidates.insert(0, nested.get("sandbox_dir", ""))
    for item in candidates:
        if item and os.path.exists(_abs(item)):
            return _abs(item)
    return ""


def _probe_row_brief(row: dict, classification: str = "") -> dict:
    answer = row.get("answer") or row.get("final_agent_answer") or ""
    question = str(row.get("question", "") or "")
    gt_answer = str(row.get("gt_answer") or row.get("ground_truth") or "")
    return {
        "classification": classification,
        "task_ref": "<redacted_probe_task>",
        "question_present": bool(question.strip()),
        "question_chars": len(question),
        "gt_answer_present": bool(gt_answer.strip()),
        "agent_answer_present": bool(str(answer or "").strip()),
        "agent_answer_chars": len(str(answer or "")),
        "is_correct": row.get("is_correct"),
        "steps": _trajectory_len(row),
        "tool_counts": _tool_counts(row),
        "runtime_contract_errors": _structured_runtime_contract_errors(row),
        "frames_viewed": _frames_viewed(row),
        "llm_calls": _llm_calls(row),
        "answer_shape_issue": _answer_shape_issue(str(answer)),
    }


def _classification_sets(verification: dict) -> dict:
    verification = verification or {}

    def ids(key: str) -> set:
        values = verification.get(key) or []
        out = set()
        for item in values:
            if isinstance(item, dict):
                value = item.get("task_id") or item.get("ref") or item.get("question_id")
            else:
                value = item
            if value:
                out.add(str(value))
        return out

    return {
        "correction": ids("corrections"),
        "regression": ids("regressions"),
    }


def build_candidate_rejection_context(probe_run_dir: str, *, max_rows: int = 12) -> dict:
    """Summarize a runnable-but-rejected candidate for the next round.

    This is not a probe repair prompt.  It is compact evidence that a previous
    candidate should not be repeated blindly.  It supports both candidate run
    directories and standalone probe rerun directories nested under a candidate.
    """
    run_dir = _abs(probe_run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"candidate rejection run is not a directory: {probe_run_dir}")

    bundle_path = _find_upward_file(run_dir, "candidate_bundle.json")
    candidate_bundle = _load_json_optional(bundle_path) if bundle_path else {}
    if not candidate_bundle:
        raise FileNotFoundError(f"candidate bundle not found in run ancestry: {probe_run_dir}")
    changed = list(candidate_bundle.get("changed_modules") or [])
    verification = _load_probe_verification_from_run(run_dir)
    if not verification:
        parent_verification = _find_upward_file(run_dir, "probe_verification.json")
        verification = _load_json_optional(parent_verification) if parent_verification else {}
    audit = _load_json_optional(os.path.join(run_dir, "probe_audit.json"))
    if not audit:
        audit = build_bundle_probe_audit(verification, candidate_bundle)
    sandbox_dir = _sandbox_dir_from_probe_run(run_dir, verification)
    rows = _iter_jsonl_rows(sandbox_dir)
    raw_trajectory_files = _sandbox_jsonl_files(sandbox_dir)
    valid_rows = [row for row in rows if row.get("question") or row.get("task_id")]
    class_sets = _classification_sets(verification)
    row_briefs = []
    for row in rows:
        task_id = str(row.get("task_id", ""))
        label = "probe_row"
        if task_id in class_sets["regression"]:
            label = "regression"
        elif task_id in class_sets["correction"]:
            label = "correction"
        elif row.get("is_correct") is False:
            label = "unchanged_or_failed"
        row_briefs.append(_probe_row_brief(row, label))

    row_briefs = sorted(
        row_briefs,
        key=lambda r: (
            {"regression": 0, "unchanged_or_failed": 1, "correction": 2, "probe_row": 3}.get(
                r.get("classification"), 9
            ),
            -int(r.get("steps") or 0),
        ),
    )[:max_rows]
    engineering_repairable = probe_result_requires_codex_repair(verification, audit)
    behavior_iterable = probe_result_requires_same_direction_behavior_iteration(verification, audit)
    next_action = classify_probe_next_action(verification, audit)
    next_action_name = next_action.get("action", "")
    rejection_type = (
        "engineering_invalid"
        if engineering_repairable
        else "same_direction_behavior_iteration"
        if behavior_iterable
        else "not_rejected_full_eval_candidate"
        if next_action_name == "full_eval_candidate"
        else "behavior_rejected"
    )
    corrections = _metric_count(verification.get("corrections"))
    if corrections == 0:
        corrections = _metric_count(audit.get("corrections"))
    regressions = _metric_count(verification.get("regressions"))
    if regressions == 0:
        regressions = _metric_count(audit.get("regressions"))
    delta = verification.get("accuracy_delta")
    structure_eval_modes = verification.get("structure_eval_modes") or {}
    bundle_changes_struct = "video_structuring" in (candidate_bundle.get("changed_modules") or [])
    structure_semantics_valid = not (
        int(structure_eval_modes.get("full_rebuild", 0) or 0) > 0
        and not bundle_changes_struct
    )
    summary = []
    if engineering_repairable:
        summary.append("Probe exposed runtime/interface invalidity; linked-bundle engineering repair is allowed.")
    elif behavior_iterable:
        summary.append("Probe was runnable but did not earn full evaluation; bounded same-direction Codex behavior iteration is allowed.")
    elif next_action_name == "full_eval_candidate":
        summary.append("Probe found net-positive behavior; regression risk is recorded, but full evaluation should decide final validity.")
    else:
        summary.append("Probe outcome could not be routed as a valid same-direction iteration; use a new diagnosis audit.")
    if regressions:
        summary.append(f"Probe found {regressions} regression(s) against the explicit reference/current-best combo.")
    if corrections == 0:
        summary.append("Probe found no correction on selected repair opportunities.")
    if delta is not None:
        summary.append(f"Probe accuracy_delta={delta}.")
    if not structure_semantics_valid:
        summary.append(
            "Probe structure provenance was invalid for a non-structuring candidate; "
            "treat its behavior metrics as infrastructure-contaminated evidence."
        )

    return {
        "schema_version": 1,
        "context_type": "candidate_rejection_context",
        "source_probe_run": run_dir,
        "candidate_run": os.path.dirname(bundle_path),
        "candidate_bundle_path": bundle_path,
        "sandbox_dir": sandbox_dir,
        # These are source files, not a compact audit representation.  The
        # feedback-resume entrypoint copies every byte into its own immutable
        # run directory before Codex is invoked.
        "raw_training_trajectory_files": raw_trajectory_files,
        "raw_training_trajectory_row_count": len(valid_rows),
        "target_modules": changed,
        "candidate": {
            "name": "MetaVideoAgent bundle",
            "changed_modules": changed,
            "bundle_fingerprint": candidate_bundle.get("bundle_fingerprint", ""),
            "mechanism_signature": candidate_bundle.get("bundle_fingerprint", ""),
        },
        "decision": {
            "rejection_type": rejection_type,
            "should_trigger_codex_engineering_repair": engineering_repairable,
            "should_trigger_codex_behavior_iteration": behavior_iterable,
            "probe_next_action": next_action,
            "use_in_next_codex_prompt": True,
            "round_level_signal": (
                not engineering_repairable
                and not behavior_iterable
                and next_action_name != "full_eval_candidate"
            ),
            "candidate_should_enter_full_eval": next_action_name == "full_eval_candidate",
            "probe_structure_semantics_valid": structure_semantics_valid,
            "do_not_overfit_probe_rows": True,
            "do_not_repeat_rejected_mechanism": (
                not engineering_repairable
                and not behavior_iterable
                and next_action_name != "full_eval_candidate"
            ),
            "mechanism_repair_budget": mechanism_repair_budget(),
        },
        "probe_metrics": {
            "probe_status": verification.get("probe_status") or audit.get("probe_status"),
            "accepted": verification.get("accepted"),
            "total_questions": verification.get("total_questions") or audit.get("total_questions"),
            "reference_correct": verification.get("reference_correct"),
            "evolved_correct": verification.get("candidate_correct"),
            "accuracy_delta": delta,
            "corrections": corrections,
            "regressions": regressions,
            "engineering_invalid": bool(
                verification.get("engineering_invalid") or audit.get("engineering_invalid")
            ),
            "runtime_contract_error_rows": int(audit.get("runtime_contract_error_rows") or 0),
            "structure_eval_modes": structure_eval_modes,
            "structure_semantics_valid": structure_semantics_valid,
            "mechanism_witness_outcome": (audit.get("mechanism_witness") or {}).get("outcome", "not_configured"),
        },
        "probe_audit_summary": {
            "issues": (audit.get("issues") or [])[:8],
            "probe_selection_note": audit.get("probe_selection_note", ""),
            "selected_repair_refs_count": len(audit.get("selected_repair_refs") or []),
            "selected_guard_refs_count": len(audit.get("selected_guard_refs") or []),
        },
        "probe_row_briefs": row_briefs,
        "summary": summary,
        "instructions_for_next_codex": [
            "Use this as negative evidence about a previous candidate, not as a request to patch exact probe answers.",
            "If rejection_type=same_direction_behavior_iteration, inspect the bundle mechanism and make a bounded general improvement using aggregate failure classes only.",
            "If rejection_type=not_rejected_full_eval_candidate, do not treat this as a rejected candidate; run full evaluation or use it only as a risk note.",
            "If rejection_type=behavior_rejected, produce a materially different mechanism or a more disciplined version justified by diagnosis and research.",
            "Keep the linked evolution direction fixed unless a new diagnosis changes it.",
            "Do not hard-code task ids, video ids, answers, options, or time ranges from probe rows.",
        ],
    }


def write_candidate_rejection_context(probe_run_dir: str, output_path: str = "") -> str:
    context = build_candidate_rejection_context(probe_run_dir)
    if not output_path:
        output_path = os.path.join(_abs(probe_run_dir), "candidate_rejection_context.json")
    write_json(_abs(output_path), context)
    return _abs(output_path)







def mechanism_repair_budget() -> int:
    """Return the bounded same-mechanism repair allowance.

    Malformed environment overrides must never disable the loop guard.
    """
    try:
        configured = int(os.environ.get("MFP_MECHANISM_REPAIR_BUDGET", "2") or 2)
    except (TypeError, ValueError):
        configured = 2
    return max(1, configured)





def _history_dir(workspace_dir: str, output_root: str = "") -> str:
    del workspace_dir
    if output_root:
        return os.path.abspath(output_root)
    return os.path.join(runtime_output_root(), "codex_reports")



def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)



def _copy_complete_probe_trajectory_artifacts(source_path: str, run_dir: str) -> list[dict]:
    """Copy raw sandbox JSONL files unchanged for a Codex feedback run.

    Do not reconstruct these files from ``build_probe_audit``: an audit has
    already made analytical choices, while Codex must be able to inspect the
    entire original execution record.  A missing raw source is fail-closed so
    a feedback run cannot silently regress to clipped summaries.
    """
    source_files = _sandbox_jsonl_files(source_path)
    if not source_files:
        raise RuntimeError(
            "probe feedback requires complete raw sandbox trajectory JSONL artifacts; "
            f"none found under {source_path!r}"
        )
    destination_dir = os.path.join(run_dir, "probe_training_trajectories")
    os.makedirs(destination_dir, exist_ok=True)
    manifest = []
    for index, source in enumerate(source_files, start=1):
        # Prefixing prevents a malformed sandbox layout with duplicate basenames
        # from overwriting any trajectory while retaining the original bytes.
        destination = os.path.join(destination_dir, f"{index:03d}_{os.path.basename(source)}")
        shutil.copy2(source, destination)
        digest_state = hashlib.sha256()
        with open(destination, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest_state.update(chunk)
        digest = digest_state.hexdigest()
        with open(destination, "rb") as handle:
            line_count = sum(1 for _ in handle)
        manifest.append({
            "source_path": os.path.abspath(source),
            "path": os.path.abspath(destination),
            "bytes": os.path.getsize(destination),
            "line_count": line_count,
            "sha256": digest,
            "copied_byte_for_byte": True,
        })
    write_json(os.path.join(run_dir, "probe_training_trajectories_manifest.json"), {
        "schema_version": 1,
        "source": os.path.abspath(source_path),
        "artifacts": manifest,
        "copy_policy": "all sandbox JSONL files copied byte-for-byte without row filtering or field trimming",
    })
    return manifest


def _existing_paths(*paths: str) -> list[str]:
    """Return existing paths only, preserving order and avoiding duplicates."""
    result = []
    for path in paths:
        raw = str(path or "").strip()
        if not raw:
            continue
        absolute = _abs(raw)
        if absolute and os.path.exists(absolute) and absolute not in result:
            result.append(absolute)
    return result


def _materialize_bundle_source_snapshot(bundle_path: str, destination: str) -> str:
    """Materialize the immutable current-best bundle code as a real source tree."""
    bundle = _load_json_optional(bundle_path)
    modules = bundle.get("modules") if isinstance(bundle, dict) else None
    records = list(modules.values()) if isinstance(modules, dict) else modules
    if not isinstance(records, (list, tuple)):
        return ""
    root = _abs(destination)
    wrote = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        rel = str(record.get("source_file") or "").replace("\\", "/").lstrip("/")
        code = str(record.get("code") or "")
        if not rel or not code or ".." in rel.split("/"):
            continue
        target = os.path.join(root, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        write_text(target, code)
        wrote += 1
    return root if wrote else ""


def _materialize_bundle_probe_forensics(
    *,
    run_dir: str,
    attempt_tag: str,
    parent_dir: str,
    parent_after: str,
    raw_candidate_trajectories: str,
    diagnosis_report: dict,
    parent_verification: dict,
    parent_audit: dict,
    machine_contract_path: str,
) -> dict:
    """Make the supervised, immutable evidence set for one bundle retry.

    The feedback child sees the *whole* previous candidate, probe traces and
    current-best/reference artifacts.  No audit-derived row selection occurs
    here: the audit is merely an additional navigational artifact.  The caller
    creates one fresh workspace per Codex attempt so an earlier report can
    never be mistaken for the current attempt's forensic analysis.
    """
    root = os.path.join(run_dir, "forensics", f"attempt_{attempt_tag}")
    reference_results = str(
        parent_verification.get("persisted_reference_minimal_results_path")
        or parent_verification.get("persisted_reference_results_path")
        or diagnosis_report.get("reference_results_path") or ""
    )
    replay = parent_verification.get("reference_probe_replay") if isinstance(parent_verification, dict) else {}
    replay = replay if isinstance(replay, dict) else {}
    candidate_replay = str(
        parent_verification.get("candidate_probe_results_path")
        or replay.get("candidate_replay_path") or ""
    )
    reference_replay = str(replay.get("reference_replay_path") or "")
    current_best_source = _materialize_bundle_source_snapshot(
        str(diagnosis_report.get("reference_bundle_path") or ""),
        os.path.join(run_dir, "forensics_current_best_bundle_source"),
    )
    parent_source_before = os.path.join(parent_dir, "source_before")
    reference_artifacts = _existing_paths(
        os.path.join(parent_dir, "probe_verification.json"),
        os.path.join(parent_dir, "probe_audit.json"),
        str(diagnosis_report.get("reference_report_path") or ""),
        str(diagnosis_report.get("reference_bundle_path") or ""),
        machine_contract_path,
        parent_source_before if os.path.isdir(parent_source_before) else "",
    )
    task_manifests = _existing_paths(str(diagnosis_report.get("distribution_manifest") or ""))
    manifest = materialize_supervised_forensic_workspace(
        root,
        candidate_source_dir=parent_after,
        reference_source_dir=current_best_source if os.path.isdir(current_best_source) else "",
        candidate_trajectory_paths=[candidate_replay or raw_candidate_trajectories],
        reference_trajectory_paths=[reference_replay or reference_results] if (reference_replay or reference_results) else [],
        reference_artifact_paths=reference_artifacts,
        task_manifest_paths=task_manifests,
        metadata={
            "mode": "bundle_probe_same_direction_self_debug",
            "parent_run": _abs(parent_dir),
            "parent_bundle_fingerprint": str(
                (_load_json_optional(os.path.join(parent_dir, "candidate_bundle.json")) or {}).get(
                    "bundle_fingerprint", ""
                )
            ),
            "probe_verification_path": _abs(os.path.join(parent_dir, "probe_verification.json")),
            "probe_audit_path": _abs(os.path.join(parent_dir, "probe_audit.json")),
            "declared_probe_total": int(parent_verification.get("total_questions") or 0),
            "declared_candidate_correct": int(
                parent_verification.get("candidate_correct") or 0
            ),
            "parent_audit_type": str(parent_audit.get("audit_type") or ""),
            "candidate_probe_replay_path": _abs(candidate_replay) if candidate_replay else "",
            "reference_probe_replay_path": _abs(reference_replay) if reference_replay else "",
            "probe_comparison_mode": str(parent_verification.get("probe_comparison_mode") or ""),
            "complete_trace_analysis_required": bool(
                parent_verification.get("probe_comparison_mode")
                == "persisted_current_best_minimal_problem"
            ),
            "current_best_source_policy": "materialized directly from immutable reference_bundle modules",
            "parent_source_before": _abs(parent_source_before) if os.path.isdir(parent_source_before) else "",
        },
    )
    return manifest


def _apply_source_snapshot(src_dir: str) -> None:
    """Apply a recorded source_after snapshot back to the workspace."""
    if not os.path.isdir(src_dir):
        return
    for root, _dirs, files in os.walk(src_dir):
        for name in files:
            src = os.path.join(root, name)
            rel = os.path.relpath(src, src_dir)
            dst = _abs(rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def write_final_summary(run_dir: str, report: dict, conclusion: dict = None) -> None:
    candidate = report.get("candidate_bundle") or {}
    sandbox = report.get("sandbox") or {}
    verification = sandbox.get("verification") or {}
    conclusion = conclusion or sandbox.get("conclusion") or {}
    next_action = report.get("probe_next_action") or {}
    lines = [
        "# Codex Evolution Summary",
        "",
        f"- target_modules: {', '.join(report.get('target_modules', []) or [])}",
        f"- changed_modules: {', '.join(candidate.get('changed_modules', []) or [])}",
        f"- status: {report.get('status', '')}",
        f"- source_restored: {report.get('source_restored', '')}",
        f"- output_file: {report.get('output_file', '')}",
        f"- candidate_rounds: {report.get('candidate_rounds', '')}",
        f"- probe_feedback_rounds_requested: {report.get('probe_feedback_rounds_requested', '')}",
        f"- probe_feedback_rounds_run: {len(report.get('probe_feedback_rounds', []) or [])}",
        f"- probe_status: {verification.get('probe_status', '')}",
        f"- verification_scope: {verification.get('verification_scope', '')}",
        f"- accuracy_delta: {verification.get('accuracy_delta', '')}",
        f"- corrections: {len(verification.get('corrections', []) or [])}",
        f"- regressions: {len(verification.get('regressions', []) or [])}",
        f"- full_verification: {(verification.get('full_verification') or {}).get('status', '')}",
        f"- probe_next_action: {next_action.get('action', '')}",
        f"- probe_next_action_reason: {next_action.get('reason', '')}",
        f"- verdict: {conclusion.get('verdict', '')}",
        f"- verdict_reason: {conclusion.get('verdict_reason', '')}",
    ]
    write_text(os.path.join(run_dir, "final_summary.md"), "\n".join(lines) + "\n")


def _ensure_source_record_files(run_dir: str) -> None:
    os.makedirs(os.path.join(run_dir, "source_before"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "source_after"), exist_ok=True)
    diff_path = os.path.join(run_dir, "source_diff.patch")
    if not os.path.exists(diff_path):
        write_text(diff_path, "")


def build_record_manifest(run_dir: str, report: dict) -> dict:
    """Check run_dir record completeness without using git."""
    _ensure_source_record_files(run_dir)
    required_files = [
        "diagnosis.json",
        "diagnosis_rollout.json",
        "prompt.txt",
        "source_diff.patch",
        "validation.json",
        "sandbox_report.json",
        "final_summary.md",
    ]
    required_dirs = ["source_before", "source_after"]
    if report.get("candidate_bundle") or report.get("status") == "completed":
        required_files.append("candidate_bundle.json")
    missing_files = [
        name for name in required_files
        if not os.path.exists(os.path.join(run_dir, name))
    ]
    missing_dirs = [
        name for name in required_dirs
        if not os.path.isdir(os.path.join(run_dir, name))
    ]
    sandbox_dir = (
        ((report.get("sandbox") or {}).get("verification") or {})
        .get("sandbox_dir", "")
    )
    sandbox_inside_run_dir = (
        not sandbox_dir
        or os.path.abspath(sandbox_dir).startswith(os.path.abspath(run_dir) + os.sep)
    )
    return {
        "schema_version": 1,
        "run_dir": run_dir,
        "status": report.get("status", ""),
        "required_files": required_files,
        "required_dirs": required_dirs,
        "missing_files": missing_files,
        "missing_dirs": missing_dirs,
        "sandbox_dir": sandbox_dir,
        "sandbox_inside_run_dir": sandbox_inside_run_dir,
        "uses_git_for_recording": False,
        "record_complete": (
            not missing_files and not missing_dirs and sandbox_inside_run_dir
        ),
        "notes": [
            "source_diff.patch is produced with difflib, not git",
            "sandbox/full-eval outputs must remain under the candidate run_dir",
        ],
    }


def finalize_run_record(run_dir: str, report: dict,
                        conclusion: dict = None) -> dict:
    """Write final_summary.md and record_manifest.json for every exit path."""
    _ensure_source_record_files(run_dir)
    write_final_summary(run_dir, report, conclusion)
    manifest = build_record_manifest(run_dir, report)
    write_json(os.path.join(run_dir, "record_manifest.json"), manifest)
    report["record_manifest"] = manifest
    return manifest






def _bundle_edit_files() -> list[str]:
    files = []
    for module in BUNDLE_MODULE_TYPES:
        for path in MODULE_SOURCE_FILES.get(module, []) or []:
            if path not in files:
                files.append(path)
    return files


def _bundle_module_records(bundle: dict) -> dict[str, dict]:
    """Return the canonical five module records from a self-contained bundle.

    Initial bundles use a list while evolved bundles use a mapping. Keeping
    the conversion here makes the source origin explicit and
    prevents a later codegen attempt from accidentally inheriting whatever
    candidate classes happen to be left in the repository checkout.
    """
    raw = (bundle or {}).get("modules") or {}
    if isinstance(raw, dict):
        return {
            module: dict(raw.get(module) or {})
            for module in BUNDLE_MODULE_TYPES
            if isinstance(raw.get(module), dict)
        }
    records = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            module = str(item.get("module_type") or "")
            if module in BUNDLE_MODULE_TYPES:
                records[module] = dict(item)
    return records


def _materialize_bundle_source_workspace(bundle: dict, edit_files: list[str]) -> dict:
    """Validate and attest a bundle without overwriting runtime ABI sources.

    Bundle code is passed to Codex through the immutable codegen context and
    is dynamically injected by the evaluator.  Writing it into files such as
    ``memory_modules.py`` replaces the base class imported by that very code,
    creating a circular import before any smoke input can execute.
    """
    records = _bundle_module_records(bundle)
    missing = [module for module in BUNDLE_MODULE_TYPES if not records.get(module)]
    if missing:
        raise RuntimeError(
            "cannot materialize immutable bundle source: missing module records="
            + ", ".join(missing)
        )
    manifest = {"source_policy": "immutable_reference_bundle", "modules": {}}
    for module in BUNDLE_MODULE_TYPES:
        record = records[module]
        code = str(record.get("code") or "")
        paths = [path for path in (MODULE_SOURCE_FILES.get(module) or []) if path in edit_files]
        if not code or not paths:
            raise RuntimeError(
                f"cannot materialize immutable bundle source for {module}: code/path missing"
            )
        manifest["modules"][module] = {
            "name": str(record.get("name") or ""),
            "source_file": paths[0],
            "source_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        }
    return manifest


def _class_change_candidates(filepath: str, original_content: str) -> list[dict]:
    """Describe changed classes without conflating support classes and strategies."""
    if not os.path.isfile(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            current_content = handle.read()
        current_tree = ast.parse(current_content)
        original_tree = ast.parse(original_content)
    except (OSError, SyntaxError):
        return []

    def classes(tree, content):
        lines = content.splitlines()
        result = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            end = node.end_lineno or len(lines)
            result[node.name] = "\n".join(lines[node.lineno - 1:end])
        return result

    before = classes(original_tree, original_content)
    after = classes(current_tree, current_content)
    candidates = []
    for name, code in after.items():
        prior = before.get(name)
        if prior == code:
            continue
        is_new = prior is None
        delta_lines = len(set(code.splitlines()).symmetric_difference(set((prior or "").splitlines())))
        candidates.append({
            "name": name,
            "code": code,
            "is_new": is_new,
            "delta_lines": delta_lines,
            # The neutral module ABI consistently names infrastructure roots
            # ``*Base``.  They are never runtime strategies by themselves.
            "is_infrastructure": name.endswith("Base"),
        })
    return candidates


def _selected_concrete_change(filepath: str, original_content: str) -> tuple[dict | None, list[dict]]:
    """Select a changed concrete class; never promote a support/base class."""
    candidates = _class_change_candidates(filepath, original_content)
    concrete = [item for item in candidates if not item["is_infrastructure"]]
    if not concrete:
        return None, candidates
    concrete.sort(key=lambda item: (bool(item["is_new"]), int(item["delta_lines"]), item["name"]), reverse=True)
    return concrete[0], candidates


def _bundle_context_artifact(artifact_id: str, title: str, purpose: str,
                             content, content_kind: str = "text") -> dict:
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, indent=2)
    return {
        "artifact_id": artifact_id,
        "title": title,
        "purpose": purpose,
        "content_kind": content_kind,
        "content": content,
        "chars": len(content),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def _read_codegen_context_file(relative_path: str, *, required: bool = False) -> str:
    try:
        with open(_abs(relative_path), "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        if required:
            raise RuntimeError(
                f"required bundle-codegen context artifact is missing: {relative_path}"
            ) from exc
        return ""


def _assert_bundle_codegen_context(artifacts: list[dict], *, context_name: str) -> None:
    """Fail closed instead of truncating a later-round code-writing packet."""
    try:
        limit = int(os.environ.get("CODEX_MAX_CONTEXT_CHARS", str(MAX_BUNDLE_CODEX_CONTEXT_CHARS)))
    except ValueError as exc:
        raise RuntimeError("CODEX_MAX_CONTEXT_CHARS must be an integer") from exc
    total = sum(int((item or {}).get("chars", 0) or 0) for item in artifacts if isinstance(item, dict))
    if total > limit:
        raise RuntimeError(
            f"{context_name} contains {total} characters, exceeding "
            f"CODEX_MAX_CONTEXT_CHARS={limit}; refusing to truncate Codex repair evidence"
        )


def _validate_bundle_context_source(records: dict[str, dict], combo: dict) -> None:
    """Ensure an annotated context really names five executable source records."""
    issues = []
    for module in BUNDLE_MODULE_TYPES:
        record = records.get(module) or {}
        selected = str(combo.get(BUNDLE_COMBO_SLOT[module]) or "")
        if not selected:
            issues.append(f"combo.{BUNDLE_COMBO_SLOT[module]} is empty")
            continue
        if str(record.get("name") or "") != selected:
            issues.append(f"{module} selected class does not match its source record")
        for field in ("source_file", "code"):
            if not str(record.get(field) or ""):
                issues.append(f"{module}.{field} is empty")
    if issues:
        raise RuntimeError("invalid immutable bundle codegen source: " + "; ".join(issues))


def build_bundle_codegen_context(diagnosis_report: dict, bundle: dict,
                                 edit_files: list[str], *,
                                 bundle_role: str = "current_best") -> dict:
    """Build the later-round equivalent of initial codegen's artifact packet.

    Every small, authoritative code-writing input is embedded and annotated.
    This avoids asking Codex to infer ABI/tool behavior from broad diagnosis
    prose or stale files in the repository.  Large supervised probe JSONL is
    intentionally kept in its forensic workspace and is linked separately by
    the probe-feedback prompt.
    """
    records = _bundle_module_records(bundle)
    combo = dict((bundle or {}).get("combo") or diagnosis_report.get("reference_combo") or {})
    _validate_bundle_context_source(records, combo)
    registry = {slot: [] for slot in ("localization", "perception")}
    contracts = {slot: {} for slot in ("localization", "perception")}
    for slot in registry:
        record = records.get(slot) or {}
        selected = str(combo.get(slot) or "")
        if selected and selected == str(record.get("name") or ""):
            registry[slot] = _tool_methods_from_code(str(record.get("code") or ""), selected)
            contracts[slot] = _tool_contracts_from_code(str(record.get("code") or ""), selected)
    source_records = []
    for module in BUNDLE_MODULE_TYPES:
        record = records.get(module) or {}
        source_records.append({
            "module_type": module,
            "selected_class": str(combo.get(BUNDLE_COMBO_SLOT[module]) or ""),
            "record_name": str(record.get("name") or ""),
            "source_file": str(record.get("source_file") or ""),
            "class_role": str(record.get("class_role") or "concrete_strategy"),
            "code_sha256": hashlib.sha256(str(record.get("code") or "").encode("utf-8")).hexdigest(),
        })
    task = diagnosis_report.get("implementation_task") or {}
    research = diagnosis_report.get("codex_deep_research") or {}
    artifacts = [
        _bundle_context_artifact(
            "current_bundle", (
                "Failed smoke candidate five-module bundle"
                if bundle_role == "failed_smoke_candidate" else "Current-best five-module bundle"
            ),
            (
                "Authoritative failed candidate combo, class records, and source code under repair."
                if bundle_role == "failed_smoke_candidate" else
                "Authoritative active combo, class records, and source code from which this evolution starts."
            ),
            {"combo": combo, "modules": records, "source_records": source_records}, "json",
        ),
        _bundle_context_artifact(
            "runtime_integration_witness", "Runtime integration witness",
            "Exact dispatch and injection facts. Tool names and parameter contracts are derived from the selected current classes, not from diagnosis prose.",
            {
                "combo": combo,
                "selected_module_records": source_records,
                "tool_registry": registry,
                "tool_parameter_contracts": contracts,
                "agent_dispatch": {
                    "plan_field": "target_tool",
                    "dispatch_rule": "Agent resolves target_tool by getattr(localization, 'tool_'+name) then perception; tool_params are validated against inspect.signature before invocation.",
                    "result_rule": "localization/perception return current module-protocol envelopes; memory stores raw request/result events; thinking receives normalized runtime context.",
                },
                "candidate_extraction_rule": "Only imports, module constants, base-class chain, and the selected concrete class are serialized. Keep candidate dependencies inside the selected class, its base chain, imports, or module constants; do not depend on arbitrary top-level helper functions or unselected classes.",
            }, "json",
        ),
        _bundle_context_artifact(
            "five_module_interface_contract", "Five-module interface contract",
            "Binding responsibilities, inputs, outputs, and fallback boundaries for all five modules.",
            _read_codegen_context_file("evolution/agent_docs/module_interface_contract.md", required=True), "text",
        ),
        _bundle_context_artifact(
            "generated_runtime_abi_witness", "Generated runtime ABI witness",
            "Binding constructor, method, return-shape, and dynamic injection facts.",
            _read_codegen_context_file("evolution/agent_docs/generated_runtime_abi_witness.json", required=True), "json",
        ),
        _bundle_context_artifact(
            "runtime_module_protocol", "Runtime module protocol",
            "Actual protocol envelopes and normalization helpers used by the execution layer.",
            _read_codegen_context_file(os.path.relpath(os.path.join(runtime_dir(), "module_protocol.py"), PROJECT_ROOT), required=True), "text",
        ),
        _bundle_context_artifact(
            "runtime_capability_manual", "Runtime capability manual",
            "Only supported runtime adapters, call shapes, failure behavior, and model usage rules.",
            _read_codegen_context_file("evolution/agent_docs/runtime_capability_manual.md", required=True), "text",
        ),
        _bundle_context_artifact(
            "runtime_capability_profiles", "Registered runtime capability profiles",
            "Literal profile IDs, transport, provenance, and provider policy.",
            _read_codegen_context_file(str(PROFILE_MANUAL_PATH), required=True), "json",
        ),
        _bundle_context_artifact(
            "evolution_direction", "Diagnosis and implementation direction",
            "Binding outer-round mechanism, module responsibilities, handoff contracts, and constraints; it sets direction but not an edit whitelist.",
            {"execution_policy": diagnosis_report.get("execution_policy") or {}, "implementation_task": task}, "json",
        ),
        _bundle_context_artifact(
            "deep_research_handoff", "Validated Codex deep-research handoff",
            "Implementation evidence supporting the diagnosis direction; cannot override runtime ABI or current-bundle facts.",
            {"research_complete": research.get("research_complete"), "validation_passed": research.get("validation_passed"), "design_brief": research.get("design_brief") or {}, "summary_text": str(research.get("summary_text") or "")}, "json",
        ),
    ]
    _assert_bundle_codegen_context(artifacts, context_name="bundle_codegen_context")
    return {
        "artifact_type": "bundle_codegen_context", "schema_version": 1,
        "purpose": "complete annotated code-writing context for a coupled five-module evolution bundle",
        "artifact_manifest": [
            {key: artifact[key] for key in ("artifact_id", "title", "purpose", "content_kind", "chars", "sha256")}
            for artifact in artifacts
        ],
        "artifacts": artifacts,
        "total_chars": sum(int(artifact["chars"]) for artifact in artifacts),
        "edit_files": list(edit_files),
    }


_SMOKE_NO_LABEL_FIELDS = {
    "answer", "answers", "candidate_answer", "final_agent_answer", "gold_answer",
    "ground_truth", "gt_answer", "is_correct", "correct", "correctness", "score",
    "question", "questions", "choices", "options", "task_id", "task_ids", "video_id",
    "video_ids", "time_reference", "time_references",
}


def _redact_bundle_smoke_no_label(value):
    """Keep runtime facts while enforcing smoke's no-answer/no-task boundary."""
    if isinstance(value, dict):
        return {
            str(key): _redact_bundle_smoke_no_label(item)
            for key, item in value.items()
            if str(key).lower() not in _SMOKE_NO_LABEL_FIELDS
        }
    if isinstance(value, list):
        return [_redact_bundle_smoke_no_label(item) for item in value]
    return value


def _read_required_bundle_context_json(path: str, artifact_id: str):
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"required bundle smoke repair artifact missing: {artifact_id}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid bundle smoke repair artifact {artifact_id}: {path}") from exc


def build_bundle_smoke_repair_context(diagnosis_report: dict, bundle: dict,
                                      edit_files: list[str], smoke: dict,
                                      failure_contract: dict, *,
                                      execution_trace_path: str = "",
                                      repair_history: list[dict] | None = None) -> dict:
    """Build the later-round counterpart of initial's complete smoke packet.

    Repair must see the *failed candidate*, not the current-best source that
    preceded it.  Inputs are embedded and annotated so a fresh Codex process
    does not need to infer engineering evidence from host-only paths.
    """
    context = build_bundle_codegen_context(
        diagnosis_report, bundle, edit_files, bundle_role="failed_smoke_candidate",
    )
    artifacts = [dict(item) for item in context.get("artifacts") or [] if isinstance(item, dict)]
    artifacts.append(_bundle_context_artifact(
        "smoke_failure_boundary", "Machine-derived smoke failure boundary",
        "Deterministic producer/consumer/capability failure facts and falsifiable success conditions for this repair.",
        _redact_bundle_smoke_no_label(failure_contract or {}), "json",
    ))
    if execution_trace_path:
        trace = _read_required_bundle_context_json(execution_trace_path, "complete_smoke_execution_trace")
        artifacts.append(_bundle_context_artifact(
            "complete_smoke_execution_trace", "Complete no-label smoke execution trace",
            "All persisted runtime steps, handoffs, and capability events from this smoke attempt. Diagnose the earliest actual divergence across fields.",
            _redact_bundle_smoke_no_label(trace), "json",
        ))
    artifacts.append(_bundle_context_artifact(
        "smoke_runtime_output", "Complete no-label smoke runtime output",
        "Runtime result, assembly facts, handoff checks, capability events, and error evidence from this exact failed candidate.",
        _redact_bundle_smoke_no_label(smoke or {}), "json",
    ))
    if repair_history:
        artifacts.append(_bundle_context_artifact(
            "earlier_repair_history", "Earlier repair history for this candidate lineage",
            "Repeated signatures require repairing the earliest unchanged boundary rather than another generic envelope patch.",
            _redact_bundle_smoke_no_label(repair_history), "json",
        ))
    _assert_bundle_codegen_context(artifacts, context_name="bundle_smoke_repair_context")
    context.update({
        "artifact_type": "bundle_smoke_repair_context", "schema_version": 1,
        "purpose": "complete no-label engineering repair context for one failed bundle smoke",
        "artifact_manifest": [
            {key: item[key] for key in ("artifact_id", "title", "purpose", "content_kind", "chars", "sha256")}
            for item in artifacts
        ],
        "artifacts": artifacts,
        "total_chars": sum(int(item["chars"]) for item in artifacts),
    })
    return context


def _render_bundle_codegen_context(context: dict) -> str:
    parts = []
    for artifact in context.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("artifact_id") or "unnamed")
        parts.append(
            f"### {artifact_id}: {artifact.get('title', '')}\n"
            f"Purpose: {artifact.get('purpose', '')}\n"
            f"Content type: {artifact.get('content_kind', '')}; sha256={artifact.get('sha256', '')}; chars={artifact.get('chars', 0)}\n"
            f"<<<BEGIN {artifact_id}>>>\n{artifact.get('content', '')}\n<<<END {artifact_id}>>>"
        )
    return "\n\n".join(parts)


def _create_bundle_codegen_staging_workspace(run_dir: str, edit_files: list[str]) -> str:
    """Give later-round Codex the same isolated write surface as initial codegen."""
    # Keep the stage outside ``run_dir``.  Probe feedback may need a byte-for-
    # byte read-only copy of that run directory; placing the stage inside it
    # would make that copy recursively include itself.
    stage = tempfile.mkdtemp(
        prefix=".bundle_codegen_stage_", dir=os.path.dirname(os.path.abspath(run_dir)),
    )
    try:
        runtime_target = os.path.join(stage, os.path.relpath(runtime_dir(), PROJECT_ROOT))
        shutil.copytree(RUNTIME_DIR, runtime_target)
        # ``RUNTIME_DIR`` already contains the materialized immutable current-best
        # source for this invocation.  Reject accidental path expansion by
        # recording the only files whose edits will be collected.
        write_text(
            os.path.join(stage, "STAGING_SCOPE.md"),
            "This isolated workspace contains the five-module MetaVideoAgent runtime.\n"
            "Edit only the five source files listed in the Codex prompt.\n"
            "The host will copy only those files back, extract concrete classes,\n"
            "then perform real dynamic injection and smoke.\n",
        )
        return stage
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _rewrite_staged_path_references(root: str, source_root: str, staged_root: str) -> None:
    """Relocate navigational paths in copied forensic metadata, never JSONL data."""
    for directory, _dirs, names in os.walk(root):
        for name in names:
            if not name.endswith((".json", ".md", ".txt")):
                continue
            path = os.path.join(directory, name)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    content = handle.read()
            except OSError:
                continue
            rewritten = content.replace(source_root, staged_root)
            if rewritten == content:
                continue
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(rewritten)


def _stage_bundle_run_artifacts(stage: str, run_dir: str) -> str:
    """Copy complete probe-forensic files into the Codex workspace unchanged.

    JSONL rows are copied byte-for-byte.  Only path-bearing JSON/Markdown/Text
    indexes are relocated, so Codex can follow their references while confined
    to its workspace-write sandbox.
    """
    source = os.path.abspath(run_dir)
    target = os.path.join(stage, "codex_run_artifacts")
    shutil.copytree(source, target)
    _rewrite_staged_path_references(target, source, target)
    return target


def _run_bundle_codex(prompt: str, *, run_dir: str, edit_files: list[str],
                      codex_cli: str, timeout: int, settings: dict,
                      stage_run_artifacts: bool = False,
                      writable_artifact_paths: list[str] | None = None) -> dict:
    """Run Codex in a workspace-write stage and collect explicit artifacts only."""
    stage = _create_bundle_codegen_staging_workspace(run_dir, edit_files)
    try:
        prompt_for_codex = prompt
        staged_outputs: dict[str, str] = {}
        for index, raw_path in enumerate(writable_artifact_paths or [], start=1):
            host_path = os.path.abspath(raw_path)
            if not host_path:
                continue
            staged_path = os.path.join(stage, "codex_outputs", f"{index:02d}_{os.path.basename(host_path)}")
            os.makedirs(os.path.dirname(staged_path), exist_ok=True)
            staged_outputs[host_path] = staged_path
            # Replace output destinations before replacing an enclosing run
            # directory, otherwise a forensic report would remain inside a
            # copied read-only artifact tree.
            prompt_for_codex = prompt_for_codex.replace(host_path, staged_path)
        if stage_run_artifacts:
            staged_run_artifacts = _stage_bundle_run_artifacts(stage, run_dir)
            prompt_for_codex = prompt_for_codex.replace(
                os.path.abspath(run_dir), staged_run_artifacts,
            )
        result = run_codex(
            prompt_for_codex, codex_cli, timeout, settings=settings,
            workdir=stage,
        )
        for rel_path in edit_files:
            staged = os.path.join(stage, rel_path)
            if not os.path.isfile(staged):
                continue
            destination = _abs(rel_path)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(staged, destination)
        for host_path, staged_path in staged_outputs.items():
            if not os.path.isfile(staged_path):
                continue
            os.makedirs(os.path.dirname(host_path), exist_ok=True)
            shutil.copy2(staged_path, host_path)
        result["execution_scope"] = "isolated_bundle_codegen_staging_workspace"
        result["sandbox_mode"] = "workspace-write"
        return result
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def build_bundle_codex_prompt(diagnosis_report: dict, edit_files: list[str],
                              codegen_context: dict | None = None) -> str:
    """Prompt Codex for a bundle, not a brief-locked per-module patch."""
    policy = diagnosis_report.get("execution_policy") or {}
    task = diagnosis_report.get("implementation_task") or {}
    research = diagnosis_report.get("codex_deep_research") or {}
    # The persisted diagnosis_for_codex handoff is the only formal bridge from
    # deep research to code generation.  Keep it visible to the first bundle
    # candidate and every same-direction self-debug iteration; otherwise the
    # research stage can validate successfully while having no effect on code.
    research_context = {
        "research_complete": research.get("research_complete"),
        "formal_research": research.get("formal_research"),
        "validation_passed": research.get("validation_passed"),
        "target_modules": list(research.get("target_modules") or []),
        "design_brief_path": research.get("design_brief_path", ""),
        "design_brief": research.get("design_brief") or {},
        "summary_text": str(research.get("summary_text") or "")[:20000],
    }
    rendered_context = _render_bundle_codegen_context(codegen_context or {})
    # New bundle runs carry these facts inside the annotated context packet.
    # Do not duplicate a potentially long research report after it: duplicate
    # instructions dilute the ABI/source facts that must dominate code edits.
    if rendered_context:
        direction_block = (
            "Read the embedded `evolution_direction` artifact. It is the binding "
            "outer-round direction, not an edit whitelist."
        )
        research_block = (
            "Read the embedded `deep_research_handoff` artifact as implementation "
            "evidence; it cannot override the current bundle or runtime ABI."
        )
    else:
        direction_block = json.dumps(
            {"execution_policy": policy, "implementation_task": task},
            ensure_ascii=False, indent=2,
        )
        research_block = json.dumps(research_context, ensure_ascii=False, indent=2)
    return f"""You are evolving one complete MetaVideoAgent five-module bundle.

This adaptive bundle brief supplies an OUTER-ROUND direction. It is not an edit whitelist.
First implementation may start from `initial_focus_modules`; after probe feedback you may
change any related module inside the five-module bundle when that is needed to repair an
observed producer→consumer→decision failure chain.

Allowed editable source files:
{json.dumps(edit_files, ensure_ascii=False, indent=2)}

Every authoritative code-writing input is embedded below, with its purpose and
checksum. Read `current_bundle` and `runtime_integration_witness` before
editing. The source files in your isolated workspace are materialized from
`current_bundle`; do not rely on any unlisted repository state.

## Bundle Codegen Context Manifest
{json.dumps((codegen_context or {}).get("artifact_manifest", []), ensure_ascii=False, indent=2)}

## Embedded Bundle Codegen Context
{rendered_context}

Evolution direction:
{direction_block}

Validated deep-research handoff:
{research_block}

Use the research as implementation evidence for this outer direction. It does
not freeze the changed module set: during probe self-debug, change any related
module necessary to repair an observed producer→consumer→decision failure.

Rules:
- Keep a complete five-slot combo. Do not edit orchestration, evaluators, datasets, artifacts,
  capability profiles, raw media, or any JSON/JSONL data.
- Add or modify a concrete runtime strategy class in each source file you actually modify.
  A shared `*Base`/helper class is infrastructure, never a combo choice: a base-only edit is
  not a serializable candidate and will be returned for repair. Preserve interfaces and use
  only registered runtime_evidence adapters with literal profile_id values.
- The generic segment builder invokes `video_structuring.addStructure` with only
  `start_sec`, `end_sec`, and `multimodal_narration` (raw VLM text or JSON).
  The neutral `VideoStructuringBase.addStructure` canonicalizes it into
  `canonical_evidence` and `retrieval_document`. If you override addStructure,
  preserve the original `multimodal_narration` and call the base persistence
  path; never erase it by writing empty description/entity/action/text fields.
  Localization must report unavailable for an empty canonical record rather
  than return an `ok` candidate card.
- Implement and preserve parseable handoff fields required by the affected boundary contracts.
- The five-module protocol is mandatory: structure records are time-bounded
  canonical evidence/media refs; localization emits a `localization_result`
  that may contain direct structure records or `perception_requests`;
  perception emits a `perception_result`; memory retains raw request/result
  events; thinking reads `get_runtime_context()` and returns an act plan or a
  finish answer. These envelopes are neutral interfaces, not an edit whitelist.
- Do not hard-code task/video identifiers, questions, answers, time windows, transcript phrases,
  or dataset-specific branches.
- At the end, add a Python class attribute `CODEX_BUNDLE_CHANGE_REASON` to every changed class.
  The harness derives the actual changed module set from the source diff; do not claim changes to
  untouched files.
- The emitted candidate is dynamically extracted. Keep every dependency of a changed concrete
  class within that class, its inherited base chain, imports, or module constants explicitly
  included by the extraction rule in `runtime_integration_witness`; do not rely on arbitrary
  top-level helper functions, an unselected class, or hidden workspace state.
"""



def bundle_smoke_failure_contract(smoke: dict, bundle: dict) -> dict:
    """Compile a task-free, machine-derived repair contract from real smoke.

    The real smoke may contain a training question in stdout.  That data is
    deliberately *not* forwarded to ordinary smoke repair.  The contract only
    carries interface, producer/consumer and capability-event facts that Codex
    can falsify by making the same generic smoke pass.
    """
    smoke = smoke if isinstance(smoke, dict) else {}
    bundle = bundle if isinstance(bundle, dict) else {}
    handoff_checks = []
    for item in smoke.get("handoff_checks") or []:
        if not isinstance(item, dict):
            continue
        handoff_checks.append({
            "producer": str(item.get("producer") or ""),
            "consumer": str(item.get("consumer") or ""),
            "required_fields": [str(value) for value in (item.get("required_fields") or [])],
            "producer_observed": bool(item.get("producer_observed")),
            "missing_fields": [str(value) for value in (item.get("missing_fields") or [])],
            "consumer_observed": bool(item.get("consumer_observed")),
            "consumption_event": bool(item.get("consumption_event")),
        })
    capability_events = []
    for item in smoke.get("capability_events") or []:
        if not isinstance(item, dict):
            continue
        capability_events.append({
            "capability": str(item.get("capability") or ""),
            "event": str(item.get("event") or ""),
            "status": str(item.get("status") or ""),
            "profile_id": str(item.get("profile_id") or ""),
            "producer_module": str(item.get("producer_module") or ""),
            "consumer_module": str(item.get("consumer_module") or ""),
        })
    error = str(smoke.get("error") or "")
    assembly_preflight = dict(smoke.get("assembly_preflight") or {})
    error_class = ""
    matched = re.search(r"(?:^|[\n: ])([A-Z][A-Za-z_]*(?:Error|Exception))(?:[: ]|$)", error)
    if matched:
        error_class = matched.group(1)
    # A missing runtime tool is the earliest actionable divergence for a
    # generated agent.  Merely reporting downstream absent handoff fields
    # makes Codex repeatedly patch envelopes that can never be produced.  The
    # no-label engineering trace is authoritative; compile its first failed
    # dispatch into a compact, task-free fact for the repair prompt.
    first_runtime_divergence = {}
    trace = smoke.get("engineering_execution_trace") or {}
    trajectory = trace.get("trajectory") if isinstance(trace, dict) else []
    if isinstance(trajectory, list):
        for ordinal, item in enumerate(trajectory, start=1):
            if not isinstance(item, dict) or item.get("step_type") != "tool_execution":
                continue
            status = str(item.get("execution_status") or "")
            error_code = str(item.get("error_code") or "")
            if status in {"ok", "success", ""} and not error_code:
                continue
            first_runtime_divergence = {
                "trajectory_step": int(item.get("step") or ordinal),
                "requested_tool": str(item.get("action") or ""),
                "execution_status": status,
                "error_code": error_code,
                # The executor emits deterministic tool-contract messages;
                # retain only this bounded engineering fact, never a model
                # observation or candidate answer.
                "dispatch_message": str(item.get("observation") or "")[:500],
            }
            break
    contract_issues = [str(value) for value in (smoke.get("contract_issues") or [])]
    for issue in assembly_preflight.get("issues") or []:
        text = "runtime assembly preflight: " + str(issue)
        if text not in contract_issues:
            contract_issues.insert(0, text)
    if first_runtime_divergence:
        contract_issues.insert(
            0,
            "runtime tool dispatch failed before a producer result: "
            f"tool={first_runtime_divergence['requested_tool']!r}, "
            f"status={first_runtime_divergence['execution_status']!r}, "
            f"error_code={first_runtime_divergence['error_code']!r}",
        )
    return {
        "artifact_type": "bundle_smoke_failure_contract", "schema_version": 1,
        "failure_kind": str(smoke.get("failure_kind") or "candidate_contract"),
        "contract_mode": "adaptive_bundle",
        "changed_modules": list(bundle.get("changed_modules") or []),
        "declared_handoffs": list(bundle.get("handoff_contracts") or []),
        "observed_handoff_checks": handoff_checks,
        "observed_contract_issues": contract_issues,
        "first_runtime_divergence": first_runtime_divergence,
        "capability_event_summary": capability_events,
        "assembly_preflight": assembly_preflight,
        "runtime_error_class": error_class,
        "required_transition": {
            "must_preserve": "all five module interfaces, declared handoffs, registered runtime adapters and active-media-window boundaries",
            "must_not": "fit or hard-code a benchmark question, video, answer, time range, transcript phrase, or provider response",
        },
        "falsifiable_success": [
            "The same real single-question smoke completes without candidate-contract errors.",
            "Every handoff that is actually triggered has a parseable producer result with its required fields and a later consumer or valid consumption event.",
            "Only declared runtime profiles are used when their conditional capability is triggered.",
        ],
    }


def _bundle_smoke_failure_signature(smoke: dict, contract: dict | None = None) -> dict:
    """Compact, task-free identity for repeated engineering failures."""
    contract = contract if isinstance(contract, dict) else {}
    divergence = contract.get("first_runtime_divergence") or {}
    preflight = (smoke or {}).get("assembly_preflight") or {}
    issues = list(contract.get("observed_contract_issues") or [])
    return {
        "failure_kind": str((smoke or {}).get("failure_kind") or contract.get("failure_kind") or "candidate_contract"),
        "first_runtime_divergence": {
            key: str(divergence.get(key) or "")
            for key in ("requested_tool", "execution_status", "error_code")
        },
        "assembly_preflight_issues": [str(value) for value in (preflight.get("issues") or [])][:4],
        "contract_issue_prefixes": [str(value)[:240] for value in issues[:3]],
    }


def _write_bundle_engineering_trace(run_dir: str, attempt: int, smoke: dict) -> str:
    """Persist the full no-label smoke trace as a separate Codex artifact."""
    trace = smoke.get("engineering_execution_trace") if isinstance(smoke, dict) else {}
    if not isinstance(trace, dict) or not trace:
        return ""
    path = os.path.join(run_dir, f"bundle_smoke_execution_trace_attempt_{attempt}.json")
    write_json(path, {
        "artifact_type": "engineering_smoke_execution_trace",
        "attempt": int(attempt),
        "trace": trace,
    })
    return path


def _persist_bundle_smoke_attempt(run_dir: str, attempt: int, bundle: dict,
                                  smoke: dict, failure_contract: dict | None = None) -> dict:
    """Persist one immutable smoke attempt before a later repair can overwrite it.

    A bundle can need several repairs.  Each smoke result, its self-contained
    candidate bundle and its no-label trace must remain inspectable by Codex
    and by the next evolution round; retaining only the final in-memory
    ``smoke`` object destroys the evidence that justified earlier repairs.
    """
    attempt_dir = os.path.join(run_dir, "bundle_smoke_attempts", f"attempt_{int(attempt)}")
    os.makedirs(attempt_dir, exist_ok=True)
    bundle_path = os.path.join(attempt_dir, "candidate_bundle_before_smoke.json")
    result_path = os.path.join(attempt_dir, "smoke_result.json")
    write_json(bundle_path, bundle)
    write_json(result_path, smoke)
    summary_trace_path = _write_bundle_engineering_trace(attempt_dir, attempt, smoke)
    # The runner's full trace contains the actual controller outputs and API
    # returns needed to repair a behavioral failure.  Copy it into the
    # immutable attempt directory before any later run can overwrite a
    # workspace-level trace path.
    source_full_trace = _abs(str((smoke or {}).get("full_execution_trace_path") or ""))
    full_trace_path = ""
    if source_full_trace and os.path.isfile(source_full_trace):
        full_trace_path = os.path.join(attempt_dir, "smoke_full_execution_trace.json")
        shutil.copy2(source_full_trace, full_trace_path)
    contract_path = ""
    if failure_contract:
        contract_path = os.path.join(attempt_dir, "failure_contract.json")
        write_json(contract_path, failure_contract)
    return {
        "attempt": int(attempt),
        "candidate_bundle_path": bundle_path,
        "smoke_result_path": result_path,
        "engineering_execution_trace_path": full_trace_path or summary_trace_path,
        "summary_execution_trace_path": summary_trace_path,
        "full_execution_trace_path": full_trace_path,
        "failure_contract_path": contract_path,
        "passed": bool((smoke or {}).get("passed")),
        "execution_layer_rerun_required": bool((smoke or {}).get("execution_layer_rerun_required")),
    }


def build_bundle_smoke_repair_prompt(contract: dict, edit_files: list[str],
                                     execution_trace_path: str = "",
                                     candidate_bundle_path: str = "",
                                     repair_history: list[dict] | None = None,
                                     codegen_context: dict | None = None) -> str:
    """Repair prompt for a real-runtime failure using no-label raw execution."""
    artifact_ids = {
        str(item.get("artifact_id") or "")
        for item in (codegen_context or {}).get("artifacts") or []
        if isinstance(item, dict)
    }
    has_embedded_smoke_evidence = bool(
        {"smoke_failure_boundary", "smoke_runtime_output"} <= artifact_ids
    )
    trace_instruction = (
        "\nThe complete deterministic no-label engineering execution trace is embedded below. "
        "Use it to locate the broken implementation/dataflow boundary; do not infer or optimize an answer.\n"
        if has_embedded_smoke_evidence and execution_trace_path else
        "\nBefore editing, read the complete deterministic engineering execution trace at "
        f"`{execution_trace_path}`. It preserves every runtime step, handoff and capability event, "
        "but deliberately removes gold labels, candidate final answers, correctness and scores. "
        "Use it to locate the broken implementation/dataflow boundary; do not infer or optimize an answer.\n"
        if execution_trace_path else
        "\nNo complete runtime trace was available because smoke failed before an Agent trajectory existed. "
        "Use the task-free contract below to repair the pre-trajectory engineering failure.\n"
    )
    bundle_instruction = (
        "\nThe complete failed candidate bundle is embedded below. Its `combo` is the actual "
        "five-class runtime selection, not a suggestion. A `*Base`/support class "
        "must never replace a selected combo slot. If you modify shared support "
        "code, preserve and, where necessary, modify the selected concrete strategy "
        "class in the same editable source so the bundle can serialize and inject it.\n"
        if has_embedded_smoke_evidence else
        "\nRead the complete current candidate bundle at "
        f"`{candidate_bundle_path}` before editing. Its `combo` is the actual "
        "five-class runtime selection, not a suggestion. A `*Base`/support class "
        "must never replace a selected combo slot. If you modify shared support "
        "code, preserve and, where necessary, modify the selected concrete strategy "
        "class in the same editable source so the bundle can serialize and inject it.\n"
        if candidate_bundle_path else ""
    )
    history_instruction = ""
    if repair_history:
        history_instruction = (
            "\nEarlier repair attempts in this same candidate lineage are listed below. "
            "If a failure signature repeats, do not apply another generic envelope-only "
            "patch: repair the earliest unchanged runtime boundary.\n"
            + json.dumps(repair_history, ensure_ascii=False, indent=2) + "\n"
        )
    rendered_context = _render_bundle_codegen_context(codegen_context or {})
    context_instruction = (
        "\nThe complete annotated code-writing context is embedded below. "
        "Read `current_bundle` and `runtime_integration_witness` before editing; "
        "they are authoritative for the failed candidate's selected classes, injected runtime source, "
        "tool names, and call signatures.\n\n"
        "## Embedded Bundle Codegen Context\n"
        f"{rendered_context}\n"
        if rendered_context else ""
    )
    return f"""You are repairing a generated five-module MetaVideoAgent bundle after a
real single-question smoke failed its engineering contract. The source files
currently contain the failed candidate. This is an engineering repair, not a
small-sample accuracy judgment.

Edit only these bundle source files:
{json.dumps(edit_files, ensure_ascii=False, indent=2)}

Machine-derived failure contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}
{trace_instruction}{bundle_instruction}{history_instruction}{context_instruction}

Repair the producer→consumer/runtime dataflow, not answer accuracy. Keep the
complete five-module bundle and use only registered runtime_evidence adapters
with literal profile IDs. Do not alter orchestration, datasets, artifacts,
profiles, raw media, or JSON/JSONL files. Do not hard-code task-specific data.
Add/update CODEX_BUNDLE_CHANGE_REASON on every class you change.
"""


def _parent_bundle_inheritance_issues(bundle: dict, parent_bundle: dict) -> list[str]:
    """Ensure a feedback child cannot lose untouched parent candidate code."""
    if not parent_bundle:
        return []
    issues = []
    child_modules = bundle.get("modules") or {}
    parent_modules = parent_bundle.get("modules") or {}
    changed = set(bundle.get("changed_modules") or [])
    for module in BUNDLE_MODULE_TYPES:
        if module in changed:
            continue
        child = child_modules.get(module) if isinstance(child_modules, dict) else {}
        parent = parent_modules.get(module) if isinstance(parent_modules, dict) else {}
        for field in ("name", "source_file", "code"):
            if str((child or {}).get(field) or "") != str((parent or {}).get(field) or ""):
                issues.append(
                    f"feedback child did not retain parent {module}.{field}; untouched module code must be byte-identical"
                )
                break
    return issues


def extract_candidate_bundle(*, edit_files: list[str], originals: dict,
                             diagnosis_report: dict,
                             retained_base_bundle: dict | None = None,
                             active_bundle: dict | None = None) -> dict:
    """Extract every changed concrete strategy into one atomic candidate artifact.

    A source diff may legitimately touch a shared ``*Base`` helper.  That is
    not a request to replace a combo slot with the helper itself.  When a
    repair changes only such support code, the already-selected concrete class
    remains the active slot and is re-serialized from the current source.
    """
    changed_records: dict[str, dict] = {}
    active_records = _bundle_module_records(active_bundle or {})
    for module_type in BUNDLE_MODULE_TYPES:
        paths = MODULE_SOURCE_FILES.get(module_type, []) or []
        for rel_path in paths:
            if rel_path not in edit_files:
                continue
            path = _abs(rel_path)
            original = originals.get(rel_path, "")
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    current = handle.read()
            except OSError:
                continue
            if current == original:
                continue
            selected, candidates = _selected_concrete_change(path, original)
            if selected is None:
                # A support-only edit is representable only if this run already
                # has an explicit concrete strategy for the same slot.  Never
                # let the historical "largest changed class" heuristic turn a
                # base/helper into a runtime combo choice.
                active = active_records.get(module_type) or {}
                active_name = str(active.get("name") or "")
                if not active_name or active_name.endswith("Base"):
                    changed_names = [item["name"] for item in candidates]
                    raise RuntimeError(
                        "candidate extraction found only infrastructure changes for "
                        f"{module_type}: {changed_names}; add or modify an explicit concrete "
                        "runtime strategy class instead of selecting a *Base helper"
                    )
                current_code = get_full_module_code(path, active_name)
                if not current_code:
                    # The active class can be in an immutable external source
                    # file.  A base-only change cannot be safely injected into
                    # it without a complete source closure, so fail closed and
                    # give the repair loop an actionable engineering fact.
                    raise RuntimeError(
                        "candidate extraction cannot serialize the active concrete "
                        f"{module_type} class {active_name!r} after a support-only edit; "
                        "modify the concrete strategy in the editable source file"
                    )
                selected = {"name": active_name, "code": current_code,
                            "is_new": False, "delta_lines": 0,
                            "is_infrastructure": False}
            class_name = str(selected["name"])
            class_code = str(selected["code"])
            changed_records[module_type] = {
                "name": class_name,
                "module_type": module_type,
                "source_file": rel_path,
                "code": get_full_module_code(path, class_name) or class_code,
                "class_role": "concrete_strategy",
                "evolution_mode": "generate_bundle_code:codex",
            }
            break
    if not changed_records:
        return {}
    # A probe-feedback iteration is an edit of the *previous candidate*, not
    # an unrelated fresh edit of current best.  Reconstituting retained slots
    # from current best here silently discarded earlier coupled changes when
    # Codex touched only one file on the next iteration.
    reference_path = _abs(str(diagnosis_report.get("reference_bundle_path") or ""))
    reference_bundle = (
        dict(retained_base_bundle)
        if isinstance(retained_base_bundle, dict) and retained_base_bundle
        else (_load_json_optional(reference_path) if reference_path else {})
    )
    reference_records = _bundle_module_records(reference_bundle)
    missing_reference = [module for module in BUNDLE_MODULE_TYPES if module not in reference_records]
    if missing_reference:
        raise RuntimeError(
            "candidate extraction requires a self-contained current-best reference bundle; "
            f"missing modules={missing_reference}, reference_bundle={reference_path!r}"
        )
    # Keep the exact code for retained slots in the candidate artifact; the
    # changed records then replace only their corresponding reference slots.
    modules = reference_records
    modules.update(changed_records)
    base_combo = dict(reference_bundle.get("combo") or diagnosis_report.get("reference_combo") or {})
    # Diagnosis and runtime use the same canonical module vocabulary.
    combo = {
        "video_structuring": base_combo.get("video_structuring") or "",
        "localization": base_combo.get("localization") or "",
        "perception": base_combo.get("perception") or "",
        "memory": base_combo.get("memory") or "",
        "thinking": base_combo.get("thinking") or "",
    }
    for module, record in changed_records.items():
        combo[BUNDLE_COMBO_SLOT[module]] = record["name"]
    task = diagnosis_report.get("implementation_task") or {}
    bundle = make_candidate_bundle(
        combo=combo,
        modules=modules,
        changed_modules=list(changed_records),
        handoff_contracts=list(task.get("handoff_contracts") or []),
        rationale={
            module: "Codex changed this module in response to the evolution brief or probe feedback."
            for module in changed_records
        },
        implementation_spec={
            "contract_mode": "adaptive_bundle",
            "target_modules": list((diagnosis_report.get("execution_policy") or {}).get("target_modules") or []),
            "handoff_contracts": list(task.get("handoff_contracts") or []),
            "runtime_capabilities": list(task.get("runtime_capabilities") or []),
            "module_responsibilities": dict(task.get("module_responsibilities") or {}),
            "module_protocol_version": "metavideoagent_module_protocol",
        },
    )
    if retained_base_bundle:
        bundle["parent_bundle_fingerprint"] = str(
            retained_base_bundle.get("bundle_fingerprint") or bundle_fingerprint(retained_base_bundle)
        )
    return bundle


def pre_validate_candidate_bundle(bundle: dict, *, workspace_dir: str = "") -> dict:
    """Validate only the serializable adaptive-bundle manifest.

    Candidate code is never accepted or rejected here.  The subsequent
    task-free/runtime smoke performs real dynamic injection and execution.
    """
    issues = validate_candidate_bundle(bundle)
    return {
        "passed": not issues,
        "validation_mode": "bundle_assembly_only_no_source_introspection",
        "error": "; ".join(issues),
        "issues": issues,
    }


def run_bundle_sandbox_verification(diagnosis_report: dict, workspace_dir: str,
                                    first_n: int, bundle: dict, sandbox_dir: str,
                                    concurrency: int = 3) -> dict:
    """Run one real bundle through the existing evaluator with all deltas injected."""
    from candidate_evaluator import CandidateEvaluator
    report_for_runtime = copy.deepcopy(diagnosis_report)
    agent = CandidateEvaluator(
        diagnosis_report=report_for_runtime, workspace_dir=workspace_dir,
        first_n=first_n, concurrency=concurrency, dry_run=False,
        sandbox_dir=sandbox_dir,
    )
    agent.evolved_bundle = bundle
    agent._strategies_tried = [{"strategy": "generate_bundle_code:codex", "success": True,
                                "module_name": bundle.get("bundle_fingerprint", "")}]
    # Bundle evolution owns every source mutation through its explicit
    # assembly, smoke, and probe loops. In-process API repair is disabled
    # because it would edit shared runtime
    # files for a provider/environment failure and invalidate candidate lineage.
    old_disable_repair = os.environ.get("SANDBOX_DISABLE_INTERNAL_ENGINEERING_REPAIR")
    os.environ["SANDBOX_DISABLE_INTERNAL_ENGINEERING_REPAIR"] = "1"
    try:
        agent._step4_verify()
    finally:
        if old_disable_repair is None:
            os.environ.pop("SANDBOX_DISABLE_INTERNAL_ENGINEERING_REPAIR", None)
        else:
            os.environ["SANDBOX_DISABLE_INTERNAL_ENGINEERING_REPAIR"] = old_disable_repair
    verification = dict(agent.verification_report or {})
    verification["candidate_bundle"] = bundle_fingerprint(bundle)
    verification["changed_modules"] = changed
    verification["handoff_contracts"] = bundle.get("handoff_contracts") or []
    verification["implementation_spec"] = dict(bundle.get("implementation_spec") or {})
    return {
        "verification": verification,
        "conclusion": agent._build_conclusion(),
        "probe_results": list(getattr(agent, "_last_probe_results", []) or []),
    }


def _reference_bundle_for_probe_replay(diagnosis_report: dict) -> dict:
    """Normalize current best into the self-contained bundle evaluator format."""
    raw = _load_json_optional(str(diagnosis_report.get("reference_bundle_path") or ""))
    if not raw:
        raise RuntimeError("reference probe replay requires diagnosis.reference_bundle_path")
    modules_raw = raw.get("modules") or {}
    records = modules_raw.values() if isinstance(modules_raw, dict) else modules_raw
    modules = {
        str(item.get("module_type") or ""): dict(item)
        for item in records or [] if isinstance(item, dict) and str(item.get("module_type") or "") in BUNDLE_MODULE_TYPES
    }
    missing = [item for item in BUNDLE_MODULE_TYPES if item not in modules]
    if missing:
        raise RuntimeError(f"reference probe replay bundle is incomplete: {missing}")
    combo_raw = raw.get("combo") or diagnosis_report.get("reference_combo") or {}
    combo = {
        "video_structuring": combo_raw.get("video_structuring") or "",
        "localization": combo_raw.get("localization") or "",
        "perception": combo_raw.get("perception") or "",
        "memory": combo_raw.get("memory") or "",
        "thinking": combo_raw.get("thinking") or "",
    }
    handoffs = list((diagnosis_report.get("implementation_task") or {}).get("handoff_contracts") or [])
    if not handoffs:
        handoffs = list(raw.get("handoff_contracts") or [])
    return make_candidate_bundle(
        combo=combo, modules=modules, changed_modules=list(BUNDLE_MODULE_TYPES),
        handoff_contracts=handoffs,
        rationale={item: "Current-best fixed-batch reference replay." for item in BUNDLE_MODULE_TYPES},
        implementation_spec={
            "contract_mode": "adaptive_bundle",
            "target_modules": list(BUNDLE_MODULE_TYPES),
            "handoff_contracts": handoffs,
            "runtime_capabilities": list((diagnosis_report.get("implementation_task") or {}).get("runtime_capabilities") or []),
            "module_responsibilities": dict((diagnosis_report.get("implementation_task") or {}).get("module_responsibilities") or {}),
            "module_protocol_version": "metavideoagent_module_protocol",
        },
    )


def _write_replay_jsonl(path: str, rows: list[dict]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return _abs(path)


def apply_live_probe_comparison(verification: dict, candidate_rows: list[dict],
                                reference_rows: list[dict], *,
                                candidate_replay_path: str,
                                reference_replay_path: str) -> dict:
    """Replace historical probe deltas with the current fixed-batch replay.

    Probe selection can consult an old full-eval report, but candidate promotion
    must compare two executions made now under the same bounded task batch.
    Keeping this transformation in one helper prevents a later caller from
    accidentally gating on the historical fields after it has generated a
    fresh reference replay.
    """
    from sandbox_evaluator import build_reference_lookup, compare_with_reference

    live_comparison = compare_with_reference(
        candidate_rows,
        build_reference_lookup(reference_rows, reference_label="current_best_probe_replay"),
        reference_label="current_best_probe_replay",
        reference_source=reference_replay_path,
    )
    historical_metrics = {
        key: verification.get(key)
        for key in (
            "reference_correct", "candidate_correct", "accuracy_delta",
            "corrections", "regressions", "comparison_valid",
        )
    }
    for key in (
        "total_questions", "reference_correct", "candidate_correct",
        "reference_correct", "reference_accuracy", "reference_total_steps",
        "reference_coverage", "candidate_correct", "candidate_accuracy",
        "candidate_total_steps", "accuracy_delta", "accuracy_delta_vs_reference",
        "steps_delta", "steps_delta_vs_reference", "corrections", "regressions",
        "efficiency_gains", "unchanged_correct", "unchanged_incorrect",
        "comparison_valid", "missing_reference", "missing_reference",
        "reference_metrics", "candidate_metrics", "delta_metrics",
    ):
        if key in live_comparison:
            verification[key] = live_comparison[key]
    verification["comparison_target"] = live_comparison.get("comparison_target", {})
    verification["live_probe_comparison"] = {
        "source": "same_batch_current_best_replay",
        "candidate_replay_path": candidate_replay_path,
        "reference_replay_path": reference_replay_path,
        "metrics": live_comparison,
        "historical_metrics_retained_for_audit": historical_metrics,
    }
    return live_comparison


def _bundle_feedback_rounds(parent_dir: str) -> int:
    """Count bundle feedback ancestry without using a module/mechanism key."""
    count, seen, current = 0, set(), _abs(parent_dir)
    while current and current not in seen and os.path.isdir(current):
        seen.add(current)
        info = _load_json_optional(os.path.join(current, "probe_failed_input.json"))
        if not isinstance(info, dict) or not info.get("probe_failed_run"):
            break
        count += 1
        current = _abs(str(info.get("probe_failed_run") or ""))
    return count


def _bundle_engineering_feedback_rounds(parent_dir: str) -> int:
    """Count resumptions of an exhausted assembly/runtime smoke candidate."""
    count, seen, current = 0, set(), _abs(parent_dir)
    while current and current not in seen and os.path.isdir(current):
        seen.add(current)
        info = _load_json_optional(os.path.join(current, "engineering_failed_input.json"))
        if not isinstance(info, dict) or not info.get("engineering_failed_run"):
            break
        count += 1
        current = _abs(str(info.get("engineering_failed_run") or ""))
    return count


def _source_tree_fingerprint(source_dir: str) -> str:
    digest = hashlib.sha256()
    root = _abs(source_dir)
    if not root or not os.path.isdir(root):
        return ""
    for current, _dirs, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(current, name)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            digest.update(relative.encode("utf-8"))
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _bundle_feedback_lineage(parent_dir: str) -> list[dict]:
    """Recover source/report/replay/route facts from immutable parent runs."""
    lineage, seen, current = [], set(), _abs(parent_dir)
    while current and current not in seen and os.path.isdir(current):
        seen.add(current)
        candidate = _load_json_optional(os.path.join(current, "candidate_bundle.json"))
        validation = _load_json_optional(os.path.join(current, "validation.json"))
        probe = _load_json_optional(os.path.join(current, "probe_verification.json"))
        forensic = _load_json_optional(os.path.join(current, "probe_failed_input.json"))
        engineering = _load_json_optional(os.path.join(current, "engineering_failed_input.json"))
        lineage.append({
            "candidate_run": current,
            "bundle_fingerprint": str(candidate.get("bundle_fingerprint") or ""),
            "source_after_fingerprint": _source_tree_fingerprint(os.path.join(current, "source_after")),
            "probe_next_action": (validation.get("probe_next_action") or {}),
            "forensic_report_path": str(((forensic.get("supervised_forensic_workspace") or {}).get("forensic_report_path") or "")),
            "reference_probe_replay": probe.get("reference_probe_replay") or {},
            "engineering_feedback": engineering,
        })
        parent = ""
        if isinstance(forensic, dict):
            parent = str(forensic.get("probe_failed_run") or "")
        if not parent and isinstance(engineering, dict):
            parent = str(engineering.get("engineering_failed_run") or "")
        current = _abs(parent)
    return list(reversed(lineage))


def _write_bundle_lineage(run_dir: str, parent_dir: str = "") -> str:
    lineage = _bundle_feedback_lineage(parent_dir) if parent_dir else []
    candidate = _load_json_optional(os.path.join(run_dir, "candidate_bundle.json"))
    validation = _load_json_optional(os.path.join(run_dir, "validation.json"))
    probe = _load_json_optional(os.path.join(run_dir, "probe_verification.json"))
    lineage.append({
        "candidate_run": _abs(run_dir),
        "bundle_fingerprint": str(candidate.get("bundle_fingerprint") or ""),
        "source_after_fingerprint": _source_tree_fingerprint(os.path.join(run_dir, "source_after")),
        "probe_next_action": validation.get("probe_next_action") or {},
        "forensic_report_path": str((((_load_json_optional(os.path.join(run_dir, "probe_failed_input.json"))).get("supervised_forensic_workspace") or {}).get("forensic_report_path") or "")),
        "reference_probe_replay": probe.get("reference_probe_replay") or {},
    })
    path = os.path.join(run_dir, "bundle_feedback_lineage.json")
    write_json(path, {"schema_version": 1, "attempts": lineage})
    return path


def _latest_bundle_engineering_trace(parent_dir: str, validation: dict, report: dict) -> str:
    """Find the immutable no-label trace for an engineering-resume candidate."""
    candidates = [
        str((report or {}).get("bundle_smoke_full_execution_trace_path") or ""),
        str((validation or {}).get("bundle_smoke_full_execution_trace_path") or ""),
        str((report or {}).get("bundle_smoke_execution_trace_path") or ""),
        str((validation or {}).get("bundle_smoke_execution_trace_path") or ""),
    ]
    candidates.extend(sorted(glob.glob(os.path.join(parent_dir, "bundle_smoke_attempts", "attempt_*", "smoke_full_execution_trace.json")), reverse=True))
    candidates.extend(sorted(glob.glob(os.path.join(parent_dir, "bundle_smoke_execution_trace_attempt_*.json")), reverse=True))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return _abs(candidate)
    return ""


def _hydrate_bundle_reference_context(diagnosis_report: dict, args: argparse.Namespace) -> dict:
    """Attach execution-only current-best paths before adaptive bundle dispatch.

    The adaptive bundle brief is intentionally model-clean, and ``main`` dispatches
    it to the bundle path before its previous reference hydration block.  Bundle
    smoke still needs the explicit current-best JSONL, bundle and report.
    Hydrate a private runtime copy here; do not write these paths back into the
    model-facing diagnosis artifact.
    """
    report = dict(diagnosis_report or {})
    # An adaptive diagnosis_for_codex artifact may carry an explicitly attested
    # current-best bundle from an explicitly attested staged run. Command-line
    # reports are supplemental hydration inputs only: their historical
    # combo_policy can contain a relocated path that no longer exists and
    # then fall back to an *initial* base bundle.  Never replace an existing
    # validated diagnosis reference with that fallback.
    attested_bundle_path = str(report.get("reference_bundle_path") or "")
    attested_bundle_is_valid = _looks_like_initial_bundle(attested_bundle_path)
    reference_results = str(getattr(args, "reference_results", "") or "")
    # Do not resolve an omitted optional path: ``_abs("")`` is the project
    # root, which is a directory and must never be treated as a JSON report.
    raw_reference_report_path = str(getattr(args, "reference_report", "") or "")
    reference_report_path = _abs(raw_reference_report_path) if raw_reference_report_path else ""
    if not reference_results and reference_report_path:
        reference_results = _reference_from_report(reference_report_path)
    ref_report = {}
    if reference_report_path and os.path.isfile(reference_report_path):
        ref_report = _load_json_optional(reference_report_path)
        combo, policy = _reference_combo_from_report(ref_report)
        if combo and not report.get("reference_combo"):
            report["reference_combo"] = combo
            report["reference_combo_policy"] = policy
        bundle_path = _reference_bundle_from_report(ref_report, reference_report_path)
        if bundle_path and not attested_bundle_is_valid:
            report["reference_bundle_path"] = bundle_path
        report["reference_report_path"] = reference_report_path
    if reference_results:
        report["reference_results_path"] = _abs(reference_results)
    if getattr(args, "distribution_manifest", ""):
        report["distribution_manifest"] = _abs(str(args.distribution_manifest))
    if getattr(args, "reference_label", ""):
        report["reference_label"] = str(args.reference_label)
    missing = [
        key for key in ("reference_results_path", "reference_report_path", "reference_bundle_path")
        if not str(report.get(key) or "") or not os.path.isfile(str(report.get(key) or ""))
    ]
    if missing:
        raise RuntimeError(
            "adaptive bundle execution requires explicit current-best artifacts: "
            + ", ".join(missing)
        )
    return report


def run_bundle_evolution(args: argparse.Namespace, diagnosis_report: dict,
                         workspace_dir: str) -> int:
    """Current adaptive evolve path with atomic source snapshots and bundle artifacts."""
    from diagnosis_execution_brief import validate_bundle_execution_brief
    issues = validate_bundle_execution_brief(diagnosis_report)
    if issues:
        raise RuntimeError("invalid adaptive execution brief: " + "; ".join(issues))
    contract_path = _abs(getattr(args, "machine_evaluation_contract", ""))
    if not contract_path or not os.path.isfile(contract_path):
        raise RuntimeError("adaptive bundle evolution requires --machine-evaluation-contract")
    machine_contract = _load_json_required(contract_path, "adaptive bundle machine evaluation contract")
    contract_issues = validate_bundle_machine_evaluation_contract(machine_contract)
    if contract_issues:
        raise RuntimeError("adaptive bundle evolution has an invalid machine contract: " + "; ".join(contract_issues))
    diagnosis_report = _hydrate_bundle_reference_context(diagnosis_report, args)
    diagnosis_report["_machine_evaluation_contract"] = machine_contract
    diagnosis_report["machine_evaluation_contract_path"] = contract_path
    edit_files = _bundle_edit_files()
    started = int(time.time())
    history_dir = _history_dir(workspace_dir, args.output_root)
    run_dir = os.path.join(history_dir, "codex_candidates", f"{started}_bundle_codex")
    os.makedirs(run_dir, exist_ok=True)
    # Keep bundle-mode records complete on every exit path.  The previous path
    # already snapshots its diagnosis rollout; without this twin artifact a
    # valid bundle smoke could be marked record-incomplete solely because it
    # used the adaptive dispatcher.
    rollout_path = _resolve_rollout_path(args.diagnosis, args.diagnosis_rollout)
    if rollout_path and os.path.isfile(rollout_path):
        shutil.copy2(rollout_path, os.path.join(run_dir, "diagnosis_rollout.json"))
    else:
        write_text(
            os.path.join(run_dir, "diagnosis_rollout.json"),
            json.dumps({"unused_reason": "diagnosis rollout not found"}, ensure_ascii=False, indent=2),
        )
    # The effective prompt cannot be constructed until the immutable reference
    # (or failed parent) source has been materialized.  Otherwise later-round
    # Codex sees only a diagnosis and not the concrete bundle it must evolve.
    prompt = ""
    report = {"status": "", "mode": "adaptive_bundle", "run_dir": run_dir,
              "changed_modules": [], "edit_files": edit_files,
              "codex_runtime_settings": args.codex_runtime_settings,
              "authoring_engine": "codex"}
    if args.dry_run:
        # Dry-run remains a consumable planning artifact.  It must not mutate
        # runtime sources, but it should expose the same prompt shape whenever
        # the declared immutable bundle is available.
        dry_bundle_path = (
            os.path.join(_abs(args.engineering_failed_run), "candidate_bundle.json")
            if getattr(args, "engineering_failed_run", "") else
            os.path.join(_abs(args.probe_failed_run), "candidate_bundle.json")
            if getattr(args, "probe_failed_run", "") else
            str(diagnosis_report.get("reference_bundle_path") or "")
        )
        dry_bundle = _load_json_optional(dry_bundle_path) if dry_bundle_path else {}
        dry_context = (
            build_bundle_codegen_context(diagnosis_report, dry_bundle, edit_files)
            if dry_bundle else {}
        )
        prompt = build_bundle_codex_prompt(diagnosis_report, edit_files, dry_context)
        write_json(os.path.join(run_dir, "diagnosis.json"), diagnosis_report)
        if dry_context:
            write_json(os.path.join(run_dir, "bundle_codegen_context.json"), dry_context)
        write_text(os.path.join(run_dir, "prompt.txt"), prompt)
        report["status"] = "dry_run"
        report["dry_run_context_available"] = bool(dry_context)
        write_json(os.path.join(run_dir, "validation.json"), {})
        write_json(os.path.join(run_dir, "sandbox_report.json"), {})
        finalize_run_record(run_dir, report)
        return 0
    # A bundle behavior iteration starts from the previous candidate source,
    # not from the mutable repository baseline.  Copy every raw sandbox JSONL
    # byte-for-byte and tell Codex to inspect the files directly; no audit
    # excerpt can replace the supervised train trajectories.
    workspace_originals = _snapshot_files(edit_files)
    feedback_parent = {}
    engineering_feedback = {}
    parent_bundle: dict = {}
    active_source_bundle: dict = {}
    forensic_manifest: dict = {}
    source_origin_manifest: dict = {}
    engineering_smoke_payload: dict = {}
    if getattr(args, "engineering_failed_run", ""):
        parent_dir = _abs(args.engineering_failed_run)
        parent_bundle = _load_json_optional(os.path.join(parent_dir, "candidate_bundle.json"))
        parent_after = os.path.join(parent_dir, "source_after")
        parent_validation_payload = _load_json_optional(os.path.join(parent_dir, "validation.json"))
        parent_report = _load_json_optional(os.path.join(parent_dir, "run_record.json"))
        if not parent_bundle or not os.path.isdir(parent_after):
            raise RuntimeError(
                "--engineering-failed-run requires candidate_bundle.json and source_after/; "
                "it is only for a persisted assembly/runtime smoke candidate"
            )
        prior_rounds = _bundle_engineering_feedback_rounds(parent_dir)
        budget = max(1, int(os.environ.get("BUNDLE_ENGINEERING_FEEDBACK_BUDGET", "2") or 2))
        if prior_rounds >= budget:
            raise RuntimeError(
                "engineering feedback budget exhausted for this candidate lineage; "
                "start a new outer diagnosis direction"
            )
        parent_validation = dict(parent_validation_payload.get("assembly_validation") or parent_validation_payload or {})
        parent_smoke = dict(parent_validation_payload.get("bundle_smoke") or parent_validation.get("bundle_smoke") or {})
        engineering_smoke_payload = parent_smoke
        trace_path = _latest_bundle_engineering_trace(parent_dir, parent_validation_payload, parent_report)
        if parent_smoke:
            engineering_contract = bundle_smoke_failure_contract(parent_smoke, parent_bundle)
        else:
            engineering_contract = {
                "artifact_type": "bundle_runtime_failure_contract", "schema_version": 1,
                "failure_kind": "candidate_contract",
                "observed_contract_issues": [str(parent_validation.get("error") or "runtime smoke was not recorded")],
            }
        _apply_source_snapshot(parent_after)
        active_source_bundle = parent_bundle
        source_origin_manifest = {
            "source_policy": "immutable_parent_candidate_snapshot",
            "parent_run": parent_dir,
            "snapshot": parent_after,
        }
        engineering_feedback = {
            "engineering_failed_run": parent_dir,
            "parent_bundle_fingerprint": parent_bundle.get("bundle_fingerprint", ""),
            "engineering_feedback_round": prior_rounds + 1,
            "engineering_feedback_budget": budget,
            "failure_contract": engineering_contract,
            "engineering_execution_trace_path": trace_path,
        }
        write_json(os.path.join(run_dir, "engineering_failed_input.json"), engineering_feedback)
    if getattr(args, "probe_failed_run", ""):
        if engineering_feedback:
            raise RuntimeError("do not mix --engineering-failed-run with --probe-failed-run")
        parent_dir = _abs(args.probe_failed_run)
        budget = int(os.environ.get("BUNDLE_FEEDBACK_BUDGET", "2") or 2)
        prior_rounds = _bundle_feedback_rounds(parent_dir)
        requested_round = int(getattr(args, "probe_feedback_round", 0) or prior_rounds + 1)
        # The CLI default is 1. After the first feedback hop it is treated as an
        # omitted-value sentinel after the first hop; ancestry remains the
        # source of truth and cannot be reset by changing module sets.
        if requested_round == 1 and prior_rounds:
            requested_round = prior_rounds + 1
        if requested_round != prior_rounds + 1 or requested_round > budget:
            raise RuntimeError(
                "bundle feedback round is outside the immutable outer-direction budget: "
                f"requested={requested_round}, expected={prior_rounds + 1}, budget={budget}"
            )
        parent_bundle = _load_json_optional(os.path.join(parent_dir, "candidate_bundle.json"))
        parent_after = os.path.join(parent_dir, "source_after")
        if not parent_bundle or not os.path.isdir(parent_after):
            raise RuntimeError(
                "--probe-failed-run requires candidate_bundle.json and source_after/"
            )
        parent_verification = _load_json_optional(os.path.join(parent_dir, "probe_verification.json"))
        parent_audit = _load_json_optional(os.path.join(parent_dir, "probe_audit.json"))
        fixed_probe_task_ids = list(
            ((parent_verification.get("probe_plan") or {}).get("selected_refs") or [])
            if isinstance(parent_verification, dict) else []
        )
        fixed_probe_task_ids = [str(value) for value in fixed_probe_task_ids if str(value)]
        if not fixed_probe_task_ids:
            raise RuntimeError(
                "bundle probe feedback requires probe_verification.probe_plan.selected_refs; "
                "refusing to rotate to a newly selected question batch"
            )
        replay_info = parent_verification.get("reference_probe_replay") if isinstance(parent_verification, dict) else {}
        if not isinstance(replay_info, dict) or not replay_info.get("passed"):
            raise RuntimeError(
                "bundle probe feedback requires a completed same-batch candidate/reference replay; "
                "rerun the exact candidate probe before supervised Codex feedback"
            )
        for key in ("candidate_replay_path", "reference_replay_path"):
            if not os.path.isfile(str(replay_info.get(key) or "")):
                raise RuntimeError(f"bundle probe feedback replay artifact missing: {key}")
        diagnosis_report["fixed_probe_task_ids"] = fixed_probe_task_ids
        raw_source = str(parent_verification.get("sandbox_dir") or parent_audit.get("sandbox_dir") or "")
        raw_artifacts = _copy_complete_probe_trajectory_artifacts(raw_source, run_dir)
        feedback_path = os.path.join(run_dir, "bundle_probe_feedback.json")
        write_json(feedback_path, {
            "artifact_type": "bundle_probe_feedback",
            "schema_version": 1,
            "feedback_mode": "automatic_codex_same_batch_replay",
            "previous_candidate": {
                "bundle_fingerprint": parent_bundle.get("bundle_fingerprint", ""),
                "changed_modules": parent_bundle.get("changed_modules") or [],
                "handoff_contracts": parent_bundle.get("handoff_contracts") or [],
            },
            # The raw JSONL remains the authoritative unabridged training
            # evidence.  These two complete reports make the runtime and
            # producer→consumer failure explicit instead of forcing Codex to
            # rediscover it from log ordering alone.
            "probe_verification": parent_verification,
            "bundle_probe_audit": parent_audit,
            "raw_training_trajectory_artifacts": raw_artifacts,
            "persisted_current_best_comparison": {
                "mode": parent_verification.get("probe_comparison_mode", ""),
                "candidate_results_path": parent_verification.get("candidate_probe_results_path", ""),
                "reference_results_path": parent_verification.get("persisted_reference_minimal_results_path")
                or parent_verification.get("persisted_reference_results_path", ""),
                "selected_task_ids": fixed_probe_task_ids,
                "reference_replay_not_required": False,
            },
        })
        parent_diff = os.path.join(parent_dir, "source_diff.patch")
        copied_parent_diff = ""
        if os.path.isfile(parent_diff):
            copied_parent_diff = os.path.join(run_dir, "parent_source_diff.patch")
            shutil.copy2(parent_diff, copied_parent_diff)
        _apply_source_snapshot(parent_after)
        active_source_bundle = parent_bundle
        source_origin_manifest = {
            "source_policy": "immutable_parent_candidate_snapshot",
            "parent_run": parent_dir,
            "snapshot": parent_after,
        }
        feedback_parent = {
            "probe_failed_run": parent_dir,
            "candidate_bundle_fingerprint": parent_bundle.get("bundle_fingerprint", ""),
            "changed_modules": parent_bundle.get("changed_modules") or [],
            "raw_training_trajectory_artifacts": raw_artifacts,
            "raw_trajectory_manifest": os.path.join(run_dir, "probe_training_trajectories_manifest.json"),
            "bundle_probe_feedback_path": feedback_path,
            "parent_source_diff_path": copied_parent_diff,
            "prior_bundle_feedback_rounds": prior_rounds,
            "bundle_feedback_round": requested_round,
            "bundle_feedback_budget": budget,
            "fixed_probe_task_ids": fixed_probe_task_ids,
        }
        write_json(os.path.join(run_dir, "probe_failed_input.json"), feedback_parent)
    if not feedback_parent and not engineering_feedback:
        reference_bundle_path = str(diagnosis_report.get("reference_bundle_path") or "")
        reference_bundle = _load_json_optional(reference_bundle_path) if reference_bundle_path else {}
        if not reference_bundle:
            _restore_files(workspace_originals)
            raise RuntimeError(
                "bundle codegen requires diagnosis.reference_bundle_path with a complete immutable current-best bundle"
            )
        try:
            source_origin_manifest = _materialize_bundle_source_workspace(reference_bundle, edit_files)
        except Exception:
            _restore_files(workspace_originals)
            raise
        source_origin_manifest["reference_bundle_path"] = _abs(reference_bundle_path)
        source_origin_manifest["reference_bundle_fingerprint"] = str(
            reference_bundle.get("bundle_fingerprint") or ""
        )
        active_source_bundle = reference_bundle
    write_json(os.path.join(run_dir, "source_origin_manifest.json"), source_origin_manifest)
    originals = _snapshot_files(edit_files)
    _copy_source_snapshot(edit_files, os.path.join(run_dir, "source_before"))
    if not active_source_bundle:
        _restore_files(workspace_originals)
        raise RuntimeError("bundle codegen did not resolve an immutable active source bundle")
    codegen_context = build_bundle_codegen_context(
        diagnosis_report, active_source_bundle, edit_files,
    )
    write_json(os.path.join(run_dir, "bundle_codegen_context.json"), codegen_context)
    if engineering_feedback:
        repair_context = build_bundle_smoke_repair_context(
            diagnosis_report, parent_bundle, edit_files, engineering_smoke_payload,
            engineering_contract,
            execution_trace_path=trace_path,
        )
        write_json(os.path.join(run_dir, "bundle_smoke_repair_context.json"), repair_context)
        prompt = build_bundle_smoke_repair_prompt(
            engineering_contract, edit_files,
            execution_trace_path=trace_path,
            candidate_bundle_path=os.path.join(parent_dir, "candidate_bundle.json"),
            codegen_context=repair_context,
        )
    else:
        prompt = build_bundle_codex_prompt(diagnosis_report, edit_files, codegen_context)
    if feedback_parent:
        # The prompt contains paths, never a lossy transcription of the
        # supervised evidence.  The workspace is fresh for this Codex attempt
        # and records every raw candidate/reference trace and source snapshot.
        forensic_manifest = _materialize_bundle_probe_forensics(
            run_dir=run_dir,
            attempt_tag="initial",
            parent_dir=parent_dir,
            parent_after=parent_after,
            raw_candidate_trajectories=raw_source,
            diagnosis_report=diagnosis_report,
            parent_verification=parent_verification,
            parent_audit=parent_audit,
            machine_contract_path=contract_path,
        )
        feedback_parent["supervised_forensic_workspace"] = {
            "manifest_path": forensic_manifest["manifest_path"],
            "forensic_report_path": forensic_manifest["forensic_report_path"],
            "candidate_task_count": forensic_manifest["candidate_task_count"],
        }
        write_json(os.path.join(run_dir, "probe_failed_input.json"), feedback_parent)
        prompt += (
            f"\n\nThis is a bundle self-debug iteration. The workspace already contains the previous candidate; "
            f"its immutable snapshot is `{os.path.join(run_dir, 'source_before')}`. Before editing, read every JSONL "
            f"listed in `{os.path.join(run_dir, 'probe_training_trajectories_manifest.json')}`; they are "
            "complete raw training trajectories copied byte-for-byte. Read the complete structured runtime audit and "
            f"verification at `{feedback_path}` and the previous source diff at `{copied_parent_diff}` when present. "
            "Diagnose concrete producer→consumer→decision failures, then adapt any related bundle modules needed for "
            "a general solution.\n"
        )
        prompt += forensic_prompt_block(forensic_manifest)
    # Persist the effective fixed-batch IDs as part of the immutable attempt,
    # not only in in-memory runtime state.
    write_json(os.path.join(run_dir, "diagnosis.json"), diagnosis_report)
    write_text(os.path.join(run_dir, "prompt.txt"), prompt)
    extraction_error = ""
    writable_codegen_artifacts = []
    if forensic_manifest:
        writable_codegen_artifacts.append(str(forensic_manifest.get("forensic_report_path") or ""))
    result = _run_bundle_codex(
        prompt, run_dir=run_dir, edit_files=edit_files,
        codex_cli=resolve_coding_agent_executable(args.codex_cli), timeout=args.timeout,
        settings=args.codex_runtime_settings,
        stage_run_artifacts=bool(feedback_parent),
        writable_artifact_paths=[path for path in writable_codegen_artifacts if path],
    )
    write_text(os.path.join(run_dir, "authoring_stdout.txt"), result.get("stdout", ""))
    write_text(os.path.join(run_dir, "authoring_stderr.txt"), result.get("stderr", ""))
    if result.get("ok"):
        try:
            bundle = extract_candidate_bundle(
                edit_files=edit_files, originals=originals, diagnosis_report=diagnosis_report,
                retained_base_bundle=parent_bundle or None,
            )
        except RuntimeError as exc:
            bundle = {}
            extraction_error = str(exc)
    else:
        bundle = {}
    forensic_validation = (
        validate_codex_forensic_report(forensic_manifest)
        if forensic_manifest else {"passed": True, "skipped": True, "reason": "initial_bundle_codegen"}
    )
    validation = pre_validate_candidate_bundle(bundle, workspace_dir=workspace_dir) if bundle else {}
    specialization_validation = (
        scan_candidate_source_for_specialization(bundle, forensic_manifest)
        if forensic_manifest and bundle else {"passed": True, "skipped": True}
    )
    if not forensic_validation.get("passed"):
        validation = dict(validation or {})
        validation["passed"] = False
        validation["forensic_report_validation"] = forensic_validation
        validation["error"] = "; ".join(filter(None, [
            str(validation.get("error") or ""),
            "supervised forensic report incomplete: " + str(forensic_validation.get("error") or "unknown"),
        ]))
    if not specialization_validation.get("passed"):
        validation = dict(validation or {})
        validation["passed"] = False
        validation["supervised_specialization_validation"] = specialization_validation
        validation["error"] = "; ".join(filter(None, [
            str(validation.get("error") or ""),
            "supervised literal crossed into candidate source",
        ]))
    codex_engineering_repairs = []

    _copy_source_snapshot(edit_files, os.path.join(run_dir, "source_after"))
    _write_source_diff(originals, os.path.join(run_dir, "source_diff.patch"))
    if not args.keep_source_edits:
        _restore_files(workspace_originals)
    if not bundle:
        report.update({
            "status": "bundle_codegen_failed",
            "codex": result,
            "candidate_extraction_error": extraction_error,
            "supervised_forensic_report_validation": forensic_validation,
            "supervised_specialization_validation": specialization_validation,
        })
        write_json(os.path.join(run_dir, "validation.json"), {})
        write_json(os.path.join(run_dir, "sandbox_report.json"), {})
        finalize_run_record(run_dir, report)
        return 1
    inheritance_issues = _parent_bundle_inheritance_issues(bundle, parent_bundle)
    if inheritance_issues:
        validation = dict(validation or {})
        validation["passed"] = False
        validation["parent_bundle_inheritance_issues"] = inheritance_issues
        validation["error"] = "; ".join(
            [str(validation.get("error") or "")] + inheritance_issues
        ).strip("; ")
    write_json(os.path.join(run_dir, "candidate_bundle.json"), bundle)
    write_json(os.path.join(run_dir, "validation.json"), {"assembly_validation": validation})
    report.update({"candidate_bundle": {key: value for key, value in bundle.items() if key != "modules"},
                   "changed_modules": bundle["changed_modules"], "codex": result,
                   "task_free_engineering_repairs": codex_engineering_repairs,
                   "validation": validation,
                   "supervised_forensic_report_validation": forensic_validation,
                   "supervised_specialization_validation": specialization_validation,
                   })
    if feedback_parent:
        report["probe_failed_input"] = feedback_parent
    if engineering_feedback:
        report["engineering_failed_input"] = engineering_feedback
    if not validation.get("passed"):
        validation["probe_next_action"] = {
            "action": "codex_engineering_repair",
            "reason": "candidate bundle assembly is incomplete; regenerate the bundle artifact before runtime smoke",
            "codex_feedback_allowed": True,
            "feedback_mode": "engineering_no_label",
        }
        write_json(os.path.join(run_dir, "validation.json"), {"assembly_validation": validation})
        report["status"] = "bundle_assembly_invalid"
        report["probe_next_action"] = validation["probe_next_action"]
        write_json(os.path.join(run_dir, "sandbox_report.json"), {})
        report["bundle_feedback_lineage_path"] = _write_bundle_lineage(
            run_dir,
            str(feedback_parent.get("probe_failed_run") or engineering_feedback.get("engineering_failed_run") or ""),
        )
        finalize_run_record(run_dir, report)
        return 0
    # Smoke is a separate, fast, real-runtime gate.  It injects every changed
    # class into the same Agent invocation, so a bundle cannot reach probe
    # merely because each file compiled in isolation.  Candidate-contract
    # failures are repairable with task-free facts; provider/asset/provenance
    # failures are explicitly retained for a same-candidate rerun instead.
    from bundle_validation import bundle_smoke_test
    smoke = {}
    smoke_repairs = []
    repair_history: list[dict] = []
    smoke_attempts = []
    smoke_repair_budget = max(
        0,
        int(os.environ.get("BUNDLE_SMOKE_REPAIR_ATTEMPTS", str(args.repair_attempts)) or 0),
    )
    for smoke_attempt in range(smoke_repair_budget + 1):
        if validation.get("passed"):
            smoke = bundle_smoke_test(
                bundle,
                workspace_dir=workspace_dir,
                distribution_manifest=getattr(args, "distribution_manifest", ""),
                reference_results=str(diagnosis_report.get("reference_results_path") or ""),
                reference_report=str(diagnosis_report.get("reference_report_path") or ""),
                reference_bundle=str(diagnosis_report.get("reference_bundle_path") or ""),
                failure_contract={
                    "handoff_contracts": bundle.get("handoff_contracts") or [],
                    "contract_mode": "adaptive_bundle",
                    "runtime_capabilities": (
                        (bundle.get("implementation_spec") or {}).get("runtime_capabilities") or []
                    ),
                },
            )
            if smoke.get("passed") or smoke.get("execution_layer_rerun_required"):
                smoke_attempts.append(_persist_bundle_smoke_attempt(
                    run_dir, smoke_attempt + 1, bundle, smoke,
                ))
                break
            repair_contract = bundle_smoke_failure_contract(smoke, bundle)
        else:
            repair_contract = {
                "artifact_type": "bundle_assembly_failure_contract", "schema_version": 1,
                "failure_kind": "candidate_contract",
                "observed_contract_issues": list(validation.get("issues") or []),
                "falsifiable_success": [
                    "The candidate bundle contains five named modules with non-empty source and a complete canonical combo."
                ],
            }
            smoke = {"passed": False, "failure_kind": "candidate_contract",
                     "error": str(validation.get("error") or "bundle assembly invalid")}
        attempt_artifacts = _persist_bundle_smoke_attempt(
            run_dir, smoke_attempt + 1, bundle, smoke, repair_contract,
        )
        smoke_attempts.append(attempt_artifacts)
        if smoke_attempt >= smoke_repair_budget:
            break
        repair_prefix = os.path.join(run_dir, f"bundle_smoke_repair_{smoke_attempt + 1}")
        trace_path = attempt_artifacts["engineering_execution_trace_path"]
        # Rebuild this packet from the candidate that *just failed*, rather
        # than reusing the original current-best context.  A repair must never
        # receive contradictory source identities when it diagnoses a runtime
        # failure produced by the newly generated bundle.
        repair_context = build_bundle_smoke_repair_context(
            diagnosis_report, bundle, edit_files, smoke, repair_contract,
            execution_trace_path=trace_path,
            repair_history=repair_history,
        )
        repair_prompt = build_bundle_smoke_repair_prompt(
            repair_contract, edit_files, execution_trace_path=trace_path,
            candidate_bundle_path=attempt_artifacts["candidate_bundle_path"],
            repair_history=repair_history,
            codegen_context=repair_context,
        )
        write_json(repair_prefix + "_contract.json", repair_contract)
        write_json(repair_prefix + "_context.json", repair_context)
        write_text(repair_prefix + "_prompt.txt", repair_prompt)
        _apply_source_snapshot(os.path.join(run_dir, "source_after"))
        repair_result = _run_bundle_codex(
            repair_prompt, run_dir=run_dir, edit_files=edit_files,
            codex_cli=resolve_coding_agent_executable(args.codex_cli), timeout=args.timeout,
            settings=args.codex_runtime_settings,
        )
        write_text(repair_prefix + "_stdout.txt", repair_result.get("stdout", ""))
        write_text(repair_prefix + "_stderr.txt", repair_result.get("stderr", ""))
        repair_accepted = bool(repair_result.get("ok"))
        if repair_accepted:
            extraction_error = ""
            try:
                repaired_bundle = extract_candidate_bundle(
                    edit_files=edit_files, originals=originals, diagnosis_report=diagnosis_report,
                    retained_base_bundle=parent_bundle or None, active_bundle=bundle,
                )
            except RuntimeError as exc:
                repaired_bundle = {}
                extraction_error = str(exc)
            if repaired_bundle:
                bundle = repaired_bundle
                validation = pre_validate_candidate_bundle(bundle, workspace_dir=workspace_dir)
            else:
                validation = {
                    "passed": False,
                    "issues": [extraction_error or "repair produced no serializable concrete bundle"],
                    "error": extraction_error or "repair produced no serializable concrete bundle",
                }
            if forensic_manifest and bundle:
                specialization_validation = scan_candidate_source_for_specialization(bundle, forensic_manifest)
                if not specialization_validation.get("passed"):
                    validation = dict(validation or {})
                    validation["passed"] = False
                    validation["supervised_specialization_validation"] = specialization_validation
                    validation["error"] = "; ".join(filter(None, [
                        str(validation.get("error") or ""),
                        "supervised literal crossed into candidate source",
                    ]))
            inherited = _parent_bundle_inheritance_issues(bundle, parent_bundle) if bundle else []
            if inherited:
                validation = dict(validation or {})
                validation["passed"] = False
                validation["parent_bundle_inheritance_issues"] = inherited
                validation["error"] = "; ".join(
                    [str(validation.get("error") or "")] + inherited
                ).strip("; ")
            _copy_source_snapshot(edit_files, os.path.join(run_dir, "source_after"))
            _write_source_diff(originals, os.path.join(run_dir, "source_diff.patch"))
            if not args.keep_source_edits:
                _restore_files(workspace_originals)
        else:
            # A failed Codex invocation cannot become a hidden source mutation
            # for the next retry; restore the last accepted candidate source.
            _apply_source_snapshot(os.path.join(run_dir, "source_after"))
        repair_status = (
            "repair_edit_accepted_pending_runtime_smoke"
            if repair_accepted and validation.get("passed") else
            "repair_edit_unserializable"
            if repair_accepted else "repair_edit_rejected"
        )
        repair_record = {
            "attempt": smoke_attempt + 1,
            "status": repair_status,
            "input_contract_path": repair_prefix + "_contract.json",
            "smoke_attempt_artifacts": attempt_artifacts,
            "engineering_execution_trace_path": trace_path,
            "codex": {key: repair_result.get(key) for key in ("ok", "returncode", "elapsed_sec")},
            "repair_evidence_mode": "embedded_no_label_smoke_context",
            "output_assembly_validation": validation,
        }
        smoke_repairs.append(repair_record)
        repair_history.append({
            "attempt": smoke_attempt + 1,
            "repair_status": repair_record["status"],
            "failure_signature": _bundle_smoke_failure_signature(smoke, repair_contract),
            "failure_contract_path": repair_prefix + "_contract.json",
            "candidate_bundle_before_smoke": attempt_artifacts["candidate_bundle_path"],
        })
        if not repair_accepted:
            break
    validation["bundle_smoke"] = smoke
    if smoke.get("execution_layer_rerun_required"):
        validation["probe_next_action"] = {
            "action": "execution_layer_rerun",
            "reason": "bundle smoke recorded provider, asset, or provenance infrastructure failure",
        }
    elif not smoke.get("passed"):
        validation["probe_next_action"] = {
            "action": "codex_engineering_repair",
            "reason": "candidate runtime smoke remained invalid after bounded repair; resume the same candidate with --engineering-failed-run",
            "codex_feedback_allowed": True,
            "feedback_mode": "engineering_no_label",
        }
    write_json(os.path.join(run_dir, "candidate_bundle.json"), bundle)
    write_json(os.path.join(run_dir, "validation.json"), {
        "assembly_validation": validation, "bundle_smoke": smoke,
        "bundle_smoke_repairs": smoke_repairs, "bundle_smoke_attempts": smoke_attempts,
    })
    report["bundle_smoke_repairs"] = smoke_repairs
    report["bundle_smoke_attempts"] = smoke_attempts
    if not smoke.get("passed"):
        final_trace_path = str((smoke_attempts[-1] if smoke_attempts else {}).get(
            "engineering_execution_trace_path", ""
        ))
        report.update({
            "status": (
                "bundle_smoke_execution_layer_rerun_required"
                if smoke.get("execution_layer_rerun_required") else "bundle_smoke_failed"
            ),
            "bundle_smoke": smoke,
            "bundle_smoke_execution_trace_path": final_trace_path,
            "probe_next_action": (
                {"action": "execution_layer_rerun", "reason": "smoke recorded provider/asset/provenance failure"}
                if smoke.get("execution_layer_rerun_required") else {
                    "action": "codex_engineering_repair",
                    "reason": "candidate runtime smoke remained invalid after bounded repair; resume the same candidate with --engineering-failed-run",
                    "codex_feedback_allowed": True,
                    "feedback_mode": "engineering_no_label",
                }
            ),
        })
        write_json(os.path.join(run_dir, "sandbox_report.json"), {"bundle_smoke": smoke})
        report["bundle_feedback_lineage_path"] = _write_bundle_lineage(
            run_dir,
            str(feedback_parent.get("probe_failed_run") or engineering_feedback.get("engineering_failed_run") or ""),
        )
        finalize_run_record(run_dir, report)
        # The candidate is complete and immutable; a caller may rerun this
        # exact smoke after repairing execution infrastructure.  Do not route
        # that condition into another Codex edit.
        # A persisted candidate engineering failure is a completed state
        # transition, not a launcher crash. Returning zero lets the staged
        # runner preserve candidate_run and route it to an explicit resume.
        return 0
    if args.skip_probe_verification:
        report.update({"status": "bundle_smoke_passed_probe_skipped", "bundle_smoke": smoke})
        # Preserve the actual smoke outcome.  ``--skip_probe_verification``
        # only suppresses a later behavioral probe; it does not make the
        # successful injected runtime execution disappear.
        write_json(os.path.join(run_dir, "sandbox_report.json"), {
            "bundle_smoke": smoke,
            "probe_skipped": True,
            "skip_reason": "--skip_probe_verification",
        })
        finalize_run_record(run_dir, report)
        return 0
    sandbox = run_bundle_sandbox_verification(
        diagnosis_report, workspace_dir, args.first_n, bundle,
        os.path.join(run_dir, "sandbox"), concurrency=args.concurrency,
    )
    verification = sandbox.get("verification") or {}
    # Probe feedback compares two real executions in the same fixed task batch.
    # Historical full-eval rows remain useful context, but cannot substitute for
    # a current-best replay under the candidate's time-bounded probe settings.
    selected_refs = [str(item) for item in ((verification.get("probe_plan") or {}).get("selected_refs") or []) if str(item)]
    reference_replay = {"passed": False, "selected_refs": selected_refs}
    if selected_refs:
        try:
            replay_report = copy.deepcopy(diagnosis_report)
            replay_report["fixed_probe_task_ids"] = selected_refs
            candidate_repeat = run_bundle_sandbox_verification(
                replay_report, workspace_dir, len(selected_refs), bundle,
                os.path.join(run_dir, "candidate_probe_repeat"), concurrency=args.concurrency,
            )
            candidate_repeat_rows = list(candidate_repeat.get("probe_results") or [])
            candidate_repeat_ids = [str(item.get("task_id") or "") for item in candidate_repeat_rows]
            replay_bundle = _reference_bundle_for_probe_replay(replay_report)
            replay = run_bundle_sandbox_verification(
                replay_report, workspace_dir, len(selected_refs), replay_bundle,
                os.path.join(run_dir, "reference_probe_replay"), concurrency=args.concurrency,
            )
            replay_rows = list(replay.get("probe_results") or [])
            replay_ids = [str(item.get("task_id") or "") for item in replay_rows]
            candidate_rows = list(sandbox.get("probe_results") or [])
            candidate_ids = [str(item.get("task_id") or "") for item in candidate_rows]
            expected_ids = set(selected_refs)
            first_correct = {str(item.get("task_id") or ""): bool(item.get("is_correct")) for item in candidate_rows}
            repeat_correct = {str(item.get("task_id") or ""): bool(item.get("is_correct")) for item in candidate_repeat_rows}
            repeatable = (
                set(candidate_repeat_ids) == expected_ids
                and len(candidate_repeat_ids) == len(selected_refs)
                and first_correct == repeat_correct
            )
            complete = (
                set(candidate_ids) == expected_ids and len(candidate_ids) == len(selected_refs)
                and set(replay_ids) == expected_ids and len(replay_ids) == len(selected_refs)
                and len(expected_ids) == len(selected_refs)
                and repeatable
            )
            replay_path = _write_replay_jsonl(
                os.path.join(run_dir, "reference_probe_replay.jsonl"), replay_rows,
            )
            candidate_replay_path = _write_replay_jsonl(
                os.path.join(run_dir, "candidate_probe_replay.jsonl"), candidate_rows,
            )
            candidate_repeat_path = _write_replay_jsonl(
                os.path.join(run_dir, "candidate_probe_repeat.jsonl"), candidate_repeat_rows,
            )
            # The historical full-eval reference is useful for selecting this
            # probe batch, but it is not a valid comparator for a stochastic,
            # bounded runtime execution.  Recompute every benefit metric from
            # the candidate and current-best executions produced above.
            live_comparison = apply_live_probe_comparison(
                verification, candidate_rows, replay_rows,
                candidate_replay_path=candidate_replay_path,
                reference_replay_path=replay_path,
            )
            reference_replay = {
                "passed": complete,
                "selected_refs": selected_refs,
                "candidate_replay_path": candidate_replay_path,
                "candidate_repeat_path": candidate_repeat_path,
                "reference_replay_path": replay_path,
                "candidate_sandbox_dir": verification.get("sandbox_dir", ""),
                "reference_sandbox_dir": (replay.get("verification") or {}).get("sandbox_dir", ""),
                "candidate_task_ids": candidate_ids,
                "candidate_repeat_task_ids": candidate_repeat_ids,
                "reference_task_ids": replay_ids,
                "repeatability": {
                    "passed": repeatable,
                    "first_correct_by_task": first_correct,
                    "repeat_correct_by_task": repeat_correct,
                },
                "live_comparison_path": "probe_verification.live_probe_comparison",
                "live_comparison_valid": bool(live_comparison.get("comparison_valid")),
            }
            if not complete or not live_comparison.get("comparison_valid"):
                verification["execution_layer_rerun_required"] = True
                verification["engineering_invalid"] = True
                verification["engineering_invalid_reason"] = (
                    "fixed-batch candidate/reference replay was incomplete, mismatched, "
                    "or could not form a valid live comparison"
                )
        except Exception as exc:
            reference_replay = {
                "passed": False,
                "selected_refs": selected_refs,
                "error": f"reference_probe_replay_failed: {type(exc).__name__}: {exc}",
            }
            verification["execution_layer_rerun_required"] = True
            verification["engineering_invalid"] = True
            verification["engineering_invalid_reason"] = reference_replay["error"]
    else:
        verification["execution_layer_rerun_required"] = True
        verification["engineering_invalid"] = True
        verification["engineering_invalid_reason"] = "candidate probe omitted fixed selected_refs; reference replay cannot be audited"
    verification["reference_probe_replay"] = reference_replay
    audit = build_bundle_probe_audit(verification, bundle)
    audit["artifact_type"] = "bundle_probe_audit"
    probe_next_action = classify_probe_next_action(verification, audit)
    validation["sandbox"] = sandbox
    validation["probe_next_action"] = probe_next_action
    validation["probe_verification"] = verification
    report.update({"sandbox": sandbox, "probe_audit": audit,
                   "probe_next_action": probe_next_action, "status": "completed"})
    write_json(os.path.join(run_dir, "validation.json"), validation)
    write_json(os.path.join(run_dir, "probe_audit.json"), audit)
    write_json(os.path.join(run_dir, "probe_verification.json"), verification)
    write_json(os.path.join(run_dir, "sandbox_report.json"), sandbox)
    report["bundle_feedback_lineage_path"] = _write_bundle_lineage(
        run_dir,
        str(feedback_parent.get("probe_failed_run") or engineering_feedback.get("engineering_failed_run") or ""),
    )
    finalize_run_record(run_dir, report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Author, validate, smoke-test, and probe a MetaVideoAgent bundle."
    )
    parser.add_argument("--diagnosis", required=True, help="Validated diagnosis execution brief.")
    parser.add_argument("--machine-evaluation-contract", default="")
    parser.add_argument("--diagnosis_rollout", default="")
    parser.add_argument("--workspace", default=os.path.abspath(os.path.join(CURRENT_DIR, DEFAULT_WORKSPACE)))
    parser.add_argument("--distribution-manifest", required=True)
    parser.add_argument("--reference-results", default="")
    parser.add_argument("--reference-report", default="")
    parser.add_argument("--reference-label", default="")
    parser.add_argument("--first_n", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true", default=True)
    parser.add_argument("--run", dest="dry_run", action="store_false")
    parser.add_argument("--keep_source_edits", action="store_true")
    parser.add_argument(
        "--codex-cli",
        default="",
        help="Codex executable or a wrapper implementing the same command contract.",
    )
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--codex-reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--repair-attempts", type=int, default=3)
    parser.add_argument("--probe-failed-run", default="")
    parser.add_argument("--engineering-failed-run", default="")
    parser.add_argument("--probe-feedback-round", type=int, default=1)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--skip_probe_verification", action="store_true")
    args = parser.parse_args()

    try:
        args.codex_runtime_settings = resolve_codex_runtime_settings(
            model=args.codex_model,
            reasoning_effort=args.codex_reasoning_effort,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.distribution_manifest = resolve_distribution_manifest_path(
        args.distribution_manifest
    )
    if int(args.first_n or 0) <= 0:
        spec = load_distribution_spec(args.distribution_manifest)
        budgets = spec.get("budgets") if isinstance(spec.get("budgets"), dict) else {}
        try:
            args.first_n = int(budgets.get("probe_first_n") or 0)
        except (TypeError, ValueError):
            args.first_n = 0
    if args.first_n <= 0:
        raise SystemExit(
            "MetaVideoAgent evolution requires --first_n or a positive "
            "budgets.probe_first_n in the distribution manifest."
        )

    require_metavideoagent_runtime("codex_evolve")
    diagnosis_report = _load_diagnosis(args.diagnosis)
    brief_issues = validate_bundle_execution_brief(diagnosis_report)
    if brief_issues:
        raise SystemExit(
            "Invalid diagnosis execution brief: " + "; ".join(brief_issues)
        )
    return run_bundle_evolution(args, diagnosis_report, _abs(args.workspace))


if __name__ == "__main__":
    raise SystemExit(main())
