"""Run an isolated full evaluation for a saved MetaVideoAgent bundle.

This runner is intentionally separate from the closed-loop evolution entrypoint:
it reuses sandbox_evaluator for real Agent execution, but writes a normalized
full-eval results JSONL and report under the candidate run directory so the next
review and diagnosis can consume exactly the resulting trajectories.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
from runtime_config import bootstrap_cli_api_env

bootstrap_cli_api_env(sys.argv[1:])
from runtime_paths import (
    default_workspace,
    require_metavideoagent_runtime,
    runtime_id,
    runtime_output_root,
    use_runtime_path,
)

RUNTIME_DIR = use_runtime_path()
from capability_registry import DEFAULT_PROFILE_IDS, resolve_profile

DEFAULT_LLM_MODEL = str(resolve_profile("llm").get("model_id") or "")

from combo_contract import require_complete_agent_combo
from evaluation_audit import build_full_eval_audit, write_audit_files
from post_full_eval_decider import (
    CURRENT_BEST_SELECTION_SPLIT,
    current_best_update_allowed,
    decide_post_full_eval,
)
from question_set_loader import (
    load_questions as load_question_set,
)
from question_set_loader import (
    require_evolution_task_contract,
)
from reference_loader import combo_from_reference
from sandbox_evaluator import (
    BUNDLE_MODULE_ORDER,
    apply_effectiveness_policy,
    build_reference_lookup,
    compare_with_reference,
    evaluate_combo_multi_video,
    inject_module_bundle,
    prebuild_isolated_video_structures,
    validate_bundle_assembly,
)
from task_identity import attach_task_identity

MODULE_TO_COMBO_KEY = {
    "video_structuring": "video_structuring",
    "thinking": "thinking",
    "memory": "memory",
    "localization": "localization",
    "perception": "perception",
}

# The train split supplies review, diagnosis, and current-best selection.
# The held-out split is read-only reporting evidence.
EVOLUTION_SPLIT = "train"
HELD_OUT_TEST_SPLIT = "test"


def _file_digest(path: str) -> str:
    if not path or not os.path.exists(path) or os.path.isdir(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _active_vlm_model_name() -> str:
    return str(resolve_profile("vlm").get("model_id") or "")


def _runtime_concurrency_limit(env_name: str, default: int, ceiling: int) -> int:
    """Mirror the execution adapter's bounded per-capability limiter.

    Reports must state the limit actually in force for this run rather than a
    stale literal, otherwise the recorded run configuration disagrees
    with the runtime API limiter.
    """
    try:
        value = int(os.environ.get(env_name, str(default)) or default)
    except ValueError:
        value = default
    return max(1, min(value, ceiling))


def _execution_capability_limits() -> dict:
    try:
        vlm_starts_per_second = max(
            0, min(100, int(os.environ.get("METAVIDEOAGENT_VLM_STARTS_PER_SECOND", "0") or 0))
        )
    except ValueError:
        vlm_starts_per_second = 0
    return {
        "vlm_max_concurrency": _runtime_concurrency_limit(
            "METAVIDEOAGENT_VLM_MAX_CONCURRENCY", 32,
            128 if vlm_starts_per_second > 0 else 32,
        ),
        "vlm_starts_per_second": vlm_starts_per_second,
        "llm_max_concurrency": _runtime_concurrency_limit(
            "METAVIDEOAGENT_LLM_MAX_CONCURRENCY", 8, 16
        ),
        "asr_max_concurrency": _runtime_concurrency_limit(
            "METAVIDEOAGENT_ASR_MAX_CONCURRENCY", 16, 16
        ),
        "embedding_max_concurrency": _runtime_concurrency_limit(
            "METAVIDEOAGENT_EMBEDDING_MAX_CONCURRENCY", 16, 16
        ),
    }


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_optional(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    return _read_json(path)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_result_rows(path: str) -> List[dict]:
    rows = []
    if not path or not os.path.exists(path):
        return rows
    paths = [path]
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "*_sandbox.jsonl")))
    for item in paths:
        with open(item, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _resolve_path(path: str, *bases: str) -> str:
    if not path:
        return ""
    candidates = [path]
    if not os.path.isabs(path):
        candidates.extend(
            os.path.normpath(os.path.join(base, path))
            for base in bases
            if base
        )
        candidates.append(os.path.abspath(path))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return path


def _load_or_build_distribution_profile(workspace: str, manifest: str = "",
                                        profile_path: str = "") -> dict:
    if profile_path:
        return _read_json(profile_path)
    if not manifest:
        return {}
    from distribution_profiler import build_observed_distribution_profile
    from distribution_spec import load_distribution_spec
    spec = load_distribution_spec(manifest)
    out_dir = os.path.join(runtime_output_root(), "profiles")
    out_path = os.path.join(out_dir, f"observed_distribution_profile_{int(time.time())}.json")
    return build_observed_distribution_profile(
        workspace,
        spec,
        output_path=out_path,
        include_execution_artifacts=False,
    )


def _resolve_reference_results(workspace: str, args, diagnosis_context: dict = None) -> tuple:
    report_path = getattr(args, "reference_report", "") or ""
    results_path = getattr(args, "reference_results", "") or ""
    if not (report_path or results_path):
        raise RuntimeError(
            "MetaVideoAgent full_eval requires --reference-report or --reference-results; "
            "the public runner does not infer reference artifacts."
        )
    label = getattr(args, "reference_label", "") or "current_best"

    if report_path:
        report_path = _resolve_path(report_path, os.getcwd())
        report = _read_json(report_path)
        if not getattr(args, "reference_label", ""):
            label = report.get("reference_label") or (
                "initial_combo"
                if report.get("validation_scope") == "initial_reference_multi_video"
                else label
            )
        results_path = (
            results_path
            or report.get("results_path", "")
            or report.get("full_eval_results", "")
            or report.get("source_full_eval_results", "")
            or report.get("sandbox_dir", "")
        )
        results_path = _resolve_path(
            results_path,
            os.path.dirname(report_path),
            workspace,
            os.getcwd(),
        )
    elif results_path:
        results_path = _resolve_path(results_path, workspace, os.getcwd())
    if results_path:
        rows = _read_result_rows(results_path)
        if not rows:
            raise RuntimeError(
                "MetaVideoAgent full_eval requires readable explicit reference rows; "
                f"resolved reference path has no rows: {results_path}"
            )
        return rows, results_path, label
    raise RuntimeError(
        "MetaVideoAgent full_eval could not resolve reference results from "
        f"reference_report={report_path!r} reference_results={results_path!r}"
    )


def _combo_from_report(path: str) -> dict:
    if not path:
        return {}
    path = _resolve_path(path, os.getcwd())
    if not os.path.exists(path):
        return {}
    report = _read_json(path)
    combo = report.get("combo") or report.get("agent_combo") or {}
    return dict(combo) if isinstance(combo, dict) else {}


def _combo_from_plan(path: str) -> dict:
    if not path:
        return {}
    path = _resolve_path(path, os.getcwd())
    if not os.path.exists(path):
        return {}
    plan = _read_json(path)
    combo = plan.get("combo") or {}
    return dict(combo) if isinstance(combo, dict) else {}


def _agent_combo_from_evolution_combo(combo: dict) -> dict:
    if not isinstance(combo, dict):
        return {}
    converted = {}
    key_map = {
        "video_structuring": "video_structuring",
        "thinking": "thinking",
        "memory": "memory",
        "localization": "localization",
        "perception": "perception",
    }
    for key, value in combo.items():
        mapped = key_map.get(key)
        if mapped and value:
            converted[mapped] = value
    return converted


def _resolve_base_combo(args, diagnosis_context: dict) -> tuple:
    def _complete(combo: dict, source: str) -> tuple:
        return require_complete_agent_combo(combo, source), source

    plan_combo = _combo_from_plan(getattr(args, "combo_plan", ""))
    if plan_combo:
        return _complete(plan_combo, "combo_plan")
    report_combo = _combo_from_report(getattr(args, "reference_report", ""))
    if report_combo:
        return _complete(report_combo, "reference_report")
    reference_results = getattr(args, "reference_results", "") or ""
    if reference_results:
        reference_results = _resolve_path(reference_results, os.getcwd())
        reference_combo = _agent_combo_from_evolution_combo(
            combo_from_reference(reference_results)
        )
        if reference_combo:
            return _complete(reference_combo, "reference_results")
    if str(getattr(args, "question_split", "") or "") == HELD_OUT_TEST_SPLIT:
        frozen_bundle = _read_json_optional(getattr(args, "candidate_bundle", "") or "")
        frozen_combo = frozen_bundle.get("combo") if isinstance(frozen_bundle, dict) else {}
        if frozen_combo:
            return _complete(frozen_combo, "frozen_candidate_bundle")
    decision = diagnosis_context.get("evolution_decision", {}) or {}
    for key in ("base_combo", "combo_base_config", "current_best_combo"):
        value = decision.get(key)
        if isinstance(value, dict) and value:
            return _complete(value, f"diagnosis.{key}")
    raise RuntimeError(
        "Full-eval requires an explicit base combo from --reference-report, "
        "--reference-results, --combo-plan, or diagnosis context. Refusing to "
        "fall back to an implicit/default combo."
    )


def _reference_bundle_candidate_paths(args, ref_report: dict,
                                      ref_report_path: str = "") -> List[str]:
    paths = []
    explicit = getattr(args, "reference_bundle", "") or ""
    if explicit:
        paths.append(explicit)

    for key in (
        "bundle_path",
        "initial_bundle_path",
        "current_best_bundle_path",
        "candidate_full_eval_bundle_path",
        "effective_initial_baseline_bundle",
    ):
        value = ref_report.get(key)
        if isinstance(value, str) and value:
            paths.append(value)

    initial_bundle = ref_report.get("initial_baseline_bundle")
    if isinstance(initial_bundle, str) and initial_bundle:
        paths.append(initial_bundle)
    elif isinstance(initial_bundle, dict):
        for key in ("path", "bundle_path", "source_path"):
            value = initial_bundle.get(key)
            if isinstance(value, str) and value:
                paths.append(value)

    combo_policy = ref_report.get("combo_policy") or {}
    if isinstance(combo_policy, dict):
        for key in (
            "current_best_bundle_path",
            "effective_bundle_path",
            "base_bundle_path",
        ):
            value = combo_policy.get(key)
            if isinstance(value, str) and value:
                paths.append(value)

    resolved = []
    seen = set()
    for path in paths:
        resolved_path = _resolve_path(
            path,
            os.path.dirname(ref_report_path) if ref_report_path else "",
            os.getcwd(),
        )
        if resolved_path and resolved_path not in seen:
            seen.add(resolved_path)
            resolved.append(resolved_path)
    return resolved


def _bundle_combo(bundle: dict) -> dict:
    try:
        return require_complete_agent_combo(bundle.get("combo") or {}, "reference bundle combo")
    except Exception:
        return {}


def _bundle_matches_combo(bundle: dict, combo: dict) -> bool:
    bundle_combo = _bundle_combo(bundle)
    return bool(bundle_combo) and bundle_combo == require_complete_agent_combo(combo, "base combo")


def _bundle_modules_by_type(bundle: dict) -> dict:
    raw = bundle.get("modules") or {}
    if isinstance(raw, dict):
        return {
            str(module_type): dict(record, module_type=record.get("module_type") or module_type)
            for module_type, record in raw.items() if isinstance(record, dict)
        }
    return {
        str(record.get("module_type") or ""): dict(record)
        for record in raw if isinstance(record, dict) and record.get("module_type")
    }


def _resolve_reference_bundle(args, ref_report: dict, ref_report_path: str,
                              base_combo: dict) -> tuple:
    for path in _reference_bundle_candidate_paths(args, ref_report, ref_report_path):
        if not path or not os.path.exists(path):
            continue
        try:
            bundle = _read_json(path)
        except Exception:
            continue
        if _bundle_matches_combo(bundle, base_combo):
            validation = validate_bundle_assembly(bundle)
            return bundle, path, validation
    return {}, "", {
        "passed": False,
        "issues": [
            "No reference bundle with full module code matched the base combo. "
            "MetaVideoAgent full-eval refuses to alias Codex bundle names to built-in registry keys."
        ],
    }


def _module_code_digest(module: dict) -> str:
    code = module.get("code", "") if isinstance(module, dict) else ""
    return hashlib.sha256(str(code).encode("utf-8")).hexdigest() if code else ""


def _structuring_implementation_digest(module: dict) -> tuple[str, str]:
    """Return the source digest used for structure-cache compatibility."""
    return _module_code_digest(module), "literal_source"


def _planned_structure_reuse_decision(base_bundle: dict, candidate_bundle: dict) -> dict:
    """Choose rebuild/reuse from the two complete bundle manifests."""
    reference_combo = (base_bundle.get("combo") or {}) if isinstance(base_bundle, dict) else {}
    candidate_combo = (candidate_bundle.get("combo") or {}) if isinstance(candidate_bundle, dict) else {}
    reference_struct = reference_combo.get("video_structuring", "")
    candidate_struct = candidate_combo.get("video_structuring", "")
    reference_module = _bundle_modules_by_type(base_bundle).get("video_structuring") or {}
    candidate_module = _bundle_modules_by_type(candidate_bundle).get("video_structuring") or {}
    reference_raw_sha = _module_code_digest(reference_module)
    candidate_raw_sha = _module_code_digest(candidate_module)
    reference_sha, reference_digest_source = _structuring_implementation_digest(reference_module)
    candidate_sha, candidate_digest_source = _structuring_implementation_digest(candidate_module)
    if not reference_struct or not candidate_struct:
        raise RuntimeError(
            "Cannot determine structure reuse from incomplete reference/candidate combo: "
            f"reference_struct={reference_struct!r}, candidate_struct={candidate_struct!r}"
        )
    if candidate_struct != reference_struct:
        rebuild, reason = True, "struct_module_name_changed"
    elif candidate_sha and reference_sha and candidate_sha != reference_sha:
        rebuild, reason = True, "struct_module_code_changed"
    elif candidate_sha and not reference_sha:
        rebuild, reason = True, "reference_struct_code_unavailable"
    else:
        rebuild, reason = False, "struct_implementation_matches_current_best"
    return {
        "rebuild_structure": rebuild,
        "reason": reason,
        "reference_struct": reference_struct,
        "candidate_struct": candidate_struct,
        "reference_struct_code_sha256": reference_sha,
        "candidate_struct_code_sha256": candidate_sha,
        "reference_struct_raw_code_sha256": reference_raw_sha,
        "candidate_struct_raw_code_sha256": candidate_raw_sha,
        "reference_struct_digest_source": reference_digest_source,
        "candidate_struct_digest_source": candidate_digest_source,
    }


def _combo_registry_status(combo: dict) -> dict:
    try:
        import module_map
    except Exception as exc:
        return {"registered": False, "missing": [], "error": str(exc)}
    registries = {
        "video_structuring": getattr(module_map, "STRUCTURING_MAP", {}),
        "thinking": getattr(module_map, "THINKING_MAP", {}),
        "memory": getattr(module_map, "WORK_MEMORY_MAP", {}),
        "localization": getattr(module_map, "LOCALIZATION_MAP", {}),
        "perception": getattr(module_map, "PERCEPTION_MAP", {}),
    }
    missing = [
        {"slot": slot, "module_name": name}
        for slot, name in (combo or {}).items()
        if slot in registries and name not in registries[slot]
    ]
    return {"registered": not missing, "missing": missing, "error": ""}


def _external_history_meta(*paths: str) -> dict:
    formal_output_root = os.path.abspath(runtime_output_root())
    external = []
    for path in paths:
        if not path:
            continue
        resolved = _resolve_path(path, os.getcwd())
        abs_path = os.path.abspath(resolved)
        if not abs_path.startswith(formal_output_root + os.sep):
            external.append(abs_path)
    return {
        "external_history_input": bool(external),
        "external_history_paths": external,
    }



def _infer_structure_suffix_from_bundle(bundle: dict, struct_name: str) -> str:
    if not bundle or not struct_name:
        return ""
    for module in bundle.get("modules") or []:
        if not isinstance(module, dict):
            continue
        if module.get("module_type") != "video_structuring":
            continue
        if module.get("name") != struct_name:
            continue
        code = module.get("code") or ""
        keyword = re.search(
            r"\bstruct_type\s*=\s*[\"']([A-Za-z0-9_-]+)[\"']",
            code,
        )
        if keyword:
            return keyword.group(1)
        explicit = re.search(
            r"(?:VideoStructuringBase|super)\s*\.\s*__init__\s*\([^\n]*?[\"']([A-Za-z0-9_]+)[\"']",
            code,
        )
        if explicit:
            return explicit.group(1)
    return ""


def _structure_suffix_for_combo(combo: dict, bundle: dict = None) -> str:
    struct_key = (combo or {}).get("video_structuring") or ""
    inferred = _infer_structure_suffix_from_bundle(bundle or {}, struct_key)
    if inferred:
        return inferred
    return struct_key


def _check_required_structure(workspace: str, video_ids: List[str],
                              combo: dict = None, structure_dir: str = "",
                              bundle: dict = None) -> List[dict]:
    suffix = _structure_suffix_for_combo(combo or {}, bundle=bundle)
    if not suffix:
        return []
    if isinstance(structure_dir, (list, tuple)):
        base_dirs = [str(item) for item in structure_dir if item]
    else:
        base_dirs = [structure_dir] if structure_dir else []
    base_dirs = base_dirs or [os.path.join(workspace, "video_structure")]
    checks = []
    for video_id in video_ids:
        chosen_dir = base_dirs[0]
        path = os.path.join(chosen_dir, f"{video_id}_{suffix}.jsonl")
        for base_dir in base_dirs:
            candidate = os.path.join(base_dir, f"{video_id}_{suffix}.jsonl")
            if os.path.exists(candidate):
                chosen_dir = base_dir
                path = candidate
                break
        checks.append(
            {
                "video_id": video_id,
                "struct_key": (combo or {}).get("video_structuring", ""),
                "storage_suffix": suffix,
                "path": path,
                "source_dir": chosen_dir,
                "exists": os.path.exists(path),
                "size": os.path.getsize(path) if os.path.exists(path) else 0,
            }
        )
    return checks


def _reference_structure_dirs(report: dict, report_path: str = "") -> List[str]:
    candidates = []
    if isinstance(report, dict):
        artifact = report.get("structure_artifact") or {}
        if isinstance(artifact, dict):
            for key in ("materialized_dir", "source_dir"):
                if artifact.get(key):
                    candidates.append(artifact[key])
        paths = report.get("paths", {}) if isinstance(report.get("paths"), dict) else {}
        for key in ("structure_dir",):
            if report.get(key):
                candidates.append(report[key])
            if paths.get(key):
                candidates.append(paths[key])
        sandbox = report.get("sandbox_dir") or paths.get("sandbox_dir")
        if sandbox:
            candidates.append(os.path.join(sandbox, "video_structure"))
        for source in report.get("sources", []) or []:
            if not isinstance(source, str) or not source:
                continue
            source_path = _resolve_path(
                source,
                os.path.dirname(os.path.abspath(report_path)) if report_path else "",
                os.getcwd(),
            )
            source_dir = os.path.dirname(os.path.abspath(source_path))
            candidates.append(os.path.join(source_dir, "sandbox", "video_structure"))
            sibling_report = os.path.join(source_dir, "initial_baseline_report.json")
            if os.path.exists(sibling_report):
                try:
                    sibling = _read_json(sibling_report)
                    if sibling.get("structure_dir"):
                        candidates.append(sibling["structure_dir"])
                    if sibling.get("sandbox_dir"):
                        candidates.append(os.path.join(sibling["sandbox_dir"], "video_structure"))
                except Exception:
                    pass
    bases = [os.path.dirname(os.path.abspath(report_path)) if report_path else "", os.getcwd()]
    resolved_dirs = []
    seen = set()
    for item in candidates:
        resolved = _resolve_path(item, *bases)
        if resolved and os.path.isdir(resolved):
            real = os.path.abspath(resolved)
            if real not in seen:
                seen.add(real)
                resolved_dirs.append(real)
    return resolved_dirs



def _prepopulate_structure_dir(source_dir, sandbox_dir: str) -> str:
    source_dirs = (
        [str(item) for item in source_dir if item]
        if isinstance(source_dir, (list, tuple))
        else ([source_dir] if source_dir else [])
    )
    source_dirs = [item for item in source_dirs if os.path.isdir(item)]
    if not source_dirs:
        return ""
    target = os.path.join(sandbox_dir, "video_structure")
    os.makedirs(target, exist_ok=True)
    for src in source_dirs:
        for name in os.listdir(src):
            src_path = os.path.join(src, name)
            dst_path = os.path.join(target, name)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            elif os.path.isfile(src_path) and not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
    return target


def _structure_artifact(structure_dir: str, combo: dict, video_ids: List[str],
                        rebuilt: bool, source_report_path: str = "",
                        source_artifact: dict = None,
                        source_structure_dirs: List[str] = None) -> dict:
    """Persist explicit structure provenance for later current-best reuse."""
    structure_dir = os.path.abspath(structure_dir) if structure_dir else ""
    files = []
    if structure_dir and os.path.isdir(structure_dir):
        for path in sorted(Path(structure_dir).glob("*.jsonl")):
            files.append({
                "name": path.name,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": _file_digest(str(path)),
            })
    return {
        "artifact_type": "metavideoagent_structure_artifact",
        "schema_version": 1,
        "materialized_dir": structure_dir,
        "source_dir": (
            structure_dir if rebuilt else str(
                (source_artifact or {}).get("materialized_dir")
                or ((source_structure_dirs or [""])[0])
            )
        ),
        "source_report_path": os.path.abspath(source_report_path) if source_report_path else "",
        "struct_module_name": (combo or {}).get("video_structuring", ""),
        "rebuilt_for_candidate": bool(rebuilt),
        "reused_from_current_best": bool(not rebuilt),
        "video_ids": list(video_ids or []),
        "jsonl_files": files,
    }


def _check_required_videos(workspace: str, video_ids: List[str]) -> List[dict]:
    checks = []
    for video_id in video_ids:
        path = os.path.join(workspace, "raw_videos", f"{video_id}.mp4")
        checks.append(
            {
                "video_id": video_id,
                "path": path,
                "exists": os.path.exists(path),
                "size": os.path.getsize(path) if os.path.exists(path) else 0,
            }
        )
    return checks


def _safe_remove(path: str, candidate_run: str) -> None:
    abs_path = os.path.abspath(path)
    abs_run = os.path.abspath(candidate_run)
    if not abs_path.startswith(abs_run + os.sep):
        raise ValueError(f"Refuse to remove path outside candidate run: {path}")
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _prepare_outputs(candidate_run: str, overwrite: bool, resume: bool = False) -> dict:
    paths = {
        "sandbox_dir": os.path.join(candidate_run, "full_eval_sandbox"),
        "results_path": os.path.join(candidate_run, "full_eval_results.jsonl"),
        "report_path": os.path.join(candidate_run, "full_eval_report.json"),
        "summary_path": os.path.join(candidate_run, "full_eval_summary.md"),
        "audit_path": os.path.join(candidate_run, "evaluation_audit.json"),
        "audit_summary_path": os.path.join(candidate_run, "evaluation_audit_summary.md"),
        "provider_audit_path": os.path.join(candidate_run, "provider_calls.jsonl"),
        "post_eval_decision_path": os.path.join(candidate_run, "post_full_eval_decision.json"),
        "provider_failure_path": os.path.join(candidate_run, "full_eval_provider_failures.json"),
    }
    existing = [path for path in paths.values() if os.path.exists(path)]
    if resume:
        # A resumed run may have only the append-only per-question sandbox
        # artifacts.  Final rows/reports are terminal artifacts: resuming over
        # them would silently replace an already measured full evaluation.
        terminal = [
            path for key, path in paths.items()
            # ``provider_calls.jsonl`` is deliberately append-only: a
            # resumed full eval must retain the start/finish audit events
            # already recorded before interruption.  It is not a completed
            # evaluation result and therefore must not block ``--resume``.
            if key not in {"sandbox_dir", "provider_audit_path"}
            and os.path.exists(path)
        ]
        if terminal:
            raise FileExistsError(
                "resume refused: final full-eval artifacts already exist; "
                f"start a new candidate-run instead: {terminal}"
            )
        if not os.path.isdir(paths["sandbox_dir"]):
            raise FileNotFoundError(
                "resume refused: no sandbox_dir exists to resume: "
                f"{paths['sandbox_dir']}"
            )
        return paths
    # A newly scoped recovery may deliberately pre-place immutable per-video
    # JSONL structures before its first evaluation.  These are inputs, not
    # prior evaluation outputs: accept only this exact empty-sandbox shape.
    sandbox_dir = paths["sandbox_dir"]
    if sandbox_dir in existing:
        allowed_preseed = os.path.join(sandbox_dir, "video_structure")
        sandbox_entries = set(os.listdir(sandbox_dir)) if os.path.isdir(sandbox_dir) else set()
        structure_entries = (
            set(os.listdir(allowed_preseed))
            if os.path.isdir(allowed_preseed) else set()
        )
        preseed_only = (
            sandbox_entries == {"video_structure"}
            and bool(structure_entries)
            and all(name.endswith(".jsonl") for name in structure_entries)
        )
        if preseed_only:
            existing.remove(sandbox_dir)
    if existing and not overwrite:
        raise FileExistsError(
            "Full-eval outputs already exist. Pass --overwrite to replace only "
            f"candidate-run outputs: {existing}"
        )
    if overwrite:
        for path in existing:
            _safe_remove(path, candidate_run)
    os.makedirs(paths["sandbox_dir"], exist_ok=True)
    return paths


def _seed_resume_sandbox(target_sandbox_dir: str, source_dirs: List[str],
                         copy_structures: bool = False) -> dict:
    """Copy immutable completed sandbox artifacts into a new resume run.

    This is deliberately a file-level provenance operation, not a result-row
    merger: the resumed evaluator later validates every copied task row against
    the exact bundle/combo and produces the only final report itself.
    """
    if not source_dirs:
        return {"artifact_type": "full_eval_resume_seed", "sources": [], "files": []}
    if os.listdir(target_sandbox_dir):
        raise RuntimeError("resume seed requires an empty target sandbox directory")
    manifest = {"artifact_type": "full_eval_resume_seed", "schema_version": 1,
                "sources": [], "files": [], "structure_sources": [],
                "structures_copied": bool(copy_structures)}
    for raw_source in source_dirs:
        source = os.path.abspath(raw_source)
        if not os.path.isdir(source):
            raise FileNotFoundError(f"resume seed sandbox does not exist: {source}")
        manifest["sources"].append(source)
        candidates = sorted(glob.glob(os.path.join(source, "*_sandbox.jsonl")))
        structure_dir = os.path.join(source, "video_structure")
        if os.path.isdir(structure_dir):
            for item in sorted(glob.glob(os.path.join(structure_dir, "*.jsonl"))):
                manifest["structure_sources"].append({
                    "source": item, "sha256": _file_digest(item),
                    "size": os.path.getsize(item),
                })
            if copy_structures:
                candidates.extend(sorted(glob.glob(os.path.join(structure_dir, "*.jsonl"))))
        for item in candidates:
            rel = (
                os.path.join("video_structure", os.path.basename(item))
                if os.path.dirname(item) == structure_dir else os.path.basename(item)
            )
            target = os.path.join(target_sandbox_dir, rel)
            if os.path.exists(target):
                raise RuntimeError(
                    "resume seed refuses duplicate artifact basename: "
                    f"{rel} from {source}"
                )
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(item, target)
            manifest["files"].append({
                "source": item, "target": target,
                "sha256": _file_digest(item), "size": os.path.getsize(item),
            })
    return manifest


def _resume_contract_path(candidate_run: str) -> str:
    return os.path.join(candidate_run, "full_eval_resume_contract.json")


def _resume_contract(candidate_bundle_path: str, split: str,
                     questions: List[dict], combo_id: str) -> dict:
    """Return the immutable identity of a resumable full-eval invocation."""
    task_ids = [str(item.get("task_id") or "") for item in questions]
    if not task_ids or any(not item for item in task_ids):
        raise RuntimeError("resume contract requires non-empty task_ids")
    return {
        "artifact_type": "full_eval_resume_contract",
        "schema_version": 1,
        "candidate_bundle_path": os.path.abspath(candidate_bundle_path),
        "candidate_bundle_sha256": _file_digest(candidate_bundle_path),
        "question_split": str(split),
        "task_ids": task_ids,
        "combo_id": str(combo_id),
    }


def _ensure_resume_contract(candidate_run: str, expected: dict,
                            resume: bool) -> tuple[str, str]:
    """Create/validate a fail-closed contract before writing Agent rows."""
    path = _resume_contract_path(candidate_run)
    if resume:
        if not os.path.isfile(path):
            raise RuntimeError(
                "resume refused: missing full_eval_resume_contract.json; "
                "this run predates resumable full-eval and must remain an "
                "archived interrupted attempt"
            )
        actual = _read_json(path)
        actual_identity = dict(actual)
        actual_combo_id = str(actual_identity.pop("combo_id", "") or "")
        expected_identity = dict(expected)
        expected_identity.pop("combo_id", None)
        if not actual_combo_id or actual_identity != expected_identity:
            raise RuntimeError(
                "resume refused: candidate bundle, split, or task set differs "
                "from the original full-eval contract"
            )
        return path, actual_combo_id
    else:
        if os.path.exists(path):
            raise FileExistsError(
                f"full-eval resume contract already exists: {path}; use --resume "
                "or choose a new candidate-run"
            )
        _write_json(path, expected)
        return path, str(expected["combo_id"])


def _load_completed_sandbox_rows(sandbox_dir: str, questions: List[dict],
                                 combo: dict,
                                 rerun_provider_failures: bool = False) -> tuple[List[dict], set[str]]:
    """Load exact completed task rows for an explicit full-eval resume.

    Rows are append-only after each Agent question completes.  Duplicate ids,
    malformed JSON, or an unexpected combo are fatal rather than silently
    selecting a row from a different evaluation pass.
    """
    expected = {str(item.get("task_id") or "") for item in questions}
    if not expected or "" in expected:
        raise RuntimeError("resume requires a complete task-id-bearing question set")
    rows_by_task = {}
    duplicates = set()
    for path in sorted(glob.glob(os.path.join(sandbox_dir, "*_sandbox.jsonl"))):
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid persisted sandbox JSONL: {path}:{line_number}"
                    ) from exc
                attach_task_identity(row, fallback_video_id=str(row.get("video_id") or ""))
                task_id = str(row.get("task_id") or "")
                if task_id not in expected:
                    continue
                if row.get("combo") != combo:
                    raise RuntimeError(
                        "resume refused: persisted row combo differs from the "
                        f"requested combo ({path}:{line_number})"
                    )
                # A terminal provider failure is an execution incident, not a
                # reusable completed result when an operator explicitly asks
                # for a provider-failure-only retry.  Keep the original row on
                # disk for provenance, but schedule this task again in the
                # new isolated resume run.
                if rerun_provider_failures and task_id in _terminal_provider_failure_task_ids([row]):
                    continue
                if task_id in rows_by_task:
                    duplicates.add(task_id)
                else:
                    rows_by_task[task_id] = row
    if duplicates:
        raise RuntimeError(
            "resume refused: duplicate task_ids in persisted sandbox rows: "
            f"{sorted(duplicates)[:5]}"
        )
    ordered = [rows_by_task[str(item.get("task_id") or "")]
               for item in questions if str(item.get("task_id") or "") in rows_by_task]
    return ordered, set(rows_by_task)


def _terminal_provider_failure_task_ids(results: List[dict]) -> List[str]:
    """Return tasks whose final execution contains a terminal API error.

    ``retrying`` records preserve an earlier failed attempt but are not terminal
    failures.  The returned IDs are persisted for an explicit later task rerun;
    full evaluation itself never silently reruns or replaces question results.
    """
    task_ids = []
    for row in results or []:
        task_id = str((row or {}).get("task_id") or "")
        if not task_id:
            continue
        events = (row or {}).get("capability_events") or []
        if any(
            isinstance(event, dict) and str(event.get("status") or "") == "api_error"
            for event in events
        ):
            task_ids.append(task_id)
    return list(dict.fromkeys(task_ids))


def _provider_failure_summary(row: dict) -> List[dict]:
    """Keep explicit-rerun provenance compact while raw attempts remain in JSONL."""
    summary = []
    for event in (row or {}).get("capability_events") or []:
        if not isinstance(event, dict) or str(event.get("status") or "") != "api_error":
            continue
        summary.append({
            "capability": str(event.get("capability") or ""),
            "event": str(event.get("event") or ""),
            "profile_id": str(event.get("profile_id") or ""),
            "start_sec": event.get("start_sec"),
            "end_sec": event.get("end_sec"),
        })
    return summary


def _load_diagnosis_context(candidate_run: str, args) -> dict:
    explicit = getattr(args, "diagnosis_report", "") or ""
    payload = (
        _read_json_optional(explicit)
        if explicit else _read_json_optional(os.path.join(candidate_run, "diagnosis.json"))
    )
    if payload.get("artifact_type") == "diagnosis_execution_brief":
        # Candidate runs persist the bounded codegen handoff. Normalize the
        # policy names expected by full-eval without reopening the full audit
        # diagnosis artifact.
        payload = dict(payload)
        payload["evolution_decision"] = dict(payload.get("execution_policy", {}) or {})
        current_best = payload.get("current_best_contract", {}) or {}
        payload.setdefault("reference_bundle_path", current_best.get("reference_bundle_path", ""))
        payload.setdefault("reference_results_path", current_best.get("reference_results_path", ""))
    return payload



def _module_artifact(module_type: str, module_name: str,
                     path: str = "", source: str = "") -> dict:
    return {
        "module_type": module_type,
        "module_name": module_name,
        "path": os.path.abspath(path) if path else "",
        "sha256": _file_digest(path) if path else "",
        "source": source,
    }




def _materialize_effective_bundle(candidate_run: str, combo: dict,
                                  candidate_bundle: dict) -> str:
    """Write the exact five-module combo evaluated by this full-eval run."""
    modules_by_type = _bundle_modules_by_type(candidate_bundle)
    modules = []
    for module_type in BUNDLE_MODULE_ORDER:
        selected = dict(modules_by_type.get(module_type) or {})
        wanted_name = (combo or {}).get(MODULE_TO_COMBO_KEY[module_type], "")
        if not selected or selected.get("name") != wanted_name:
            raise RuntimeError(
                "Cannot materialize effective full-eval bundle for "
                f"{module_type}={wanted_name!r}"
            )
        selected.pop("_artifact_path", None)
        selected.pop("_artifact_container", None)
        modules.append(selected)
    path = os.path.join(candidate_run, "candidate_full_eval_bundle.json")
    payload = dict(candidate_bundle)
    payload.update({
        "combo": combo,
        "modules": modules,
        "evaluated_at": int(time.time()),
        "source_bundle_fingerprint": candidate_bundle.get("bundle_fingerprint", ""),
    })
    _write_json(path, payload)
    return path


def _dedupe_results(results: List[dict], questions: List[dict],
                    strict: bool = True) -> tuple[List[dict], List[str]]:
    by_task = {}
    for item in results:
        attach_task_identity(item, fallback_video_id=item.get("video_id", ""))
        by_task[item.get("task_id", "")] = item
    ordered = []
    missing = []
    for task in questions:
        task_id = task.get("task_id", "")
        if task_id in by_task:
            ordered.append(by_task[task_id])
        else:
            missing.append(task_id)
    if missing and strict:
        raise RuntimeError(f"Full evaluation missing {len(missing)} task results")
    return ordered, missing


def _write_summary(path: str, report: dict) -> None:
    audit = report.get("evaluation_audit") or {}
    candidate_cost = audit.get("candidate_cost") or {}
    reference_cost = audit.get("reference_cost") or {}
    candidate_avg = candidate_cost.get("per_question_avg") or {}
    reference_avg = reference_cost.get("per_question_avg") or {}
    candidate_total = candidate_cost.get("totals") or {}
    reference_total = reference_cost.get("totals") or {}
    lines = [
        "# Full Evaluation Summary",
        "",
        f"- candidate: `{report.get('candidate', {}).get('name', '')}`",
        f"- combo_id: `{report.get('combo_id', '')}`",
        f"- scope: `{report.get('validation_scope', '')}`",
        f"- total: `{report.get('total_questions')}`",
        f"- expected_total: `{report.get('expected_total_questions')}`",
        f"- evaluation_complete: `{report.get('evaluation_complete')}`",
        f"- status: `{report.get('status')}`",
        f"- engineering_invalid: `{report.get('engineering_invalid')}`",
        f"- reference_label: `{report.get('reference_label', 'baseline')}`",
        f"- reference_correct: `{report.get('reference_correct')}`",
        f"- candidate_correct: `{report.get('candidate_correct')}`",
        f"- accuracy_delta_vs_reference: `{report.get('accuracy_delta_vs_reference', report.get('accuracy_delta'))}`",
        f"- comparison_valid: `{report.get('comparison_valid')}`",
        f"- corrections: `{len(report.get('corrections', []) or [])}`",
        f"- regressions: `{len(report.get('regressions', []) or [])}`",
        f"- final_success: `{(report.get('final_success_criteria') or {}).get('passed')}`",
        f"- post_eval_next_action: `{(report.get('post_full_eval_decision') or {}).get('next_action')}`",
        f"- post_eval_update_current_best: `{(report.get('post_full_eval_decision') or {}).get('update_current_best')}`",
        f"- cost_ratio_vs_reference: `{audit.get('cost_ratio_vs_reference')}`",
        f"- candidate_per_question: frames=`{candidate_avg.get('frames')}`, vlm_calls=`{candidate_avg.get('vlm_calls')}`, llm_calls=`{candidate_avg.get('llm_calls')}`, tokens=`{candidate_avg.get('total_tokens')}`, task_latency_sec=`{candidate_avg.get('latency_sec')}`",
        f"- candidate_totals: frames=`{candidate_total.get('frames')}`, vlm_calls=`{candidate_total.get('vlm_calls')}`, llm_calls=`{candidate_total.get('llm_calls')}`, tokens=`{candidate_total.get('total_tokens')}`, task_latency_sec=`{candidate_total.get('latency_sec')}` (per-task latency sum; not wall-clock)",
        f"- reference_per_question: frames=`{reference_avg.get('frames')}`, vlm_calls=`{reference_avg.get('vlm_calls')}`, llm_calls=`{reference_avg.get('llm_calls')}`, tokens=`{reference_avg.get('total_tokens')}`, task_latency_sec=`{reference_avg.get('latency_sec')}`",
        f"- reference_totals: frames=`{reference_total.get('frames')}`, vlm_calls=`{reference_total.get('vlm_calls')}`, llm_calls=`{reference_total.get('llm_calls')}`, tokens=`{reference_total.get('total_tokens')}`, task_latency_sec=`{reference_total.get('latency_sec')}` (per-task latency sum; not wall-clock)",
        f"- audit_recommended_next_focus: `{audit.get('recommended_next_focus')}`",
        f"- results_path: `{report.get('results_path', '')}`",
        f"- sandbox_dir: `{report.get('sandbox_dir', '')}`",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _metavideoagent_ledger_path(args) -> str:
    run_id = getattr(args, "metavideoagent_run_id", "") or ""
    if not run_id:
        return ""
    output_root = getattr(args, "metavideoagent_output_root", "") or ""
    metavideoagent_dir = os.path.join(PROJECT_ROOT, "execution")
    if metavideoagent_dir not in sys.path:
        sys.path.insert(0, metavideoagent_dir)
    try:
        from ledger import ledger_path
        return ledger_path(run_id, output_root)
    except Exception:
        return ""


def _record_metavideoagent_ledger(args, report: dict, decision: dict = None,
                             decision_path: str = "") -> str:
    run_id = getattr(args, "metavideoagent_run_id", "") or ""
    if not run_id:
        return ""
    output_root = getattr(args, "metavideoagent_output_root", "") or ""
    metavideoagent_dir = os.path.join(PROJECT_ROOT, "execution")
    if metavideoagent_dir not in sys.path:
        sys.path.insert(0, metavideoagent_dir)
    try:
        from ledger import record_reference
    except Exception as exc:
        report.setdefault("ledger_warning", f"failed to import MetaVideoAgent ledger: {exc}")
        return ""
    can_update_best = current_best_update_allowed(report, decision or {})
    # Every completed full evaluation is an auditable next-round input. A
    # rejected candidate is recorded as candidate_full_eval without changing
    # the ledger current best, rather than being silently omitted.
    if report.get("evaluation_complete", True) is not True:
        return ""
    decision = decision or {}
    # The decision's preference is necessary but not sufficient; the policy
    # gate also verifies that the evaluated split is the evolution split.
    update_current_best = bool(decision.get("update_current_best") is True and can_update_best)
    ledger = record_reference(
        run_id,
        "current_best" if bool(update_current_best) else "candidate_full_eval",
        report.get("report_path", ""),
        report.get("results_path", ""),
        combo=report.get("combo", {}),
        metrics={
            "total_questions": report.get("total_questions"),
            "correct": report.get("candidate_correct"),
            "accuracy_delta": report.get("accuracy_delta"),
            "corrections": len(report.get("corrections", []) or []),
            "regressions": len(report.get("regressions", []) or []),
            "cost_ratio_vs_reference": (
                (report.get("evaluation_audit", {}) or {})
                .get("cost_ratio_vs_reference")
            ),
            "api_cost_per_question": (
                ((report.get("evaluation_audit", {}) or {})
                .get("candidate_cost", {}) or {}).get("per_question_avg", {})
            ),
        },
        output_root=output_root,
        manifest_path=getattr(args, "distribution_manifest", "") or "",
        split=getattr(args, "question_split", "") or "",
        dataset_version=report.get("dataset_version", ""),
        update_current_best=bool(update_current_best),
        decision=decision,
        decision_path=decision_path,
        bundle_path=(
            report.get("current_best_bundle_path", "")
            or report.get("candidate_full_eval_bundle_path", "")
            or (report.get("combo_policy", {}) or {}).get("effective_bundle_path", "")
        ),
        candidate_bundle_path=report.get("candidate_bundle", ""),
        module_artifacts=report.get("module_artifacts", {}),
        structure_artifact=report.get("structure_artifact", {}),
        selection_split=(report.get("current_best_selection") or {}).get("this_eval_split", ""),
        train_diagnostic_report_path=(
            (report.get("current_best_selection") or {}).get("train_diagnostic_report_path", "")
        ),
        train_diagnostic_results_path=(
            (report.get("current_best_selection") or {}).get("train_diagnostic_results_path", "")
        ),
    )
    return ledger.get("path", "")


def run(args: argparse.Namespace) -> dict:
    require_metavideoagent_runtime("full_eval_runner")
    workspace = args.workspace
    candidate_run = os.path.abspath(args.candidate_run)
    candidate_bundle_path = getattr(args, "candidate_bundle", "") or os.path.join(
        candidate_run, "candidate_bundle.json"
    )
    if not os.path.isfile(candidate_bundle_path):
        raise ValueError("MetaVideoAgent full eval requires a self-contained candidate_bundle.json")
    candidate_bundle = _read_json(candidate_bundle_path)
    split = str(getattr(args, "question_split", EVOLUTION_SPLIT) or EVOLUTION_SPLIT)
    if split not in {EVOLUTION_SPLIT, HELD_OUT_TEST_SPLIT}:
        raise ValueError(f"Unsupported full-eval question split: {split!r}")
    from bundle_contract import validate_candidate_bundle
    bundle_issues = (
        []
        if split == HELD_OUT_TEST_SPLIT
        and validate_bundle_assembly(candidate_bundle).get("passed")
        else validate_candidate_bundle(candidate_bundle)
    )
    if bundle_issues:
        raise ValueError("Invalid candidate bundle: " + "; ".join(bundle_issues))
    changed_modules = list(candidate_bundle.get("changed_modules") or [])
    diagnosis_context = _load_diagnosis_context(candidate_run, args)

    if split == HELD_OUT_TEST_SPLIT:
        train_diagnostic = getattr(args, "train_diagnostic_report", "")
        if not train_diagnostic or not os.path.isfile(train_diagnostic):
            raise ValueError(
                "held-out reporting requires --train-diagnostic-report from "
                "this frozen candidate's completed evolution-split evaluation"
            )
        train_diagnostic_payload = _read_json_optional(train_diagnostic)
        if (
            train_diagnostic_payload.get("question_split") != EVOLUTION_SPLIT
            or train_diagnostic_payload.get("evaluation_complete") is not True
        ):
            raise ValueError(
                "--train-diagnostic-report must be a completed train full eval"
            )
        paired_bundles = [
            str(train_diagnostic_payload.get(key) or "")
            for key in (
                "candidate_bundle", "candidate_full_eval_bundle_path",
                "effective_initial_baseline_bundle", "initial_baseline_bundle",
            )
        ]
        paired_digests = {
            _file_digest(path) for path in paired_bundles if path and os.path.isfile(path)
        }
        paired_digests.update({
            str(train_diagnostic_payload.get(key) or "").strip()
            for key in (
                "candidate_bundle_sha256", "candidate_full_eval_bundle_sha256",
            )
            if str(train_diagnostic_payload.get(key) or "").strip()
        })
        if not paired_digests:
            raise ValueError(
                "--train-diagnostic-report does not attest a frozen bundle digest"
            )
        if _file_digest(candidate_bundle_path) not in paired_digests:
            raise ValueError(
                "--train-diagnostic-report belongs to a different frozen bundle"
            )
    questions, video_ids, split_spec = load_question_set(
        workspace,
        video_id=args.video_id,
        distribution_manifest=getattr(args, "distribution_manifest", ""),
        preferred_splits=(split,),
    )
    if split == HELD_OUT_TEST_SPLIT:
        require_evolution_task_contract(questions, require_evidence=False)
    unexpected_splits = sorted({
        str(question.get("split"))
        for question in questions
        if question.get("split") and str(question.get("split")) != split
    })
    if unexpected_splits:
        raise RuntimeError(
            f"Full eval resolved rows outside requested {split!r} split: "
            f"{unexpected_splits}"
        )
    if len({q.get("task_id") for q in questions}) != len(questions):
        raise RuntimeError("Duplicate task_id detected in full-eval question set")
    requested_task_ids = [
        str(value or "").strip()
        for value in (getattr(args, "task_id", []) or [])
        if str(value or "").strip()
    ]
    if requested_task_ids:
        if len(set(requested_task_ids)) != len(requested_task_ids):
            raise ValueError("--task-id values must be unique")
        available_by_id = {
            str(item.get("task_id") or ""): item
            for item in questions
        }
        missing_task_ids = [
            task_id for task_id in requested_task_ids
            if task_id not in available_by_id
        ]
        if missing_task_ids:
            raise ValueError(
                "--task-id is outside the selected split/video scope: "
                f"{missing_task_ids}"
            )
        # Explicit task recovery is intentionally an isolated evaluation
        # artifact.  It never expands a user-selected repair set to other
        # rows that merely contain historical transient provider events.
        questions = [available_by_id[task_id] for task_id in requested_task_ids]
    scheduled_questions = list(questions)

    validation = validate_bundle_assembly(candidate_bundle)
    base_combo, base_combo_source = _resolve_base_combo(args, diagnosis_context)
    ref_report_path = _resolve_path(getattr(args, "reference_report", "") or "", os.getcwd())
    ref_report = _read_json_optional(ref_report_path)
    if split == HELD_OUT_TEST_SPLIT and not getattr(args, "reference_bundle", ""):
        args.reference_bundle = candidate_bundle_path
    if split == HELD_OUT_TEST_SPLIT and not (
        getattr(args, "reference_report", "") or getattr(args, "reference_results", "")
    ):
        reference_rows, reference_source, reference_label = (
            [], "held_out_reporting_without_comparison_reference", "none"
        )
    else:
        reference_rows, reference_source, reference_label = _resolve_reference_results(
            workspace,
            args,
            diagnosis_context,
        )
    if not reference_rows and split != HELD_OUT_TEST_SPLIT:
        raise RuntimeError(
            "MetaVideoAgent full eval requires readable explicit reference rows; "
            f"got reference_source={reference_source!r}."
        )
    reference_lookup = build_reference_lookup(reference_rows, reference_label=reference_label)
    missing_reference_refs = [] if split == HELD_OUT_TEST_SPLIT else [
        q.get("task_id", "") for q in questions
        if q.get("task_id", "") not in reference_lookup
        and q.get("question", "") not in reference_lookup
    ]
    base_combo_registry_status = _combo_registry_status(base_combo)
    base_bundle, base_bundle_path, base_bundle_validation = _resolve_reference_bundle(
        args, ref_report, ref_report_path, base_combo
    )
    structure_reuse_decision = _planned_structure_reuse_decision(base_bundle, candidate_bundle)
    planned_rebuild_structure = structure_reuse_decision["rebuild_structure"]
    # A reproducibility run may deliberately materialize fresh isolated
    # libraries even when its candidate preserves the structuring class.  Keep
    # reuse as the default; only this explicit flag opts into fresh VLM-backed
    # structure artifacts for every video before answers start.
    effective_rebuild_structure = bool(
        split == HELD_OUT_TEST_SPLIT
        or planned_rebuild_structure
        or getattr(args, "force_fresh_structure", False)
    )
    if args.rebuild_structure and not planned_rebuild_structure:
        raise ValueError(
            "--rebuild-structure was requested, but the injected candidate combo "
            "matches the current-best struct implementation; use "
            "--force-fresh-structure for an explicit reproducibility rebuild."
        )
    base_combo_injectable = bool(
        base_combo_registry_status.get("registered")
        or (base_bundle_path and base_bundle_validation.get("passed"))
    )
    explicit_structure_dirs = [
        os.path.abspath(str(path))
        for path in (getattr(args, "reference_structure_dir", []) or [])
        if str(path).strip()
    ]
    reference_structure_dirs = (
        explicit_structure_dirs
        if explicit_structure_dirs
        else _reference_structure_dirs(ref_report, ref_report_path)
    )
    structure_checks = _check_required_structure(
        workspace, video_ids, base_combo, structure_dir=reference_structure_dirs,
        bundle=base_bundle,
    )
    raw_video_checks = _check_required_videos(workspace, video_ids)
    missing_structures = [
        item for item in structure_checks if not item["exists"] or item["size"] <= 0
    ]
    missing_videos = [
        item for item in raw_video_checks if not item["exists"] or item["size"] <= 0
    ]

    readiness = {
        "candidate_run": candidate_run,
        "candidate_bundle": candidate_bundle_path,
        "runtime_dir": RUNTIME_DIR,
        "workspace_dir": os.path.abspath(workspace),
        "output_root": runtime_output_root(),
        "candidate": {
            "name": "MetaVideoAgent bundle",
            "bundle_fingerprint": candidate_bundle.get("bundle_fingerprint", ""),
            "changed_modules": changed_modules,
        },
        "diagnosis_context_path": (
            getattr(args, "diagnosis_report", "") or
            os.path.join(candidate_run, "diagnosis.json")
        ),
        "evolution_decision": diagnosis_context.get("evolution_decision", {}) or {},
        "assembly_validation": validation,
        "video_ids": video_ids,
        "total_questions": len(questions),
        "unique_task_ids": len({q.get("task_id") for q in questions}),
        "reference_label": reference_label,
        "reference_source": reference_source,
        "base_combo": base_combo,
        "base_combo_source": base_combo_source,
        "base_combo_registry_status": base_combo_registry_status,
        "base_bundle_path": base_bundle_path,
        "base_bundle_validation": base_bundle_validation,
        "base_combo_injectable": base_combo_injectable,
        "missing_reference_refs": missing_reference_refs,
        "question_split": split,
        "question_split_spec": {
            "manifest_path": split_spec.get("manifest_path", "") if isinstance(split_spec, dict) else "",
            "split_names": sorted((split_spec.get("splits", {}) or {}).keys()) if isinstance(split_spec, dict) else [],
        },
        "llm_model": args.llm_model,
        "vlm_model": _active_vlm_model_name(),
        "rebuild_structure": effective_rebuild_structure,
        "structure_reuse_decision": structure_reuse_decision,
        "structure_source": (
            "rebuilt_isolated_candidate_video_structuring"
            if effective_rebuild_structure and "video_structuring" in changed_modules
            else ("rebuilt_isolated_requested" if effective_rebuild_structure else "reference_or_existing_structure")
        ),
        "structure_workers": args.structure_workers,
        "raw_video_checks": raw_video_checks,
        "structure_checks": structure_checks,
        "reference_structure_dir": reference_structure_dirs[0] if reference_structure_dirs else "",
        "reference_structure_dirs": reference_structure_dirs,
        "ready": (
            bool(validation.get("passed"))
            and base_combo_injectable
            and not missing_videos
            and (split == HELD_OUT_TEST_SPLIT or not missing_reference_refs)
            and (
                # A fresh isolated structure rebuild is explicit and complete
                # for the whole bundle, regardless of which changed module is
                # listed first in its provenance metadata.
                effective_rebuild_structure
                or not missing_structures
            )
        ),
    }

    if args.check_only:
        return readiness
    if not readiness["ready"]:
        raise RuntimeError(f"Full-eval readiness failed: {readiness}")

    evaluation_dir = (
        os.path.join(candidate_run, "held_out_evaluation")
        if split == HELD_OUT_TEST_SPLIT else candidate_run
    )
    resume_requested = bool(getattr(args, "resume", False))
    resume_seed_dirs = list(getattr(args, "resume_seed_sandbox", []) or [])
    if resume_seed_dirs and not resume_requested:
        raise ValueError("--resume-seed-sandbox requires --resume")
    # Seeding bootstraps a brand-new, isolated run from immutable per-video
    # artifacts. It then follows the same strict resume path as an interrupted
    # run. Existing candidate-run artifacts are never overwritten.
    seed_manifest = {}
    if resume_seed_dirs:
        paths = _prepare_outputs(evaluation_dir, overwrite=args.overwrite, resume=False)
        seed_manifest = _seed_resume_sandbox(
            paths["sandbox_dir"], resume_seed_dirs,
            copy_structures=bool(getattr(args, "resume_seed_structures", False)),
        )
        _write_json(os.path.join(evaluation_dir, "full_eval_resume_seed.json"), seed_manifest)
    else:
        paths = _prepare_outputs(
            evaluation_dir, overwrite=args.overwrite, resume=resume_requested
        )
    # Persist one start/end record for every execution-provider request,
    # including the pre-answer candidate structure build.  Full-eval reports
    # already retain per-question capability events; this append-only audit
    # makes the build phase equally reproducible and is isolated per split.
    os.environ["RUNTIME_PROVIDER_AUDIT_PATH"] = paths["provider_audit_path"]
    requested_combo_id = (
        args.combo_id
        or f"Evolved_bundle_{int(time.time())}"
    )
    resume_contract_path, combo_id = _ensure_resume_contract(
        evaluation_dir,
        _resume_contract(
            candidate_bundle_path, split, scheduled_questions, requested_combo_id
        ),
        resume=resume_requested and not bool(resume_seed_dirs),
    )
    if reference_structure_dirs and not effective_rebuild_structure:
        _prepopulate_structure_dir(reference_structure_dirs, paths["sandbox_dir"])
    os.environ["METAVIDEOAGENT_LLM_MODEL"] = args.llm_model
    os.environ.setdefault("VIDEO_STRUCT_REBUILD_WORKERS", str(args.structure_workers))
    if effective_rebuild_structure:
        os.environ["SANDBOX_DISABLE_FULL_STRUCTURE_BUILD"] = "0"
        os.environ["SANDBOX_ALLOW_FULL_STRUCTURE_REBUILD"] = "1"
        os.environ["SANDBOX_STRUCTURE_CHROMA_PER_VIDEO"] = "1"
        os.environ["VIDEO_STRUCT_RESUME_BUILD"] = "0"
    else:
        os.environ.setdefault("SANDBOX_DISABLE_FULL_STRUCTURE_BUILD", "1")
        os.environ.setdefault("SANDBOX_ALLOW_FULL_STRUCTURE_REBUILD", "0")
        os.environ.setdefault("SANDBOX_STRUCTURE_CHROMA_PER_VIDEO", "0")
        os.environ.setdefault("VIDEO_STRUCT_RESUME_BUILD", "0")

    import module_map

    module_map_refs = {
        "structuring": module_map.STRUCTURING_MAP,
        "thinking": module_map.THINKING_MAP,
        "memory": module_map.WORK_MEMORY_MAP,
        "localization": module_map.LOCALIZATION_MAP,
        "perception": module_map.PERCEPTION_MAP,
    }
    original_maps = {key: dict(value) for key, value in module_map_refs.items()}
    module_artifacts = {}

    try:
        ok, agent_combo, custom_configs, candidate_injection = inject_module_bundle(
            candidate_bundle, module_map_refs
        )
        if not ok:
            raise RuntimeError(f"Candidate bundle injection failed: {candidate_injection}")
        injected_candidate_names = {
            item: agent_combo[MODULE_TO_COMBO_KEY[item]]
            for item in BUNDLE_MODULE_ORDER
        }
        module_artifacts = {
            item: _module_artifact(
                item, injected_candidate_names[item],
                path=candidate_bundle_path, source="candidate_bundle",
            )
            for item in BUNDLE_MODULE_ORDER
        }
        if agent_combo.get("video_structuring", "") != structure_reuse_decision["candidate_struct"]:
            raise RuntimeError(
                "Injected full-eval combo disagrees with planned structure decision: "
                f"injected={agent_combo.get('video_structuring', '')!r}, "
                f"planned={structure_reuse_decision['candidate_struct']!r}"
            )
        if "video_structuring" in injected_candidate_names:
            custom_configs.setdefault(injected_candidate_names["video_structuring"], {})
        for struct_name in {
            agent_combo.get("video_structuring", ""),
            injected_candidate_names.get("video_structuring", ""),
        }:
            if struct_name and effective_rebuild_structure:
                custom_configs.setdefault(struct_name, {})
                custom_configs[struct_name].setdefault("physics", {})
                physics = custom_configs[struct_name]["physics"]
                physics["parallel_workers"] = args.structure_workers
                physics.setdefault("parallel_workers_full_coverage", args.structure_workers)
                physics.setdefault("parallel_workers_interval_entity_ocr", args.structure_workers)
                physics.setdefault("parallel_workers_actionable_handoff", args.structure_workers)
                physics.setdefault("parallel_batch_delay", 0.2)
        resumed_rows, resumed_task_ids = ([], set())
        if resume_requested:
            resumed_rows, resumed_task_ids = _load_completed_sandbox_rows(
                paths["sandbox_dir"], scheduled_questions, agent_combo,
                rerun_provider_failures=bool(
                    getattr(args, "resume_rerun_provider_failures", False)
                ),
            )
        pending_questions = [
            item for item in scheduled_questions
            if str(item.get("task_id") or "") not in resumed_task_ids
        ]
        # A resume preserves an existing candidate structure artifact for its
        # video. Only a video with no artifact is rebuilt. This prevents a
        # partial interruption from consuming VLM/ASR calls again.
        structure_prebuild = []
        force_rebuild_by_video = {}
        if effective_rebuild_structure:
            structure_dir = os.path.join(paths["sandbox_dir"], "video_structure")
            struct_name = str(agent_combo.get("video_structuring") or "")
            pending_video_ids = {
                str(item.get("video_id") or "") for item in pending_questions
            }
            missing_structure_video_ids = []
            for video_id in pending_video_ids:
                artifact = os.path.join(structure_dir, f"{video_id}_{struct_name}.jsonl")
                needs_build = not (
                    os.path.isfile(artifact) and os.path.getsize(artifact) > 0
                )
                force_rebuild_by_video[video_id] = needs_build
                if needs_build:
                    missing_structure_video_ids.append(video_id)
            # Formal train/test full eval has an explicit two-phase schedule:
            # materialize every required video library first, then release the
            # complete question pool to the global answer scheduler.
            if missing_structure_video_ids:
                structure_prebuild = prebuild_isolated_video_structures(
                    workspace,
                    missing_structure_video_ids,
                    agent_combo,
                    paths["sandbox_dir"],
                    custom_configs=custom_configs,
                )
                force_rebuild_by_video = {
                    video_id: False for video_id in pending_video_ids
                }
        fresh_rows = evaluate_combo_multi_video(
            workspace,
            pending_questions,
            agent_combo,
            combo_id,
            paths["sandbox_dir"],
            custom_configs=custom_configs,
            force_rebuild=False,
            force_rebuild_by_video=force_rebuild_by_video,
            isolate_structure=True,
            concurrency=args.concurrency,
            # Structure artifacts are prepopulated above before any question
            # starts.  A shared answer worker pool therefore keeps all
            # question slots busy across videos instead of serializing by
            # video group.  The evaluator automatically falls back to the
            # safe grouped path if a future candidate requires rebuilding.
            global_question_scheduler=True,
        ) if pending_questions else []
        results = resumed_rows + fresh_rows
        # Provider/API failures are per-question execution incidents, not a
        # reason to discard a completed multi-video evaluation.  Adapters have
        # already retried the identical API request once.  Persist any terminal
        # failures for an explicitly requested task rerun; do not make the
        # full-eval runner silently rerun or replace individual results.
        provider_failure_task_ids = _terminal_provider_failure_task_ids(results)
        provider_failure_record = {
            "artifact_type": "full_eval_provider_failures",
            "schema_version": 1,
            "task_rerun_managed_by": "explicit_or_external_runner",
            "task_ids": provider_failure_task_ids,
            "failures": {
                task_id: _provider_failure_summary(next(
                    (row for row in results if str(row.get("task_id") or "") == task_id
                ), {}))
                for task_id in provider_failure_task_ids
            },
            "sandbox_dir": paths["sandbox_dir"],
        }
    finally:
        for key, backup in original_maps.items():
            module_map_refs[key].clear()
            module_map_refs[key].update(backup)

    ordered_results, missing_result_task_ids = _dedupe_results(
        results, scheduled_questions, strict=False
    )
    # Always persist the explicit rerun list, including the empty case, so later
    # operators can distinguish "no provider failure" from missing provenance.
    _write_json(paths["provider_failure_path"], provider_failure_record)
    _write_jsonl(paths["results_path"], ordered_results)
    structure_artifact = _structure_artifact(
        os.path.join(paths["sandbox_dir"], "video_structure"),
        agent_combo,
        video_ids,
        rebuilt=effective_rebuild_structure,
        source_report_path=ref_report_path,
        source_artifact=ref_report.get("structure_artifact") or {},
        source_structure_dirs=reference_structure_dirs,
    )
    effective_bundle_path = _materialize_effective_bundle(
        evaluation_dir,
        agent_combo,
        candidate_bundle,
    )

    report = compare_with_reference(
        ordered_results,
        reference_lookup,
        require_reference=split != HELD_OUT_TEST_SPLIT,
        reference_label=reference_label,
        reference_source=reference_source,
    )
    report.update(
        {
            "combo_id": combo_id,
            "combo": agent_combo,
            "combo_policy": {
                "base_combo_source": base_combo_source,
                "base_combo": base_combo,
                "base_bundle_path": base_bundle_path,
                "effective_bundle_path": effective_bundle_path,
                "source_diagnosis": readiness.get("diagnosis_context_path", ""),
            },
            "structure_reuse_decision": structure_reuse_decision,
            "structure_prebuild": structure_prebuild,
            "candidate": {
                "name": "MetaVideoAgent bundle",
                "bundle_fingerprint": candidate_bundle.get("bundle_fingerprint", ""),
                "changed_modules": changed_modules,
            },
            "module_artifacts": module_artifacts,
            "candidate_run": candidate_run,
            "evaluation_dir": evaluation_dir,
            "reference_report_path": ref_report_path,
            "candidate_bundle": candidate_bundle_path,
            "candidate_bundle_sha256": _file_digest(candidate_bundle_path),
            "runtime_dir": RUNTIME_DIR,
            "runtime_id": runtime_id(),
            "workspace_dir": os.path.abspath(workspace),
            "output_root": runtime_output_root(),
            "validation_scope": (
                "held_out_test_multi_video"
                if split == HELD_OUT_TEST_SPLIT else "full_multi_video"
            ),
            "video_ids": video_ids,
            "question_split": split,
            "reference_label": reference_label,
            "reference_source": reference_source,
            "history_input_policy": _external_history_meta(
                reference_source,
                getattr(args, "reference_report", "") or "",
                getattr(args, "reference_results", "") or "",
            ),
            "expected_total_questions": len(scheduled_questions),
            "resume": {
                "enabled": resume_requested,
                "contract_path": resume_contract_path,
                "seed_manifest_path": (
                    os.path.join(candidate_run, "full_eval_resume_seed.json")
                    if resume_seed_dirs else ""
                ),
                "seed_source_count": len(resume_seed_dirs),
                "resumed_question_count": len(resumed_rows),
                "executed_question_count": len(pending_questions),
            },
            "missing_result_task_ids": missing_result_task_ids,
            "concurrency": args.concurrency,
            "llm_model": args.llm_model,
            "runtime_models": {
                "vlm_profile_id": DEFAULT_PROFILE_IDS["vlm"],
                "llm_profile_id": DEFAULT_PROFILE_IDS["llm"],
                "llm_model": args.llm_model,
                "asr_profile_id": DEFAULT_PROFILE_IDS["asr"],
                "embedding_profile_id": DEFAULT_PROFILE_IDS["embedding"],
            },
            "execution_capability_limits": _execution_capability_limits(),
            "rebuild_structure": effective_rebuild_structure,
            "structure_source": readiness.get("structure_source", ""),
            "structure_workers": args.structure_workers if effective_rebuild_structure else 0,
            "structure_inflight_workers": (
                _runtime_concurrency_limit(
                    "METAVIDEOAGENT_STRUCTURE_INFLIGHT_WORKERS",
                    args.structure_workers,
                    256,
                )
                if effective_rebuild_structure else 0
            ),
            "sandbox_dir": paths["sandbox_dir"],
            "structure_dir": os.path.join(paths["sandbox_dir"], "video_structure"),
            "structure_artifact": structure_artifact,
            "candidate_full_eval_bundle_path": effective_bundle_path,
            "candidate_full_eval_bundle_sha256": _file_digest(effective_bundle_path),
            "results_path": paths["results_path"],
            "report_path": paths["report_path"],
            "summary_path": paths["summary_path"],
            "evaluation_audit_path": paths["audit_path"],
            "evaluation_audit_summary_path": paths["audit_summary_path"],
            "provider_audit_path": paths["provider_audit_path"],
            "post_full_eval_decision_path": paths["post_eval_decision_path"],
            "provider_failure_path": paths["provider_failure_path"],
            "provider_failures": provider_failure_record,
            "assembly_validation": validation,
            "readiness": readiness,
            "observed_distribution_profile": _load_or_build_distribution_profile(
                workspace,
                getattr(args, "distribution_manifest", ""),
                getattr(args, "distribution_profile", ""),
            ),
        }
    )
    apply_effectiveness_policy(
        report,
        expected_total_questions=len(scheduled_questions),
        result_rows=ordered_results,
        require_complete=True,
        scope=(
            "held_out_test_multi_video"
            if split == HELD_OUT_TEST_SPLIT else "full_multi_video"
        ),
    )
    audit = build_full_eval_audit(
        report,
        candidate_rows=ordered_results,
        reference_rows=reference_rows,
        sandbox_dir=paths["sandbox_dir"],
    )
    write_audit_files(
        audit,
        paths["audit_path"],
        paths["audit_summary_path"],
    )
    report["evaluation_audit"] = {
        "path": paths["audit_path"],
        "summary_path": paths["audit_summary_path"],
        "cost_ratio_vs_reference": audit.get("cost_ratio_vs_reference"),
        "candidate_cost": audit.get("candidate_cost", {}),
        "reference_cost": audit.get("reference_cost", {}),
        "recommended_next_focus": (
            audit.get("decision_guidance", {}) or {}
        ).get("recommended_next_focus", ""),
        "issue_summary": audit.get("issue_summary", []),
    }
    # Keep a compact, stable top-level metrics contract for both train and
    # held-out test full evaluation.  The detailed per-task evidence remains
    # in evaluation_audit.json; this section is intentionally aggregate only.
    report["execution_metrics"] = {
        "schema_version": 1,
        "latency_semantics": "sum_of_per_task_latency_sec_not_wall_clock",
        "candidate": audit.get("candidate_cost", {}),
        "reference": audit.get("reference_cost", {}),
        "cost_ratio_vs_reference": audit.get("cost_ratio_vs_reference"),
    }
    report["current_best_selection"] = {
        "policy": "evolution_split_only",
        "selection_split": CURRENT_BEST_SELECTION_SPLIT,
        "this_eval_split": split,
        "eligible_to_update_current_best": split == CURRENT_BEST_SELECTION_SPLIT,
        "train_diagnostic_report_path": (
            os.path.abspath(getattr(args, "train_diagnostic_report", ""))
            if getattr(args, "train_diagnostic_report", "") else ""
        ),
        "train_diagnostic_results_path": "",
    }
    if report["current_best_selection"]["train_diagnostic_report_path"]:
        train_diagnostic = _read_json_optional(report["current_best_selection"]["train_diagnostic_report_path"])
        train_results = str(train_diagnostic.get("results_path") or "")
        if train_results and os.path.isfile(train_results):
            report["current_best_selection"]["train_diagnostic_results_path"] = os.path.abspath(train_results)

    # Provider failures remain in the persisted audit for trajectory review.
    if report.get("evaluation_complete", True) is not True:
        report["full_eval_runtime_failure"] = {
            "requires_rerun": True,
            "reason": "incomplete_full_eval",
            "evaluation_audit_path": paths["audit_path"],
        }
        _write_json(paths["report_path"], report)
        _write_summary(paths["summary_path"], report)
        raise RuntimeError(
            "full eval is incomplete; rerun this same candidate before post-decision"
        )

    if split == HELD_OUT_TEST_SPLIT:
        decision = {
            "schema_version": 1,
            "decision_source": "held_out_reporting_only",
            "update_current_best": False,
            "state_transition": "held_out_no_state_transition",
            "next_action": "report_only",
            "rationale": (
                "Held-out results are retained for reporting and are not visible "
                "to review, diagnosis, code generation, or current-best selection."
            ),
        }
        _write_json(paths["post_eval_decision_path"], decision)
        report["post_full_eval_decision"] = decision
    else:
        ledger_snapshot = _read_json_optional(_metavideoagent_ledger_path(args))
        # Some staged MetaVideoAgent invocations intentionally omit
        # --metavideoagent-run-id so that an explicit full evaluation cannot mutate
        # the ledger.  They still provide an explicit current-best reference.
        # That reference is sufficient for the acceptance comparison: do not
        # mistake an unavailable ledger for absence of a current best and
        # accept a lower-scoring candidate as the first executed reference.
        if not (ledger_snapshot.get("current_best") if isinstance(ledger_snapshot, dict) else None):
            reference_correct = report.get("reference_correct")
            reference_total = report.get("total_questions")
            if isinstance(reference_correct, (int, float)) and isinstance(reference_total, (int, float)):
                ledger_snapshot = {
                    "current_best": {
                        "label": "explicit_reference_for_post_full_eval",
                        "report_path": report.get("reference_report_path", ""),
                        "results_path": report.get("reference_results_path", ""),
                        "metrics": {
                            "correct": int(reference_correct),
                            "total_questions": int(reference_total),
                            "accuracy_delta": 0,
                            "regressions": 0,
                        },
                        "synthetic": True,
                    },
                    "synthetic_from_explicit_reference": True,
                }
        decision = decide_post_full_eval(
            report,
            audit,
            ledger_snapshot,
            call_llm=not getattr(args, "skip_post_eval_decision", False),
        )
        _write_json(paths["post_eval_decision_path"], decision)
        report["post_full_eval_decision"] = decision
        if current_best_update_allowed(report, decision):
            current_best_bundle_path = os.path.join(candidate_run, "current_best_bundle.json")
            shutil.copy2(effective_bundle_path, current_best_bundle_path)
            report["current_best_bundle_path"] = current_best_bundle_path
            report["combo_policy"]["current_best_bundle_path"] = current_best_bundle_path
        ledger_path = _record_metavideoagent_ledger(
            args, report,
            decision=decision,
            decision_path=paths["post_eval_decision_path"],
        )
        if ledger_path:
            report["current_best_ledger"] = ledger_path

    _write_json(paths["report_path"], report)
    _write_summary(paths["summary_path"], report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an isolated full evaluation for a MetaVideoAgent bundle")
    parser.add_argument("--workspace", default=default_workspace())
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--candidate-bundle", default="",
                        help="Self-contained five-module candidate bundle")
    parser.add_argument("--api-key", default="", help="Optional provider credential override for this run")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--diagnosis-report", default="",
                        help="Optional diagnosis JSON; defaults to candidate-run/diagnosis.json")
    parser.add_argument("--video-id", default="ALL")
    parser.add_argument("--task-id", action="append", default=[],
                        help=("Optional exact task ID to evaluate; may be repeated. "
                              "Creates an isolated, explicitly scoped recovery run."))
    parser.add_argument("--question-split", choices=(EVOLUTION_SPLIT, HELD_OUT_TEST_SPLIT),
                        default=EVOLUTION_SPLIT,
                        help=("Dataset split to evaluate. train drives evolution; test is "
                              "a read-only held-out report for a frozen agent."))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--rebuild-structure", action="store_true")
    parser.add_argument("--force-fresh-structure", action="store_true",
                        help=("Explicitly rebuild fresh isolated per-video structure libraries before "
                              "answers even when the candidate preserves the structuring implementation."))
    parser.add_argument("--structure-workers", type=int, default=4)
    parser.add_argument("--combo-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help=("Resume an interrupted run with the exact saved "
                              "resume contract; completed sandbox rows and "
                              "matching candidate structures are reused."))
    parser.add_argument("--resume-seed-sandbox", action="append", default=[],
                        help=("Sandbox directory from a completed compatible "
                              "per-video run to seed into a new --resume run; "
                              "may be supplied multiple times."))
    parser.add_argument("--resume-seed-structures", action="store_true",
                        help=("Also copy seeded structure JSONLs. Required only "
                              "when the seeded run still has pending questions."))
    parser.add_argument("--resume-rerun-provider-failures", action="store_true",
                        help=("With --resume, treat persisted rows with terminal provider "
                              "API failures as pending so only those tasks are re-executed; "
                              "all other exact rows remain seeded."))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--distribution-profile", default="",
                        help="Optional label-blind observed_distribution_profile.json to carry into the report")
    parser.add_argument("--distribution-manifest", default="",
                        help="Optional distribution manifest; human descriptions are stripped before profiling")
    parser.add_argument("--reference-results", default="",
                        help="Optional current-best results JSONL or sandbox dir to compare against")
    parser.add_argument("--reference-report", default="",
                        help="Optional current-best full-eval report; results_path/sandbox_dir will be used")
    parser.add_argument("--reference-structure-dir", action="append", default=[],
                        help=("Explicit verified per-video structure directory to materialize before "
                              "question execution. Takes precedence over directories inferred from "
                              "--reference-report; may be repeated."))
    parser.add_argument("--reference-bundle", default="",
                        help="Optional JSON bundle with full module code for the reference/base combo")
    parser.add_argument("--reference-label", default="",
                        help="Reference label for reports, e.g. initial_combo/current_best")
    parser.add_argument("--combo-plan", default="",
                        help="Optional initial_combo_plan.json used as the base combo before injecting candidate modules")
    parser.add_argument("--metavideoagent-run-id", default="",
                        help="Optional run id used to update the evolution-split current-best ledger")
    parser.add_argument("--metavideoagent-output-root", default="",
                        help="Optional execution output root for the current-best ledger")
    parser.add_argument("--train-diagnostic-report", default="",
                        help=("Completed evolution-split report paired with a read-only held-out "
                              "evaluation of the same frozen candidate."))
    parser.add_argument("--skip-post-eval-decision", action="store_true",
                        help="Skip LLM post-full-eval decision and use heuristic fallback only")
    args = parser.parse_args()

    report = run(args)
    if args.check_only:
        print("FULL_EVAL_CHECK")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ready") else 1

    print("FULL_EVAL_DONE")
    for key in (
        "report_path",
        "results_path",
        "sandbox_dir",
        "reference_correct",
        "candidate_correct",
        "total_questions",
        "accuracy_delta_vs_reference",
    ):
        print(f"{key}={report.get(key)}")
    print(f"corrections={len(report.get('corrections', []) or [])}")
    print(f"regressions={len(report.get('regressions', []) or [])}")
    decision = report.get("post_full_eval_decision") or {}
    print(f"post_eval_next_action={decision.get('next_action')}")
    print(f"post_eval_update_current_best={decision.get('update_current_best')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
