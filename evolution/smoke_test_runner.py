"""Run one isolated MetaVideoAgent bundle smoke trajectory.

The runner emits structured progress and result records on standard output for
the Codex automatic workflow.
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time

# Set path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)
from runtime_paths import default_workspace, use_runtime_path

tasks_dir = use_runtime_path()
if tasks_dir not in sys.path:
    sys.path.insert(0, tasks_dir)

from combo_contract import require_complete_evolution_combo
from distribution_spec import load_distribution_spec
from question_set_loader import load_questions as load_question_set
from reference_loader import combo_from_reference
from structure_artifacts import (
    reference_structure_artifact_dir as _reference_structure_artifact_dir,
)
from task_sanitizer import sanitize_task_for_evolution


class SmokeExecutionLayerError(RuntimeError):
    """Typed runner/input fault that must never enter Codex repair."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _extract_time_reference_bounds(time_reference) -> tuple:
    starts, ends = [], []
    if isinstance(time_reference, (int, float)):
        starts.append(float(time_reference))
        ends.append(float(time_reference))
    elif isinstance(time_reference, str):
        import re
        nums = re.findall(r"\d+(?:\.\d+)?", time_reference)
        vals = [float(x) for x in nums]
        if len(vals) >= 2:
            starts.append(vals[0])
            ends.append(vals[1])
    elif isinstance(time_reference, list):
        for item in time_reference:
            bounds = _extract_time_reference_bounds(item)
            if bounds:
                starts.append(bounds[0])
                ends.append(bounds[1])
    elif isinstance(time_reference, dict):
        for key in ("start_sec", "start", "begin"):
            if key in time_reference:
                starts.append(float(time_reference[key]))
                break
        for key in ("end_sec", "end", "finish"):
            if key in time_reference:
                ends.append(float(time_reference[key]))
                break
    if not starts or not ends:
        return ()
    return max(0.0, min(starts)), max(ends)


def _single_time_reference_bounds(time_reference) -> tuple:
    """Return one executable interval only when the reference is singular.

    MetaVideoAgent smoke is an engineering gate, not a coverage evaluation.  A
    fixed, short, single-reference training example keeps the media budget and
    evidence boundary stable across every evolution round.  Multi-interval
    references are deliberately excluded rather than merged into a broad clip.
    """
    if isinstance(time_reference, dict):
        return _extract_time_reference_bounds(time_reference)
    if isinstance(time_reference, (list, tuple)):
        if len(time_reference) == 2 and not isinstance(
            time_reference[0], (list, tuple, dict)
        ):
            return _extract_time_reference_bounds(time_reference)
        if len(time_reference) != 1:
            return ()
        return _extract_time_reference_bounds(time_reference[0])
    return _extract_time_reference_bounds(time_reference)


def _matches_preferred_time_reference(actual, selector: str) -> bool:
    """Compare an environment selector to normalized list/dict time references.

    Dataset adapters preserve JSON intervals as Python lists, while environment
    variables are necessarily strings.  Direct equality made an explicit
    ``SMOKE_TIME_REFERENCE='[[0, 30]]'`` silently fall back to the shortest
    unrelated task.  Parse JSON selectors first and compare a canonical JSON
    representation. A literal string match also supports non-JSON selectors.
    """
    raw = str(selector or "").strip()
    if not raw:
        return False
    if actual == raw:
        return True
    try:
        wanted = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    try:
        return json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == json.dumps(
            wanted, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return False


def _read_json_optional(path: str) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return {}


def _initial_bundle_path_from_report(report_path: str) -> str:
    report = _read_json_optional(report_path)
    if not report:
        return ""
    combo_policy = report.get("combo_policy") or {}
    direct_paths = [
        report.get("current_best_bundle_path", ""),
        report.get("candidate_full_eval_bundle_path", ""),
        combo_policy.get("current_best_bundle_path", ""),
        combo_policy.get("effective_bundle_path", ""),
        report.get("effective_initial_baseline_bundle", ""),
        report.get("initial_baseline_bundle", ""),
        report.get("bundle_path", ""),
        combo_policy.get("base_bundle_path", ""),
    ]
    candidates = []
    report_dir = os.path.dirname(os.path.abspath(report_path))
    for direct in direct_paths:
        if not isinstance(direct, str) or not direct:
            continue
        candidates.append(direct)
        if not os.path.isabs(direct):
            candidates.append(os.path.join(report_dir, direct))
            candidates.append(os.path.join(_PROJECT_ROOT, direct))
    candidates.append(os.path.join(report_dir, "current_best_bundle.json"))
    candidates.append(os.path.join(report_dir, "initial_baseline_bundle.json"))
    for source in report.get("sources", []) or []:
        if isinstance(source, str):
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(source)), "initial_baseline_bundle.json"))
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(source)), "current_best_bundle.json"))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return ""


def _prepare_isolated_smoke_workspace(real_workspace_dir: str, log):
    """Create an isolated workspace that can only receive explicit artifacts."""
    temp_workspace = tempfile.TemporaryDirectory(prefix="codex_smoke_workspace_")
    workspace_dir = temp_workspace.name
    for shared_name in ("raw_videos", "extracted_frames"):
        src = os.path.abspath(os.path.join(real_workspace_dir, shared_name))
        dst = os.path.join(workspace_dir, shared_name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
            log("INIT", f"linked {shared_name}: {src}")
    os.makedirs(os.path.join(workspace_dir, "video_structure"), exist_ok=True)
    return temp_workspace, workspace_dir


def _reuse_reference_structure_for_smoke(agent, reference_structure_dir: str, log) -> None:
    """Copy the one required current-best JSONL into the isolated smoke workspace."""
    db_path = getattr(agent.structuring, "db_path", "")
    if not db_path:
        raise RuntimeError("smoke structuring module exposes no db_path")
    source_db = os.path.join(reference_structure_dir, os.path.basename(db_path))
    if not os.path.isfile(source_db) or os.path.getsize(source_db) <= 0:
        available = []
        try:
            available = sorted(name for name in os.listdir(reference_structure_dir) if name.endswith(".jsonl"))
        except OSError:
            pass
        raise RuntimeError(
            "current-best structure artifact is missing the required JSONL: "
            f"expected={os.path.basename(db_path)}, available={available[:12]}"
        )
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    shutil.copy2(source_db, db_path)
    source_chroma = os.path.join(reference_structure_dir, "chroma_db")
    target_chroma = os.path.join(os.path.dirname(db_path), "chroma_db")
    if os.path.isdir(source_chroma) and not os.path.exists(target_chroma):
        shutil.copytree(source_chroma, target_chroma)
    log("BUILD_DB", f"reused current-best artifact: {os.path.basename(source_db)}")


def _find_artifact_module(ref_report: dict, module_type: str,
                          module_name: str) -> dict:
    artifacts = (ref_report or {}).get("module_artifacts", {}) or {}
    item = artifacts.get(module_type) or {}
    path = item.get("path", "")
    if not path or not os.path.exists(path):
        return {}
    candidates = []
    if path.endswith(".jsonl"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return {}
    else:
        candidates.append(_read_json_optional(path))
    for module in candidates:
        if module.get("module_type") == module_type and module.get("name") == module_name:
            module.setdefault("_artifact_path", os.path.abspath(path))
            module.setdefault("_artifact_container", "jsonl" if path.endswith(".jsonl") else "json")
            return module
    return {}


def _localization_smoke_contract_issues(trajectory, module_code: str,
                                        raw_video_path: str) -> list[str]:
    """Detect target-output failures before a runnable candidate reaches probe.

    This is intentionally about dataflow, not answer correctness: preserved
    thinking must be able to progress from a localization observation to a
    different downstream action.  It also verifies that a candidate claiming
    ASR uses the documented extraction boundary instead of a PATH-dependent
    shell binary.
    """
    tool_steps = [
        step for step in (trajectory or [])
        if isinstance(step, dict) and step.get("step_type") == "tool_execution"
    ]
    issues = []
    localization_tools = [
        str(step.get("tool") or "") for step in tool_steps
        if "window_search" in str(step.get("tool") or "")
    ]
    localization_observations = [
        str(step.get("observation") or "")
        for step in tool_steps
        if "window_search" in str(step.get("tool") or "")
    ]
    has_structured_windows = any(
        re.search(
            r"(?:JSON_CANDIDATES|candidate_windows|time_ranges)\s*[:=]"
            r".*?\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]",
            observation,
            re.I | re.S,
        )
        for observation in localization_observations
    )
    if len(localization_tools) >= 3 and len(set(localization_tools[-3:])) == 1 and not has_structured_windows:
        issues.append(
            "Localization repeatedly returned non-actionable retrieval output: "
            "preserved thinking could not extract structured candidate windows and repeated the same tool."
        )
    drops_available_visual_candidates = any(
        "candidate windows selected: 0" in observation.lower()
        and re.search(r'"candidates"\s*:\s*\[\s*\{', observation)
        for observation in localization_observations
    )
    if drops_available_visual_candidates:
        issues.append(
            "Localization discarded available retrieval candidates while reporting zero candidate windows; "
            "it must return the selected intervals in the preserved thinking contract instead of prose-only fallback."
        )

    return issues


def _bundle_handoff_checks(trajectory, capability_events, contracts) -> list[dict]:
    """Verify declared bundle handoffs on the one real smoke trajectory.

    Declarative contracts are not sufficient: each declared producer must emit an
    observation carrying its required fields and a later consumer must either
    run or record a valid runtime consumption event.
    """
    tool_steps = [
        step for step in (trajectory or [])
        if isinstance(step, dict) and step.get("step_type") == "tool_execution"
    ]
    checks = []
    for contract in contracts or []:
        if not isinstance(contract, dict):
            continue
        producer, consumer = str(contract.get("producer") or ""), str(contract.get("consumer") or "")
        fields = [str(value) for value in (contract.get("required_fields") or []) if str(value)]
        producer_index, observation, module_result = None, "", {}
        for index, step in enumerate(tool_steps):
            if str(step.get("producer_module") or "") == producer:
                producer_index, observation = index, str(step.get("observation") or "")
                module_result = step.get("module_result") if isinstance(step.get("module_result"), dict) else {}
                break
        # Memory is stateful rather than a tool.  Its packet is consumed by
        # thinking at the next loop boundary, so there is deliberately no
        # synthetic ``tool_execution`` row to locate.  Accept only the
        # framework-recorded consumption event carrying an explicit field
        # inventory; this keeps the handoff auditable without pretending the
        # memory was a sensor/tool call.
        memory_consumption = next((
            event for event in (capability_events or [])
            if isinstance(event, dict) and event.get("event") == "consumer_consumed"
            and event.get("status") == "ok" and event.get("contract_valid") is True
            and str(event.get("producer_module") or "") == producer
            and str(event.get("consumer") or event.get("consumer_module") or "") == consumer
            and str(event.get("output_protocol") or "") == "memory_context"
        ), None)
        event_fields = set(memory_consumption.get("output_fields") or []) if memory_consumption else set()
        if producer == "memory" and memory_consumption is not None:
            producer_index = -1
        parsed = {}
        try:
            parsed = json.loads(observation) if observation.strip().startswith("{") else {}
        except json.JSONDecodeError:
            parsed = {}
        # The formal ABI stores perception-specific values under the stable
        # result envelope's ``evidence`` member.  Prefer the lossless module
        # result retained by working memory; only use rendered text for previous
        # trajectories.  Treating every declared field as a top-level string
        # made a real perception→thinking handoff look absent.
        evidence = module_result.get("evidence") if isinstance(module_result.get("evidence"), dict) else {}
        previous_evidence = parsed.get("evidence") if isinstance(parsed.get("evidence"), dict) else {}
        missing_fields = [
            field for field in fields
            if field not in module_result
            and field not in evidence
            and field not in parsed
            and field not in previous_evidence
            and field not in observation
            and field not in event_fields
        ]
        later_consumer_run = any(
            str(step.get("producer_module") or "") == consumer
            for step in tool_steps[(producer_index + 1 if producer_index is not None else 0):]
        )
        consumed_event = any(
            isinstance(event, dict) and event.get("event") == "consumer_consumed"
            and event.get("status") == "ok" and event.get("contract_valid") is True
            and str(event.get("producer_module") or "") == producer
            and str(event.get("consumer") or event.get("consumer_module") or "") == consumer
            for event in (capability_events or [])
        )
        checks.append({
            "producer": producer, "consumer": consumer, "required_fields": fields,
            "producer_observed": producer_index is not None,
            "missing_fields": missing_fields,
            "consumer_observed": later_consumer_run or consumed_event,
            "consumption_event": consumed_event,
            "passed": producer_index is not None and not missing_fields and (later_consumer_run or consumed_event),
        })
    return checks


def _structure_build_handoff_step(db_path: str, contracts: list[dict] | None = None,
                                  source_mode: str = "isolated_build") -> dict:
    """Expose the materialized isolated structure input as a producer event.

    Video structuring runs before the question-answering loop, so it is not a
    tool-execution row in ``agent.run()``. Treating that absence as a missing
    structuring-to-localization handoff would incorrectly reject rebuilt
    stores. This adapter reads only the isolated candidate JSONL that was
    produced in this smoke *or* explicitly copied from the attested scoped
    reference artifact.  Both are real timestamped producer outputs available
    to localization; only their provenance differs.  Recording that distinction
    prevents a valid reuse-only smoke from masquerading as an absent handoff.
    """
    # Start with the runtime's observable schema, then add every field declared
    # by the candidate's structure-build producer contract. The isolated smoke
    # record is the authoritative materialized output.
    fields = {
        "transcript_sentence", "screen_text", "translated_keywords", "source",
        "segment_boundaries",
    }
    for contract in contracts or []:
        if not isinstance(contract, dict):
            continue
        if str(contract.get("producer") or "") != "video_structuring":
            continue
        fields.update(
            str(field) for field in (contract.get("required_fields") or [])
            if isinstance(field, str) and field.strip()
        )
    fields = tuple(sorted(fields))
    observed = {field: [] for field in fields}

    def normalized_record(record: dict) -> dict:
        """Expose deterministic schema views of native builder evidence.

        The generated segment builder owns the media calls and persists native
        ASR/VLM observations.  A structuring candidate may enrich those
        observations before storage, but smoke must also recognize the same
        evidence when the builder supplies a source-native record. This is a
        no-model, no-question transformation: every derived value is bounded
        by the record's own timestamped text and interval.
        """
        normalized = dict(record)
        native_text = []
        for key in ("asr_text", "transcript", "multimodal_narration", "description"):
            text = str(record.get(key) or "").strip()
            if text:
                native_text.append(text)
        screen = record.get("screen_text")
        if isinstance(screen, list):
            native_text.extend(str(item).strip() for item in screen if str(item).strip())
        elif str(screen or "").strip():
            native_text.append(str(screen).strip())
        joined = " ".join(native_text).strip()
        if "retrieval_document" in fields and not str(normalized.get("retrieval_document") or "").strip() and joined:
            normalized["retrieval_document"] = joined
        if "action_semantics" in fields and not normalized.get("action_semantics") and joined:
            tokens = re.findall(r"[^\W_]+", joined.lower(), flags=re.UNICODE)
            normalized["action_semantics"] = list(dict.fromkeys(token for token in tokens if len(token) > 2))
        if "semantic_subwindows" in fields and not normalized.get("semantic_subwindows"):
            try:
                start, end = float(record.get("start_sec")), float(record.get("end_sec"))
            except (TypeError, ValueError):
                start = end = 0.0
            if end > start:
                normalized["semantic_subwindows"] = [{"start_sec": start, "end_sec": end}]
        return normalized

    record_count = 0
    try:
        with open(db_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                # The optional synthetic micro-record validates the call
                # interface, but must not stand in for real media-derived
                # producer output.
                if str(record.get("source") or "") == "smoke_test_micro_add":
                    continue
                record_count += 1
                record = normalized_record(record)
                for field in fields:
                    if field in record and record.get(field) not in (None, "", [], {}):
                        observed[field].append(record.get(field))
    except OSError:
        return {}
    if record_count <= 0:
        return {}
    evidence = {field: values[:3] for field, values in observed.items() if values}
    reused_reference = str(source_mode or "") == "reused_scoped_reference"
    action = "reuse_scoped_video_structure" if reused_reference else "build_video_structure"
    thought = (
        "Smoke harness recorded the scoped, attested current-best structure input."
        if reused_reference else
        "Smoke harness recorded the isolated candidate structure build."
    )
    return {
        "protocol_version": "metavideoagent_module_protocol",
        "step_type": "tool_execution",
        "step": 0,
        "thought": thought,
        "action": action,
        "observation": json.dumps(evidence, ensure_ascii=False, default=str),
        "execution_status": "ok",
        "error_code": "",
        "producer_module": "video_structuring",
        "output_protocol": "timestamped_structure_records",
        "output_contract_valid": bool(evidence),
        "evidence_event_ids": [],
        "active_time_windows": [],
        "tool_params_summary": {"persisted_record_count": record_count, "source_mode": source_mode},
        "tool_request": {"kind": "smoke_structure_input", "tool_name": action,
                         "source_mode": source_mode},
        "module_result": {
            "protocol_version": "metavideoagent_module_protocol",
            "kind": "structure_reuse_result" if reused_reference else "structure_build_result",
            "status": "ok",
            "producer_module": "video_structuring",
            "tool_name": action,
            "evidence": evidence,
        },
    }


_SMOKE_SUPERVISION_KEYS = {
    "answer", "gt_answer", "gold_answer", "final_answer", "final_agent_answer",
    "is_correct", "correct", "correctness", "score", "reward",
}


def _redact_smoke_supervision(value, answer_text: str = ""):
    """Preserve execution shape while removing answer supervision.

    Engineering smoke is allowed to expose the real request and every runtime
    step to Codex, because that is how a generated bundle can be debugged.  It
    must not expose a gold label, a candidate terminal answer, or an accuracy
    signal: those belong exclusively to the supervised probe.  This function
    is deterministic and deliberately does not summarize or judge a trace.
    """
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted_answer_supervision>"
                if str(key).lower() in _SMOKE_SUPERVISION_KEYS
                else _redact_smoke_supervision(item, answer_text)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_smoke_supervision(item, answer_text) for item in value]
    if isinstance(value, str) and answer_text:
        return value.replace(answer_text, "<redacted_candidate_answer>")
    return value


def _engineering_execution_trace(test_task: dict, trajectory: list,
                                 capability_events: list, answer_text: str) -> dict:
    """Return the complete no-label trace supplied to engineering repair."""
    request = {
        key: test_task.get(key)
        for key in ("question", "choices", "time_reference")
        if key in test_task
    }
    # This artifact is the authoritative no-label debugging view consumed by
    # Codex repair.  Keep the real request and every agent/runtime event, but
    # deterministically remove answer supervision.  Do not replace this with
    # an LLM summary: the repair loop needs the actual producer/consumer data
    # flow in order to find the first broken boundary.
    return {
        "artifact_type": "engineering_smoke_execution_trace", "schema_version": 1,
        "trace_policy": (
            "complete runtime trajectory and capability events; gold labels and "
            "correctness fields and the candidate terminal answer are redacted; "
            "engineering repair diagnoses only execution/dataflow"
        ),
        "runtime_request": _redact_smoke_supervision(request, answer_text),
        "trajectory": _redact_smoke_supervision(trajectory, answer_text),
        "capability_events": _redact_smoke_supervision(capability_events, answer_text),
        "candidate_final_answer": "<redacted_candidate_answer>",
    }


def _runtime_bundle_assembly_preflight(bundle: dict, combo: dict,
                                        module_map_refs: dict) -> dict:
    """Inspect the *injected* classes before spending a real smoke trajectory.

    This is intentionally not an AST/source gate.  It asks the same registries
    used by ``MetaVideoAgent`` which class will actually be constructed and
    records its public tool surface.  In particular, a neutral ``*Base`` class
    is infrastructure and cannot silently become a combo strategy.
    """
    bundle = bundle if isinstance(bundle, dict) else {}
    modules = bundle.get("modules") or {}
    map_keys = {
        "video_structuring": "structuring", "localization": "localization",
        "perception": "perception", "memory": "memory", "thinking": "thinking",
    }
    details, issues = {}, []
    for module_type in bundle.get("changed_modules") or []:
        module_type = str(module_type)
        record = modules.get(module_type) if isinstance(modules, dict) else {}
        expected_name = str((record or {}).get("name") or "")
        selected_name = str((combo or {}).get(module_type) or "")
        registry = module_map_refs.get(map_keys.get(module_type, ""), {})
        cls = registry.get(selected_name) if isinstance(registry, dict) else None
        tool_surface = sorted(
            name[5:] for name in dir(cls) if name.startswith("tool_") and callable(getattr(cls, name, None))
        ) if isinstance(cls, type) else []
        details[module_type] = {
            "declared_class": expected_name,
            "selected_class": selected_name,
            "injected_class": getattr(cls, "__name__", ""),
            "class_role": str((record or {}).get("class_role") or ""),
            "tool_surface": tool_surface,
        }
        if expected_name != selected_name:
            issues.append(
                f"{module_type}: bundle class {expected_name!r} differs from runtime combo {selected_name!r}"
            )
        if not isinstance(cls, type):
            issues.append(f"{module_type}: selected class {selected_name!r} was not dynamically injected")
            continue
        if selected_name.endswith("Base") or getattr(cls, "__name__", "").endswith("Base"):
            issues.append(
                f"{module_type}: runtime combo selected infrastructure class {selected_name!r}, not a concrete strategy"
            )
        if str((record or {}).get("class_role") or "") not in {"", "concrete_strategy"}:
            issues.append(
                f"{module_type}: bundle declares non-strategy class_role={record.get('class_role')!r}"
            )
    return {"passed": not issues, "issues": issues, "effective_modules": details}


def _answer_contract_issue(answer, task: dict) -> str:
    """Validate completion without consulting a gold answer.

    MetaVideoAgent tasks are multiple-choice.  A smoke must prove that the candidate
    reaches an evaluable terminal decision, not that it is correct.  Therefore
    only the visible option syntax is used here; no labels or correctness data
    are read.
    """
    text = str(answer or "").strip()
    if not text:
        return "final answer is empty"
    folded = re.sub(r"\s+", " ", text).strip().lower()
    failure_tokens = (
        "uncertain", "unknown", "cannot determine", "can't determine",
        "insufficient evidence", "task failed", "failed to find answer",
        "no answer", "unable to answer", "Unable to determine", "Insufficient evidence", "Unable to answer",
    )
    if folded in failure_tokens or any(token in folded for token in failure_tokens):
        return f"final answer is a non-evaluable failure token: {text[:160]}"
    choices = task.get("choices") or []
    if isinstance(choices, dict):
        labels = [str(key).strip() for key in choices]
        values = [str(value).strip() for value in choices.values()]
    else:
        labels = [chr(ord("A") + index) for index, _ in enumerate(choices)]
        values = [str(value).strip() for value in choices]
    if not labels:
        return "task has no visible choices for answer parsing"
    normalized_values = {value.lower() for value in values if value}
    if folded in {label.lower() for label in labels} or folded in normalized_values:
        return ""
    label_set = {label.upper() for label in labels}
    explicit_match = re.search(
        r"(?:answer|option)\s*(?:is|:)?\s*[\(\[]?([A-Za-z])[\)\]]?",
        text,
        re.IGNORECASE,
    )
    if explicit_match and explicit_match.group(1).upper() in label_set:
        return ""
    for option_match in re.finditer(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", text):
        if option_match.group(1).upper() in label_set:
            return ""
    return "final answer cannot be parsed as one of the visible options"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_module", default="")
    parser.add_argument("--class_name", default="")
    parser.add_argument("--bundle-json", default="", help="Current candidate_bundle.json for atomic bundle smoke")
    parser.add_argument("--workspace", default=default_workspace())
    parser.add_argument("--distribution-manifest", default="")
    parser.add_argument("--question-split", default="train")
    parser.add_argument("--reference-results", default="")
    parser.add_argument("--reference-report", default="")
    parser.add_argument("--reference-bundle", default="")
    parser.add_argument("--expected-capabilities", default="")
    parser.add_argument("--expected-profiles-json", default="{}")
    parser.add_argument("--require-consumer-event", action="store_true")
    parser.add_argument("--failure-contract-json", default="{}")
    parser.add_argument("--trace-output", default="",
                        help="Persist the complete credential-free smoke execution trace here")
    args = parser.parse_args()
    expected_capabilities = {
        value.strip() for value in args.expected_capabilities.split(",") if value.strip()
    }
    try:
        expected_profiles = json.loads(args.expected_profiles_json or "{}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --expected-profiles-json: {exc}")
    expected_profiles = {
        str(capability): str(profile_id)
        for capability, profile_id in (expected_profiles or {}).items()
        if str(capability).strip() and str(profile_id).strip()
    }
    try:
        failure_contract = json.loads(args.failure_contract_json or "{}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --failure-contract-json: {exc}")
    if not isinstance(failure_contract, dict):
        raise SystemExit("--failure-contract-json must be an object")
    if not args.bundle_json:
        raise SystemExit("--bundle-json is required")
    reference_report = _read_json_optional(args.reference_report)
    if not args.reference_results:
        raise SystemExit(
            "MetaVideoAgent smoke test requires --reference-results so the smoke combo "
            "matches the current explicit reference."
        )

    # Log assistance
    def log(tag, msg):
        print(f"[SMOKE:{tag}] {msg}", flush=True)

    bundle_candidate = _read_json_optional(args.bundle_json) if args.bundle_json else {}
    if args.bundle_json and not bundle_candidate:
        raise SmokeExecutionLayerError("candidate_bundle_invalid", "cannot load candidate bundle JSON")
    changed_modules = list(bundle_candidate.get("changed_modules") or [])
    changed_modules = [str(item) for item in changed_modules if str(item)]
    if bundle_candidate and not changed_modules:
        raise SmokeExecutionLayerError("candidate_bundle_invalid", "candidate bundle has no changed_modules")
    # A bundle smoke is one real agent trajectory; this is only the display
    # target used by previous logs and the structuring rebuild decision.
    primary_target = args.target_module or ("video_structuring" if "video_structuring" in changed_modules else changed_modules[0])
    log("INIT", f"target={primary_target}, class={args.class_name or 'bundle'}")

    # ─── Load the evolution module code ───
    log("PHASE", "loading_module")
    modules = bundle_candidate.get("modules") or {}
    evolved = {}
    # An adaptive bundle is self-contained. Inject every canonical module
    # carried by the candidate; changed_modules is provenance only.
    candidate_module_types = (
        "video_structuring", "localization", "perception", "memory", "thinking",
    )
    for module_type in candidate_module_types:
        item = modules.get(module_type) or {}
        if not isinstance(item, dict) or not item.get("name") or not item.get("code"):
            raise SmokeExecutionLayerError(
                "candidate_bundle_invalid",
                f"candidate bundle has no executable code for {module_type}",
            )
        record = dict(item)
        record["module_type"] = module_type
        evolved[module_type] = [record]
    module_dict = (evolved.get(primary_target) or [{}])[-1]
    module_code = str(module_dict.get("code") or "")

    # Dynamically register module classes
    import module_map
    from agent import MetaVideoAgent
    from run_video_qa import VideoQAEnv, build_video_structure
    from sandbox_evaluator import (
        inject_evolved_modules,
        inject_module_bundle,
        map_combo_to_agent_config,
    )

    # ─── Position dataset ───
    real_workspace_dir = os.path.abspath(args.workspace)
    # Every smoke run is isolated.  Non-structuring candidates must receive
    # the current-best artifact explicitly rather than silently using or
    # rebuilding the mutable workspace structure DB.
    temp_workspace, workspace_dir = _prepare_isolated_smoke_workspace(
        real_workspace_dir, log,
    )
    trace_path = os.path.abspath(args.trace_output) if args.trace_output else os.path.join(
        real_workspace_dir, "smoke_traces",
        "smoke_%s_%s.json" % (int(time.time() * 1000), os.getpid()),
    )
    # A changed structuring module must build a fresh isolated store.  Copying
    # the current-best JSONL/Chroma here would validate the old producer while
    # the candidate localization consumes it, which is neither a real handoff
    # nor a valid smoke of the changed module.
    preferred_time_ref = os.environ.get("SMOKE_TIME_REFERENCE", "").strip()
    test_task = None
    try:
        manifest_spec = (
            load_distribution_spec(args.distribution_manifest)
            if args.distribution_manifest else {}
        )
        if args.distribution_manifest and not (manifest_spec.get("splits") or {}):
            raise SmokeExecutionLayerError(
                "distribution_manifest_invalid",
                "distribution manifest declares no train split",
            )
        tasks, _video_ids, split_spec = load_question_set(
            real_workspace_dir,
            video_id="ALL",
            distribution_manifest=args.distribution_manifest,
            preferred_splits=(args.question_split,),
        )
    except Exception as e:
        raise SmokeExecutionLayerError("smoke_dataset_unavailable", f"failed to load smoke dataset: {e}") from e
    sanitized_tasks = [sanitize_task_for_evolution(item) for item in tasks]
    if preferred_time_ref:
        for item in sanitized_tasks:
            if (
                _matches_preferred_time_reference(item.get("time_reference"), preferred_time_ref)
                and _single_time_reference_bounds(item.get("time_reference", ""))
            ):
                test_task = item
                break
    if test_task is None and sanitized_tasks:
        def _smoke_sort_key(item):
            bounds = _single_time_reference_bounds(item.get("time_reference", ""))
            return (bounds[1], bounds[0], str(item.get("task_id", "")))
        single_reference_tasks = [
            item for item in sanitized_tasks
            if _single_time_reference_bounds(item.get("time_reference", ""))
        ]
        if single_reference_tasks:
            test_task = sorted(single_reference_tasks, key=_smoke_sort_key)[0]
    if not test_task:
        if args.distribution_manifest:
            log("ERROR", "configured smoke split resolved zero questions")
        else:
            log("ERROR", "empty dataset")
        raise SmokeExecutionLayerError("smoke_dataset_empty", "configured smoke split resolved zero questions")
    video_id = test_task.get("video_id", "")
    if not video_id:
        raise SmokeExecutionLayerError("smoke_task_invalid", "smoke task has no video_id")

    log("INIT", f"question={test_task.get('question', '?')[:80]}")
    log("INIT", f"video_id={video_id}")
    log("INIT", "smoke_selection=single_time_reference_shortest task_id=%s" % (
        str(test_task.get("task_id", "")),
    ))
    log("INIT", f"workspace={real_workspace_dir}")
    if args.distribution_manifest:
        log("INIT", f"distribution_manifest={args.distribution_manifest}")
    if args.reference_results:
        log("INIT", f"reference_results={args.reference_results}")

    # ─── Build combo + inject ───
    log("PHASE", "injecting_module")
    module_map_refs = {
        "structuring": module_map.STRUCTURING_MAP,
        "thinking": module_map.THINKING_MAP,
        "memory": module_map.WORK_MEMORY_MAP,
        "localization": module_map.LOCALIZATION_MAP,
        "perception": module_map.PERCEPTION_MAP,
    }

    evolved_combo_config = require_complete_evolution_combo(
        combo_from_reference(args.reference_results),
        "smoke reference combo",
    )
    reference_combo_config = dict(evolved_combo_config)
    for module_type, records in evolved.items():
        evolved_combo_config[module_type] = records[-1].get("name", "")

    custom_configs = {}
    bundle_path = args.reference_bundle or _initial_bundle_path_from_report(args.reference_report)
    if bundle_path:
        bundle = _read_json_optional(bundle_path)
        ok, bundle_combo, bundle_custom_configs, bundle_validation = inject_module_bundle(
            bundle, module_map_refs
        )
        if not ok:
            raise SmokeExecutionLayerError(
                "reference_bundle_invalid",
                f"reference initial bundle injection failed: {bundle_validation}",
            )
        custom_configs.update(bundle_custom_configs)
        log("INJECT", f"reference_bundle={bundle_path}")
        log("INJECT", f"reference_bundle_combo={bundle_combo}")
        bundle_as_evolution = {
            "video_structuring": bundle_combo.get("video_structuring", ""),
            "thinking": bundle_combo.get("thinking", ""),
            "memory": bundle_combo.get("memory", ""),
            "localization": bundle_combo.get("localization", ""),
            "perception": bundle_combo.get("perception", ""),
        }
        reference_delta_modules = {}
        for module_type, reference_name in reference_combo_config.items():
            if not reference_name or reference_name == bundle_as_evolution.get(module_type):
                continue
            if module_type in evolved:
                continue
            artifact_module = _find_artifact_module(
                reference_report,
                module_type,
                reference_name,
            )
            if artifact_module:
                reference_delta_modules.setdefault(module_type, []).append(artifact_module)
                continue
            raise SmokeExecutionLayerError(
                "reference_bundle_artifact_missing",
                "reference combo requires a module not embedded in its self-contained bundle/report: "
                f"{module_type}={reference_name}",
            )
        if reference_delta_modules:
            custom_configs.update(
                inject_evolved_modules(
                    reference_delta_modules,
                    module_map_refs,
                )
            )
            for module_type, modules in reference_delta_modules.items():
                module_name = modules[-1].get("name", "")
                combo_key = {
                    "video_structuring": "video_structuring",
                    "thinking": "thinking",
                    "memory": "memory",
                    "localization": "localization",
                    "perception": "perception",
                }.get(module_type)
                if combo_key and module_name:
                    bundle_combo[combo_key] = module_name
    else:
        raise SmokeExecutionLayerError(
            "reference_bundle_missing",
            "MetaVideoAgent smoke requires an explicit injectable reference bundle; "
            "refusing builtin alias fallback",
        )

    agent_combo = map_combo_to_agent_config(evolved_combo_config)
    custom_configs.update(
        inject_evolved_modules(evolved, module_map_refs)
    )
    log("INJECT", f"combo={evolved_combo_config}")
    if bundle_candidate:
        log("INJECT", f"candidate_bundle_changed_modules={changed_modules}")
        assembly_preflight = _runtime_bundle_assembly_preflight(
            bundle_candidate, evolved_combo_config, module_map_refs,
        )
        log("INJECT", "runtime_preflight=" + json.dumps(assembly_preflight, ensure_ascii=False))
        if not assembly_preflight["passed"]:
            result = {
                "passed": False,
                "failure_kind": "candidate_contract",
                "error": "runtime bundle assembly preflight failed: " + " | ".join(assembly_preflight["issues"]),
                "contract_issues": list(assembly_preflight["issues"]),
                "assembly_preflight": assembly_preflight,
                "capability_events": [],
                "engineering_execution_trace": {},
            }
            log("RESULT", json.dumps(result, ensure_ascii=False))
            return

    # Capture [MODULE:*] logs on stdout. The Agent creation phase must be overridden,
    # Because video_structuring's [MODULE:INIT] is usually output in __init__.
    import io
    module_log_buffer = io.StringIO()

    class TeeStdout:
        """Write to original stdout and buffer at the same time"""
        def __init__(self, original, buffer):
            self.original = original
            self.buffer = buffer
        def write(self, s):
            self.original.write(s)
            self.buffer.write(s)
        def flush(self):
            self.original.flush()
            self.buffer.flush()

    original_stdout = sys.stdout
    sys.stdout = TeeStdout(original_stdout, module_log_buffer)

    # ─── Create Agent ───
    log("PHASE", "creating_agent")
    env = VideoQAEnv(workspace_dir=workspace_dir, video_id=video_id)
    # A behavioral smoke is a one-question, time-reference-scoped runtime
    # check.  It must never hand the complete raw video to a perception
    # adapter merely because the reusable structure library was built for the
    # full video.  The selected task's normalized time reference is the sole
    # executable media boundary for this smoke trajectory.
    bounds = _single_time_reference_bounds(test_task.get("time_reference", ""))
    if not bounds:
        raise SmokeExecutionLayerError(
            "smoke_time_reference_missing",
            "selected smoke task has no executable time_reference bounds",
        )
    smoke_start, smoke_end = bounds
    original_video_length = float(getattr(env, "video_length_secs", 0.0) or 0.0)
    if original_video_length > 0:
        smoke_start = max(0.0, min(smoke_start, original_video_length))
        smoke_end = max(smoke_start, min(smoke_end, original_video_length))
    if smoke_end <= smoke_start:
        raise SmokeExecutionLayerError(
            "smoke_time_reference_invalid",
            "selected smoke task time_reference has no positive media interval",
        )
    setter = getattr(env, "set_active_time_windows", None)
    if not callable(setter):
        raise SmokeExecutionLayerError(
            "smoke_active_window_unsupported",
            "runtime environment does not expose set_active_time_windows",
        )
    active_windows = setter([(smoke_start, smoke_end)], scope="smoke_time_reference")
    log("INIT", "active_time_reference=%.3fs-%.3fs windows=%s" % (
        smoke_start, smoke_end, active_windows,
    ))
    if "video_structuring" in changed_modules and os.environ.get(
        "CODEX_SMOKE_TIME_REF_ONLY", "1"
    ).lower() not in ("0", "false", "no"):
        if bounds:
            start, end = smoke_start, smoke_end
            # A behavioral smoke is intentionally bounded to the selected
            # task's time_reference.  The environment itself enforces that
            # boundary, so a non-zero default here only schedules builder
            # clips that cannot yield frames and makes a valid one-window
            # smoke look like a partial build failure.  Temporal context is
            # still available when explicitly requested by the caller.
            padding = float(os.environ.get("CODEX_SMOKE_TIME_REF_PADDING", "0") or 0)
            original_len = float(getattr(env, "video_length_secs", 0.0) or 0.0)
            window_start = max(0.0, float(start) - padding)
            capped_len = min(original_len, max(1.0, float(end) + padding))
            os.environ["STRUCTURE_BUILD_START_SEC"] = f"{window_start:.3f}"
            os.environ["STRUCTURE_BUILD_END_SEC"] = f"{capped_len:.3f}"
            if 0 < capped_len < original_len:
                env.video_length_secs = capped_len
                log("BUILD_DB", f"time-ref build window: {window_start:.1f}s - {capped_len:.1f}s of {original_len:.1f}s")
    agent = MetaVideoAgent(
        env,
        agent_combo,
        custom_configs=custom_configs,
    )
    log("AGENT", "created successfully")
    if "video_structuring" in changed_modules:
        requires_vector_db = bool(getattr(agent.structuring, "requires_vector_db", False))
        collection_state = (
            bool(getattr(agent.structuring, "collection", None))
            if requires_vector_db else "not_required"
        )
        print(
            "[MODULE:INIT] module=video_structuring class=%s collection=%s origin=smoke_harness_observation"
            % (agent.structuring.__class__.__name__, collection_state)
        )

    # ─── Library building (video_structuring uses an isolated workspace when evolving and cannot pollute the baseline) ───
    force_rebuild = ("video_structuring" in changed_modules)
    structure_handoff_step = {}
    if force_rebuild:
        log("PHASE", "building_db")
        db_path = getattr(agent.structuring, 'db_path', None)
        if os.environ.get("CODEX_SMOKE_DISABLE_FULL_BUILD", "0").lower() not in ("0", "false", "no"):
            log("ERROR", "candidate structuring smoke requires an isolated real build")
            sys.exit(1)
        # Count a real build from the injected structuring class.  The
        # isolated workspace starts empty, so no current-best records can be
        # mistaken for candidate output.
        record_count = [0]
        build_start = time.time()
        original_add = agent.structuring.addStructure

        def _counted_add(*a, **kw):
            record_count[0] += 1
            elapsed = time.time() - build_start
            if record_count[0] % 5 == 0 or record_count[0] <= 3:
                log("BUILD_DB", f"record {record_count[0]}, elapsed {elapsed:.1f}s")
            return original_add(*a, **kw)
        agent.structuring.addStructure = _counted_add

        build_video_structure(env, agent.structuring)
        build_elapsed = time.time() - build_start
        log("BUILD_DB", f"completed: {record_count[0]} records in {build_elapsed:.1f}s")

        # ChromaDB supplemental rebuild: build_video_structure skips VLM because JSONL already exists,
        # But the isolated ChromaDB may be empty. Call rebuild_chroma_from_jsonl to rebuild the vector index from JSONL.
        if hasattr(agent.structuring, 'rebuild_chroma_from_jsonl'):
            import chromadb as _chromadb
            chroma_db_path_new = os.path.join(workspace_dir, "video_structure", "chroma_db")
            video_id_local = getattr(agent.structuring, 'video_id', '')
            struct_type_local = getattr(agent.structuring, 'struct_type', '')
            if getattr(agent.structuring, 'requires_vector_db', False):
                agent.structuring.chroma_client = _chromadb.PersistentClient(path=chroma_db_path_new)
                col_name = f"{video_id_local}_{struct_type_local}".replace("-", "_")[:63]
                agent.structuring.collection = agent.structuring.chroma_client.get_or_create_collection(name=col_name)
            if agent.structuring.collection is None or agent.structuring.collection.count() == 0:
                agent.structuring.rebuild_chroma_from_jsonl()
            else:
                log("BUILD_DB", f"isolated ChromaDB ready: {agent.structuring.collection.count()} records")

        if os.environ.get("CODEX_SMOKE_MICRO_ADDSTRUCTURE", "1").lower() not in ("0", "false", "no"):
            try:
                micro_payload = json.dumps({
                    "clip_start_time": "99990s",
                    "clip_end_time": "99991s",
                    "subject_registry": {},
                    "clip_description": "Smoke test synthetic clip: a person verifies the evolved video structure module.",
                    "action_inventory": ["verify module", "write structure"],
                    "semantic_tags": ["smoke_test", "module_validation"],
                    "entity_list": ["synthetic tester"]
                }, ensure_ascii=False)
                agent.structuring.addStructure(
                    start_sec=99990.0,
                    end_sec=99991.0,
                    description=micro_payload,
                    raw_model_output=micro_payload,
                    source="smoke_test_micro_add",
                )
                log("BUILD_DB", "micro addStructure passed in isolated workspace")
            except Exception as e:
                log("ERROR", f"micro addStructure failed: {e}")
                sys.exit(1)

        try:
            probe = agent.structuring.retrieveStructure(
                query="opening caption title text with year",
                start_sec=0.0,
                end_sec=60.0,
                top_k=8,
            )
            log("BUILD_DB", f"retrieve probe ok: {len(str(probe))} chars")
            print(
                "[MODULE:RETRIEVE] module=video_structuring mode=%s total_chars=%d origin=smoke_harness_observation"
                % ("vector" if getattr(agent.structuring, "collection", None) is not None else "linear_fallback", len(str(probe)))
            )
        except Exception as e:
            log("ERROR", f"retrieve probe failed: {e}")
            sys.exit(1)
        if db_path:
            structure_handoff_step = _structure_build_handoff_step(
                db_path, bundle_candidate.get("handoff_contracts") or [],
            )
            if structure_handoff_step:
                log("BUILD_DB", "recorded isolated structuring→localization producer handoff")
    else:
        reference_structure_dir = _reference_structure_artifact_dir(
            reference_report, args.reference_report,
        )
        if not reference_structure_dir:
            raise SmokeExecutionLayerError(
                "reference_structure_artifact_missing",
                "current-best reference report has no reusable structure artifact",
            )
        try:
            _reuse_reference_structure_for_smoke(
                agent, reference_structure_dir, log,
            )
        except Exception as exc:
            raise SmokeExecutionLayerError(
                "reference_structure_artifact_unusable",
                f"could not reuse current-best structure artifact: {exc}",
            ) from exc
        # The isolated DB now contains exactly the report-attested structure
        # artifact (and, for time-reference smoke, only its scoped subset).
        # It is a real producer input consumed by localization, so expose a
        # deterministic producer row just as the rebuild branch does.
        db_path = getattr(agent.structuring, "db_path", None)
        if db_path:
            structure_handoff_step = _structure_build_handoff_step(
                db_path, bundle_candidate.get("handoff_contracts") or [],
                source_mode="reused_scoped_reference",
            )
            if structure_handoff_step:
                log("BUILD_DB", "recorded scoped reused structuring→localization producer handoff")

    # ─── Run Agent ───
    log("PHASE", "agent_running")
    agent_start = time.time()

    import runtime_evidence
    evidence_token = runtime_evidence.begin_evidence_run()
    trace_token = runtime_evidence.begin_runtime_trace()
    try:
        answer, trajectory = agent.run(test_task["question"], extra_info=test_task)
        capability_events = runtime_evidence.get_evidence_events()
        runtime_trace = runtime_evidence.get_runtime_trace()
    finally:
        runtime_evidence.reset_evidence_run(evidence_token)
        runtime_evidence.reset_runtime_trace(trace_token)

    if structure_handoff_step:
        trajectory = [structure_handoff_step] + list(trajectory or [])

    sys.stdout = original_stdout
    agent_elapsed = time.time() - agent_start

    steps = len([s for s in trajectory if s.get("step_type") == "tool_execution"])
    log("AGENT", f"completed: {steps} steps in {agent_elapsed:.1f}s")

    # ─── Critical path log inspection ───
    module_logs = module_log_buffer.getvalue()
    module_log_lines = [line for line in module_logs.split("\n") if "[MODULE:" in line]

    def persist_full_trace(result):
        """Write the raw runtime evidence needed by a later repair.

        This artifact intentionally keeps model outputs and normalized module
        envelopes, but never reads environment variables or serializes API
        credentials.  It is written for both passed and failed smokes.
        """
        document = {
            "artifact_type": "smoke_full_execution_trace",
            "schema_version": 1,
            "candidate_bundle_fingerprint": str((bundle_candidate or {}).get("bundle_fingerprint") or ""),
            "changed_modules": list(changed_modules),
            # The agent receives only question/choices/time scope, but the
            # evaluator task row also carries gold fields.  Persist the same
            # execution-safe view here so a smoke trace can never become an
            # accidental supervision source for a later repair/codegen step.
            "task": {
                key: test_task.get(key)
                for key in ("task_id", "video_id", "question", "choices", "time_reference")
                if key in test_task
            },
            "trajectory": trajectory,
            "capability_events": capability_events,
            "runtime_trace": runtime_trace,
            "module_stdout": module_logs,
            "smoke_result": result,
        }
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, default=str)
        result["full_execution_trace_path"] = trace_path

    log("VERIFY", f"captured {len(module_log_lines)} module log lines")
    for ml in module_log_lines[:10]:
        log("MODULE_LOG", ml.strip()[:150])

    # Critical path verification by module type
    path_warnings = []
    nonfatal_warnings = []
    handoff_checks = _bundle_handoff_checks(
        trajectory, capability_events,
        (bundle_candidate.get("handoff_contracts") or []) if bundle_candidate else [],
    )
    engineering_trace = _engineering_execution_trace(
        test_task, trajectory, capability_events, str(answer or ""),
    )
    # The trajectory is the source of truth, but these execution-layer facts
    # make an inter-module break directly inspectable without asking Codex to
    # reconstruct it from stdout.  They are still no-label evidence.
    engineering_trace.update({
        "runtime_summary": {
            "tool_execution_steps": steps,
            "agent_elapsed_sec": round(agent_elapsed, 6),
        },
        "module_logs": _redact_smoke_supervision(module_log_lines, str(answer or "")),
        "handoff_checks": _redact_smoke_supervision(handoff_checks, str(answer or "")),
    })
    answer_issue = _answer_contract_issue(answer, test_task)
    if answer_issue:
        path_warnings.append("Final-answer contract failed: " + answer_issue)
    for check in handoff_checks:
        if not check["passed"]:
            path_warnings.append(
                "Bundle handoff was not realized on the smoke trajectory: "
                f"{check['producer']}→{check['consumer']}; "
                f"producer_observed={check['producer_observed']}, "
                f"missing_fields={check['missing_fields']}, "
                f"consumer_observed={check['consumer_observed']}"
            )
    if "video_structuring" in changed_modules:
        # Check 1: Whether the collection is initialized
        init_logs = [line for line in module_log_lines if "[MODULE:INIT]" in line]
        if init_logs:
            if "collection=False" in init_logs[-1] or "collection=None" in init_logs[-1]:
                path_warnings.append(
                    "ChromaDB collection is not initialized (collection=False/None)."
                    "Vector retrieval is unavailable and will degrade to linear full-text scanning."
                    "Check whether the requires_vector_db parameter in __init__ is True.")
        else:
            path_warnings.append(
                "[MODULE:INIT] log not found. The code may be missing critical path log output.")

        # Check 2: Whether retrieveStructure takes the vector path
        retrieve_logs = [line for line in module_log_lines if "[MODULE:RETRIEVE]" in line]
        if retrieve_logs:
            vector_calls = [line for line in retrieve_logs if "mode=vector" in line]
            fallback_calls = [line for line in retrieve_logs if "mode=linear_fallback" in line]
            if not vector_calls and fallback_calls:
                nonfatal_warnings.append(
                    f"retrieveStructure used linear_fallback for all {len(fallback_calls)} calls; "
                    "vector retrieval was not exercised. Verify the collection, query, and indexed records."
                )
            # Check whether the number of characters returned is too large
            for rl in retrieve_logs:
                if "total_chars=" in rl:
                    try:
                        chars = int(rl.split("total_chars=")[1].split(",")[0].split(")")[0])
                        if chars > 10000:
                            path_warnings.append(
                                f"retrieveStructure returned {chars} characters (>10000); "
                                "limit top_k or bound the rendered result."
                            )
                            break
                    except (ValueError, IndexError):
                        pass
        else:
            path_warnings.append(
                "[MODULE:RETRIEVE] log not found."
                "retrieveStructure() may be missing logs or never called.")

    elif primary_target in ("perception", "localization"):
        tool_logs = [line for line in module_log_lines if "[MODULE:TOOL]" in line]
        if not tool_logs:
            # Generated modules are not required to print this optional log line.
            # The normalized module-result trajectory and capability events are
            # the execution source of truth.  Do not reject a real scoped VLM
            # call merely because the optional display log is absent.
            observed_tool_result = any(
                isinstance(step, dict)
                and step.get("step_type") == "tool_execution"
                and str(step.get("producer_module") or "") == primary_target
                and str((step.get("module_result") or {}).get("status") or "") == "ok"
                for step in trajectory
            )
            if observed_tool_result:
                nonfatal_warnings.append(
                    f"[MODULE:TOOL] log absent, but the trajectory confirms {primary_target} tool execution."
                )
            else:
                path_warnings.append(
                    f"[MODULE:TOOL] log absent and the trajectory does not confirm {primary_target} tool execution."
                )
        if primary_target == "localization":
            path_warnings.extend(_localization_smoke_contract_issues(
                trajectory,
                module_code,
                getattr(env, "raw_video_path", ""),
            ))

    observed_capabilities = {
        str(item.get("capability") or "")
        for item in capability_events
        if item.get("event") in {"inference", "transcription"} and item.get("status") in {"ok", "no_content", "no_speech"}
    }
    missing_capabilities = expected_capabilities - observed_capabilities
    if missing_capabilities:
        # Smoke uses one generic runtime task and must not turn an untriggered
        # conditional capability into a synthetic failure.  Assembly validation
        # already proves the registered adapter binding; probe checks whether
        # the capability is actually reached on its selected mechanism rows.
        nonfatal_warnings.append(
            "Declared conditional runtime capabilities were not triggered by this smoke task: "
            + ", ".join(sorted(missing_capabilities))
        )
    observed_profiles = {}
    for item in capability_events:
        if (
            item.get("event") not in {"inference", "transcription"}
            or item.get("status") not in {"ok", "no_content", "no_speech"}
            or not item.get("profile_id")
        ):
            continue
        observed_profiles.setdefault(str(item.get("capability") or ""), set()).add(
            str(item.get("profile_id") or "")
        )
    mismatched_profiles = {
        capability: {"expected": expected, "observed": sorted(observed_profiles.get(capability, set()))}
        for capability, expected in expected_profiles.items()
        if capability in observed_capabilities and expected not in observed_profiles.get(capability, set())
    }
    if mismatched_profiles:
        path_warnings.append(
            "Declared capability profile did not match the actual runtime event: "
            + json.dumps(mismatched_profiles, ensure_ascii=False, sort_keys=True)
        )
    if args.require_consumer_event and not any(
        item.get("event") == "consumer_consumed" and item.get("status") == "ok"
        and item.get("producer_module") in set(changed_modules)
        and item.get("contract_valid") is True
        for item in capability_events
    ):
        path_warnings.append(
            "Target output was not consumed with a valid protocol by the preserved thinking/runtime path on the smoke trajectory."
        )
    # Output verification results
    if path_warnings:
        log("VERIFY", f"⚠️ {len(path_warnings)} critical-path warning(s):")
        for i, w in enumerate(path_warnings):
            log("VERIFY_WARN", f"[{i+1}] {w}")
        # Critical path failure = smoke test failed
        result = {
            "passed": False,
            "steps": steps,
            "elapsed": agent_elapsed,
            "contract_issues": path_warnings,
            "capability_events": capability_events,
            "expected_capabilities": sorted(expected_capabilities),
            "expected_profiles": expected_profiles,
            "failure_contract": failure_contract,
            "handoff_checks": handoff_checks,
            "engineering_execution_trace": engineering_trace,
            "answer": str(answer or ""),
            "error": "Critical-path verification failed: " + " | ".join(path_warnings),
        }
        persist_full_trace(result)
        log("RESULT", json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    else:
        if nonfatal_warnings:
            log("VERIFY", f"⚠️ {len(nonfatal_warnings)} non-fatal critical path warnings:")
            for i, w in enumerate(nonfatal_warnings):
                log("VERIFY_WARN", f"[{i+1}] {w}")
        log("VERIFY", "✅ All critical paths verified passed")

    # ─── Output trajectory summary ───
    log("PHASE", "completed")
    for i, step in enumerate(trajectory[-5:]):
        st = step.get("step_type", "?")
        if st == "tool_execution":
            tool = step.get("tool", "?")
            obs_preview = str(step.get("observation", ""))[:100]
            log("TRACE", f"step {i}: tool={tool}, obs={obs_preview}")
        elif st == "reasoning":
            decision = step.get("decision", "?")
            log("TRACE", f"step {i}: reasoning decision={decision}")

    # final result
    result = {
        "passed": True,
        "steps": steps,
        "elapsed": agent_elapsed,
        "capability_events": capability_events,
        "expected_capabilities": sorted(expected_capabilities),
        "expected_profiles": expected_profiles,
        "handoff_checks": handoff_checks,
        "failure_contract": failure_contract,
        "engineering_execution_trace": engineering_trace,
        "answer": str(answer or ""),
    }
    persist_full_trace(result)
    log("RESULT", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except SmokeExecutionLayerError as e:
        print(f"[SMOKE:ERROR] {e}", flush=True)
        print(
            f"[SMOKE:RESULT] {json.dumps({'passed': False, 'failure_kind': 'execution_layer', 'execution_layer_rerun_required': True, 'execution_error_code': e.code, 'error': str(e)[:1000]})}",
            flush=True,
        )
        sys.exit(2)
    except Exception as e:
        import traceback
        print(f"[SMOKE:CRASH] {type(e).__name__}: {e}", flush=True)
        print(f"[SMOKE:TRACEBACK] {traceback.format_exc()[-1000:]}", flush=True)
        print(f"[SMOKE:RESULT] {json.dumps({'passed': False, 'error': str(e)[:500]})}", flush=True)
        sys.exit(1)
