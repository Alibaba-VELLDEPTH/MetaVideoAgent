"""Per-iteration context assembly for MetaVideoAgent evolution.

The context file is the contract between orchestration, Teacher/Diagnosis, and
Codex.  It records what the current reference is, what earlier candidates did,
which artifacts exist, and what the next diagnosis/code pass must decide.  It
does not replace Teacher reviews or full reports; it points to them and stores
bounded summaries so prompts do not depend on path guessing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List

try:
    from runtime_paths import metavideoagent_run_dir
except ImportError:  # pragma: no cover - script-mode fallback
    from .runtime_paths import metavideoagent_run_dir


SCHEMA_VERSION = 1


def read_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def file_digest(path: str) -> str:
    if not path or not os.path.exists(path) or os.path.isdir(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_meta(path: str) -> Dict[str, Any]:
    if not path:
        return {"path": "", "exists": False}
    abs_path = os.path.abspath(path)
    exists = os.path.exists(abs_path)
    return {
        "path": abs_path,
        "exists": exists,
        "sha256": file_digest(abs_path) if exists and os.path.isfile(abs_path) else "",
    }


def _reference_results_path(report: Dict[str, Any]) -> str:
    return (
        report.get("results_path")
        or report.get("full_eval_results")
        or report.get("source_full_eval_results")
        or report.get("sandbox_dir")
        or ""
    )


def _first_combo_from_results(results_path: str) -> Dict[str, Any]:
    for row in _iter_result_rows(results_path, limit=64) or []:
        combo = (
            row.get("architecture_combo")
            or row.get("combo")
            or row.get("agent_combo")
            or {}
        )
        if isinstance(combo, dict) and combo:
            return combo
    return {}


def _bundle_from_report(report: Dict[str, Any], report_path: str) -> str:
    path = (
        report.get("current_best_bundle_path")
        or report.get("candidate_full_eval_bundle_path")
        or ((report.get("combo_policy") or {}).get("current_best_bundle_path", ""))
        or ((report.get("combo_policy") or {}).get("effective_bundle_path", ""))
        or report.get("initial_baseline_bundle")
        or report.get("bundle_path")
        or ((report.get("combo_policy") or {}).get("base_bundle_path", ""))
        or ""
    )
    if path and os.path.exists(path):
        return os.path.abspath(path)
    if path and report_path:
        joined = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(report_path)), path))
        if os.path.exists(joined):
            return joined
    if report_path:
        candidate = os.path.join(os.path.dirname(os.path.abspath(report_path)), "initial_baseline_bundle.json")
        if os.path.exists(candidate):
            return candidate
    for source in report.get("sources", []) or []:
        if not isinstance(source, str):
            continue
        candidate = os.path.join(os.path.dirname(os.path.abspath(source)), "initial_baseline_bundle.json")
        if os.path.exists(candidate):
            return candidate
    return ""


def _iter_result_rows(path: str, limit: int = 500) -> Iterable[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return
    paths = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".jsonl"):
                paths.append(os.path.join(path, name))
    else:
        paths = [path]
    yielded = 0
    for item in paths:
        if yielded >= limit:
            return
        try:
            with open(item, "r", encoding="utf-8") as f:
                for line in f:
                    if yielded >= limit:
                        return
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yielded += 1
                    yield row
        except OSError:
            continue


def _cluster_failure_text(text: str) -> str:
    value = (text or "").lower()
    if any(k in value for k in ("connection error", "apiconnectionerror", "read timed out", "timeout")):
        return "api_connection_or_timeout"
    if any(k in value for k in ("rate limit", "429")):
        return "api_rate_limit"
    if any(k in value for k in ("invalid token", "401", "unauthorized")):
        return "api_auth"
    if any(k in value for k in ("readonly database", "database is locked", "chroma")):
        return "storage_or_chroma"
    if any(k in value for k in ("does not match any options", "not match any option")):
        return "option_mapping_or_unmatched_answer"
    if any(k in value for k in ("error", "exception", "traceback")):
        return "runtime_error"
    return "semantic_or_unknown"


def _result_behavior_metrics(results_path: str) -> Dict[str, Any]:
    rows = list(_iter_result_rows(results_path, limit=500) or [])
    if not rows:
        return {
            "row_sample_count": 0,
            "latency_avg": None,
            "tool_steps_avg": None,
            "api_call_count": None,
            "runtime_error_count": 0,
            "failure_clusters": {},
        }
    latencies = []
    tool_steps = []
    api_calls = 0
    runtime_error_count = 0
    clusters: Dict[str, int] = {}
    for row in rows:
        for key in ("elapsed_sec", "duration_sec", "latency_sec", "runtime_sec"):
            if isinstance(row.get(key), (int, float)):
                latencies.append(float(row[key]))
                break
        traj = row.get("trajectory") or []
        if isinstance(traj, list):
            tool_steps.append(len(traj))
            for step in traj:
                if not isinstance(step, dict):
                    continue
                text = json.dumps(step, ensure_ascii=False).lower()[:2000]
                if re.search(r"\b(llm|vlm|api|model|provider|endpoint|evolution_llm)\b", text):
                    api_calls += 1
        text = "\n".join(str(row.get(k, "")) for k in ("answer", "final_agent_answer", "error"))
        cluster = _cluster_failure_text(text)
        if cluster != "semantic_or_unknown" or row.get("error"):
            runtime_error_count += 1 if cluster != "semantic_or_unknown" else 0
            clusters[cluster] = clusters.get(cluster, 0) + 1
    return {
        "row_sample_count": len(rows),
        "latency_avg": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "tool_steps_avg": round(sum(tool_steps) / len(tool_steps), 3) if tool_steps else None,
        "api_call_count": api_calls if api_calls else None,
        "runtime_error_count": runtime_error_count,
        "failure_clusters": clusters,
    }


def _metrics_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    total = report.get("total_questions") or report.get("expected_total_questions") or 0
    correct = (
        report.get("candidate_correct")
        if report.get("candidate_correct") is not None
        else report.get("correct")
    )
    accuracy = None
    if total and correct is not None:
        try:
            accuracy = float(correct) / float(total)
        except Exception:
            accuracy = None
    criteria = report.get("final_success_criteria", {}) or {}
    results_path = _reference_results_path(report)
    behavior = _result_behavior_metrics(results_path)
    return {
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_delta": report.get("accuracy_delta"),
        "corrections": len(report.get("corrections", []) or []),
        "regressions": len(report.get("regressions", []) or []),
        "passed": criteria.get("passed"),
        "verdict": report.get("verdict") or report.get("status"),
        "cost_reference": report.get("cost_reference") or report.get("cost_baseline") or {},
        "cost_summary": report.get("cost_summary") or report.get("cost_reference") or report.get("cost_baseline") or {},
        **behavior,
    }


def _compact_combo_report(report_path: str, label: str = "") -> Dict[str, Any]:
    report = read_json(report_path)
    if not report:
        return {
            "label": label,
            "report": _path_meta(report_path),
            "results": {"path": "", "exists": False},
            "combo": {},
            "metrics": {},
        }
    results_path = _reference_results_path(report)
    if results_path and not os.path.isabs(results_path):
        joined = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(report_path)), results_path))
        if os.path.exists(joined):
            results_path = joined
    combo = report.get("combo") or report.get("agent_combo") or {}
    if not combo:
        combo = _first_combo_from_results(results_path)
    return {
        "label": label or report.get("reference_label") or "reference",
        "report": _path_meta(report_path),
        "results": _path_meta(results_path),
        "combo": combo,
        "candidate": report.get("candidate") or {},
        "initial_baseline_bundle": _path_meta(_bundle_from_report(report, report_path)),
        "metrics": _metrics_from_report(report),
        "validation_scope": report.get("validation_scope", ""),
        "question_split": report.get("question_split", ""),
        "video_ids": report.get("video_ids", []),
    }


def load_current_best(run_id: str, output_root: str = "") -> Dict[str, Any]:
    path = os.path.join(metavideoagent_run_dir(run_id, output_root), "current_best_ledger.json")
    ledger = read_json(path)
    if not ledger:
        return {"path": path, "current_best": {}, "entries": []}
    ledger["path"] = path
    return ledger


def evolution_memory_index_path(run_id: str, output_root: str = "") -> str:
    return os.path.join(metavideoagent_run_dir(run_id, output_root), "evolution_memory_index.json")


def load_evolution_memory_index(run_id: str, output_root: str = "") -> Dict[str, Any]:
    """Read the run-scoped candidate history written by the orchestrator.

    MetaVideoAgent later rounds consume only artifacts from the active run.
    """
    path = evolution_memory_index_path(run_id, output_root)
    payload = read_json(path)
    if not payload:
        return {
            "artifact_type": "metavideoagent_evolution_memory",
            "schema_version": 1,
            "run_id": run_id,
            "records": [],
            "module_summary": {},
            "path": path,
        }
    payload.setdefault("records", [])
    payload.setdefault("module_summary", {})
    payload["path"] = path
    return payload


def _artifact_refs(paths: Iterable[str]) -> List[Dict[str, Any]]:
    return [_path_meta(path) for path in paths if path]


def build_iter_context(
    *,
    run_id: str,
    iter_index: int,
    stage: str,
    workspace: str,
    distribution_manifest: str,
    reference_report_path: str,
    reference_label: str,
    output_root: str = "",
    profile_path: str = "",
    profile: Dict[str, Any] | None = None,
    initial_baseline_bundle: str = "",
    previous_diagnosis: str = "",
    evolution_review: str = "",
    full_eval_report: str = "",
    candidate_run: str = "",
    candidate_bundle: str = "",
    candidate_rejection_context: str = "",
    deep_research_summary: str = "",
    extra_artifacts: Iterable[str] = (),
    check_only: bool = False,
    attempt_tag: str = "",
) -> Dict[str, Any]:
    """Create a bounded context object for an iteration.

    Check-only plans are written under ``checks/`` so they cannot overwrite a
    completed or in-progress formal ``iter_N/iter_context.json``.
    """
    base_dir = metavideoagent_run_dir(run_id, output_root)
    iter_dir = os.path.join(
        base_dir,
        "checks", f"iter_{iter_index}"
    ) if check_only else os.path.join(base_dir, f"iter_{iter_index}")
    if attempt_tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(attempt_tag)).strip("._")
        if not safe_tag:
            raise ValueError("attempt_tag must contain at least one safe filename character")
        iter_dir = os.path.join(iter_dir, safe_tag)
    os.makedirs(iter_dir, exist_ok=True)

    reference = _compact_combo_report(reference_report_path, label=reference_label)
    current_best = load_current_best(run_id, output_root)
    evolution_memory = load_evolution_memory_index(run_id, output_root)
    profile_payload = profile or read_json(profile_path)
    channel_schema = profile_payload.get("information_channel_schema", {}) or {}
    profile_summary = {
        "path": profile_path or profile_payload.get("output_path", ""),
        "primary_channel": (
            (profile_payload.get("information_channel_hypothesis", {}) or {})
            .get("primary_observed_channel", "")
        ),
        "information_channel_schema": channel_schema,
        "dominant_information_channel": channel_schema.get("dominant_information_channel", ""),
        "channel_confidence": channel_schema.get("channel_confidence"),
        "channel_scores": (
            (profile_payload.get("information_channel_hypothesis", {}) or {})
            .get("channel_scores", {})
        ),
        "sampling": profile_payload.get("sampling", {}),
        "redacted_prompt_fields": (
            (profile_payload.get("distribution_spec", {}) or {})
            .get("redacted_prompt_fields", [])
        ),
    }

    history_entries = current_best.get("entries", [])
    candidate_rejection_payload = read_json(candidate_rejection_context)
    memory_records = evolution_memory.get("records", [])
    recent_records = memory_records[-8:]
    # Historical probe/engineering failures remain in evolution_memory where
    # their schema and evidence tier are explicit.  candidate_rejections is
    # reserved for the current codex_evolve rejection-context contract.
    candidate_rejections = [candidate_rejection_payload] if candidate_rejection_payload else []
    context = {
        "schema_version": SCHEMA_VERSION,
        "created_at": int(time.time()),
        "run_id": run_id,
        "iter_index": iter_index,
        "stage": stage,
        "attempt_tag": attempt_tag,
        "check_only": bool(check_only),
        "workspace": os.path.abspath(workspace),
        "distribution_manifest": os.path.abspath(distribution_manifest) if distribution_manifest else "",
        "run_dir": base_dir,
        "iter_dir": iter_dir,
        "reference": reference,
        "current_best": current_best.get("current_best", {}),
        "current_best_ledger_path": current_best.get("path", ""),
        "history": {
            "current_best_entries": history_entries[-8:],
            "evolution_memory_index_path": evolution_memory.get("path", ""),
            "recent_candidate_records": recent_records,
            "module_summary": evolution_memory.get("module_summary", {}),
            "artifact_refs": _artifact_refs([
                previous_diagnosis,
                evolution_review,
                full_eval_report,
                candidate_bundle,
                candidate_rejection_context,
                *list(extra_artifacts or []),
            ]),
        },
        "initial_baseline": {
            "bundle": _path_meta(initial_baseline_bundle or (reference.get("initial_baseline_bundle", {}) or {}).get("path", "")),
        },
        "previous_round": {
            "diagnosis": _path_meta(previous_diagnosis),
            "evolution_review": _path_meta(evolution_review),
            "full_eval_report": _path_meta(full_eval_report),
            "candidate_run": _path_meta(candidate_run),
            "candidate_bundle": _path_meta(candidate_bundle),
            "candidate_rejection_context": _path_meta(candidate_rejection_context),
        },
        "candidate_rejections": candidate_rejections,
        "observed_distribution_profile": profile_summary,
        "deep_research": {
            "summary": deep_research_summary[:4000] if deep_research_summary else "",
        },
        "diagnosis_requirements": {
            "current_best_policy": (
                "Diagnose and modify the explicit current_best/reference combo, "
                "not a rejected candidate. Treat candidate history as evidence "
                "about mechanisms to preserve or avoid."
            ),
            "history_policy": (
                "Use recent_candidate_records and module_summary to distinguish "
                "full-eval evidence, probe-only evidence, and engineering failures. "
                "Do not repeat a rejected direction without a new causal rationale."
            ),
            "base_combo_policy": (
                "Use combo_base=current_best and base_combo_policy=keep_current_best. "
                "The next candidate must implement the diagnosis-selected module-set "
                "direction while retaining the accepted current best as its base."
            ),
            "rejected_directions": "Explicitly list rejected implementation directions from review, full-eval, and probe evidence.",
            "probe_plan": "Emit repair_probe and regression_guard refs without using hidden type/category labels.",
        },
    }
    context_path = os.path.join(iter_dir, "iter_context.json")
    context["path"] = context_path
    write_json(context_path, context)
    return context


def inject_iter_context(report: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Attach compact iteration context fields to a diagnosis/report object."""
    if not context:
        return report
    report["iter_context_path"] = context.get("path", "")
    report["iter_context"] = {
        "run_id": context.get("run_id"),
        "iter_index": context.get("iter_index"),
        "stage": context.get("stage"),
        "reference": context.get("reference", {}),
        "current_best": context.get("current_best", {}),
        "evolution_memory": {
            "index_path": ((context.get("history", {}) or {}).get("evolution_memory_index_path", "")),
            "recent_candidate_records": ((context.get("history", {}) or {}).get("recent_candidate_records", []))[:8],
            "module_summary": ((context.get("history", {}) or {}).get("module_summary", {})),
        },
        "previous_round": context.get("previous_round", {}),
        "candidate_rejections": context.get("candidate_rejections", [])[:5],
        "observed_distribution_profile": context.get("observed_distribution_profile", {}),
        "diagnosis_requirements": context.get("diagnosis_requirements", {}),
    }
    return report
