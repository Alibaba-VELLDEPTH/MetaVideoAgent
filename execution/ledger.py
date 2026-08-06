"""Run-scoped current-best ledger for MetaVideoAgent evolution."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict

try:
    from .paths import ensure_dir
    from .paths import run_dir as metavideoagent_run_dir
except ImportError:  # script-mode fallback
    from paths import ensure_dir
    from paths import run_dir as metavideoagent_run_dir


def ledger_path(run_id: str, output_root: str = "") -> str:
    return os.path.join(metavideoagent_run_dir(run_id, output_root), "current_best_ledger.json")


def evolution_memory_path(run_id: str, output_root: str = "") -> str:
    """Return the run-scoped, append-only candidate history index.

    The current-best ledger answers "which combo is the execution reference?".
    This companion index answers "which candidates were tried, why did they
    help or fail, and which artifacts explain that conclusion?"  Keeping the
    two concerns separate prevents a rejected candidate from replacing the
    current reference while still making it available to later diagnosis.
    """
    return os.path.join(metavideoagent_run_dir(run_id, output_root), "evolution_memory_index.json")


def load_ledger(run_id: str, output_root: str = "") -> Dict:
    path = ledger_path(run_id, output_root)
    if not os.path.exists(path):
        return {
            "artifact_type": "metavideoagent_current_best_ledger",
            "schema_version": 1,
            "run_id": run_id,
            "entries": [],
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evolution_memory(run_id: str, output_root: str = "") -> Dict:
    path = evolution_memory_path(run_id, output_root)
    if not os.path.exists(path):
        return {
            "artifact_type": "metavideoagent_evolution_memory",
            "schema_version": 1,
            "run_id": run_id,
            "records": [],
            "module_summary": {},
        }
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("artifact_type", "metavideoagent_evolution_memory")
    payload.setdefault("schema_version", 1)
    payload.setdefault("run_id", run_id)
    payload.setdefault("records", [])
    payload.setdefault("module_summary", {})
    payload["path"] = path
    return payload


def _file_digest(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_meta(path: str) -> Dict[str, str]:
    return {
        "path": os.path.abspath(path) if path else "",
        "sha256": _file_digest(path),
    }


def _compact(value: Any, *, depth: int = 0) -> Any:
    """Bound history payloads so later prompts never ingest raw trajectories."""
    if depth >= 3:
        return "<truncated>"
    if isinstance(value, str):
        return value[:1600]
    if isinstance(value, list):
        return [_compact(item, depth=depth + 1) for item in value[:12]]
    if isinstance(value, dict):
        return {
            str(key): _compact(item, depth=depth + 1)
            for key, item in list(value.items())[:24]
            if key not in {"code", "evolved_micro_reviews", "delta_reviews", "question_reviews"}
        }
    return value


def _read_json(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _candidate_identity(candidate: Dict, full_report: Dict, candidate_run: str,
                        candidate_bundle_path: str = "") -> Dict:
    candidate = candidate or {}
    full_report = full_report or {}
    if candidate.get("artifact_type") != "candidate_bundle":
        raise ValueError("evolution history accepts candidate_bundle artifacts only")
    return {
        "combo_id": full_report.get("combo_id", ""),
        "name": "MetaVideoAgent bundle",
        "target_modules": list(candidate.get("changed_modules") or []),
        "changed_modules": list(candidate.get("changed_modules") or []),
        "bundle_fingerprint": candidate.get("bundle_fingerprint", ""),
        # The signature is a compact mechanism-level guard.  It deliberately
        # excludes candidate code and task data, so later diagnosis can avoid
        # repeating an exhausted repair without turning a module into a ban.
        "mechanism_signature": candidate.get("bundle_fingerprint", ""),
        "candidate_run": os.path.abspath(candidate_run) if candidate_run else "",
        "candidate_bundle": _path_meta(candidate_bundle_path),
    }


def _review_summary(review: Dict) -> Dict:
    if not review:
        return {}
    # The review runner creates this compact schema specifically for later
    # diagnosis.  Never reconstruct macro/micro review prose here.
    return _compact({
        "review_status": review.get("review_status") or review.get("status") or "",
        "mechanism_summary": review.get("mechanism_summary") or {},
    })


def _probe_mechanism_summary(rejection: Dict) -> Dict:
    """Retain only executable probe facts, not audit prose or candidate code."""
    rejection = rejection or {}
    audit = rejection.get("probe_audit") or rejection.get("audit") or {}
    if not isinstance(audit, dict):
        audit = {}
    issues = audit.get("issues") or rejection.get("issues") or []
    return _compact({
        "action": rejection.get("action") or rejection.get("route") or "",
        "reason": rejection.get("reason") or rejection.get("failure_reason") or "",
        "probe_status": rejection.get("probe_status") or audit.get("probe_status") or "",
        "engineering_invalid": bool(rejection.get("engineering_invalid") or audit.get("engineering_invalid")),
        "accuracy_delta": rejection.get("accuracy_delta", audit.get("accuracy_delta")),
        "corrections": rejection.get("corrections", audit.get("corrections")),
        "regressions": rejection.get("regressions", audit.get("regressions")),
        "issue_codes": [
            str(item.get("code") or "") for item in issues if isinstance(item, dict)
        ][:12],
        "capability_summary": audit.get("capability_summary") or {},
        "consumer_contract_summary": audit.get("consumer_contract_summary") or {},
        "mechanism_witness_outcome": (audit.get("mechanism_witness") or {}).get("outcome", ""),
        "mechanism_signature": (rejection.get("candidate") or {}).get("mechanism_signature", ""),
    })


def _rebuild_module_summary(records: list[Dict]) -> Dict:
    summary: Dict[str, Dict] = {}
    for record in records:
        for module in (record.get("candidate") or {}).get("target_modules") or []:
            bucket = summary.setdefault(module, {
                "attempts": 0,
                "became_current_best": 0,
                "full_eval_non_best": 0,
                "probe_rejected": 0,
                "engineering_invalid": 0,
                "recent_record_ids": [],
                "recent_mechanism_signatures": [],
            })
            bucket["attempts"] += 1
            status = record.get("outcome", {}).get("status", "")
            if status == "current_best":
                bucket["became_current_best"] += 1
            elif status == "full_eval_non_best":
                bucket["full_eval_non_best"] += 1
            elif status == "probe_rejected":
                bucket["probe_rejected"] += 1
            elif status == "engineering_invalid":
                bucket["engineering_invalid"] += 1
            record_id = record.get("record_id", "")
            if record_id:
                bucket["recent_record_ids"].append(record_id)
            signature = str((record.get("candidate") or {}).get("mechanism_signature") or "")
            if signature:
                bucket["recent_mechanism_signatures"].append(signature)
    for bucket in summary.values():
        bucket["recent_record_ids"] = bucket["recent_record_ids"][-8:]
        bucket["recent_mechanism_signatures"] = list(dict.fromkeys(
            bucket["recent_mechanism_signatures"][-8:]
        ))
    return summary


def record_evolution_outcome(
    run_id: str,
    *,
    iteration: int,
    candidate_run: str = "",
    candidate_bundle_path: str = "",
    parent_report_path: str = "",
    parent_results_path: str = "",
    parent_label: str = "",
    diagnosis_path: str = "",
    research_report_path: str = "",
    full_eval_report_path: str = "",
    review_path: str = "",
    rejection_context_path: str = "",
    outcome_status: str = "",
    output_root: str = "",
) -> Dict:
    """Upsert a candidate outcome into the run-scoped evolution memory.

    A record is keyed by candidate run (or the available report path) and can
    be written incrementally: probe rejection first, then later full-eval and
    review artifacts.  It deliberately stores compact summaries plus paths,
    rather than duplicating raw trajectory JSONL.
    """
    memory = load_evolution_memory(run_id, output_root)
    full_report = _read_json(full_eval_report_path)
    candidate_payload = _read_json(candidate_bundle_path)
    review = _read_json(review_path)
    rejection = _read_json(rejection_context_path)
    decision = full_report.get("post_full_eval_decision") or {}
    candidate = _candidate_identity(candidate_payload, full_report, candidate_run, candidate_bundle_path)
    implementation_spec = candidate_payload.get("implementation_spec", {}) if isinstance(candidate_payload, dict) else {}
    if not isinstance(implementation_spec, dict):
        implementation_spec = {}
    key_source = candidate.get("candidate_run") or os.path.abspath(full_eval_report_path or rejection_context_path or candidate_bundle_path)
    record_id = "candidate:" + key_source if key_source else f"iteration:{iteration}:{int(time.time())}"
    if not outcome_status:
        if decision.get("update_current_best") is True:
            outcome_status = "current_best"
        elif full_report:
            outcome_status = "full_eval_non_best"
        elif rejection:
            outcome_status = "probe_rejected"
        else:
            outcome_status = "candidate_recorded"
    record = {
        "record_id": record_id,
        "iteration": int(iteration),
        "updated_at": int(time.time()),
        "candidate": candidate,
        "runtime_contract": _compact({
            "capability_declaration": [
                str(item.get("capability") or "")
                for item in (implementation_spec.get("capability_contract") or [])
                if isinstance(item, dict) and str(item.get("capability") or "")
            ],
            "consumer_contract": implementation_spec.get("consumer_contract") or {},
        }),
        "lineage": {
            "parent_report": _path_meta(parent_report_path),
            "parent_results": _path_meta(parent_results_path),
            "parent_label": parent_label,
        },
        "artifacts": {
            "diagnosis": _path_meta(diagnosis_path),
            "codex_deep_research": _path_meta(research_report_path),
            "full_eval_report": _path_meta(full_eval_report_path),
            "review": _path_meta(review_path),
            "candidate_rejection_context": _path_meta(rejection_context_path),
        },
        "outcome": {
            "status": outcome_status,
            "evidence_tier": (
                "full_eval_review" if review and full_report
                else "full_eval" if full_report
                else "probe" if rejection
                else "engineering"
            ),
            "post_full_eval_decision": _compact(decision),
            "metrics": _compact({
                "candidate_correct": full_report.get("candidate_correct"),
                "reference_correct": full_report.get("reference_correct"),
                "accuracy_delta": full_report.get("accuracy_delta"),
                "corrections": full_report.get("corrections") or [],
                "regressions": full_report.get("regressions") or [],
            }),
            "review_summary": _review_summary(review),
            "probe_rejection_summary": _probe_mechanism_summary(rejection),
        },
    }
    records = memory.setdefault("records", [])
    existing_idx = next((idx for idx, item in enumerate(records) if item.get("record_id") == record_id), None)
    if existing_idx is None:
        records.append(record)
    else:
        # Preserve a previously discovered path if the incremental caller did
        # not have that artifact available yet.
        previous = records[existing_idx]
        for group in ("lineage", "artifacts"):
            for name, meta in record[group].items():
                # lineage.parent_label is intentionally a scalar, while the
                # remaining lineage/artifact entries use path metadata.
                if not isinstance(meta, dict):
                    continue
                previous_meta = (previous.get(group, {}) or {}).get(name) or {}
                if (
                    not meta.get("path")
                    and isinstance(previous_meta, dict)
                    and previous_meta.get("path")
                ):
                    record[group][name] = previous_meta
        records[existing_idx] = record
    records.sort(key=lambda item: (int(item.get("iteration") or 0), int(item.get("updated_at") or 0)))
    memory["module_summary"] = _rebuild_module_summary(records)
    memory["updated_at"] = int(time.time())
    path = evolution_memory_path(run_id, output_root)
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    memory["path"] = path
    return memory


def _entry_score(entry: Dict) -> tuple:
    metrics = entry.get("metrics") or {}
    return (
        int(metrics.get("correct") or metrics.get("candidate_correct") or 0),
        int(metrics.get("total_questions") or 0),
        int(metrics.get("accuracy_delta") or 0),
        -int(metrics.get("regressions") or 0),
    )


def _same_record(existing: Dict, label: str, report_path: str, results_path: str) -> bool:
    return (
        existing.get("label") == label
        and os.path.abspath(existing.get("report_path") or "") == os.path.abspath(report_path or "")
        and os.path.abspath(existing.get("results_path") or "") == os.path.abspath(results_path or "")
    )


def record_reference(run_id: str, label: str, report_path: str, results_path: str,
                     combo: Dict | None = None, metrics: Dict | None = None,
                     output_root: str = "", manifest_path: str = "",
                     split: str = "", dataset_version: str = "",
                     update_current_best: bool = True,
                     finalized: bool = False,
                     decision: Dict | None = None,
                     decision_path: str = "",
                     bundle_path: str = "",
                     candidate_bundle_path: str = "",
                     module_artifacts: Dict | None = None,
                     structure_artifact: Dict | None = None,
                     selection_split: str = "",
                     train_diagnostic_report_path: str = "",
                     train_diagnostic_results_path: str = "") -> Dict:
    """Record an auditable evolution-split reference and its artifacts."""
    if update_current_best and selection_split and selection_split != "train":
        raise ValueError("current best may only be selected from the evolution split")
    ledger = load_ledger(run_id, output_root)
    entry = {
        "timestamp": int(time.time()),
        "label": label,
        "report_path": os.path.abspath(report_path) if report_path else "",
        "results_path": os.path.abspath(results_path) if results_path else "",
        "combo": combo or {},
        "metrics": metrics or {},
        "decision": decision or {},
        "decision_path": os.path.abspath(decision_path) if decision_path else "",
        "finalized": bool(finalized),
        "artifacts": {
            "bundle": {
                "path": os.path.abspath(bundle_path) if bundle_path else "",
                "sha256": _file_digest(bundle_path),
            },
            "candidate_bundle": {
                "path": os.path.abspath(candidate_bundle_path) if candidate_bundle_path else "",
                "sha256": _file_digest(candidate_bundle_path),
            },
            "modules": module_artifacts or {},
            "structure": structure_artifact or {},
        },
        "run_scope": {
            "manifest_path": os.path.abspath(manifest_path) if manifest_path else "",
            "manifest_sha256": _file_digest(manifest_path),
            "split": split,
            "dataset_version": dataset_version,
            "output_root": os.path.abspath(output_root) if output_root else "",
        },
        "selection": {
            "selection_split": selection_split or split,
            "selection_role": "current_best_selection" if update_current_best else "diagnostic_or_candidate_measurement",
            "train_diagnostic_report_path": os.path.abspath(train_diagnostic_report_path) if train_diagnostic_report_path else "",
            "train_diagnostic_results_path": os.path.abspath(train_diagnostic_results_path) if train_diagnostic_results_path else "",
        },
    }
    ledger.setdefault("run_scope", entry["run_scope"])
    entries = ledger.setdefault("entries", [])
    existing_idx = next(
        (idx for idx, old in enumerate(entries) if _same_record(old, label, report_path, results_path)),
        None,
    )
    if existing_idx is None:
        entries.append(entry)
    else:
        entries[existing_idx].update(entry)
        entry = entries[existing_idx]
    if update_current_best:
        current = ledger.get("current_best")
        # The caller has already applied the post-full-eval acceptance policy.
        # Do not re-rank against historical deltas here: an accuracy_delta is
        # relative to each candidate's own parent and is not comparable across
        # generations. In particular, an accepted equal-accuracy/lower-cost
        # candidate must replace the prior best even when its delta is zero.
        if current and _entry_score(entry) < _entry_score(current):
            ledger.setdefault("current_best_policy_overrides", []).append({
                "timestamp": int(time.time()),
                "label": label,
                "candidate_score": _entry_score(entry),
                "current_best_score": _entry_score(current),
                "reason": "explicit_post_eval_acceptance_overrides_noncomparable_historical_score",
            })
        ledger["current_best"] = entry
    if finalized:
        ledger["finalized_combo"] = entry
        ledger["finalized_at"] = int(time.time())
    path = ledger_path(run_id, output_root)
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
    ledger["path"] = path
    return ledger
