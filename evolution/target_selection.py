"""Deterministic evidence ledger for MetaVideoAgent diagnosis.

Diagnosis, not a hand-written score threshold, chooses the next module.  This
module only preserves the factual distinction between current-best failures,
candidate corrections, candidate regressions, and unchanged failures.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable

MODULES = (
    "video_structuring",
    "localization",
    "perception",
    "memory",
    "thinking",
)

_WINDOW_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)s?\s*-\s*(-?\d+(?:\.\d+)?)s?\s*\]"
)
_STRUCTURE_FAILURE_TOKENS = (
    "no structure", "missing structure", "missing jsonl", "empty structure",
    "chroma", "index error", "database error", "no records",
)
_UNCERTAIN_TOKENS = (
    "uncertain", "cannot determine", "unable to determine", "not enough evidence",
)


def canonical_task_ref(value: Any, *, video_id: str = "") -> str:
    """Normalize task ids/time references so hash/no-hash forms deduplicate."""
    if isinstance(value, (list, tuple)):
        interval = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return f"{video_id}::{interval}" if video_id else interval
    text = str(value or "").strip()
    if not text:
        return ""
    parts = text.split("::")
    if len(parts) >= 2:
        video = parts[0] or video_id
        raw_interval = parts[1]
        try:
            interval = json.dumps(json.loads(raw_interval), ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError, json.JSONDecodeError):
            interval = re.sub(r"\s+", "", raw_interval)
        return f"{video}::{interval}" if video else interval
    return text


def _video_id(value: Any, fallback: str = "") -> str:
    text = str(value or "")
    return text.split("::", 1)[0] if "::" in text else fallback


def _intervals(value: Any) -> list[tuple[float, float]]:
    text = str(value or "")
    parts = text.split("::")
    raw = parts[1] if len(parts) >= 2 else text
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    intervals: list[tuple[float, float]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    intervals.append((float(item[0]), float(item[1])))
                except (TypeError, ValueError):
                    continue
    return intervals


def _search_windows(trajectory: Iterable[Any]) -> list[tuple[float, float]]:
    def as_number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def structured_windows(value: Any) -> list[tuple[float, float]]:
        if isinstance(value, str):
            try:
                return structured_windows(json.loads(value))
            except (TypeError, ValueError, json.JSONDecodeError):
                return []
        if isinstance(value, dict):
            lowered = {str(key).lower(): item for key, item in value.items()}
            start = next((lowered.get(key) for key in ("start_sec", "start", "start_time")), None)
            end = next((lowered.get(key) for key in ("end_sec", "end", "end_time")), None)
            left, right = as_number(start), as_number(end)
            found = [(left, right)] if left is not None and right is not None else []
            for key in (
                "time_ranges", "windows", "candidate_windows", "intervals",
                "evidence_windows", "retrieved_windows", "results",
            ):
                found.extend(structured_windows(lowered.get(key)))
            return found
        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                left, right = as_number(value[0]), as_number(value[1])
                if left is not None and right is not None:
                    return [(left, right)]
            found = []
            for item in value:
                found.extend(structured_windows(item))
            return found
        return []

    windows: list[tuple[float, float]] = []
    for step in trajectory or []:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").lower()
        if "search" not in action and "window" not in action:
            continue
        for field in ("observation", "action_input", "plan_output", "result", "output"):
            value = step.get(field)
            windows.extend(structured_windows(value))
            if isinstance(value, str):
                windows.extend((float(start), float(end)) for start, end in _WINDOW_RE.findall(value))
    return sorted(set(windows))


def _has_structure_failure(trajectory: Iterable[Any]) -> bool:
    text = "\n".join(
        str(step.get("observation") or "")
        for step in trajectory or [] if isinstance(step, dict)
    ).lower()
    return any(token in text for token in _STRUCTURE_FAILURE_TOKENS)


def _verification_state(trajectory: Iterable[Any]) -> tuple[bool, bool]:
    observations = []
    for step in trajectory or []:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").lower()
        if "verify" in action or "perception" in action:
            observations.append(str(step.get("observation") or "").lower())
    text = "\n".join(observations)
    return bool(observations), any(token in text for token in _UNCERTAIN_TOKENS)


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _direct_attribution(entry: Dict[str, Any], task_ref: str) -> tuple[str, str]:
    """Classify only when the trajectory itself provides a direct signal."""
    trajectory = entry.get("trajectory") or []
    if _has_structure_failure(trajectory):
        return "video_structuring", "structure_artifact_failure"
    expected = _intervals(task_ref)
    windows = _search_windows(trajectory)
    if expected and windows:
        covered = any(_overlaps(window, target) for window in windows for target in expected)
        if not covered:
            return "localization", "retrieval_miss"
        verified, uncertain = _verification_state(trajectory)
        if verified and uncertain:
            return "perception", "verification_uncertain"
        if verified:
            return "thinking", "evidence_present_final_wrong"
        return "thinking", "retrieval_not_consumed"
    return "", "insufficient_trace_evidence"


def _review_delta_map(review: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in review.get("delta_reviews") or []:
        if not isinstance(item, dict):
            continue
        ref = canonical_task_ref(
            item.get("task_id") or item.get("time_reference"),
            video_id=str(item.get("video_id") or ""),
        )
        change = str(item.get("change_type") or "").lower()
        if ref and change in {"fix", "regression"}:
            result[ref] = change
    return result


def _recent_module_outcomes(iter_context: Dict[str, Any]) -> dict[str, str]:
    memory = (iter_context or {}).get("evolution_memory", {}) or {}
    records = memory.get("records") or []
    latest: Dict[str, tuple[int, str]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        outcome = str(((record.get("outcome") or {}).get("status") or ""))
        timestamp = int(record.get("updated_at") or 0)
        for target in (record.get("candidate") or {}).get("target_modules") or []:
            target = str(target)
            if target in MODULES and (target not in latest or timestamp >= latest[target][0]):
                latest[target] = (timestamp, outcome)
    return {target: outcome for target, (_timestamp, outcome) in latest.items()}


def _micro_review_map(review: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mapped = {}
    for item in review.get("evolved_micro_reviews") or []:
        if not isinstance(item, dict):
            continue
        ref = canonical_task_ref(
            item.get("task_id") or item.get("time_reference"),
            video_id=str(item.get("video_id") or ""),
        )
        if ref:
            mapped[ref] = item
    return mapped


def _validated_mfps(micro: Dict[str, Any]) -> list[Dict[str, Any]]:
    validation = micro.get("minimal_failure_point_validation") or {}
    if validation.get("direct_repair_eligible") is not True:
        return []
    points = micro.get("minimal_failure_points") or []
    point_validations = validation.get("point_validations") or []
    valid_indexes = {
        int(item.get("index")) for item in point_validations
        if isinstance(item, dict) and item.get("valid") is True
        and isinstance(item.get("index"), int)
    }
    return [
        point for index, point in enumerate(points)
        if index in valid_indexes and isinstance(point, dict)
        and point.get("root_module") in MODULES
    ]


def _validated_repair_actions(micro: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Expose only contract-valid micro repair mechanisms to diagnosis.

    These actions are not a target vote.  They preserve the Teacher's
    trace-grounded explanation of *what behavior should change*, so diagnosis
    can synthesize a joint direction instead of rediscovering everything from
    action-name heuristics.
    """
    actions = micro.get("concrete_repair_actions") or []
    if not isinstance(actions, list):
        return []
    validation = micro.get("concrete_repair_action_validation") or {}
    per_action = validation.get("action_validations") or []
    if isinstance(per_action, list):
        valid_indexes = {
            int(row.get("index")) for row in per_action
            if isinstance(row, dict) and row.get("valid") is True
            and isinstance(row.get("index"), int)
        }
        return [
            action for index, action in enumerate(actions)
            if index in valid_indexes and isinstance(action, dict)
            and str(action.get("module") or "") in MODULES
        ]
    return actions if validation.get("valid") is True else []


def _failure_capsule(point: Dict[str, Any], *, module: str) -> Dict[str, Any]:
    """Return the bounded, task/path-free portion allowed into diagnosis.

    Full points stay with the machine ledger.  This capsule retains the input
    condition and evidence handoff needed for a concrete design without task
    identifiers, answer text, artifact paths, or raw trajectory observations.
    """
    return {
        "root_module": module,
        "failure_kind": str(point.get("failure_kind") or ""),
        "trigger_window_seconds": round(
            float((point.get("active_window") or [0, 0])[1])
            - float((point.get("active_window") or [0, 0])[0]), 3,
        ) if isinstance(point.get("active_window"), (list, tuple)) and len(point.get("active_window")) == 2 else 0.0,
        "evidence_event_count": len(point.get("evidence_event_ids") or []),
        "capability_needs": [str(value) for value in (point.get("capability_needs") or []) if str(value)][:4],
        "observed_failure": str(point.get("observed_output") or "")[:360],
        "expected_evidence": str(point.get("expected_evidence") or "")[:360],
        "consumer_failure": str(point.get("downstream_effect") or "")[:360],
        "counterfactual_success": str(point.get("counterfactual_success") or "")[:360],
    }


def _consumer_signature(point: Dict[str, Any], *, events: Iterable[Any] = ()) -> str:
    """Return a stable, content-free producer/consumer handoff signature."""
    point_event_ids = {str(item) for item in (point.get("evidence_event_ids") or []) if str(item)}
    for event in events or ():
        if not isinstance(event, dict):
            continue
        if event.get("event") != "consumer_consumed" or event.get("status") != "ok":
            continue
        source_ids = {str(item) for item in (event.get("source_event_ids") or []) if str(item)}
        if point_event_ids and not point_event_ids.intersection(source_ids):
            continue
        consumer = str(event.get("consumer_module") or "").strip()
        if consumer in MODULES:
            return consumer
    text = str(point.get("downstream_effect") or "").lower()
    for module in MODULES:
        if module in text:
            return module
    return "preserved_consumer"


def _mfp_cluster_key(point: Dict[str, Any], *, module: str, events: Iterable[Any] = ()) -> str:
    """Cluster direct MFPs by reusable mechanism, never by task content."""
    capabilities = ",".join(sorted({
        str(item).strip() for item in (point.get("capability_needs") or [])
        if str(item).strip()
    })) or "no_declared_capability"
    return " | ".join((
        module,
        str(point.get("failure_kind") or "unknown_failure"),
        _consumer_signature(point, events=events),
        capabilities,
    ))


def _weak_failure_signature(entry: Dict[str, Any], task_ref: str,
                            effect: Dict[str, Any]) -> Dict[str, str]:
    """Describe every unresolved wrong rollout without inventing causal repair.

    This is intentionally a deterministic trace summary.  It is useful for
    diagnosis to see the broad failure distribution, but is never promoted to
    direct repair evidence unless a validated MFP or direct trace attribution
    independently supports that claim.
    """
    trajectory = entry.get("trajectory") or []
    module, cause_kind = _direct_attribution(entry, task_ref)
    actions = " ".join(
        str(step.get("action") or step.get("producer_module") or "").lower()
        for step in trajectory if isinstance(step, dict)
    )
    observations = "\n".join(
        str(step.get("observation") or "")
        for step in trajectory if isinstance(step, dict)
    ).strip()
    events = entry.get("capability_events") or entry.get("runtime_evidence") or []
    if not isinstance(events, list):
        events = []
    capabilities = sorted({
        str(item.get("capability") or "").strip()
        for item in events if isinstance(item, dict) and str(item.get("capability") or "").strip()
    })
    consumer_state = (
        "consumer_consumed" if any(
            isinstance(item, dict) and item.get("event") == "consumer_consumed"
            and item.get("status") == "ok"
            for item in events
        ) else "consumer_not_recorded"
    )
    if "verify" in actions or "perception" in actions:
        stage = "verification_or_perception"
    elif "search" in actions or "window" in actions or "retriev" in actions:
        stage = "localization_or_retrieval"
    elif "think" in actions or "answer" in actions:
        stage = "reasoning_or_answer"
    elif not trajectory:
        stage = "missing_runtime_trace"
    else:
        stage = "other_runtime_stage"
    quality = (
        "empty_output" if not observations else
        "uncertain_or_low_confidence" if any(token in observations.lower() for token in _UNCERTAIN_TOKENS) else
        "observed_output"
    )
    effect_text = " ".join(str(effect.get(key) or "") for key in (
        "observed_behavior", "verification", "candidate_method_delta", "method_or_tool",
    )).lower()
    if any(token in effect_text for token in ("empty", "missing", "absent", "no evidence", "no evidence", "is empty", "Missing")):
        semantic_category = "evidence_absent_or_unusable"
    elif any(token in effect_text for token in ("not consumed", "handoff", "Not consumed", "not delivered")):
        semantic_category = "evidence_not_consumed"
    elif any(token in effect_text for token in ("reason", "option", "answer", "reasoning", "Options", "answer")):
        semantic_category = "reasoning_mapping_mismatch"
    elif any(token in effect_text for token in ("conflict", "inconsistent", "conflict", "Contradiction")):
        semantic_category = "evidence_conflict"
    else:
        semantic_category = "trace_state_unresolved"
    return {
        "suggested_module": module or "unattributed",
        "trace_cause_kind": cause_kind,
        "failure_stage": stage,
        "evidence_quality": quality,
        "capability_signature": ",".join(capabilities) or "no_capability_event",
        "consumer_state": consumer_state,
        "semantic_category": semantic_category,
        "effect_verdict": str(effect.get("effect_verdict") or "not_recorded"),
    }


def _cluster_summary(clusters: Dict[str, Dict[str, Any]], *, module: str) -> Dict[str, Any]:
    rows = [item for item in clusters.values() if item.get("root_module") == module]
    return {
        "cluster_count": len(rows),
        "task_count": len({ref for item in rows for ref in item.get("refs", [])}),
        "video_count": len({video for item in rows for video in item.get("video_ids", [])}),
        "failure_kinds": sorted({str(item.get("failure_kind") or "") for item in rows if item.get("failure_kind")}),
        "cluster_ids": sorted(str(item.get("cluster_id") or "") for item in rows),
    }


def _review_candidate_targets(review: Dict[str, Any]) -> list[str]:
    """Return the reviewed candidate bundle's changed modules.

    This is provenance for corrections/regressions, not a recommendation for the
    next diagnosis target.  Keeping that distinction explicit prevents an
    accepted candidate from becoming an implicit target-selection prior.
    """
    review = review if isinstance(review, dict) else {}
    lineage = review.get("candidate_lineage") or {}
    context = (review.get("candidate_context") or {}).get("evolution_info") or {}
    values = lineage.get("target_modules") or context.get("changed_modules") or []
    return [str(value) for value in values if str(value) in MODULES]


def _unique_refs(items: Iterable[Dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for item in items:
        ref = str(item.get("task_ref") or item.get("task_id") or item.get("time_reference") or "").strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _target_comparison(
    *,
    module_refs: Dict[str, list[Dict[str, Any]]],
    mfp_clusters: Dict[str, Dict[str, Any]],
    weak_failure_clusters: Dict[str, Dict[str, Any]],
    regression_guards: list[Dict[str, Any]],
    correction_evidence: list[Dict[str, Any]],
    unchanged_failure_evidence: list[Dict[str, Any]],
    reviewed_candidate_targets: list[str],
    prior_outcomes: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Build factual, non-ranking comparison facts for every module.

    Counts are derived from the entire ledger.  They deliberately do not
    compute a score or a recommended target: the diagnosis LLM must choose and
    justify the causal mechanism, while this table prevents it from treating a
    speculative fault distribution as direct evidence.
    """
    comparison: Dict[str, Dict[str, Any]] = {}
    for module in MODULES:
        direct_repair = list(module_refs.get(module, []))
        blocked_upstream = [
            item for item in unchanged_failure_evidence
            if str(item.get("module") or "") == module
            and str((item.get("causal_effect_point") or {}).get("effect_verdict") or "")
            in {"blocked_upstream", "blocked_downstream", "not_triggered", "no_effect"}
        ]
        # Corrections/regressions belong to the reviewed bundle's linked changed
        # set. This is provenance, not attribution of a question to one module.
        candidate_regressions = regression_guards if module in reviewed_candidate_targets else []
        candidate_corrections = correction_evidence if module in reviewed_candidate_targets else []
        comparison[module] = {
            "direct_repair": {
                "count": len(_unique_refs(direct_repair)),
                "refs": _unique_refs(direct_repair),
                "cause_kinds": sorted({str(item.get("cause_kind") or "") for item in direct_repair if item.get("cause_kind")}),
            },
            "direct_regression_stabilization": {
                "count": len(_unique_refs(candidate_regressions)),
                "refs": _unique_refs(candidate_regressions),
                "candidate_target_provenance": reviewed_candidate_targets,
            },
            "candidate_correction_preservation": {
                "count": len(_unique_refs(candidate_corrections)),
                "refs": _unique_refs(candidate_corrections),
                "candidate_target_provenance": reviewed_candidate_targets,
            },
            "blocked_upstream": {
                "count": len(_unique_refs(blocked_upstream)),
                "refs": _unique_refs(blocked_upstream),
            },
            # These are mechanism-distribution facts, not a ranking score.
            # refs remain machine-only; diagnosis receives only a sanitized
            # aggregate summary built from this table.
            "validated_mfp_coverage": _cluster_summary(mfp_clusters, module=module),
            "weak_failure_distribution": {
                "cluster_count": len([
                    item for item in weak_failure_clusters.values()
                    if item.get("suggested_module") == module
                ]),
                "task_count": len({
                    ref for item in weak_failure_clusters.values()
                    if item.get("suggested_module") == module
                    for ref in item.get("refs", [])
                }),
                "failure_stages": sorted({
                    str(item.get("failure_stage") or "")
                    for item in weak_failure_clusters.values()
                    if item.get("suggested_module") == module and item.get("failure_stage")
                }),
                "evidence_qualities": sorted({
                    str(item.get("evidence_quality") or "")
                    for item in weak_failure_clusters.values()
                    if item.get("suggested_module") == module and item.get("evidence_quality")
                }),
                "capability_signatures": sorted({
                    str(item.get("capability_signature") or "")
                    for item in weak_failure_clusters.values()
                    if item.get("suggested_module") == module and item.get("capability_signature")
                }),
                "consumer_states": sorted({
                    str(item.get("consumer_state") or "")
                    for item in weak_failure_clusters.values()
                    if item.get("suggested_module") == module and item.get("consumer_state")
                }),
                "semantic_categories": sorted({
                    str(item.get("semantic_category") or "")
                    for item in weak_failure_clusters.values()
                    if item.get("suggested_module") == module and item.get("semantic_category")
                }),
            },
            "history_outcome": str(prior_outcomes.get(module) or ""),
        }
    return comparison


def build_target_selection(trajectories: Dict[str, list[Dict[str, Any]]],
                           review: Dict[str, Any] | None = None,
                           iter_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build factual evidence categories without selecting an evolution target."""
    review = review or {}
    iter_context = iter_context or {}
    delta = _review_delta_map(review)
    micro_by_ref = _micro_review_map(review)
    module_refs: Dict[str, list[Dict[str, str]]] = defaultdict(list)
    # Keep evidence tiers separate.  A trajectory-derived attribution is a
    # useful distribution signal, but it is not a validated minimal failure
    # point and must never be presented as direct repair evidence.
    direct_mfp_evidence: list[Dict[str, str]] = []
    trace_supported_evidence: list[Dict[str, str]] = []
    regression_guards: list[Dict[str, Any]] = []
    correction_evidence: list[Dict[str, Any]] = []
    unchanged_failure_evidence: list[Dict[str, Any]] = []
    unresolved_repairs: list[str] = []
    failure_capsules: list[Dict[str, Any]] = []
    mfp_clusters: Dict[str, Dict[str, Any]] = {}
    review_action_clusters: Dict[str, Dict[str, Any]] = {}
    weak_failure_clusters: Dict[str, Dict[str, Any]] = {}
    reviewed_candidate_targets = _review_candidate_targets(review)
    trajectory_total = 0
    trajectory_correct = 0
    trajectory_wrong = 0

    for task_id, entries in (trajectories or {}).items():
        if not entries or not isinstance(entries[0], dict):
            continue
        entry = entries[0]
        trajectory_total += 1
        if bool(entry.get("is_correct")):
            trajectory_correct += 1
        else:
            trajectory_wrong += 1
        task_ref = canonical_task_ref(
            task_id or entry.get("task_id") or entry.get("time_reference"),
            video_id=str(entry.get("video_id") or ""),
        )
        if not task_ref:
            continue
        micro = micro_by_ref.get(task_ref, {})
        change = delta.get(task_ref, "") or str(micro.get("change_type") or "")
        effect = micro.get("causal_effect_point") or {}
        common = {
            "task_ref": task_ref,
            "video_id": _video_id(task_ref, str(entry.get("video_id") or "")),
            "change_type": change,
            "causal_effect_point": effect,
        }
        if change == "regression":
            regression_guards.append(common)
            continue
        if change == "fix":
            correction_evidence.append(common)
            continue
        if bool(entry.get("is_correct")):
            continue
        for action in _validated_repair_actions(micro):
            module = str(action.get("module") or "")
            mechanism = str(action.get("repair_mechanism") or "")
            handoff = str(action.get("input_output_or_handoff") or "")
            key = " | ".join((module, mechanism, handoff))
            cluster = review_action_clusters.setdefault(key, {
                "modules": [module], "repair_mechanism": mechanism,
                "handoff_or_state_to_repair": handoff,
                "falsifiable_runtime_signals": [], "generalization_scopes": [],
                "observed_failure_examples": [], "refs": [], "video_ids": [], "count": 0,
            })
            for action_field, cluster_field in (
                ("current_observed_failure", "observed_failure_examples"),
                ("falsifiable_runtime_signal", "falsifiable_runtime_signals"),
                ("generalization_scope", "generalization_scopes"),
            ):
                value = str(action.get(action_field) or "")
                if value and value not in cluster[cluster_field]:
                    cluster[cluster_field].append(value)
            if task_ref not in cluster["refs"]:
                cluster["refs"].append(task_ref)
            if common["video_id"] and common["video_id"] not in cluster["video_ids"]:
                cluster["video_ids"].append(common["video_id"])
            cluster["count"] += 1
        mfps = _validated_mfps(micro)
        weak_signature = _weak_failure_signature(entry, task_ref, effect)
        weak_key = " | ".join(str(weak_signature.get(key) or "") for key in (
            "suggested_module", "trace_cause_kind", "failure_stage", "evidence_quality",
            "capability_signature", "consumer_state", "semantic_category", "effect_verdict",
        ))
        weak_cluster = weak_failure_clusters.setdefault(weak_key, {
            "cluster_id": weak_key,
            **weak_signature,
            "refs": [],
            "video_ids": [],
        })
        if task_ref not in weak_cluster["refs"]:
            weak_cluster["refs"].append(task_ref)
        if common["video_id"] and common["video_id"] not in weak_cluster["video_ids"]:
            weak_cluster["video_ids"].append(common["video_id"])
        trace_module = str(weak_signature.get("suggested_module") or "")
        if trace_module in MODULES:
            trace_supported_evidence.append({
                **common,
                "module": trace_module,
                "cause_kind": str(weak_signature.get("trace_cause_kind") or "trace_supported"),
                "evidence_tier": "trace_supported",
            })
        if mfps:
            # Keep every independently validated point.  A point remains
            # direct evidence only for its own root module; the cluster index
            # lets later stages reason about broad mechanisms without seeing
            # question content.
            for point in mfps:
                point_module = str(point.get("root_module") or "")
                if point_module not in MODULES:
                    continue
                point_evidence = {
                    **common,
                    "module": point_module,
                    "cause_kind": "validated_minimal_failure_point",
                    "mfp_cluster_id": _mfp_cluster_key(
                        point, module=point_module,
                        events=entry.get("capability_events") or entry.get("runtime_evidence") or [],
                    ),
                }
                direct_mfp_evidence.append(point_evidence)
                module_refs[point_module].append(point_evidence)
                cluster_id = point_evidence["mfp_cluster_id"]
                cluster = mfp_clusters.setdefault(cluster_id, {
                    "cluster_id": cluster_id,
                    "root_module": point_module,
                    "failure_kind": str(point.get("failure_kind") or "unknown_failure"),
                    "consumer_signature": _consumer_signature(
                        point, events=entry.get("capability_events") or entry.get("runtime_evidence") or [],
                    ),
                    "capability_needs": sorted({
                        str(item).strip() for item in (point.get("capability_needs") or [])
                        if str(item).strip()
                    }),
                    "refs": [],
                    "video_ids": [],
                })
                if task_ref not in cluster["refs"]:
                    cluster["refs"].append(task_ref)
                if common["video_id"] and common["video_id"] not in cluster["video_ids"]:
                    cluster["video_ids"].append(common["video_id"])
                failure_capsules.append(_failure_capsule(point, module=point_module))
            if (micro.get("change_type") or "") == "keep_wrong":
                unchanged_failure_evidence.append({
                    **common,
                    "module": str(mfps[0].get("root_module") or ""),
                    "cause_kind": "validated_minimal_failure_point",
                })
            # The direct MFPs have already been added above; avoid adding a
            # duplicate heuristic/first-point repair record below.
            continue
        elif micro:
            # A completed review without a validated MFP remains an audit fact,
            # never direct repair support.  Diagnosis may still make bounded
            # exploration from the unresolved evidence.
            module, cause_kind = "", "mfp_missing_or_invalid"
        else:
            module, cause_kind = _direct_attribution(entry, task_ref)
        if not module:
            unresolved_repairs.append(task_ref)
            evidence = {
                **common,
                "module": "",
                "cause_kind": "causal_effect_only" if effect else "insufficient_trace_evidence",
            }
            # The row remains visible as an unresolved failure.  Its weak
            # distribution record above carries any trace-level hypothesis.
            if change == "keep_wrong":
                unchanged_failure_evidence.append(evidence)
            continue
        evidence = {**common, "module": module, "cause_kind": cause_kind}
        if (micro.get("change_type") or "") == "keep_wrong":
            unchanged_failure_evidence.append(evidence)

    scores = {}
    for module in MODULES:
        refs = module_refs.get(module, [])
        unique_refs = {item["task_ref"] for item in refs}
        videos = {item["video_id"] for item in refs if item.get("video_id")}
        scores[module] = {
            "eligible_ref_count": len(unique_refs),
            "eligible_video_count": len(videos),
            "refs": sorted(unique_refs),
            "cause_kinds": sorted({item["cause_kind"] for item in refs}),
        }

    prior_outcomes = _recent_module_outcomes(iter_context)
    comparison = _target_comparison(
        module_refs=module_refs,
        mfp_clusters=mfp_clusters,
        weak_failure_clusters=weak_failure_clusters,
        regression_guards=regression_guards,
        correction_evidence=correction_evidence,
        unchanged_failure_evidence=unchanged_failure_evidence,
        reviewed_candidate_targets=reviewed_candidate_targets,
        prior_outcomes=prior_outcomes,
    )
    return {
        "schema_version": 1,
        "ledger_status": "ready",
        "diagnosis_selects_target": True,
        "evidence_policy": {
            "regressions_are_guard_only": True,
            "corrections_are_preserve_evidence": True,
            "unchanged_failures_require_causal_explanation": True,
            "prior_module_outcomes_are_guards_not_bans": True,
            "validated_mfp_is_probe_witness_only": True,
            "trace_supported_distribution_selects_target_mechanism": True,
        },
        "module_scores": scores,
        "trajectory_summary": {
            "total_questions": trajectory_total,
            "correct": trajectory_correct,
            "incorrect": trajectory_wrong,
        },
        "target_comparison": comparison,
        "reviewed_candidate_targets": reviewed_candidate_targets,
        "direct_mfp_evidence": direct_mfp_evidence,
        "trace_supported_evidence": trace_supported_evidence,
        "candidate_correction_evidence": correction_evidence,
        "regression_guards": regression_guards,
        "unchanged_failure_evidence": unchanged_failure_evidence,
        "unresolved_repairs": sorted(set(unresolved_repairs)),
        # Full task-level capsule text remains audit-only.  The diagnosis
        # prompt consumes the mechanism summaries below, never these samples.
        "diagnosis_failure_capsules": failure_capsules,
        "mfp_mechanism_cluster_index": {
            "schema_version": 1,
            "clusters": [mfp_clusters[key] for key in sorted(mfp_clusters)],
        },
        "validated_review_repair_mechanisms": {
            "schema_version": 1,
            "clusters": [review_action_clusters[key] for key in sorted(review_action_clusters)],
        },
        "weak_failure_distribution": {
            "schema_version": 1,
            "total_wrong_task_refs": len({
                ref for item in weak_failure_clusters.values() for ref in item.get("refs", [])
            }),
            "clusters": [weak_failure_clusters[key] for key in sorted(weak_failure_clusters)],
        },
        "prior_module_outcomes": prior_outcomes,
    }
