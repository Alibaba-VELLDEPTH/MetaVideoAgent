"""Machine checks for causal post-evolution micro reviews."""

from __future__ import annotations

from typing import Any, Dict

OUTCOME_ROLES = {
    "repair_evidence",
    "candidate_correction_evidence",
    "regression_guard",
    "unchanged_failure_evidence",
    "stable_no_failure",
}
EFFECT_VERDICTS = {
    "helpful",
    "harmful",
    "no_effect",
    "blocked_upstream",
    "blocked_downstream",
    "not_triggered",
    "stable_no_failure",
}
MFP_REQUIRED_CHANGE_TYPES = {"keep_wrong", "regression"}
MFP_MODULES = {
    "video_structuring", "localization", "perception", "thinking", "memory",
}
REPAIR_ACTION_REQUIRED_CHANGE_TYPES = {"keep_wrong", "regression"}
FAILURE_CHAIN_ROLES = {
    "root_cause", "contributing", "downstream_unable_to_recover", "not_observed",
}
CHANGE_TYPE_TO_OUTCOME_ROLE = {
    "fix": "candidate_correction_evidence",
    "regression": "regression_guard",
    "keep_wrong": "unchanged_failure_evidence",
    "keep_correct": "stable_no_failure",
}


def failure_point_issues(micro: Dict[str, Any]) -> list[str]:
    """Validate the causal effect record emitted for one reviewed rollout."""
    if not isinstance(micro, dict):
        return ["micro review is not an object"]
    if micro.get("success") is not True or micro.get("dry_run"):
        return ["micro review was not completed by the Teacher"]
    expected_role = CHANGE_TYPE_TO_OUTCOME_ROLE.get(str(micro.get("change_type") or ""))
    if not expected_role:
        return ["micro review has no supported deterministic change_type"]
    initial_single = micro.get("review_mode") == "initial_reference_single_baseline"
    if not initial_single and (not micro.get("reference_available") or not str(micro.get("reference_task_id") or "").strip()):
        return ["micro review is missing the current-best reference trajectory"]
    point = micro.get("causal_effect_point") or {}
    if not isinstance(point, dict):
        return ["missing causal_effect_point"]
    if point.get("outcome_role") not in OUTCOME_ROLES:
        return ["causal_effect_point has an invalid outcome_role"]
    if point.get("outcome_role") != expected_role:
        return ["causal_effect_point outcome_role disagrees with deterministic change_type"]
    if point.get("effect_verdict") not in EFFECT_VERDICTS:
        return ["causal_effect_point has an invalid effect_verdict"]
    for field in ("candidate_method_delta", "step", "method_or_tool", "observed_behavior", "verification"):
        if not str(point.get(field) or "").strip():
            return [f"causal_effect_point is missing {field}"]
    # This causal record is the macro-consumption floor. MFP validation remains
    # separate; only a validated MFP can become direct repair evidence.
    return []


def _valid_window(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    try:
        return float(value[1]) > float(value[0]) >= 0.0
    except (TypeError, ValueError):
        return False


def _window_from_event(event: Dict[str, Any]) -> tuple[float, float] | None:
    """Return the bounded media interval carried by one runtime event."""
    try:
        start = float(event.get("start_sec"))
        end = float(event.get("end_sec"))
    except (TypeError, ValueError):
        return None
    return (start, end) if end > start >= 0.0 else None


def _windows_overlap(left: Any, right: Any) -> bool:
    if not (_valid_window(left) and _valid_window(right)):
        return False
    return max(float(left[0]), float(right[0])) < min(float(left[1]), float(right[1]))


def minimal_failure_point_issues(
        micro: Dict[str, Any], trajectory: list[Dict[str, Any]] | None = None,
        capability_events: list[Dict[str, Any]] | None = None) -> list[str]:
    """Validate a concrete, trace-anchorable Minimal Failure Point.

    Previous reviews remain consumable at macro level, but an unanchored MFP is
    never eligible as direct repair evidence.  Contextual validation is used
    by the review runner once it has the actual sandbox trajectory.
    """
    if not isinstance(micro, dict):
        return ["micro review is not an object"]
    change_type = str(micro.get("change_type") or "")
    points = micro.get("minimal_failure_points") or []
    if change_type not in MFP_REQUIRED_CHANGE_TYPES:
        return [] if not points else _mfp_shape_issues(points)
    if not isinstance(points, list) or not points:
        return ["missing minimal_failure_points for an unresolved/regressed result"]
    issues = _mfp_shape_issues(points)
    if issues or trajectory is None:
        return issues
    events_by_id = {
        str(event.get("evidence_event_id") or ""): event
        for event in (capability_events or []) if isinstance(event, dict)
    }
    tool_steps = [
        step for step in trajectory
        if isinstance(step, dict) and step.get("step_type") == "tool_execution"
    ]
    for index, point in enumerate(points):
        root_module = str(point.get("root_module") or "")
        anchor_kind = str(point.get("trace_anchor_kind") or "tool_execution")
        step_index = int(point.get("trajectory_step_index") or 0)
        lifecycle_anchor = anchor_kind in {
            "structure_artifact", "reasoning_decision", "memory_state",
        }
        if lifecycle_anchor:
            expected_module = {
                "structure_artifact": "video_structuring",
                "reasoning_decision": "thinking",
                "memory_state": "memory",
            }.get(anchor_kind)
            if expected_module and root_module != expected_module:
                issues.append(f"minimal_failure_points[{index}] trace_anchor_kind disagrees with root_module")
            if step_index not in (0,):
                issues.append(f"minimal_failure_points[{index}] lifecycle anchor must use trajectory_step_index=0")
            if anchor_kind == "reasoning_decision" and not any(
                isinstance(step, dict) and step.get("step_type") in {"reasoning", "decision_provenance"}
                for step in trajectory
            ):
                issues.append(f"minimal_failure_points[{index}] reasoning_decision has no recorded reasoning/decision")
            if anchor_kind == "memory_state" and not any(
                isinstance(step, dict) and (
                    step.get("step_type") in {"memory", "reasoning", "decision_provenance"}
                    or "memory" in str(step).lower()
                ) for step in trajectory
            ):
                issues.append(f"minimal_failure_points[{index}] memory_state has no recorded memory anchor")
            step = None
        else:
            if step_index < 1 or step_index > len(tool_steps):
                issues.append(f"minimal_failure_points[{index}] trajectory_step_index is not in trajectory")
                continue
            step = tool_steps[step_index - 1]
            producer = str(step.get("producer_module") or "")
            if producer and producer != root_module:
                issues.append(f"minimal_failure_points[{index}] root_module disagrees with trajectory producer")
        step_event_ids = {
            str(value) for value in ((step or {}).get("evidence_event_ids") or [])
        }
        point_event_ids = {str(value) for value in (point.get("evidence_event_ids") or [])}
        point_window = point.get("active_window")
        step_windows = (step or {}).get("active_time_windows") or []
        if step_windows and not any(_windows_overlap(point_window, window) for window in step_windows):
            issues.append(f"minimal_failure_points[{index}] active_window is outside the trajectory step window")
        if step is not None and point_event_ids and not point_event_ids.issubset(step_event_ids):
            issues.append(f"minimal_failure_points[{index}] references events outside its trajectory step")
        for event_id in point_event_ids:
            event = events_by_id.get(event_id)
            if event is None:
                issues.append(f"minimal_failure_points[{index}] references missing capability event")
                break
            event_window = _window_from_event(event)
            if event_window and not _windows_overlap(point_window, event_window):
                issues.append(f"minimal_failure_points[{index}] active_window does not cover the referenced event interval")
                break
            if event.get("capability") in {"ocr", "asr", "vlm"}:
                if str(event.get("status") or "") in {"api_error", "asset_error", "unavailable"}:
                    issues.append(f"minimal_failure_points[{index}] mistakes an execution-layer failure for a repair point")
                    break
        needs = {str(value) for value in (point.get("capability_needs") or []) if str(value)}
        event_capabilities = {
            str((events_by_id.get(event_id) or {}).get("capability") or "")
            for event_id in point_event_ids
        }
        if needs and event_capabilities and not needs.intersection(event_capabilities):
            issues.append(f"minimal_failure_points[{index}] capability_needs disagrees with referenced runtime evidence")
        # A media/retrieval MFP may claim a downstream effect only when the
        # framework recorded that a preserved consumer received this output.
        # Thinking/memory are themselves the handoff/root and can legitimately
        # have no media event to consume.
        if root_module not in {"thinking", "memory", "video_structuring"}:
            consumed = any(
                event.get("event") == "consumer_consumed"
                and event.get("status") == "ok"
                and event.get("contract_valid") is True
                and str(event.get("producer_module") or "") == root_module
                and point_event_ids.intersection({
                    str(value) for value in (event.get("source_event_ids") or [])
                })
                for event in (capability_events or []) if isinstance(event, dict)
            )
            if not consumed:
                issues.append(f"minimal_failure_points[{index}] has no later preserved-consumer event for its evidence")
    return issues


def _mfp_shape_issues(points: Any) -> list[str]:
    if not isinstance(points, list):
        return ["minimal_failure_points must be a list"]
    issues = []
    for index, point in enumerate(points):
        prefix = f"minimal_failure_points[{index}]"
        if not isinstance(point, dict):
            issues.append(f"{prefix} must be an object")
            continue
        for field in (
            "mfp_id", "root_module", "failure_kind", "observed_output",
            "expected_evidence", "downstream_effect", "counterfactual_success",
        ):
            if not str(point.get(field) or "").strip():
                issues.append(f"{prefix}.{field} is empty")
        if point.get("root_module") not in MFP_MODULES:
            issues.append(f"{prefix}.root_module is invalid")
        anchor_kind = str(point.get("trace_anchor_kind") or "tool_execution")
        if anchor_kind not in {"tool_execution", "structure_artifact", "reasoning_decision", "memory_state"}:
            issues.append(f"{prefix}.trace_anchor_kind is invalid")
        try:
            step_index = int(point.get("trajectory_step_index") or 0)
            if anchor_kind == "tool_execution" and step_index < 1:
                issues.append(f"{prefix}.trajectory_step_index is invalid")
            if anchor_kind != "tool_execution" and step_index != 0:
                issues.append(f"{prefix}.trajectory_step_index must be 0 for lifecycle anchor")
        except (TypeError, ValueError):
            issues.append(f"{prefix}.trajectory_step_index is invalid")
        if not _valid_window(point.get("active_window")):
            issues.append(f"{prefix}.active_window is invalid")
        if not isinstance(point.get("evidence_event_ids") or [], list):
            issues.append(f"{prefix}.evidence_event_ids must be a list")
        if "capability_needs" not in point or not isinstance(point.get("capability_needs"), list):
            issues.append(f"{prefix}.capability_needs must be a list")
    return issues


def concrete_repair_action_issues(micro: Dict[str, Any]) -> list[str]:
    """Validate actionable, evidence-bound micro-level repair proposals.

    These are review evidence, not a hard-coded evolution target.  Requiring
    an observed failure, interface/handoff, and falsifiable signal prevents a
    generic "improve thinking" sentence from being promoted into macro input.
    """
    if not isinstance(micro, dict):
        return ["micro review is not an object"]
    actions = micro.get("concrete_repair_actions") or []
    change_type = str(micro.get("change_type") or "")
    if change_type in REPAIR_ACTION_REQUIRED_CHANGE_TYPES and not actions:
        return ["missing concrete_repair_actions for unresolved/regressed result"]
    if not actions:
        return []
    if not isinstance(actions, list):
        return ["concrete_repair_actions must be a list"]
    issues = []
    for index, action in enumerate(actions):
        prefix = f"concrete_repair_actions[{index}]"
        if not isinstance(action, dict):
            issues.append(f"{prefix} must be an object")
            continue
        if str(action.get("module") or "") not in MFP_MODULES:
            issues.append(f"{prefix}.module is invalid")
        for field in (
            "failure_link", "current_observed_failure", "repair_mechanism",
            "input_output_or_handoff", "falsifiable_runtime_signal",
            "generalization_scope",
        ):
            if not str(action.get(field) or "").strip():
                issues.append(f"{prefix}.{field} is empty")
    return issues


def failure_chain_issues(chain: Any, *, trajectory: list[Dict[str, Any]] | None = None) -> list[str]:
    """Validate a multi-module micro-review chain without forcing blame.

    A module may be ``not_observed`` when a prior edge prevented it from
    running.  Every other role needs an actual trajectory anchor, preventing
    the macro review from turning a wrong answer into five unsupported claims.
    """
    if not isinstance(chain, list) or not chain:
        return ["failure_chain must be a non-empty list"]
    tool_steps = [step for step in (trajectory or []) if isinstance(step, dict)
                  and step.get("step_type") == "tool_execution"]
    issues = []
    seen = set()
    roots = 0
    for index, item in enumerate(chain):
        prefix = f"failure_chain[{index}]"
        if not isinstance(item, dict):
            issues.append(f"{prefix} must be an object")
            continue
        module = str(item.get("module") or "")
        role = str(item.get("role") or "")
        if module not in MFP_MODULES:
            issues.append(f"{prefix}.module is invalid")
        if role not in FAILURE_CHAIN_ROLES:
            issues.append(f"{prefix}.role is invalid")
        if module in seen:
            issues.append(f"{prefix}.module is duplicated")
        seen.add(module)
        roots += role == "root_cause"
        if not str(item.get("rationale") or "").strip():
            issues.append(f"{prefix}.rationale is empty")
        if role == "not_observed":
            if not str(item.get("blocked_by") or "").strip():
                issues.append(f"{prefix}.blocked_by is empty for not_observed")
            continue
        anchor_kind = str(item.get("trace_anchor_kind") or "tool_execution")
        if anchor_kind not in {"tool_execution", "structure_artifact", "reasoning_decision", "memory_state"}:
            issues.append(f"{prefix}.trace_anchor_kind is invalid")
        try:
            step_index = int(item.get("trajectory_step_index") or 0)
        except (TypeError, ValueError):
            step_index = 0
        if anchor_kind == "tool_execution":
            if step_index < 1:
                issues.append(f"{prefix}.trajectory_step_index is invalid")
            elif tool_steps and step_index > len(tool_steps):
                issues.append(f"{prefix}.trajectory_step_index is outside trajectory")
        else:
            expected_module = {
                "structure_artifact": "video_structuring",
                "reasoning_decision": "thinking",
                "memory_state": "memory",
            }.get(anchor_kind)
            if step_index != 0:
                issues.append(f"{prefix}.trajectory_step_index must be 0 for lifecycle anchor")
            if expected_module and module != expected_module:
                issues.append(f"{prefix}.trace_anchor_kind disagrees with module")
        if not isinstance(item.get("active_window"), (list, tuple)) or not _valid_window(item.get("active_window")):
            issues.append(f"{prefix}.active_window is invalid")
    if roots != 1:
        issues.append("failure_chain must contain exactly one root_cause")
    return issues


def derive_failure_chain(trajectory: list[Dict[str, Any]] | None,
                         minimal_failure_points: list[Dict[str, Any]] | None = None) -> list[dict]:
    """Derive conservative module roles from actual trace records.

    Teacher output can enrich this structure, but this deterministic fallback
    never invents an unobserved module failure.
    """
    tool_steps = [step for step in (trajectory or []) if isinstance(step, dict)
                  and step.get("step_type") == "tool_execution"]
    observed = []
    for index, step in enumerate(tool_steps, start=1):
        module = str(step.get("producer_module") or "")
        if module not in MFP_MODULES or module in {row["module"] for row in observed}:
            continue
        windows = step.get("active_time_windows") or []
        window = next((value for value in windows if _valid_window(value)), None)
        observed.append({"module": module, "trajectory_step_index": index, "active_window": window})
    mfp_modules = [str(point.get("root_module") or "") for point in (minimal_failure_points or []) if isinstance(point, dict)]
    root = next((module for module in mfp_modules if module in MFP_MODULES), observed[0]["module"] if observed else "")
    # Never manufacture a step/window merely to make a chain look complete.
    # An unanchored review remains an explicit incomplete audit item and may
    # not become adaptive diagnosis evidence.
    if not root or not any(item["module"] == root for item in observed):
        return []
    chain = []
    for item in observed:
        role = "root_cause" if item["module"] == root else "contributing"
        chain.append({**item, "role": role, "rationale": "observed runtime producer in the reviewed trajectory"})
    observed_modules = {item["module"] for item in chain}
    for module in sorted(MFP_MODULES - observed_modules):
        chain.append({"module": module, "role": "not_observed", "blocked_by": root,
                      "rationale": "no tool execution by this module after the upstream failure"})
    return chain


def causal_effect_coverage(micro_reviews: list[Dict[str, Any]] | Any) -> dict:
    """Return deterministic coverage for the current review contract."""
    if not isinstance(micro_reviews, list):
        return {
            "required": 0,
            "valid": 0,
            "complete": False,
            "invalid": [{"task_id": "", "issues": ["question_reviews is not a list"]}],
        }
    invalid = []
    for micro in micro_reviews or []:
        issues = failure_point_issues(micro)
        if issues:
            invalid.append({
                "task_id": micro.get("task_id") or micro.get("time_reference") or "",
                "issues": issues,
            })
    total = len(micro_reviews or [])
    return {
        "required": total,
        "valid": total - len(invalid),
        "complete": bool(total) and not invalid,
        "invalid": invalid[:16],
    }


def minimal_failure_point_coverage(micro_reviews: list[Dict[str, Any]] | Any) -> dict:
    """Summarize MFP availability without making macro review fail-closed."""
    if not isinstance(micro_reviews, list):
        return {"required": 0, "valid": 0, "invalid": [], "by_module": {}, "by_failure_kind": {}}
    required = 0
    valid = 0
    invalid = []
    by_module: dict[str, int] = {}
    by_failure_kind: dict[str, int] = {}
    for micro in micro_reviews:
        if not isinstance(micro, dict) or str(micro.get("change_type") or "") not in MFP_REQUIRED_CHANGE_TYPES:
            continue
        required += 1
        validation = micro.get("minimal_failure_point_validation") or {}
        points = micro.get("minimal_failure_points") or []
        if validation.get("direct_repair_eligible") is True and isinstance(points, list):
            valid += 1
            point_validations = validation.get("point_validations") or []
            valid_indexes = {
                int(item.get("index")) for item in point_validations
                if isinstance(item, dict) and item.get("valid") is True
                and isinstance(item.get("index"), int)
            }
            iterable = [
                point for index, point in enumerate(points) if index in valid_indexes
            ]
            for point in iterable:
                if not isinstance(point, dict):
                    continue
                module = str(point.get("root_module") or "unknown")
                kind = str(point.get("failure_kind") or "unknown")
                by_module[module] = by_module.get(module, 0) + 1
                by_failure_kind[kind] = by_failure_kind.get(kind, 0) + 1
        else:
            invalid.append({
                "task_id": micro.get("task_id") or micro.get("time_reference") or "",
                "issues": list(validation.get("issues") or ["unattributed_or_incomplete"]),
            })
    return {
        "required": required,
        "valid": valid,
        "invalid": invalid[:16],
        "by_module": by_module,
        "by_failure_kind": by_failure_kind,
    }


def review_is_consumable(review: Dict[str, Any]) -> bool:
    """Return whether a real completed review can enter later diagnosis.

    Individual teacher micro reviews can be incomplete with weaker models.  The
    diagnosis path consumes the valid records that exist plus current-best and
    historical evidence; it must not discard an otherwise useful round.
    """
    if not isinstance(review, dict):
        return False
    macro = review.get("macro_review") or {}
    # A dry run is a wiring check only.  Its rows deliberately contain no
    # Teacher diagnosis and must never be promoted into a real diagnosis
    # input merely because the dry-run wrapper wrote a syntactically valid
    # macro object.
    if review.get("dry_run") is True or macro.get("dry_run") is True:
        return False
    return macro.get("success") is True
