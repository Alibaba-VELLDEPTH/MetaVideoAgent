"""
Post-evolution Teacher review runner.

This stage runs after an evolved combo has produced full/sandbox trajectories.
It compares the pre-evolution current-best trajectories with the
post-evolution result trajectories. Once post-decision accepts that result,
the latter is the new current best; MetaVideoAgent runs never inherit stale review
cache from another run.
"""

import argparse
import concurrent.futures
import glob
import json
import os
import time
import traceback
from typing import Dict, List, Optional

try:
    from answer_normalizer import extract_gold_answer, judge_answer
    from config import META_VLM_MAX_CONCURRENT, TEACHER_REVIEW_MAX_CONCURRENT
    from question_set_loader import questions_by_ref
    from reference_loader import load_reference_by_ref_and_question
    from review_contract import (
        causal_effect_coverage,
        concrete_repair_action_issues,
        derive_failure_chain,
        failure_chain_issues,
        minimal_failure_point_coverage,
        minimal_failure_point_issues,
    )
    from runtime_paths import require_metavideoagent_runtime, runtime_output_root
    from sandbox_evaluator import write_engineering_invalid_artifacts
    from task_identity import identity_from_entry, judge_exact_answer
    from teacher_agent import TeacherAgent, _sanitize_teacher_output_terms
except ImportError:  # pragma: no cover - package import
    from .answer_normalizer import extract_gold_answer, judge_answer
    from .config import META_VLM_MAX_CONCURRENT, TEACHER_REVIEW_MAX_CONCURRENT
    from .question_set_loader import questions_by_ref
    from .reference_loader import load_reference_by_ref_and_question
    from .review_contract import (
        causal_effect_coverage,
        concrete_repair_action_issues,
        derive_failure_chain,
        failure_chain_issues,
        minimal_failure_point_coverage,
        minimal_failure_point_issues,
    )
    from .runtime_paths import require_metavideoagent_runtime, runtime_output_root
    from .sandbox_evaluator import write_engineering_invalid_artifacts
    from .task_identity import identity_from_entry, judge_exact_answer
    from .teacher_agent import TeacherAgent, _sanitize_teacher_output_terms


def _read_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bundle_path_from_report(report: dict) -> str:
    """Return the executable bundle recorded by a formal eval report.

    A report can describe either an accepted current best or an unevaluated
    parent/reference.  Keep this lookup centralized so review metadata never
    pairs one run's results with another run's bundle.
    """
    if not isinstance(report, dict):
        return ""
    policy = report.get("combo_policy") or {}
    for value in (
        report.get("current_best_bundle_path"),
        report.get("effective_initial_baseline_bundle"),
        report.get("initial_baseline_bundle"),
        report.get("initial_bundle_path"),
        report.get("bundle_path"),
        report.get("candidate_full_eval_bundle_path"),
        report.get("candidate_bundle"),
        policy.get("current_best_bundle_path"),
        policy.get("effective_bundle_path"),
        policy.get("base_bundle_path"),
    ):
        value = str(value or "")
        if value:
            return value
    return ""


def _review_state_contexts(verification_report: dict,
                           verification_report_path: str,
                           evolved_results_path: str,
                           reference_results_path: str,
                           initial_reference_review: bool,
                           candidate_bundle_path: str,
                           combo_id: str) -> tuple[dict, dict, dict]:
    """Separate parent reference, reviewed candidate and accepted current best.

    Review is used both before and after post-full-eval acceptance.  In the
    latter case the reviewed candidate is the new current best; in the former
    case the parent reference remains current best.  Do not combine fields
    from the two states in one ``current_best_context``.
    """
    verification_report = verification_report or {}
    parent_report_path = str(verification_report.get("reference_report_path") or "")
    parent_report = _read_json(parent_report_path)
    parent_bundle_path = _bundle_path_from_report(parent_report)
    parent_combo_id = str(parent_report.get("combo_id") or "")
    post_decision = verification_report.get("post_full_eval_decision") or {}
    current_best_override = verification_report.get("current_best_override") or {}
    override_accepted = bool(
        isinstance(current_best_override, dict)
        and current_best_override.get("accepted") is True
    )
    accepted = bool(
        initial_reference_review
        or post_decision.get("update_current_best") is True
        or override_accepted
        or (
            verification_report.get("accepted") is True
            and post_decision.get("state_transition") in (
                "accepted_new_current_best",
            )
        )
    )
    candidate_combo_id = str(
        (current_best_override.get("combo_id") if override_accepted else "")
        or verification_report.get("combo_id") or combo_id or ""
    )
    accepted_bundle_path = (
        str(verification_report.get("current_best_bundle_path") or "")
        if accepted else ""
    ) or (
        str(current_best_override.get("bundle_path") or "")
        if override_accepted else ""
    ) or candidate_bundle_path

    reference_context = {
        "reference_results_path": reference_results_path,
        "reference_report_path": parent_report_path,
        "reference_bundle_path": parent_bundle_path,
        "reference_combo_id": parent_combo_id,
        "reference_role": (
            "not_applicable_unique_initial_current_best"
            if initial_reference_review else "current_best_before_evolution"
        ),
    }
    candidate_context = {
        "candidate_results_path": evolved_results_path,
        "candidate_report_path": verification_report_path,
        "candidate_bundle_path": candidate_bundle_path,
        "candidate_combo_id": candidate_combo_id,
        "post_full_eval_accepted": accepted,
    }
    if initial_reference_review:
        current_best_context = {
            "mode": "unique_initial_current_best",
            "current_best_results_path": evolved_results_path,
            "current_best_report_path": verification_report_path,
            "current_best_bundle_path": accepted_bundle_path,
            "current_best_combo_id": candidate_combo_id,
            "comparison_available": False,
        }
    elif accepted:
        current_best_context = {
            "mode": "accepted_post_evolution_current_best",
            "current_best_results_path": evolved_results_path,
            "current_best_report_path": verification_report_path,
            "current_best_bundle_path": accepted_bundle_path,
            "current_best_combo_id": candidate_combo_id,
            "comparison_available": True,
        }
    else:
        current_best_context = {
            "mode": "pre_evolution_current_best_candidate_pending",
            "current_best_results_path": reference_results_path,
            "current_best_report_path": parent_report_path,
            "current_best_bundle_path": parent_bundle_path,
            "current_best_combo_id": parent_combo_id,
            "comparison_available": True,
        }
    return reference_context, candidate_context, current_best_context


def _iter_jsonl(path: str):
    """Stream JSONL records so review does not materialize large eval files."""
    if not path or not os.path.exists(path):
        return
    if os.path.isdir(path):
        for item in sorted(glob.glob(os.path.join(path, "*_sandbox.jsonl"))):
            yield from _iter_jsonl(item)
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _external_history_meta(*paths: str) -> dict:
    formal_output_root = os.path.abspath(runtime_output_root())
    external = []
    for path in paths:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(formal_output_root + os.sep):
            external.append(abs_path)
    return {
        "external_history_input": bool(external),
        "external_history_paths": external,
    }


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


def _manifest_path_from_profile(profile: dict) -> str:
    if not isinstance(profile, dict):
        return ""
    spec = profile.get("distribution_spec", {}) or {}
    return spec.get("manifest_path", "") or profile.get("manifest_path", "")


def _discover_video_ids(workspace_dir: str, distribution_manifest: str = "") -> List[str]:
    if not distribution_manifest:
        raise ValueError("evolution review requires a distribution manifest")
    _by_ref, _q_to_ref, ids, _spec = questions_by_ref(
        workspace_dir,
        video_id="ALL",
        distribution_manifest=distribution_manifest,
        preferred_splits=("train",),
    )
    return ids


def _normalize_video_ids(workspace_dir: str, video_id: str,
                         distribution_manifest: str = "") -> List[str]:
    if not video_id or video_id in ("ALL", "all", "multi_video"):
        return _discover_video_ids(workspace_dir, distribution_manifest)
    return [v.strip() for v in str(video_id).split(",") if v.strip()]


def _load_dataset(workspace_dir: str, video_id: str,
                  distribution_manifest: str = "") -> Dict[str, dict]:
    if not distribution_manifest:
        raise ValueError("evolution review requires a distribution manifest")
    by_ref, _question_to_ref, _ids, _spec = questions_by_ref(
        workspace_dir,
        video_id=video_id,
        distribution_manifest=distribution_manifest,
        preferred_splits=("train",),
    )
    return by_ref




def _row_matches_combo(entry: dict, combo_id: str = "") -> bool:
    if not combo_id:
        return True
    return entry.get("combo_id") == combo_id


def _normalize_evolved_entry(entry: dict, baseline_by_question: Dict[str, dict],
                             dataset_by_ref: Dict[str, dict]) -> Optional[dict]:
    """Normalize either full_eval_results rows or sandbox trajectory rows."""
    if "task_meta" in entry:
        meta = entry.get("task_meta", {})
        ident = identity_from_entry(entry)
        return {
            "task_id": ident["task_id"],
            "video_id": ident["video_id"],
            "time_reference": ident["time_reference"],
            "question": ident["question"],
            "gt_answer": extract_gold_answer(entry),
            "answer": entry.get("final_agent_answer") or entry.get("answer", ""),
            "trajectory": entry.get("trajectory", []),
            "combo": entry.get("architecture_combo", entry.get("combo", {})),
            "raw": entry,
        }

    question = entry.get("question", "")
    baseline = baseline_by_question.get(question, {})
    meta = baseline.get("task_meta", {})
    ident = identity_from_entry(baseline) if baseline else {}
    time_ref = ident.get("time_reference", "")
    video_id = ident.get("video_id", "")
    task_id = ident.get("task_id", "")
    if not task_id:
        # Last fallback: match through dataset question text.
        for tid, ds in dataset_by_ref.items():
            if ds.get("question", "") == question:
                task_id = tid
                video_id = ds.get("video_id", "")
                time_ref = ds.get("time_reference", "")
                meta = ds
                break
    if not question:
        question = meta.get("question", "")
    if not task_id or not question:
        return None
    return {
        "task_id": task_id,
        "video_id": video_id,
        "time_reference": time_ref,
        "question": question,
        "gt_answer": extract_gold_answer(entry),
        "answer": entry.get("answer", ""),
        "trajectory": entry.get("trajectory", []),
        "combo": entry.get("combo", {}),
        "raw": entry,
    }


def _answer_is_correct(answer: str, gt_or_task) -> bool:
    if isinstance(gt_or_task, dict):
        return bool(judge_answer(answer, task=gt_or_task).get("is_correct"))
    return judge_exact_answer(answer, gt_or_task)


def apply_authoritative_answer_metadata(micro_review: dict, evolved_entry: dict) -> dict:
    """Align teacher metadata with the full-eval answer contract.

    Teacher review receives a compact question string and may not see the
    choice list needed to interpret an answer such as ``F``.  Full-eval rows
    retain that task metadata, so they are the sole authority for correctness.
    The teacher still supplies causal analysis; this only normalizes redundant
    answer-status metadata before it becomes diagnosis evidence.
    """
    result = dict(micro_review or {})
    correct = _answer_is_correct(
        evolved_entry.get("answer", ""),
        evolved_entry.get("raw") or evolved_entry,
    )
    result["evolved_correct"] = correct
    result["answer_status"] = "correct" if correct else "incorrect"
    return result


def align_deterministic_causal_role(micro_review: dict) -> dict:
    """Synchronize the one causal field determined by the scored outcome.

    ``change_type`` is computed from the paired full-eval rows after the
    Teacher response arrives.  ``causal_effect_point.outcome_role`` is merely
    the schema label for that same deterministic outcome; leaving an LLM's
    stale label in place turns an otherwise trace-grounded review into a
    spurious contract failure.  This does *not* alter any attributed cause,
    evidence event, window, failure point, or repair proposal.
    """
    result = dict(micro_review or {})
    expected = {
        "fix": "candidate_correction_evidence",
        "regression": "regression_guard",
        "keep_wrong": "unchanged_failure_evidence",
        "keep_correct": "stable_no_failure",
    }.get(str(result.get("change_type") or ""))
    point = result.get("causal_effect_point")
    if expected and isinstance(point, dict):
        normalized_point = dict(point)
        normalized_point["outcome_role"] = expected
        result["causal_effect_point"] = normalized_point
    return result


def _change_type(baseline_entry: dict, evolved_entry: dict) -> str:
    reference_answer = (
        baseline_entry.get("final_agent_answer")
        or baseline_entry.get("answer")
        or ""
    )
    reference_correct = _answer_is_correct(
        reference_answer, baseline_entry
    )
    evolved_correct = _answer_is_correct(
        evolved_entry.get("answer", ""),
        evolved_entry.get("raw") or evolved_entry,
    )
    if not reference_correct and evolved_correct:
        return "fix"
    if reference_correct and not evolved_correct:
        return "regression"
    if reference_correct and evolved_correct:
        return "keep_correct"
    return "keep_wrong"


def _evidence_is_strong(review: dict) -> bool:
    return (
        review.get("success")
        and review.get("answer_status") == "correct"
        and review.get("evidence_quality") in ("strong", "direct")
        and review.get("stability") == "stable"
    )


def _evidence_was_reviewed(review: dict) -> bool:
    if review.get("dry_run"):
        return False
    return review.get("answer_status") not in ("", None, "not_reviewed")


def _review_ref(review: dict) -> str:
    return review.get("task_id") or review.get("time_reference", "")


def build_retention_review(metrics: dict, micro_reviews: list,
                           delta_reviews: list, evolution_info: dict) -> dict:
    """Deterministic candidate-retention evidence for the next diagnosis.

    It summarizes whether the previous candidate should be retained or improved
    while the next DiagnosisAgent chooses a concrete
    programmable target. It never blocks code evolution.
    """
    # The first executed combo is the unique current best, not a candidate
    # awaiting a retention decision. Its wrong answers are diagnosis evidence;
    # there is no predecessor against which a correction or regression exists.
    if (evolution_info or {}).get("review_mode") == "initial_reference_single_baseline":
        unchanged_summary = summarize_unchanged_failures(micro_reviews)
        return {
            "default_policy": "unique_initial_current_best_under_diagnosis",
            "candidate_retention_allowed": True,
            "combo_decision": {
                "previous_candidate_action": "not_applicable_no_predecessor",
                "switch_target": False,
                "target_modules": [],
                "kept_modules": [],
                "rejected_directions": [],
                "probe_plan": {
                    "repair_probe": [], "regression_guard": [],
                    "failure_clusters": unchanged_summary.get("refs_by_category", {}),
                    "previous_candidate_corrections": [],
                    "previous_candidate_regressions": [],
                },
                "cost_optimization_required": False,
                "cost_audit_issue_codes": [],
                "rationale": "The unique initial current best has no predecessor or candidate delta.",
            },
            "candidate_bundle": evolution_info or {},
            "post_evolution_status": "initial_reference_under_review",
            "net_delta": None,
            "corrections": [],
            "regressions": [],
            "strong_fix_refs": [],
            "fragile_or_weak_fix_refs": [],
            "evidence_probes": [],
            "cost_optimization_required": False,
            "cost_ratio_vs_reference": None,
            "reason": "Unique initial current best: analyze defects without pre/post retention semantics.",
        }

    fixes = [r for r in micro_reviews if r.get("change_type") == "fix"]
    regressions = [r for r in micro_reviews if r.get("change_type") == "regression"]
    strong_fixes = [r for r in fixes if _evidence_is_strong(r)]
    fragile_fixes = [
        _review_ref(r) for r in fixes
        if _evidence_was_reviewed(r) and not _evidence_is_strong(r)
    ]
    regression_refs = [_review_ref(r) for r in regressions]
    correction_refs = [_review_ref(r) for r in fixes]
    unchanged_summary = summarize_unchanged_failures(micro_reviews)
    net_delta = metrics.get("net_delta")
    regression_count = metrics.get("regression") or 0
    cost_audit = metrics.get("cost_audit") or {}
    cost_issue_codes = {
        item.get("code")
        for item in cost_audit.get("issue_summary", []) or []
        if isinstance(item, dict)
    }
    cost_ratio = cost_audit.get("cost_ratio_vs_reference")
    has_cost_explosion = (
        "cost_explosion" in cost_issue_codes
        or (isinstance(cost_ratio, (int, float)) and cost_ratio >= 1.8)
    )

    accepted_as_current_best = (
        evolution_info.get("post_evolution_status") == "accepted_new_current_best"
    )
    retention_allowed = True
    default_policy = "preserve"
    reason = "Post-evolution result has no deterministic preservation warning."

    if accepted_as_current_best:
        default_policy = "preserve_accepted_current_best_with_regression_guards"
        reason = (
            "Post-decision accepted this post-evolution combo as current best. "
            "Corrections are preservation evidence and regressions are guards for the next diagnosis."
        )
    elif (net_delta is not None and net_delta <= 0) and regression_count > 0:
        retention_allowed = False
        default_policy = "rejected_candidate_evidence_only"
        reason = (
            "Post-decision retained the pre-evolution current best. This candidate's "
            "corrections and regressions remain next-round evidence, not a module ban."
        )
    elif fragile_fixes:
        retention_allowed = False
        default_policy = "rejected_candidate_evidence_only"
        reason = (
            "one or more corrections are fragile or weakly evidenced; retention "
            "requires next-round diagnosis/probe evidence before preserving the "
            "candidate as current best."
        )
    elif net_delta is not None and net_delta > 0 and has_cost_explosion:
        retention_allowed = True
        default_policy = "preserve_and_optimize_cost"
        reason = (
            "candidate has net accuracy gain but cost audit shows expensive or "
            "non-convergent behavior; preserve gains while reducing cost next."
        )

    evidence_probes = []
    if not retention_allowed:
        evidence_probes = [
            {
                "name": "reference_vs_candidate_on_corrections",
                "time_references": correction_refs,
                "task_ids": correction_refs,
                "purpose": "verify whether corrections are causally caused by the candidate",
            },
            {
                "name": "reference_vs_candidate_on_regressions",
                "time_references": regression_refs,
                "task_ids": regression_refs,
                "purpose": "verify whether regressions are caused by the candidate",
            },
            {
                "name": "pre_vs_post_evolution_guard_probe",
                "time_references": sorted(set(correction_refs + regression_refs)),
                "task_ids": sorted(set(correction_refs + regression_refs)),
                "purpose": "verify corrections and regressions against the pre-evolution current best",
            },
        ]

    candidate_modules = list(evolution_info.get("changed_modules") or [])
    candidate_items = [
        {
            "module_type": module,
            "reason": "module changed in the post-evolution bundle under review",
        }
        for module in candidate_modules
    ]
    if accepted_as_current_best:
        previous_candidate_action = "preserve_and_improve_current_best"
        kept_modules = candidate_items
        rejected_directions = []
    elif not retention_allowed:
        previous_candidate_action = "evidence_only_rejected_candidate"
        kept_modules = []
        rejected_directions = []
    elif net_delta is not None and net_delta > 0 and regression_count == 0:
        previous_candidate_action = "keep"
        kept_modules = candidate_items
        rejected_directions = []
    else:
        previous_candidate_action = "modify"
        kept_modules = candidate_items
        rejected_directions = []

    combo_decision = {
        "previous_candidate_action": previous_candidate_action,
        "switch_target": False,
        "target_modules": candidate_modules,
        "kept_modules": kept_modules,
        "rejected_directions": rejected_directions,
        "probe_plan": {
            "repair_probe": correction_refs,
            "regression_guard": regression_refs,
            "failure_clusters": unchanged_summary.get("refs_by_category", {}),
            "previous_candidate_corrections": correction_refs,
            "previous_candidate_regressions": regression_refs,
            "evidence_probes": evidence_probes,
        },
        "cost_optimization_required": bool(has_cost_explosion),
        "cost_audit_issue_codes": sorted(code for code in cost_issue_codes if code),
        "rationale": reason,
    }

    return {
        "default_policy": default_policy,
        "candidate_retention_allowed": retention_allowed,
        "combo_decision": combo_decision,
        "candidate_bundle": evolution_info,
        "post_evolution_status": evolution_info.get("post_evolution_status", "candidate_under_review"),
        "net_delta": net_delta,
        "corrections": correction_refs,
        "regressions": regression_refs,
        "strong_fix_refs": [_review_ref(r) for r in strong_fixes],
        "fragile_or_weak_fix_refs": fragile_fixes,
        "evidence_probes": evidence_probes,
        "cost_optimization_required": bool(has_cost_explosion),
        "cost_ratio_vs_reference": cost_ratio,
        "reason": reason,
    }


MODULE_SLOTS = [
    "video_structuring", "localization", "perception", "thinking", "memory",
]

UNCHANGED_FAILURE_TYPES = [
    "evidence_absent",
    "wrong_temporal_localization",
    "evidence_retrieved_not_used",
    "vlm_perception_miss",
    "option_mapping_error",
    "counting_or_aggregation_error",
    "unattributed_or_incomplete",
]


def _compact_refs(reviews: list, change_type: str = "") -> list:
    refs = []
    for item in reviews:
        if change_type and item.get("change_type") != change_type:
            continue
        ref = _review_ref(item)
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def classify_unchanged_failure(review: dict) -> dict:
    """Classify a wrong current-best outcome from validated Teacher evidence.

    A review schema does not require a previous ``unchanged_failure_category``
    field.  New micro reviews instead carry an anchored MFP with a structured
    failure kind.  Treating every such row as ``unattributed_or_incomplete``
    discarded the actual cross-question distribution before diagnosis could
    consume it.  Only *validated* MFPs may contribute here; an invalid anchor
    remains an explicit audit omission and never becomes a repair claim.
    """
    explicit = str(
        review.get("unchanged_failure_category")
        or review.get("fault_type")
        or ""
    ).strip()
    if explicit in UNCHANGED_FAILURE_TYPES and explicit != "unattributed_or_incomplete":
        return {
            "category": explicit,
            "confidence": "teacher_explicit",
            "reason": "Teacher supplied an explicit supported unchanged-failure category",
        }

    validation = review.get("minimal_failure_point_validation") or {}
    points = review.get("minimal_failure_points") or []
    valid_indices = set()
    rows = validation.get("point_validations")
    if isinstance(rows, list):
        valid_indices = {
            int(item.get("index"))
            for item in rows
            if isinstance(item, dict) and item.get("valid") is True
            and isinstance(item.get("index"), int)
        }
    elif validation.get("valid") is True or validation.get("direct_repair_eligible") is True:
        valid_indices = set(range(len(points)))

    def category_from_failure_kind(value: object) -> str:
        kind = str(value or "").strip().lower()
        if not kind:
            return ""
        # More specific, answer-level failures take precedence over a broad
        # upstream retrieval label when a micro review names both.
        if "option" in kind and ("map" in kind or "match" in kind):
            return "option_mapping_error"
        if any(token in kind for token in ("count", "aggregation", "ordinal", "sequence")):
            return "counting_or_aggregation_error"
        if "temporal" in kind or "time_range" in kind:
            return "wrong_temporal_localization"
        if any(token in kind for token in (
            "visual_misread", "perception", "ocr", "asr", "verification_uncertain",
        )):
            return "vlm_perception_miss"
        if any(token in kind for token in (
            "evidence_not_consumed", "retrieval_not_consumed", "premature_termination",
            "hallucinated_answer", "evidence_present_final_wrong",
        )):
            return "evidence_retrieved_not_used"
        if any(token in kind for token in (
            "retrieval_miss", "evidence_absent", "structure_artifact_failure",
        )):
            return "evidence_absent"
        return ""

    for index, point in enumerate(points):
        if index not in valid_indices or not isinstance(point, dict):
            continue
        category = category_from_failure_kind(point.get("failure_kind"))
        if category:
            return {
                "category": category,
                "confidence": "teacher_validated_mfp",
                "reason": (
                    "Mapped from validated Teacher minimal_failure_point "
                    f"failure_kind={point.get('failure_kind', '')!s}"
                ),
            }

    return {
        "category": "unattributed_or_incomplete",
        "confidence": "insufficient_evidence",
        "reason": "Teacher review did not provide a causal failure attribution",
    }


def summarize_unchanged_failures(micro_reviews: list) -> dict:
    summary = {category: [] for category in UNCHANGED_FAILURE_TYPES}
    for item in micro_reviews or []:
        if item.get("change_type") != "keep_wrong":
            continue
        cls = item.get("unchanged_failure_classification") or {}
        # Previous artifacts often persisted the old blanket
        # ``unattributed_or_incomplete`` fallback.  Recompute that fallback
        # from the now-available validated MFPs instead of preserving stale
        # loss of attribution; a genuinely explicit non-default category is
        # still retained verbatim.
        category = cls.get("category")
        if not category or category == "unattributed_or_incomplete":
            category = classify_unchanged_failure(item)["category"]
        if category not in summary:
            summary[category] = []
        ref = _review_ref(item)
        if ref and ref not in summary[category]:
            summary[category].append(ref)
    return {
        "taxonomy": UNCHANGED_FAILURE_TYPES,
        "counts": {k: len(v) for k, v in summary.items()},
        "refs_by_category": summary,
    }


def refresh_review_deterministic_aggregates(review: dict) -> dict:
    """Refresh deterministic fields for a completed review without re-calling a model.

    This is intentionally safe for existing artifacts: it only recomputes the
    taxonomy from already-persisted micro reviews and removes the impossible
    ``[\"\"]`` changed-module placeholder in an initial-reference review.
    """
    review = dict(review or {})
    micros = review.get("evolved_micro_reviews") or []
    if isinstance(micros, list):
        for micro in micros:
            if isinstance(micro, dict) and micro.get("change_type") == "keep_wrong":
                micro["unchanged_failure_classification"] = classify_unchanged_failure(micro)
        review["unchanged_failure_summary"] = summarize_unchanged_failures(micros)
    context = review.get("candidate_context")
    if isinstance(context, dict):
        evolution_info = context.get("evolution_info")
        if isinstance(evolution_info, dict) and evolution_info.get("review_mode") == "initial_reference_single_baseline":
            evolution_info["changed_modules"] = []
    return review


def _valid_repair_actions(item: dict) -> list[dict]:
    """Return only repair actions that passed the micro-review contract."""
    if not isinstance(item, dict):
        return []
    actions = item.get("concrete_repair_actions") or []
    if not isinstance(actions, list):
        return []
    validation = item.get("concrete_repair_action_validation") or {}
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
        ]
    return []


def _review_modules(item: dict) -> list[str]:
    """Collect every module explicitly implicated by a micro/delta review.

    A MetaVideoAgent failure is commonly a handoff failure across multiple
    components. This helper collects every responsible module named by the current
    review contract.
    """
    if not isinstance(item, dict):
        return []
    modules: list[str] = []

    def add(value):
        value = str(value or "")
        if value in MODULE_SLOTS and value not in modules:
            modules.append(value)

    for key in ("fault_module", "improvement_target_module", "target_module"):
        add(item.get(key))
    improvement = item.get("improvement") or {}
    if isinstance(improvement, dict):
        add(improvement.get("target_module"))
    for step in item.get("fault_steps") or []:
        if isinstance(step, dict):
            add(step.get("fault_module"))
    for link in item.get("failure_chain") or []:
        if isinstance(link, dict) and link.get("role") != "not_observed":
            add(link.get("module"))
    for point in item.get("minimal_failure_points") or []:
        if isinstance(point, dict):
            add(point.get("root_module"))
    for action in _valid_repair_actions(item):
        add(action.get("module"))
    return modules


def _module_responsibility_from_reviews(micro_reviews: list,
                                        delta_reviews: list) -> dict:
    counts = {module: {"score": 0, "evidence": []} for module in MODULE_SLOTS}
    for item in list(micro_reviews or []) + list(delta_reviews or []):
        if not isinstance(item, dict):
            continue
        ref = _review_ref(item)
        evidence_weight = 2 if item.get("change_type") in ("fix", "regression") else 1

        def add(module, role_weight):
            if module not in counts:
                return
            counts[module]["score"] += evidence_weight * role_weight
            if ref and ref not in counts[module]["evidence"]:
                counts[module]["evidence"].append(ref)

        add(str(item.get("fault_module") or ""), 2)
        for step in item.get("fault_steps") or []:
            if isinstance(step, dict):
                add(str(step.get("fault_module") or ""), 3 if step.get("is_root_cause") else 1)
        chain_is_valid = bool((item.get("failure_chain_validation") or {}).get("valid") is True)
        if chain_is_valid:
            for link in item.get("failure_chain") or []:
                if not isinstance(link, dict):
                    continue
                role = str(link.get("role") or "")
                add(str(link.get("module") or ""), {
                    "root_cause": 4, "contributing": 2,
                    "downstream_unable_to_recover": 1,
                }.get(role, 0))
        for action in _valid_repair_actions(item):
            add(str(action.get("module") or ""), 2)

    output = {}
    max_score = max([v["score"] for v in counts.values()] or [0])
    for module, info in counts.items():
        score = info["score"]
        if score and score == max_score:
            level = "high"
        elif score:
            level = "medium"
        else:
            level = "low"
        output[module] = {
            "level": level,
            "evidence_refs": info["evidence"][:6],
            "reason": (
                "derived from validated failure-chain roles and repair actions"
                if score else "no direct post-evolution review evidence"
            ),
        }
    return output


def build_macro_evidence_ledger(micro_reviews: list, delta_reviews: list,
                                *, initial_reference: bool = False) -> dict:
    """Build a complete but bounded macro input from all micro reviews.

    Raw micro reviews stay in ``evolved_micro_reviews`` as the audit record.
    Macro review receives every question's outcome/validation status plus
    deterministic mechanism clusters, rather than a second copy of all long
    Gold/Student/Divergence prose.  This is aggregation, not prefix trimming:
    every task contributes to an inventory and to its applicable clusters.
    """
    inventory = []
    action_clusters: dict[str, dict] = {}
    chain_clusters: dict[str, dict] = {}
    fault_clusters: dict[str, dict] = {}
    omissions = []

    def add_ref(cluster: dict, ref: str, video_id: str):
        if ref and ref not in cluster["evidence_refs"]:
            cluster["evidence_refs"].append(ref)
        if video_id and video_id not in cluster["video_ids"]:
            cluster["video_ids"].append(video_id)

    for item in micro_reviews or []:
        if not isinstance(item, dict):
            continue
        ref = _review_ref(item)
        video_id = str(item.get("video_id") or "")
        chain_validation = item.get("failure_chain_validation") or {}
        mfp_validation = item.get("minimal_failure_point_validation") or {}
        valid_actions = _valid_repair_actions(item)
        inventory.append({
            "task_ref": ref,
            "video_id": video_id,
            "outcome": item.get("current_best_outcome") or item.get("change_type", ""),
            "answer_status": item.get("answer_status", ""),
            "teacher_completed": item.get("success") is True and not item.get("dry_run"),
            "failure_chain_valid": chain_validation.get("valid") is True,
            "mfp_direct_repair_eligible": mfp_validation.get("direct_repair_eligible") is True,
            "valid_repair_action_count": len(valid_actions),
            "fault_modules": _review_modules(item),
        })
        if chain_validation.get("valid") is not True:
            omissions.append({"task_ref": ref, "kind": "invalid_failure_chain"})
        if item.get("change_type") in {"keep_wrong", "regression"} and not valid_actions:
            omissions.append({"task_ref": ref, "kind": "missing_or_invalid_repair_action"})

        if chain_validation.get("valid") is True:
            chain = item.get("failure_chain") or []
            signature = " -> ".join(
                f"{link.get('module')}:{link.get('role')}"
                for link in chain if isinstance(link, dict)
            )
            if signature:
                cluster = chain_clusters.setdefault(signature, {
                    "failure_chain": chain, "evidence_refs": [], "video_ids": [], "count": 0,
                })
                add_ref(cluster, ref, video_id)
                cluster["count"] += 1

        module = str(item.get("fault_module") or "")
        fault_type = str(item.get("fault_type") or "")
        if module in MODULE_SLOTS or fault_type:
            key = f"{module}::{fault_type}"
            cluster = fault_clusters.setdefault(key, {
                "fault_module": module, "fault_type": fault_type,
                "evidence_refs": [], "video_ids": [], "count": 0,
            })
            add_ref(cluster, ref, video_id)
            cluster["count"] += 1

        for action in valid_actions:
            module = str(action.get("module") or "")
            mechanism = str(action.get("repair_mechanism") or "")
            handoff = str(action.get("input_output_or_handoff") or "")
            key = f"{module}::{mechanism}::{handoff}"
            cluster = action_clusters.setdefault(key, {
                "modules": [module] if module else [],
                "repair_mechanism": mechanism,
                "handoff_or_state_to_repair": handoff,
                "falsifiable_runtime_signals": [],
                "generalization_scopes": [],
                "observed_failure_examples": [],
                "evidence_refs": [], "video_ids": [], "count": 0,
            })
            for field, output_key in (
                ("current_observed_failure", "observed_failure_examples"),
                ("falsifiable_runtime_signal", "falsifiable_runtime_signals"),
                ("generalization_scope", "generalization_scopes"),
            ):
                value = str(action.get(field) or "")
                if value and value not in cluster[output_key]:
                    cluster[output_key].append(value)
            add_ref(cluster, ref, video_id)
            cluster["count"] += 1

    delta_summary: dict[str, dict] = {}
    for delta in delta_reviews or []:
        if not isinstance(delta, dict):
            continue
        kind = str(delta.get("change_type") or "unknown")
        row = delta_summary.setdefault(kind, {"count": 0, "evidence_refs": []})
        row["count"] += 1
        ref = _review_ref(delta)
        if ref and ref not in row["evidence_refs"]:
            row["evidence_refs"].append(ref)
    return {
        "review_semantics": (
            "unique_initial_current_best" if initial_reference
            else "paired_current_best_candidate_comparison"
        ),
        "all_question_inventory": inventory,
        "cross_question_repair_mechanisms": list(action_clusters.values()),
        "validated_failure_chain_clusters": list(chain_clusters.values()),
        "fault_clusters": list(fault_clusters.values()),
        "micro_review_omissions": omissions,
        "delta_summary": delta_summary,
        "micro_review_count": len(inventory),
    }


def build_programmable_macro_defaults(summary_payload: dict) -> dict:
    """Build deterministic evidence fields for post-evolution macro review.

    Review and diagnosis are intentionally separate in the MetaVideoAgent pipeline.
    The review stage may summarize evidence for the next DiagnosisAgent, but it
    must not pre-select a concrete evolution target or produce a code contract.
    """
    micro = summary_payload.get("evolved_micro_reviews", [])
    deltas = summary_payload.get("delta_reviews", [])
    retention = summary_payload.get("retention_review", {})
    evolution_info = (
        summary_payload.get("candidate_context", {}).get("evolution_info", {})
        or retention.get("candidate_bundle", {})
    )
    candidate_modules = list(evolution_info.get("changed_modules") or [])
    fixes = _compact_refs(micro, "fix")
    regressions = _compact_refs(micro, "regression")
    keep_wrong = _compact_refs(micro, "keep_wrong")
    unchanged_summary = (
        summary_payload.get("unchanged_failure_summary")
        or summarize_unchanged_failures(micro)
    )
    return {
        "review_evidence_seed": {
            "candidate_modules_under_review": candidate_modules,
            "correction_refs": fixes,
            "regression_refs": regressions,
            "unchanged_failure_refs": keep_wrong,
            "cost_optimization_required": bool(
                retention.get("cost_optimization_required")
            ),
            "retention_default_policy": retention.get("default_policy", ""),
            "retention_reason": retention.get("reason", ""),
        },
        "unchanged_failure_analysis": unchanged_summary,
        "module_responsibility": _module_responsibility_from_reviews(micro, deltas),
    }


def ensure_programmable_macro_fields(macro_review: dict,
                                     summary_payload: dict) -> dict:
    """Ensure macro_review contains evidence fields needed by diagnosis."""
    macro = dict(macro_review or {})
    defaults = build_programmable_macro_defaults(summary_payload)
    for key, value in defaults.items():
        existing = macro.get(key)
        if existing in (None, "", [], {}):
            macro[key] = value
    # Keep an LLM's explanatory wording, but never permit it to omit a
    # module from the responsibility table that diagnosis will consume.
    existing_responsibility = macro.get("module_responsibility") or {}
    if not isinstance(existing_responsibility, dict):
        existing_responsibility = {}
    for module, fallback in (defaults.get("module_responsibility") or {}).items():
        if not isinstance(existing_responsibility.get(module), dict):
            existing_responsibility[module] = fallback
    macro["module_responsibility"] = existing_responsibility
    # The LLM may select representative examples, but it must not replace the
    # deterministic full micro-review taxonomy/counts/refs used by diagnosis.
    # Preserve its module implications/reasons while pinning factual coverage.
    macro_analysis = macro.get("unchanged_failure_analysis") or {}
    canonical_analysis = defaults["unchanged_failure_analysis"]
    if not isinstance(macro_analysis, dict):
        macro_analysis = {}
    macro_analysis["taxonomy"] = canonical_analysis.get("taxonomy", [])
    macro_analysis["counts"] = canonical_analysis.get("counts", {})
    macro_analysis["refs_by_category"] = canonical_analysis.get("refs_by_category", {})
    macro["unchanged_failure_analysis"] = macro_analysis
    macro["evidence_integrity"] = {
        "unchanged_failure_statistics_source": "deterministic_evolved_micro_reviews",
        "macro_module_implications_are_interpretive": True,
    }
    ledger = summary_payload.get("macro_evidence_ledger") or {}
    if not macro.get("current_best_failure_summary"):
        wrong = [
            row for row in (ledger.get("all_question_inventory") or [])
            if isinstance(row, dict) and row.get("outcome") in {
                "keep_wrong", "regression", "current_best_wrong",
            }
        ]
        macro["current_best_failure_summary"] = {
            "scope": ledger.get("review_semantics") or "review",
            "overall_failure_chain": [
                cluster.get("failure_chain") for cluster in
                (ledger.get("validated_failure_chain_clusters") or [])
                if isinstance(cluster, dict)
            ],
            "dominant_observed_defects": [
                cluster for cluster in (ledger.get("fault_clusters") or [])
                if isinstance(cluster, dict)
            ],
            "evidence_refs": [row.get("task_ref", "") for row in wrong if row.get("task_ref")],
        }
    if not macro.get("bundle_level_improvement_opportunities"):
        macro["bundle_level_improvement_opportunities"] = [
            {
                "modules": cluster.get("modules") or [],
                "observed_pattern": cluster.get("observed_failure_examples") or [],
                "evidence_refs": cluster.get("evidence_refs") or [],
                "handoff_or_state_to_repair": cluster.get("handoff_requirements") or [],
                "generic_repair_direction": cluster.get("repair_mechanism", ""),
                "falsifiable_runtime_signal": cluster.get("falsifiable_runtime_signals") or [],
                "generalization_scope": cluster.get("generalization_scopes") or [],
            }
            for cluster in (ledger.get("cross_question_repair_mechanisms") or [])
            if isinstance(cluster, dict)
        ]
    macro.setdefault("success", bool(macro_review))
    return macro


def macro_review_issues(macro_review: dict, summary_payload: dict) -> list[str]:
    """Validate macro structure without blocking a partially useful review.

    A bad macro must be visible to DiagnosisAgent as an audit omission, not
    silently accepted as an authoritative summary.  The deterministic ledger
    remains available downstream even if this validation is partial.
    """
    if not isinstance(macro_review, dict):
        return ["macro_review is not an object"]
    issues = []
    ledger = summary_payload.get("macro_evidence_ledger") or {}
    known_refs = {
        str(row.get("task_ref") or "")
        for row in (ledger.get("all_question_inventory") or [])
        if isinstance(row, dict) and str(row.get("task_ref") or "")
    }
    current = macro_review.get("current_best_failure_summary") or {}
    if not isinstance(current, dict):
        issues.append("current_best_failure_summary is missing")
    else:
        for field in ("scope", "overall_failure_chain", "dominant_observed_defects", "evidence_refs"):
            if current.get(field) in (None, "", []):
                issues.append(f"current_best_failure_summary.{field} is empty")
        for ref in current.get("evidence_refs") or []:
            if str(ref) not in known_refs:
                issues.append("current_best_failure_summary contains unknown evidence_ref")
                break
    opportunities = macro_review.get("bundle_level_improvement_opportunities") or []
    if not isinstance(opportunities, list):
        issues.append("bundle_level_improvement_opportunities is not a list")
    for index, item in enumerate(opportunities if isinstance(opportunities, list) else []):
        prefix = f"bundle_level_improvement_opportunities[{index}]"
        if not isinstance(item, dict):
            issues.append(f"{prefix} is not an object")
            continue
        modules = item.get("modules") or []
        if not isinstance(modules, list) or not any(module in MODULE_SLOTS for module in modules):
            issues.append(f"{prefix}.modules has no valid module")
        for field in ("observed_pattern", "handoff_or_state_to_repair", "generic_repair_direction", "falsifiable_runtime_signal"):
            if item.get(field) in (None, "", []):
                issues.append(f"{prefix}.{field} is empty")
        for ref in item.get("evidence_refs") or []:
            if str(ref) not in known_refs:
                issues.append(f"{prefix} contains unknown evidence_ref")
                break
    responsibility = macro_review.get("module_responsibility") or {}
    for module in MODULE_SLOTS:
        entry = responsibility.get(module)
        if not isinstance(entry, dict):
            issues.append(f"module_responsibility.{module} is missing")
    return issues


def build_mechanism_summary(summary_payload: dict, macro_review: dict) -> dict:
    """Model-safe macro conclusions for diagnosis, preserving review semantics."""
    macro = macro_review or {}
    responsibility = macro.get("module_responsibility") or {}
    return {
        "review_semantics": (
            (summary_payload.get("current_best_context") or {}).get("mode", "")
        ),
        "candidate_modules": list(
            (summary_payload.get("candidate_context", {}).get("evolution_info", {}) or {}).get("changed_modules", [])
        ),
        "outcome": {
            "net_delta": (summary_payload.get("metrics") or {}).get("net_delta"),
            "overall_verdict": str(macro.get("overall_verdict") or ""),
            "verdict_reason": str(macro.get("verdict_reason") or ""),
        },
        "current_best_failure_summary": macro.get("current_best_failure_summary") or {},
        "macro_validation": macro.get("validation") or {},
        "bundle_level_improvement_opportunities": macro.get(
            "bundle_level_improvement_opportunities") or [],
        "validated_cross_question_repair_mechanisms": (
            (summary_payload.get("macro_evidence_ledger") or {}).get(
                "cross_question_repair_mechanisms") or []
        ),
        "validated_failure_chain_clusters": (
            (summary_payload.get("macro_evidence_ledger") or {}).get(
                "validated_failure_chain_clusters") or []
        ),
        "micro_review_omissions": (
            (summary_payload.get("macro_evidence_ledger") or {}).get(
                "micro_review_omissions") or []
        ),
        "mechanisms_to_preserve": [
            str(item) for item in ((macro.get("candidate_retention_evidence") or {}).get("stable_mechanisms") or [])
        ],
        "regression_guards": [
            str(item) for item in ((macro.get("candidate_retention_evidence") or {}).get("regression_guards") or [])
        ],
        "module_signals": {
            module: {
                "level": value.get("level", ""),
                "reason": str(value.get("reason") or ""),
                "evidence_refs": value.get("evidence_refs") or [],
            }
            for module, value in responsibility.items() if isinstance(value, dict)
        },
        "unchanged_failure_mechanisms": [
            item for item in ((macro.get("unchanged_failure_analysis") or {}).get("module_implications") or [])
        ],
        "minimal_failure_point_coverage": {
            key: value for key, value in (summary_payload.get("minimal_failure_point_coverage") or {}).items()
            if key in {"required", "valid", "by_module", "by_failure_kind"}
        },
    }


def review_completion_status(macro_review: dict, causal_coverage: dict) -> tuple[str, str]:
    """Return executable review status separately from micro-review coverage."""
    macro_success = bool((macro_review or {}).get("success") is True)
    micro_complete = bool((causal_coverage or {}).get("complete") is True)
    return (
        "complete" if macro_success else "macro_review_invalid",
        "complete" if micro_complete else "partially_incomplete",
    )


class EvolutionReviewRunner:
    def __init__(self, workspace_dir: str, video_id: str,
                 evolved_results_path: str, verification_report_path: str = "",
                 candidate_bundle_path: str = "", source_diff_path: str = "",
                 out_dir: str = "", dry_run: bool = False,
                 combo_id: str = "", concurrency: int = 1,
                 observed_distribution_profile: dict = None,
                 distribution_manifest: str = "",
                 reference_results_path: str = "",
                 reference_label: str = "reference",
                 initial_reference_review: bool = False,
                 resume_review_path: str = ""):
        self.workspace_dir = workspace_dir
        self.observed_distribution_profile = observed_distribution_profile or {}
        self.distribution_manifest = distribution_manifest or _manifest_path_from_profile(
            self.observed_distribution_profile
        )
        self.video_ids = _normalize_video_ids(
            workspace_dir, video_id, self.distribution_manifest
        )
        self.video_id = video_id or ("multi_video" if len(self.video_ids) > 1 else (self.video_ids[0] if self.video_ids else ""))
        self.evolved_results_path = evolved_results_path
        self.verification_report_path = verification_report_path
        self.candidate_bundle_path = candidate_bundle_path
        self.source_diff_path = source_diff_path
        self.out_dir = out_dir or self._default_out_dir()
        self.dry_run = dry_run
        self.combo_id = combo_id
        # A micro review is independently scheduled and uses the bounded selected VLM,
        # ASR and GLM calls.  Formal review permits the requested 16 workers;
        # this is deliberately separate from execution-layer LLM concurrency.
        self.concurrency = max(
            1, min(int(concurrency or 1), META_VLM_MAX_CONCURRENT, TEACHER_REVIEW_MAX_CONCURRENT)
        )
        self.reference_results_path = reference_results_path
        self.reference_label = reference_label or "reference"
        self.initial_reference_review = bool(initial_reference_review)
        self.resume_review_path = os.path.abspath(resume_review_path) if resume_review_path else ""
        self.structure_dir = ""

    def _default_out_dir(self) -> str:
        ts = int(time.time())
        return os.path.join(runtime_output_root(), "evolution_reviews", f"{ts}_{self.video_id}")

    def run(self) -> dict:
        os.makedirs(self.out_dir, exist_ok=True)
        dataset_by_ref = _load_dataset(
            self.workspace_dir, self.video_id, self.distribution_manifest
        )
        if not self.reference_results_path and not self.initial_reference_review:
            raise RuntimeError(
                "MetaVideoAgent evolution review requires explicit --reference-results; "
                "the public workflow does not infer reference trajectories."
            )
        if self.initial_reference_review:
            baseline_by_ref, baseline_by_question = {}, {}
        else:
            baseline_by_ref, baseline_by_question = load_reference_by_ref_and_question(
                self.reference_results_path,
                reference_label=self.reference_label,
            )
        baseline_reviews = {}
        evolved_rows = _iter_jsonl(self.evolved_results_path)
        verification_report = _read_json(self.verification_report_path)
        evaluation_audit_path = (
            verification_report.get("evaluation_audit_path", "")
            or ((verification_report.get("evaluation_audit") or {}).get("path", ""))
            or os.path.join(os.path.dirname(os.path.abspath(self.verification_report_path)), "evaluation_audit.json")
        )
        evaluation_audit = _read_json(evaluation_audit_path)
        self.structure_dir = (
            verification_report.get("structure_dir", "")
            or os.path.join(verification_report.get("sandbox_dir", ""), "video_structure")
        )
        candidate_artifact = _read_json(self.candidate_bundle_path)
        if self.initial_reference_review and not candidate_artifact:
            for key in (
                "effective_initial_baseline_bundle", "initial_baseline_bundle",
                "initial_bundle_path", "bundle_path",
            ):
                value = str(verification_report.get(key) or "")
                if value and os.path.isfile(value):
                    candidate_artifact = _read_json(value)
                    break
        if not self.initial_reference_review and candidate_artifact.get("artifact_type") != "candidate_bundle":
            raise ValueError("evolution review requires a self-contained candidate_bundle artifact")
        candidate_bundle = candidate_artifact if not self.initial_reference_review else {}
        if not self.initial_reference_review:
            changed_modules = list(candidate_bundle.get("changed_modules") or [])
        else:
            changed_modules = []
        # Provider events remain reviewable trajectory evidence.  A complete
        # full eval must therefore still receive micro/macro diagnosis; only
        # a genuinely incomplete run has no stable task set to review.
        if verification_report.get("evaluation_complete", True) is not True:
            invalid = write_engineering_invalid_artifacts(
                self.out_dir, verification_report,
                source_path=self.verification_report_path,
            )
            output = {
                "timestamp": int(time.time()),
                "video_id": self.video_id,
                "video_ids": self.video_ids,
                "out_dir": self.out_dir,
                "skipped": True,
                "skip_reason": "engineering_invalid",
                "engineering_invalid_report": invalid,
                "macro_review": {
                    "success": False,
                    "skipped": True,
                    "engineering_invalid": True,
                    "verdict_reason": invalid.get("reason", ""),
                },
            }
            review_path = os.path.join(self.out_dir, "teacher_evolution_review.json")
            with open(review_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            self._write_engineering_invalid_markdown(invalid)
            output["review_path"] = review_path
            return output

        normalized = []
        for row in evolved_rows:
            if not _row_matches_combo(row, self.combo_id):
                continue
            item = _normalize_evolved_entry(row, baseline_by_question, dataset_by_ref)
            if item:
                normalized.append(item)
        normalized.sort(key=lambda x: (x.get("video_id", ""), x.get("time_reference", ""), x.get("task_id", "")))

        # Preserve only completed Teacher evidence from an exact prior review.
        # This turns a transient quota failure into a bounded replay instead of
        # discarding successful micro reviews or silently mixing trajectories.
        resumed_micro_by_task = {}
        resumed_delta_by_task = {}
        resume_meta = {}
        if self.resume_review_path:
            existing = _read_json(self.resume_review_path)
            if not existing:
                raise RuntimeError(
                    f"--resume-review is not a readable review artifact: {self.resume_review_path}"
                )
            prior_context = existing.get("candidate_context") or {}
            prior_results = str(prior_context.get("candidate_results_path") or "")
            prior_report = str(prior_context.get("verification_report_path") or "")
            if not prior_results or os.path.abspath(prior_results) != os.path.abspath(self.evolved_results_path):
                raise RuntimeError(
                    "--resume-review candidate_results_path does not match --evolved-results"
                )
            if not prior_report or os.path.abspath(prior_report) != os.path.abspath(self.verification_report_path):
                raise RuntimeError(
                    "--resume-review verification_report_path does not match --verification-report"
                )
            for micro in existing.get("evolved_micro_reviews") or []:
                if not isinstance(micro, dict):
                    continue
                task_id = str(micro.get("task_id") or "")
                # Dry-run rows are preflight placeholders, not completed
                # Teacher evidence.  They must be replayed rather than
                # suppressing a real retry after a provider failure.
                if (
                    task_id
                    and micro.get("success") is True
                    and micro.get("dry_run") is not True
                ):
                    if task_id in resumed_micro_by_task:
                        raise RuntimeError(f"--resume-review has duplicate successful task_id: {task_id}")
                    resumed_micro_by_task[task_id] = micro
            for delta in existing.get("delta_reviews") or []:
                if isinstance(delta, dict) and delta.get("success") is True:
                    task_id = str(delta.get("task_id") or "")
                    if task_id:
                        resumed_delta_by_task[task_id] = delta
            current_task_ids = {str(item.get("task_id") or "") for item in normalized}
            unknown_task_ids = sorted(set(resumed_micro_by_task) - current_task_ids)
            if unknown_task_ids:
                raise RuntimeError(
                    "--resume-review has successful task_ids absent from current evolved results: "
                    + ", ".join(unknown_task_ids[:3])
                )
            resume_meta = {
                "source_review": self.resume_review_path,
                "source_timestamp": existing.get("timestamp"),
                "preserved_successful_micro_reviews": len(resumed_micro_by_task),
                "preserved_successful_delta_reviews": len(resumed_delta_by_task),
            }

        current_best_override = verification_report.get("current_best_override") or {}
        accepted_current_best = bool(
            (verification_report.get("post_full_eval_decision") or {}).get("update_current_best") is True
            or (
                isinstance(current_best_override, dict)
                and current_best_override.get("accepted") is True
            )
        )
        evolution_info = {
            "target_modules": changed_modules,
            "strategy": (candidate_bundle.get("implementation_spec") or {}).get("contract_mode", ""),
            "changed_modules": changed_modules,
            "pre_evolution_role": "current_best_before_evolution",
            "post_evolution_status": (
                "accepted_new_current_best"
                if accepted_current_best
                else "rejected_candidate_under_review"
            ),
            "post_evolution_role": (
                "current_best_after_evolution"
                if accepted_current_best
                else "rejected_post_evolution_candidate"
            ),
        }
        if self.initial_reference_review:
            evolution_info.update({
                "review_mode": "initial_reference_single_baseline",
                "target_modules": [],
                "pre_evolution_role": "not_applicable_no_predecessor",
                "post_evolution_role": "initial_reference_execution",
                "post_evolution_status": "initial_reference_under_review",
                "comparison_available": False,
                "current_best_semantics": (
                    "This is the unique executed current best. There is no "
                    "pre-evolution predecessor and no candidate delta to infer."
                ),
            })

        change_counts = {
            "fix": 0, "regression": 0,
            "keep_correct": 0, "keep_wrong": 0,
        }

        review_inputs = []
        preserved_results = []
        for idx, item in enumerate(normalized):
            task_id = item.get("task_id", "")
            baseline_entry = baseline_by_ref.get(task_id, {})
            baseline_review = baseline_reviews.get(task_id)
            ct = (_change_type(baseline_entry, item) if baseline_entry else
                  ("keep_correct" if _answer_is_correct(item.get("answer", ""), item.get("raw") or item) else "keep_wrong"))
            if ct in change_counts:
                change_counts[ct] += 1
            prior_micro = resumed_micro_by_task.get(str(task_id))
            # Reuse only evidence whose authoritative comparison outcome is
            # still identical; otherwise rerun the task instead of combining
            # incompatible review states.
            if prior_micro is not None and prior_micro.get("change_type") == ct:
                preserved_results.append(
                    (idx, prior_micro, resumed_delta_by_task.get(str(task_id)))
                )
            else:
                review_inputs.append((idx, item, baseline_entry, baseline_review, ct))

        def review_one(payload):
            idx, item, baseline_entry, baseline_review, ct = payload
            tr = item["time_reference"]
            try:
                return _review_one_inner(payload)
            except Exception as exc:
                micro = {
                    "success": False,
                    "review_error": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                    "task_id": item.get("task_id", ""),
                    "video_id": item.get("video_id", ""),
                    "time_reference": tr,
                    "question": item.get("question", ""),
                    "gt_answer": item.get("gt_answer", ""),
                    "evolved_answer": item.get("answer", ""),
                    "evolved_correct": _answer_is_correct(
                        item.get("answer", ""), item.get("raw") or item
                    ),
                    "answer_status": "review_failed",
                    "evidence_quality": "review_failed",
                    "stability": "review_failed",
                    "root_cause": f"Teacher micro-review failed: {type(exc).__name__}: {exc}",
                    "change_type": ct,
                }
                return idx, _sanitize_teacher_output_terms(micro, preserve_trace=False), None

        def _review_one_inner(payload):
            idx, item, baseline_entry, baseline_review, ct = payload
            tr = item["time_reference"]
            evolved_correct = _answer_is_correct(
                item.get("answer", ""), item.get("raw") or item
            )
            reference_correct = (
                _answer_is_correct(
                    baseline_entry.get("final_agent_answer")
                    or baseline_entry.get("answer", ""),
                    baseline_entry,
                )
                if baseline_entry else None
            )
            if not baseline_entry and not self.initial_reference_review:
                micro = {
                    "success": False,
                    "review_error": "missing_reference_trajectory",
                    "error_message": "candidate row has no matching current-best trajectory",
                    "task_id": item.get("task_id", ""),
                    "video_id": item.get("video_id", ""),
                    "time_reference": tr,
                    "change_type": "unknown",
                }
            elif self.dry_run:
                micro = {
                    "success": True,
                    "dry_run": True,
                    "task_id": item.get("task_id", ""),
                    "video_id": item.get("video_id", ""),
                    "time_reference": tr,
                    "question": item["question"],
                    "gt_answer": item["gt_answer"],
                    "evolved_answer": item.get("answer", ""),
                    "evolved_correct": _answer_is_correct(
                        item.get("answer", ""), item.get("raw") or item
                    ),
                    "answer_status": "not_reviewed",
                    "evidence_quality": "not_reviewed",
                    "stability": "not_reviewed",
                    "root_cause": "dry-run: model review skipped",
                }
            else:
                teacher = TeacherAgent(
                    self.workspace_dir,
                    item.get("video_id", self.video_ids[0] if self.video_ids else self.video_id),
                    structure_dir=self.structure_dir,
                )
                role_by_change = {
                    "fix": "candidate_correction_evidence",
                    "regression": "regression_guard",
                    "keep_wrong": "unchanged_failure_evidence",
                    "keep_correct": "stable_no_failure",
                }
                item_evolution_info = dict(evolution_info)
                item_evolution_info["expected_outcome_role"] = role_by_change.get(
                    ct, "repair_evidence"
                )
                item_evolution_info["authoritative_change_type"] = ct
                item_evolution_info["authoritative_evolved_correct"] = evolved_correct
                item_evolution_info["authoritative_reference_correct"] = reference_correct
                micro = teacher.review_evolved_trajectory(
                    question=item["question"],
                    gt_answer=item["gt_answer"],
                    time_reference=tr,
                    evolved_trajectory=item.get("trajectory", []),
                    evolved_answer=item.get("answer", ""),
                    evolution_info=item_evolution_info,
                    baseline_review=baseline_review,
                    reference_trajectory=(baseline_entry.get("trajectory", []) if baseline_entry else []),
                    reference_answer=(
                        baseline_entry.get("final_agent_answer")
                        or baseline_entry.get("answer", "")
                    ) if baseline_entry else "",
                    capability_events=item.get("raw", {}).get("capability_events", []),
                )
            micro["task_id"] = item.get("task_id", "")
            micro["video_id"] = item.get("video_id", "")
            micro["time_reference"] = tr
            micro["change_type"] = ct
            micro["review_mode"] = "initial_reference_single_baseline" if self.initial_reference_review else "paired_candidate_comparison"
            micro["current_best_outcome"] = (
                ("current_best_correct" if evolved_correct else "current_best_wrong")
                if self.initial_reference_review else ct
            )
            micro["comparison_semantics"] = (
                "unique_current_best_no_predecessor"
                if self.initial_reference_review else "paired_candidate_comparison"
            )
            micro["reference_available"] = bool(baseline_entry)
            micro["reference_task_id"] = item.get("task_id", "") if baseline_entry else ""
            micro = apply_authoritative_answer_metadata(micro, item)
            micro = align_deterministic_causal_role(micro)
            mfp_issues = minimal_failure_point_issues(
                micro,
                trajectory=item.get("trajectory", []),
                capability_events=item.get("raw", {}).get("capability_events", []),
            )
            points = micro.get("minimal_failure_points") or []
            point_validations = []
            for index, point in enumerate(points if isinstance(points, list) else []):
                prefix = f"minimal_failure_points[{index}]"
                point_issues = [issue for issue in mfp_issues if issue.startswith(prefix)]
                # Shape-level failures without an indexed prefix make the
                # affected point unusable; a bad sibling must not.
                if not isinstance(point, dict):
                    point_issues = point_issues or ["point is not an object"]
                point_validations.append({
                    "index": index,
                    "mfp_id": str(point.get("mfp_id") or "") if isinstance(point, dict) else "",
                    "valid": not point_issues,
                    "issues": point_issues,
                })
            micro["minimal_failure_point_validation"] = {
                "valid": not mfp_issues,
                "issues": mfp_issues,
                "point_validations": point_validations,
                "direct_repair_eligible": any(item["valid"] for item in point_validations),
            }
            action_issues = concrete_repair_action_issues(micro)
            action_validations = []
            for index, action in enumerate(micro.get("concrete_repair_actions") or []):
                if not isinstance(action, dict):
                    action_validations.append({
                        "index": index, "valid": False,
                        "issues": ["action is not an object"],
                    })
                    continue
                one_action_micro = dict(micro)
                one_action_micro["concrete_repair_actions"] = [action]
                one_action_issues = concrete_repair_action_issues(one_action_micro)
                action_validations.append({
                    "index": index,
                    "valid": not one_action_issues,
                    "issues": one_action_issues,
                })
            micro["concrete_repair_action_validation"] = {
                "valid": not action_issues,
                "issues": action_issues,
                "action_count": len(micro.get("concrete_repair_actions") or []),
                "action_validations": action_validations,
            }
            teacher_chain = micro.get("failure_chain")
            teacher_chain_issues = failure_chain_issues(
                teacher_chain, trajectory=item.get("trajectory", [])
            )
            # Never replace an invalid Teacher attribution with a synthetic
            # causal diagnosis.  Keep a deterministic producer flow only as
            # audit context; it is not promoted to a valid failure chain.
            if teacher_chain_issues:
                micro["failure_chain"] = []
                micro["deterministic_observed_module_flow"] = derive_failure_chain(
                    item.get("trajectory", []), points if isinstance(points, list) else [],
                )
            else:
                micro["failure_chain"] = teacher_chain
                micro["deterministic_observed_module_flow"] = []
            micro["failure_chain_validation"] = {
                "valid": not teacher_chain_issues,
                "issues": teacher_chain_issues,
                "roles": {
                    row.get("module"): row.get("role")
                    for row in (teacher_chain or []) if isinstance(row, dict)
                },
            }

            delta = None
            if ct in ("fix", "regression") and baseline_entry:
                if self.dry_run:
                    delta = {
                        "success": True,
                        "dry_run": True,
                        "time_reference": tr,
                        "change_type": ct,
                        "diff_summary": "dry-run: delta model review skipped",
                        "reference_answer": (
                            baseline_entry.get("final_agent_answer")
                            or baseline_entry.get("answer", "")
                        ),
                        "evolved_answer": item.get("answer", ""),
                    }
                else:
                    # Use a fresh TeacherAgent for each worker to avoid shared client state.
                    teacher = TeacherAgent(
                        self.workspace_dir,
                        item.get("video_id", self.video_ids[0] if self.video_ids else self.video_id),
                        structure_dir=self.structure_dir,
                    )
                    delta = teacher.compare_evolution(
                        question=item["question"],
                        gt_answer=item["gt_answer"],
                        time_reference=tr,
                        baseline_trajectory=baseline_entry.get("trajectory", []),
                        evolved_trajectory=item.get("trajectory", []),
                        reference_answer=(
                            baseline_entry.get("final_agent_answer")
                            or baseline_entry.get("answer", "")
                        ),
                        evolved_answer=item.get("answer", ""),
                        evolution_info=evolution_info,
                    )
                delta["task_id"] = item.get("task_id", "")
                delta["video_id"] = item.get("video_id", "")
                delta["time_reference"] = tr
                delta["change_type"] = ct
            micro = _sanitize_teacher_output_terms(micro, preserve_trace=False)
            delta = _sanitize_teacher_output_terms(delta, preserve_trace=False) if delta is not None else None
            return idx, micro, delta

        review_results = list(preserved_results)
        if self.concurrency == 1 or len(review_inputs) <= 1:
            for payload in review_inputs:
                review_results.append(review_one(payload))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = [pool.submit(review_one, payload) for payload in review_inputs]
                for future in concurrent.futures.as_completed(futures):
                    review_results.append(future.result())

        review_results.sort(key=lambda x: x[0])
        evolved_micro_reviews = [micro for _, micro, _ in review_results]
        for micro in evolved_micro_reviews:
            if micro.get("change_type") == "keep_wrong":
                micro["unchanged_failure_classification"] = (
                    micro.get("unchanged_failure_classification")
                    or classify_unchanged_failure(micro)
                )
        delta_reviews = [delta for _, _, delta in review_results if delta is not None]
        causal_coverage = causal_effect_coverage(evolved_micro_reviews)
        mfp_coverage = minimal_failure_point_coverage(evolved_micro_reviews)
        failure_chain_distribution = {}
        for micro in evolved_micro_reviews:
            if not (micro.get("failure_chain_validation") or {}).get("valid"):
                continue
            chain = micro.get("failure_chain") or []
            modules = tuple(
                f"{item.get('module')}:{item.get('role')}" for item in chain
                if isinstance(item, dict) and item.get("module") and item.get("role")
            )
            if modules:
                failure_chain_distribution[" → ".join(modules)] = (
                    failure_chain_distribution.get(" → ".join(modules), 0) + 1
                )

        code_context = self._build_code_context(candidate_artifact)
        metrics = {
            "total": len(normalized),
            **change_counts,
            "net_delta": change_counts["fix"] - change_counts["regression"],
            "verification_accuracy_delta": verification_report.get("accuracy_delta"),
            "cost_audit": {
                "path": evaluation_audit_path if os.path.exists(evaluation_audit_path) else "",
                "cost_ratio_vs_reference": evaluation_audit.get("cost_ratio_vs_reference"),
                "candidate_cost": evaluation_audit.get("candidate_cost", {}),
                "reference_cost": evaluation_audit.get("reference_cost", {}),
                "issue_summary": evaluation_audit.get("issue_summary", []),
                "recommended_next_focus": (
                    evaluation_audit.get("decision_guidance", {}) or {}
                ).get("recommended_next_focus", ""),
            },
        }
        per_video_metrics = {}
        for item, micro in zip(normalized, evolved_micro_reviews):
            vid = item.get("video_id", "")
            stat = per_video_metrics.setdefault(
                vid,
                {"total": 0, "fix": 0, "regression": 0, "keep_correct": 0, "keep_wrong": 0},
            )
            stat["total"] += 1
            ct = micro.get("change_type")
            if ct in stat:
                stat[ct] += 1
        unchanged_failure_summary = summarize_unchanged_failures(
            evolved_micro_reviews
        )
        retention_review = build_retention_review(
            metrics, evolved_micro_reviews, delta_reviews, evolution_info
        )
        macro_evidence_ledger = build_macro_evidence_ledger(
            evolved_micro_reviews, delta_reviews,
            initial_reference=self.initial_reference_review,
        )
        reference_state, candidate_state, current_best_state = _review_state_contexts(
            verification_report=verification_report,
            verification_report_path=self.verification_report_path,
            evolved_results_path=self.evolved_results_path,
            reference_results_path=self.reference_results_path,
            initial_reference_review=self.initial_reference_review,
            candidate_bundle_path=self.candidate_bundle_path,
            combo_id=self.combo_id,
        )
        summary_payload = {
            "metrics": metrics,
            "observed_distribution_profile": self.observed_distribution_profile,
            "distribution_leakage_policy": (
                "Use only the label-blind observed distribution profile. Do not name "
                "or infer human semantic video types."
            ) if self.observed_distribution_profile else "",
            "reference_context": {
                "cached_micro_reviews": len(baseline_reviews),
                "reference_label": self.reference_label,
                **reference_state,
                "comparison_roles": {
                    "reference": (
                        "not_applicable_unique_initial_current_best"
                        if self.initial_reference_review else "current_best_before_evolution"
                    ),
                    "evolved": evolution_info.get("post_evolution_role"),
                },
                "reference_results_path": self.reference_results_path,
                "reference_combo_source": (
                    self.evolved_results_path if self.initial_reference_review
                    else self.reference_results_path or "explicit_reference_required"
                ),
            },
            "history_input_policy": _external_history_meta(
                self.reference_results_path,
                self.evolved_results_path,
                self.verification_report_path,
                self.candidate_bundle_path,
                self.source_diff_path,
            ),
            "candidate_context": {
                "initial_reference_only": self.initial_reference_review,
                "semantic_role": (
                    "unique_current_best_execution" if self.initial_reference_review
                    else "accepted_current_best_under_review" if accepted_current_best
                    else "candidate_under_review"
                ),
                "candidate_results_path": self.evolved_results_path,
                "verification_report_path": self.verification_report_path,
                "combo_id_filter": self.combo_id,
                "evolution_info": evolution_info,
                **candidate_state,
            },
            "current_best_context": current_best_state,
            "reviewed_execution_context": {
                "role": (
                    "unique_current_best_execution" if self.initial_reference_review
                    else "post_evolution_candidate_execution"
                ),
                "comparison_available": not self.initial_reference_review,
                "results_path": self.evolved_results_path,
                "combo_id": self.combo_id,
            },
            "per_video_metrics": per_video_metrics,
            "evaluation_audit": evaluation_audit,
            "code_context": code_context,
            "evolved_micro_reviews": evolved_micro_reviews,
            "macro_evidence_ledger": macro_evidence_ledger,
            "causal_effect_coverage": causal_coverage,
            "minimal_failure_point_coverage": mfp_coverage,
            "failure_chain_distribution": failure_chain_distribution,
            "unchanged_failure_summary": unchanged_failure_summary,
            "delta_reviews": delta_reviews,
            "retention_review": retention_review,
        }
        if resume_meta:
            summary_payload["micro_review_resume"] = {
                **resume_meta,
                "rerun_micro_reviews": len(review_inputs),
                "merged_micro_reviews": len(evolved_micro_reviews),
            }
        if self.dry_run:
            macro_review = {
                "success": True,
                "dry_run": True,
                "overall_verdict": "not_reviewed",
                "verdict_reason": "dry-run: macro model review skipped",
            }
        else:
            teacher = TeacherAgent(
                self.workspace_dir,
                self.video_ids[0] if self.video_ids else self.video_id,
                structure_dir=self.structure_dir,
            )
            macro_review = teacher.summarize_post_evolution(summary_payload)
        macro_review = ensure_programmable_macro_fields(
            macro_review, summary_payload
        )
        macro_validation_issues = macro_review_issues(macro_review, summary_payload)
        macro_review["validation"] = {
            "valid": not macro_validation_issues,
            "issues": macro_validation_issues,
            "policy": "invalid macro fields are audit omissions; deterministic ledger remains available to diagnosis",
        }

        output = {
            "timestamp": int(time.time()),
            "video_id": self.video_id,
            "video_ids": self.video_ids,
            "out_dir": self.out_dir,
            **summary_payload,
            "macro_review": macro_review,
            "mechanism_summary": build_mechanism_summary(summary_payload, macro_review),
        }
        # A successful macro review is the executable completion boundary.
        # Weak/incomplete micro reviews remain explicit audit omissions but
        # must not block diagnosis from using the valid paired evidence that
        # did arrive.
        output["review_status"], output["micro_review_status"] = review_completion_status(
            macro_review, causal_coverage,
        )
        output = refresh_review_deterministic_aggregates(output)
        output = _sanitize_teacher_output_terms(output, preserve_trace=False)

        review_path = os.path.join(self.out_dir, "teacher_evolution_review.json")
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        self._write_markdown(output)
        output["review_path"] = review_path
        return output

    def _build_code_context(self, candidate_artifact: dict) -> dict:
        diff_text = ""
        if self.source_diff_path and os.path.exists(self.source_diff_path):
            with open(self.source_diff_path, "r", encoding="utf-8") as f:
                diff_text = f.read(6000)
        if candidate_artifact:
            modules = candidate_artifact.get("modules") or {}
            if isinstance(modules, list):
                modules = {
                    str(item.get("module_type") or ""): item
                    for item in modules if isinstance(item, dict)
                }
            changed = candidate_artifact.get("changed_modules") or list(modules)
            return {
                "candidate_name": "MetaVideoAgent bundle",
                "module_type": "bundle",
                "changed_modules": changed,
                "handoff_contracts": candidate_artifact.get("handoff_contracts") or [],
                "candidate_code_heads": {
                    item: str((modules.get(item) or {}).get("code") or "")[:3000]
                    for item in changed
                },
                "source_diff_head": diff_text,
            }
        return {"candidate_name": "", "module_type": "bundle", "source_diff_head": diff_text}

    def run_macro_only(self, existing_review_path: str) -> dict:
        """Re-run only the Teacher macro review from a completed review artifact."""
        existing = _read_json(existing_review_path)
        required = (
            "metrics", "evolved_micro_reviews", "causal_effect_coverage",
            "delta_reviews", "candidate_context",
        )
        missing = [key for key in required if key not in existing]
        if missing:
            raise RuntimeError(
                "macro-only review requires a completed review artifact with "
                f"{missing}; refusing to regenerate or infer micro reviews"
            )
        source_micro = existing.get("evolved_micro_reviews") or []
        if not source_micro:
            raise RuntimeError("macro-only review source has no evolved_micro_reviews")
        # Weak Teacher micro reviews are audit omissions, not a reason to block
        # the macro synthesis or later diagnosis. The downstream ledger keeps
        # only complete causal records.
        summary_payload = {
            key: existing.get(key)
            for key in (
                "metrics", "observed_distribution_profile",
                "distribution_leakage_policy", "reference_context",
                "history_input_policy", "candidate_context", "current_best_context", "reviewed_execution_context", "per_video_metrics",
                "evaluation_audit", "code_context", "evolved_micro_reviews", "causal_effect_coverage",
                "minimal_failure_point_coverage", "macro_evidence_ledger",
                "unchanged_failure_summary", "delta_reviews", "retention_review",
            )
        }
        summary_payload["unchanged_failure_summary"] = summarize_unchanged_failures(source_micro)
        if self.dry_run:
            macro_review = {
                "success": True,
                "dry_run": True,
                "overall_verdict": "not_reviewed",
                "verdict_reason": "dry-run: macro model review skipped",
            }
        else:
            teacher = TeacherAgent(
                self.workspace_dir,
                self.video_ids[0] if self.video_ids else self.video_id,
                structure_dir=self.structure_dir,
            )
            macro_review = teacher.summarize_post_evolution(summary_payload)
        macro_review = ensure_programmable_macro_fields(macro_review, summary_payload)
        macro_validation_issues = macro_review_issues(macro_review, summary_payload)
        macro_review["validation"] = {
            "valid": not macro_validation_issues,
            "issues": macro_validation_issues,
            "policy": "invalid macro fields are audit omissions; deterministic ledger remains available to diagnosis",
        }
        output = {
            "timestamp": int(time.time()),
            "video_id": existing.get("video_id", self.video_id),
            "video_ids": existing.get("video_ids", self.video_ids),
            "out_dir": self.out_dir,
            **summary_payload,
            "macro_review": macro_review,
            "mechanism_summary": build_mechanism_summary(summary_payload, macro_review),
            "macro_only_source_review": {
                "path": os.path.abspath(existing_review_path),
                "source_timestamp": existing.get("timestamp"),
                "micro_review_count": len(source_micro),
                "delta_review_count": len(existing.get("delta_reviews") or []),
            },
        }
        output["review_status"] = (
            "complete" if macro_review.get("success") is True else "macro_review_invalid"
        )
        output["micro_review_status"] = existing.get("micro_review_status", "")
        output = refresh_review_deterministic_aggregates(output)
        output = _sanitize_teacher_output_terms(output, preserve_trace=False)
        review_path = os.path.join(self.out_dir, "teacher_evolution_review.json")
        os.makedirs(self.out_dir, exist_ok=True)
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        self._write_markdown(output)
        output["review_path"] = review_path
        return output

    def _write_markdown(self, output: dict) -> None:
        macro = output.get("macro_review", {})
        metrics = output.get("metrics", {})
        lines = [
            "# Teacher Evolution Review",
            "",
            f"- Total: {metrics.get('total')}",
            f"- Fixes: {metrics.get('fix')}",
            f"- Regressions: {metrics.get('regression')}",
            f"- Keep correct: {metrics.get('keep_correct')}",
            f"- Keep wrong: {metrics.get('keep_wrong')}",
            f"- Net delta: {metrics.get('net_delta')}",
            f"- Cost ratio vs reference: `{(metrics.get('cost_audit') or {}).get('cost_ratio_vs_reference')}`",
            f"- Cost recommended focus: `{(metrics.get('cost_audit') or {}).get('recommended_next_focus')}`",
            "",
            "## Cost Audit Issues",
            "",
            json.dumps((metrics.get("cost_audit") or {}).get("issue_summary", []), ensure_ascii=False, indent=2),
            "",
            "## Macro Review",
            "",
            json.dumps(macro, ensure_ascii=False, indent=2),
        ]
        path = os.path.join(self.out_dir, "teacher_evolution_macro.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _write_engineering_invalid_markdown(self, invalid: dict) -> None:
        lines = [
            "# Teacher Evolution Review Skipped",
            "",
            "- reason: engineering/API invalid evaluation result",
            f"- source_report: `{invalid.get('source_report_path', '')}`",
            f"- invalid_report: `{invalid.get('engineering_invalid_report_path', '')}`",
            "",
            "## Issue Types",
            "",
            json.dumps(invalid.get("issue_types", []), ensure_ascii=False, indent=2),
            "",
            "## Fix Guidance",
            "",
            json.dumps(invalid.get("fix_guidance", {}), ensure_ascii=False, indent=2),
        ]
        path = os.path.join(self.out_dir, "teacher_evolution_macro.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run post-evolution Teacher review")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--video-id", default="ALL")
    parser.add_argument("--video-ids", nargs="*", default=[],
                        help="Optional multi-video set. Overrides --video-id when provided.")
    parser.add_argument("--evolved-results", default="")
    parser.add_argument("--verification-report", default="")
    parser.add_argument("--candidate-bundle", default="")
    parser.add_argument("--source-diff", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--combo-id", default="",
                        help="Only review evolved rows with this combo_id. Use this when a sandbox jsonl contains multiple runs.")
    parser.add_argument("--concurrency", type=int, default=min(META_VLM_MAX_CONCURRENT, TEACHER_REVIEW_MAX_CONCURRENT),
                        help=("Parallel micro-review workers (default/ceiling: 16); "
                              "macro review remains serialized."))
    parser.add_argument("--dry-run", action="store_true",
                        help="Only normalize inputs and write review files; do not call model APIs.")
    parser.add_argument("--distribution-profile", default="",
                        help="Optional label-blind observed_distribution_profile.json for macro review context.")
    parser.add_argument("--distribution-manifest", required=True,
                        help="Distribution manifest; human descriptions are stripped before profiling.")
    parser.add_argument("--reference-results", default="",
                        help="Explicit reference/current-best results JSONL or sandbox dir.")
    parser.add_argument("--reference-label", default="reference",
                        help="Reference label used for delta review, e.g. initial_combo/current_best.")
    parser.add_argument("--initial-reference-review", action="store_true",
                        help="Review one executed initial reference without inventing a pre/post self comparison.")
    parser.add_argument("--macro-only-from", default="",
                        help="Completed teacher_evolution_review.json whose micro/delta reviews are reused; only reruns macro review.")
    parser.add_argument("--resume-review", default="",
                        help=("Matching completed review artifact. Preserve its success=true micro reviews, "
                              "rerun only failed/incomplete tasks, then regenerate macro review."))
    args = parser.parse_args()
    require_metavideoagent_runtime("evolution_review_runner")
    if not args.initial_reference_review and not args.candidate_bundle:
        parser.error("MetaVideoAgent review requires --candidate-bundle")
    if not args.macro_only_from and not args.evolved_results:
        parser.error("--evolved-results is required unless --macro-only-from is supplied")
    observed_distribution_profile = _load_or_build_distribution_profile(
        args.workspace, args.distribution_manifest, args.distribution_profile)

    runner = EvolutionReviewRunner(
        workspace_dir=args.workspace,
        video_id=",".join(args.video_ids) if args.video_ids else args.video_id,
        evolved_results_path=args.evolved_results,
        verification_report_path=args.verification_report,
        candidate_bundle_path=args.candidate_bundle,
        source_diff_path=args.source_diff,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        combo_id=args.combo_id,
        concurrency=args.concurrency,
        observed_distribution_profile=observed_distribution_profile,
        distribution_manifest=args.distribution_manifest,
        reference_results_path=args.reference_results,
        reference_label=args.reference_label,
        initial_reference_review=args.initial_reference_review,
        resume_review_path=args.resume_review,
    )
    result = runner.run_macro_only(args.macro_only_from) if args.macro_only_from else runner.run()
    print(json.dumps({
        "review_path": result.get("review_path"),
        "metrics": result.get("metrics"),
    }, ensure_ascii=False, indent=2))
    if result.get("review_status") != "complete":
        # A missing/invalid macro review is not consumable. Incomplete micros
        # are already retained as audit omissions under micro_review_status.
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
