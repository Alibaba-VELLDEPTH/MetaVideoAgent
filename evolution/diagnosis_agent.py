"""evolution LLM diagnosis and deterministic brief inputs for MetaVideoAgent evolution.

Diagnosis consumes already-completed review, current-best trajectories,
run-scoped memory, and optional probe-rejection evidence. It never launches
Teacher review, reads mutable question-review caches, or inspects raw media.
The first LLM selects one target and writes an audited evolution design; the
second LLM compiles the target-locked implementation brief.
"""

import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)
from runtime_capability_context import (
    DEFAULT_EXECUTION_LLM_PROFILE_ID,
    DEFAULT_VLM_PROFILE_ID,
    build_runtime_capability_context,
)
from runtime_paths import (
    default_workspace,
    metavideoagent_run_dir,
    require_metavideoagent_runtime,
    runtime_output_root,
    use_runtime_path,
)

_RUNTIME_DIR = use_runtime_path()

from config import (
    EVOLUTION_LLM_API_KEY,
    EVOLUTION_LLM_BASE_URL,
    EVOLUTION_LLM_MODEL,
    EVOLUTION_LLM_REQUEST_OPTIONS,
)
from review_contract import review_is_consumable
from target_selection import build_target_selection, canonical_task_ref
from task_identity import (
    identity_from_entry,
    make_task_id,
)
from task_sanitizer import sanitize_task_for_evolution

TEACHER_ONLY_TOOL_TERMS = {
    "DIAG_VIDEO_AUDIT": "Video review",
    "DIAG_DETAIL_VIDEO_AUDIT": "Detailed video review",
    "AUDIT_VIDEO": "Video review",
    "AUDIT_DETAIL_VIDEO": "Detailed video review",
    "AUDIT_ENHANCED_VIDEO": "Enhanced video review",
    "AUDIT_STRUCTURE": "Structure library review",
    "AUDIT_TRANSCRIPT": "Subtitle/audio text review",
    "watch_video": "Video review",
    "video_review": "Video review",
    "teacher_video_verification": "Video review",
    "detailed_watch": "Detailed video review",
    "detail_video_review": "Detailed video review",
    "fine_video_review": "Detailed video review",
    "teacher_detailed_video_verification": "Detailed video review",
    "enhanced_watch": "Enhanced video review",
    "enhanced_video_review": "Enhanced video review",
    "teacher_enhanced_video_verification": "Enhanced video review",
    "browse_struct_db": "Structure library review",
    "structure_review": "Structure library review",
    "teacher_structure_db_audit": "Structure library review",
    "read_transcript": "Subtitle/audio text review",
    "transcript_review": "Subtitle/audio text review",
    "teacher_transcript_audit": "Subtitle/audio text review",
    "listen_audio": "Subtitle/audio text review",
}

DISALLOWED_RUNTIME_TOOL_TERMS = {
    "frame_inspect": "visual perception ability",
    "single_frame_inspect": "visual perception ability",
    "inspect_frames": "visual perception ability",
    "runtime_visual_observation_tool": "visual perception ability",
    "runtime_visual_observation_capability": "visual perception ability",
    "clip_search": "Structure retrieval/location capabilities",
    "frame_clip_search": "Structure retrieval/location capabilities",
    "search_evidence": "Evidence retrieval/location capabilities",
    "runtime_localization_or_structure_retrieval_tool": "Structure retrieval/location capabilities",
    "runtime_localization_or_structure_retrieval_capability": "Structure retrieval/location capabilities",
    "runtime_evidence_retrieval_capability": "Evidence retrieval/location capabilities",
    "global_browse": "Global structure browsing capabilities",
    "runtime_broad_structure_retrieval_tool": "Global structure browsing capabilities",
    "runtime_broad_structure_retrieval_capability": "Global structure browsing capabilities",
    "audio_analysis": "Audio/subtitle understanding ability",
    "runtime_audio_or_transcript_evidence_tool": "Audio/subtitle understanding ability",
    "runtime_audio_or_transcript_capability": "Audio/subtitle understanding ability",
}

def _normalize_runtime_capability_needs(needs: object) -> list:
    """Bind diagnosis capability declarations to their registered contracts.

    Profile selection is still part of the diagnosis design, but a profile's
    input provenance is an execution-layer fact rather than free-form LLM
    prose.  Normalizing it here prevents harmless explanatory suffixes (for
    example ``"bounded_text_context (structured evidence)"``) from making an
    otherwise valid diagnosis impossible to compile into a brief.
    """
    if not isinstance(needs, list):
        return []
    profiles = {
        str(profile.get("profile_id") or ""): profile
        for profile in build_runtime_capability_context().get("profiles", [])
        if isinstance(profile, dict) and profile.get("profile_id")
    }
    normalized = []
    for raw_need in needs:
        if not isinstance(raw_need, dict):
            normalized.append(raw_need)
            continue
        need = raw_need
        capability = str(need.get("capability") or "")
        if capability == "vlm":
            need["profile_id"] = DEFAULT_VLM_PROFILE_ID
        elif capability == "llm":
            # The configured diagnosis model is not an execution-combo
            # capability. Generated thinking code must keep GLM5.2 as the
            # final decision maker.
            need["profile_id"] = DEFAULT_EXECUTION_LLM_PROFILE_ID
        profile = profiles.get(str(need.get("profile_id") or ""))
        if profile and str(profile.get("capability") or "") == str(need.get("capability") or ""):
            provenance = str(profile.get("input_provenance") or "")
            if provenance:
                need["input_provenance"] = provenance
        normalized.append(need)
    return normalized


def _ledger_task_refs(ledger: dict, key: str) -> list:
    """Return stable, de-duplicated task references from one ledger bucket."""
    refs = []
    for item in (ledger.get(key) or []):
        value = (
            item.get("task_ref") or item.get("task_id") or item.get("time_reference")
            if isinstance(item, dict) else item
        )
        value = str(value or "").strip()
        if value and value not in refs:
            refs.append(value)
    return refs


def _trace_refs_for_targets(ledger: dict, targets: list[str]) -> list:
    """Return machine-only trace-supported rows for the selected direction."""
    refs = []
    for cluster in ((ledger.get("weak_failure_distribution") or {}).get("clusters") or []):
        if not isinstance(cluster, dict) or str(cluster.get("suggested_module") or "") not in targets:
            continue
        for ref in cluster.get("refs") or []:
            ref = str(ref or "").strip()
            if ref and ref not in refs:
                refs.append(ref)
    return refs


def complete_evidence_strategy(design: dict, ledger: dict, *, targets: list[str]) -> dict:
    """Complete the machine-only evidence contract from the factual ledger.

    The LLM selects one or more modules and the causal design. It must not be trusted to
    reproduce every historical task reference individually, because those
    refs are evaluation-only and never reach research or Codex prompts. This
    helper preserves any explanatory text the LLM supplied while deriving exhaustive
    reference coverage from the already validated ledger.
    """
    completed = dict(design or {})
    ledger = ledger if isinstance(ledger, dict) else {}
    strategy = dict(completed.get("evidence_strategy") or {})
    comparisons = [
        (ledger.get("target_comparison") or {}).get(str(target), {})
        for target in targets
    ]
    if not comparisons or any(not comparison for comparison in comparisons):
        raise ValueError("evidence ledger is missing target_comparison for a selected module")
    trace_refs = _trace_refs_for_targets(ledger, targets)
    direct_mfp_refs = list(dict.fromkeys(
        str(ref).strip()
        for comparison in comparisons
        for ref in ((comparison.get("direct_repair") or {}).get("refs") or [])
        if str(ref).strip()
    ))
    # Broad trace distribution selects the diagnosis mechanism. MFP rows are
    # supplementary probe witnesses, never target-selection votes.
    if trace_refs:
        target_refs = trace_refs
        evidence_basis = "distribution_mechanism"
    elif direct_mfp_refs:
        target_refs = direct_mfp_refs
        evidence_basis = "narrow_mfp_repair"
    else:
        stabilization_refs = list(dict.fromkeys(
            str(ref).strip()
            for comparison in comparisons
            for ref in ((comparison.get("direct_regression_stabilization") or {}).get("refs") or [])
            if str(ref).strip()
        ))
        if stabilization_refs:
            # Regressions remain guard-only operationally.  The separate
            # basis records why this module may be stabilized without
            # relabelling a regression as a repair target.
            target_refs = []
            evidence_basis = "regression_stabilization"
        else:
            # Do not silently attach unrelated current-best failures to an
            # exploratory target.  A missing LLM rationale must fail brief
            # validation rather than becoming a fabricated justification.
            evidence_basis = "bounded_exploration"
            target_refs = []
    corrections = _ledger_task_refs(ledger, "candidate_correction_evidence")
    regressions = _ledger_task_refs(ledger, "regression_guards")
    unchanged = _ledger_task_refs(ledger, "unchanged_failure_evidence")
    causal = str(completed.get("causal_hypothesis") or completed.get("design_summary") or "").strip()

    # Exact coverage is deterministic; only the LLM's causal explanations are
    # retained as natural-language content.
    strategy["target_evidence_refs"] = target_refs
    strategy["optional_mfp_witness_refs"] = direct_mfp_refs
    strategy["target_evidence_basis"] = evidence_basis
    strategy["selection_basis"] = evidence_basis
    strategy["preserve_correction_refs"] = corrections
    strategy["regression_guard_refs"] = regressions
    strategy["unchanged_failure_refs"] = unchanged
    strategy.setdefault(
        "repair_hypothesis",
        causal or f"Address current-best failures attributed to: {', '.join(targets)}.",
    )
    strategy.setdefault(
        "correction_preservation",
        "Preserve candidate correction mechanisms as constraints while implementing the selected module-set change.",
    )
    strategy.setdefault(
        "regression_protection",
        "Treat every regression reference as a mandatory probe guard; do not accept an improvement that reintroduces one.",
    )
    strategy.setdefault(
        "unchanged_failure_interpretation",
        "Use unchanged failures as causal evidence of whether the selected mechanism was blocked, untriggered, or ineffective.",
    )
    strategy["source"] = "deterministic_evidence_ledger_completion"
    completed["evidence_strategy"] = strategy
    return completed


def sanitize_tool_terms_for_evolution(value):
    """Convert Teacher/previous tool names into capability-level wording."""
    if isinstance(value, dict):
        sanitized = {}
        for k, v in value.items():
            if k == "react_trace":
                continue
            else:
                sanitized[k] = sanitize_tool_terms_for_evolution(v)
        return sanitized
    if isinstance(value, list):
        return [sanitize_tool_terms_for_evolution(v) for v in value]
    if not isinstance(value, str):
        return value
    text = value
    replacements = {**TEACHER_ONLY_TOOL_TERMS, **DISALLOWED_RUNTIME_TOOL_TERMS}
    for old, new in sorted(replacements.items(), key=lambda item: -len(item[0])):
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return text

try:
    from utils import format_json as _runtime_format_json
except Exception as _format_json_import_error:
    _runtime_format_json = None


def format_json(raw_text: str, max_tokens: int = 4096) -> dict | None:
    """Best-effort JSON repair helper.

    In MetaVideoAgent runs, `utils` resolves to the execution runtime module. That
    module may import optional video dependencies such as cv2.  Diagnosis must
    not fail at import time just because this optional repair helper is
    unavailable.
    """
    if _runtime_format_json is None:
        return None
    try:
        return _runtime_format_json(raw_text, max_tokens=max_tokens)
    except Exception:
        return None

# Diagnosis is deliberately independent from Codex codegen. The evolution LLM forms
# a cross-question, evidence-grounded module-set design; the deterministic brief
# compiler turns that design into an executable five-module bundle contract.
# Diagnosis may require a longer context than execution-time answering.  The
# environment override is intentionally diagnosis-only: it must not alter the
# formal execution LLM profile used by full evaluation.
DIAGNOSIS_MODEL = EVOLUTION_LLM_MODEL
API_TIMEOUT_SEC = int(os.environ.get("DIAGNOSIS_API_TIMEOUT_SEC", "300") or "300")
LLM_MAX_RETRIES = int(os.environ.get("DIAGNOSIS_LLM_MAX_RETRIES", "2") or "2")
# Keep the response budget configurable and large enough for the complete
# adaptive bundle design and its structured evidence contract.
DIAGNOSIS_OUTPUT_MAX_TOKENS = int(
    os.environ.get("DIAGNOSIS_OUTPUT_MAX_TOKENS", "20000") or "20000"
)

class DiagnosisAgent:
    """evolution LLM-backed diagnosis agent for MetaVideoAgent evolution."""

    MAX_ROUNDS = 15

    def __init__(self, workspace_dir: str, video_id: str = None,
                 evolution_memory: str = None,
                 previous_evolution_result: dict = None,
                 previous_diagnosis: dict = None,
                 evolution_review_path: str = None,
                 iteration: int = None,
                 observed_distribution_profile: dict = None,
                 distribution_manifest: str = "",
                 reference_results_path: str = "",
                 reference_label: str = "",
                 iter_context: dict = None,
                 iter_context_path: str = "",
                 brief_validation_feedback: list[str] | None = None,
                 diagnosis_attempt: int = 1):
        self.workspace_dir = workspace_dir
        # Iteration path management
        from workspace_paths import IterationPaths
        self.iter_paths = IterationPaths(workspace_dir, iteration=iteration)
        self.distribution_manifest = distribution_manifest or (
            (previous_diagnosis or {}).get("distribution_manifest", "")
            if isinstance(previous_diagnosis, dict) else ""
        )
        self.video_ids = [video_id] if video_id else self._detect_video_ids()
        self.video_id = video_id or ("multi_video" if len(self.video_ids) > 1 else (self.video_ids[0] if self.video_ids else ""))
        self.evolution_memory = evolution_memory or ""
        self.previous_evolution_result = previous_evolution_result
        self.previous_diagnosis = previous_diagnosis  # Last round of diagnostic report (my previous output)
        self.evolution_review_path = evolution_review_path or (
            self._find_latest_evolution_review()
            if previous_evolution_result else ""
        )
        self.evolution_review = self._load_evolution_review(self.evolution_review_path)
        if evolution_review_path and not review_is_consumable(self.evolution_review):
            raise RuntimeError(
                "Diagnosis requires an explicit completed evolution review with "
                "macro_review.success=true; incomplete micro reviews are audit-only omissions"
            )
        previous_reference_path = ""
        previous_reference_label = ""
        if isinstance(previous_diagnosis, dict):
            previous_reference_path = previous_diagnosis.get("reference_results_path", "")
            previous_reference_label = previous_diagnosis.get("reference_label", "")
        self.reference_results_path = reference_results_path or previous_reference_path
        # A formal review already records the exact current-best result that
        # it audited.  Reuse that run-scoped path only when the caller did not
        # supply a reference; never search a shared cache or a "latest" run.
        if not self.reference_results_path and self.evolution_review:
            review_context = self.evolution_review.get("current_best_context") or {}
            reviewed_path = str(review_context.get("current_best_results_path") or "")
            if reviewed_path and os.path.exists(reviewed_path):
                self.reference_results_path = reviewed_path
                previous_reference_label = "current_best_from_review"
        self.reference_label = (
            reference_label
            or previous_reference_label
            or "reference"
        )
        self.iter_context_path = iter_context_path or ""
        self.iter_context = iter_context or self._load_evolution_review(self.iter_context_path)
        self.brief_validation_feedback = [
            str(item).strip() for item in (brief_validation_feedback or [])
            if str(item).strip()
        ][:24]
        self.diagnosis_attempt = max(1, int(diagnosis_attempt or 1))
        if not self.reference_results_path:
            raise RuntimeError(
                "MetaVideoAgent diagnosis requires explicit reference_results_path; "
                "the public workflow does not infer reference trajectories."
            )
        self.observed_distribution_profile = observed_distribution_profile or (
            (previous_diagnosis or {}).get("observed_distribution_profile", {})
            if isinstance(previous_diagnosis, dict) else {}
        )
        # Compute the cost budget from the explicit current reference.
        from cost_calculator import (
            compute_reference_cost,
            format_cost_budget,
        )
        from reference_loader import load_reference_trajectories
        self._cost_baseline = compute_reference_cost(
            load_reference_trajectories(
                self.reference_results_path,
                reference_label=self.reference_label,
            )
        )
        self._cost_budget_text = format_cost_budget(self._cost_baseline)
        from evolution_phase_policy import for_iteration
        self.evolution_phase_policy = for_iteration(getattr(self.iter_paths, "iteration", None))

        # Trajectory cache (grouped by time_reference)
        self._trajectories = None      # dict[str, list[dict]]
        self._dataset = None           # dict[str, dict]  time_ref → dataset entry
        self._target_selection = {}

    def _has_candidate_rejections(self) -> bool:
        """Whether iter_context carries probe-rejected runnable candidates."""
        return bool(
            isinstance(self.iter_context, dict)
            and (self.iter_context.get("candidate_rejections") or [])
        )

    def _diagnosis_entry_mode(self) -> str:
        """Classify the fixed diagnosis input context for audit metadata."""
        if self.previous_evolution_result:
            return "post_full_eval_review"
        if self.evolution_review and self._has_candidate_rejections():
            return "probe_rejection_rediagnosis"
        if self.evolution_review:
            return "review_to_diagnosis"
        if self._has_candidate_rejections():
            return "probe_rejection_rediagnosis"
        if self.evolution_memory.strip():
            return "memory_augmented"
        return "initial_reference_diagnosis"


    def _format_iter_context_for_prompt(self) -> str:
        if not self.iter_context:
            return ""
        compact_rejections = []
        for item in (self.iter_context.get("candidate_rejections") or [])[:3]:
            if not isinstance(item, dict):
                continue
            audit = item.get("probe_audit_summary") or {}
            compact_rejections.append({
                "context_type": item.get("context_type", ""),
                "outcome_status": str((item.get("decision") or {}).get("status") or ""),
                "probe_metrics": item.get("probe_metrics", {}),
                "probe_issues": (audit.get("issues") or [])[:6],
            })
        selection = self._target_selection or {}
        payload = {
            "run_id": self.iter_context.get("run_id"),
            "iter_index": self.iter_context.get("iter_index"),
            "reference_label": self.reference_label,
            "current_best_preserve_contract": {
                "base": "current_best",
                "unselected_modules_remain_current_best_unless_required_by_linked_change": True,
            },
            "historical_module_outcomes": selection.get("prior_module_outcomes", {}),
            "candidate_rejections": compact_rejections,
            "diagnosis_requirements": self.iter_context.get("diagnosis_requirements", {}),
            "evolution_phase_policy": self.evolution_phase_policy,
        }
        return (
            "## Per-iteration evolution context\n"
            "This is audit/outcome/guard context only. Do not infer or reuse an earlier "
            "target set, candidate name, implementation hint, or historical combo as "
            "the next direction. The deterministic evidence ledger is the sole target-selection "
            "evidence. The current best is the only execution baseline.\n"
            "If candidate_rejections are present, treat them as failed runnable "
            "candidate outcome/guard evidence, not as a target recommendation.\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)[:14000]}"
        )

    def _external_history_meta(self) -> dict:
        formal_output_root = os.path.abspath(runtime_output_root())
        paths = []
        for path in (self.reference_results_path, self.evolution_review_path):
            if not path:
                continue
            abs_path = os.path.abspath(path)
            if not abs_path.startswith(formal_output_root + os.sep):
                paths.append(abs_path)
        return {
            "external_history_input": bool(paths),
            "external_history_paths": paths,
        }

    def _build_target_selection(self) -> dict:
        """Build the machine-only evidence ledger before LLM diagnosis."""
        self._target_selection = build_target_selection(
            self._load_trajectories(),
            review=self.evolution_review,
            iter_context=self.iter_context,
        )
        return self._target_selection

    def _format_target_selection_for_prompt(self) -> str:
        """Render a model-safe summary, never the machine-only ledger itself."""
        selection = self._target_selection or {}
        if not selection:
            return ""
        comparison = selection.get("target_comparison", {}) or {}
        modules = {}
        for module in ("video_structuring", "localization", "perception", "memory", "thinking"):
            facts = comparison.get(module, {}) or {}
            repair = facts.get("direct_repair", {}) or {}
            regression = facts.get("direct_regression_stabilization", {}) or {}
            correction = facts.get("candidate_correction_preservation", {}) or {}
            blocked = facts.get("blocked_upstream", {}) or {}
            mfp_coverage = facts.get("validated_mfp_coverage", {}) or {}
            weak = facts.get("weak_failure_distribution", {}) or {}
            modules[module] = {
                "direct_repair_count": int(repair.get("count") or 0),
                "direct_repair_cause_kinds": list(repair.get("cause_kinds") or []),
                "candidate_regression_guard_count": int(regression.get("count") or 0),
                "candidate_correction_preserve_count": int(correction.get("count") or 0),
                "blocked_or_untriggered_count": int(blocked.get("count") or 0),
                "validated_mfp_mechanism_clusters": int(mfp_coverage.get("cluster_count") or 0),
                "validated_mfp_failure_kinds": list(mfp_coverage.get("failure_kinds") or []),
                "validated_mfp_video_coverage": int(mfp_coverage.get("video_count") or 0),
                "weak_failure_cluster_count": int(weak.get("cluster_count") or 0),
                "weak_failure_stages": list(weak.get("failure_stages") or []),
                "weak_evidence_qualities": list(weak.get("evidence_qualities") or []),
                "weak_capability_signatures": list(weak.get("capability_signatures") or []),
                "weak_consumer_states": list(weak.get("consumer_states") or []),
                "weak_semantic_categories": list(weak.get("semantic_categories") or []),
                "history_outcome": str(facts.get("history_outcome") or ""),
            }
        mechanism_index = selection.get("mfp_mechanism_cluster_index", {}) or {}
        mfp_mechanisms = []
        for item in mechanism_index.get("clusters", []) or []:
            if not isinstance(item, dict):
                continue
            mfp_mechanisms.append({
                "root_module": str(item.get("root_module") or ""),
                "failure_kind": str(item.get("failure_kind") or ""),
                "consumer_signature": str(item.get("consumer_signature") or ""),
                "capability_needs": list(item.get("capability_needs") or []),
                "task_coverage": len(item.get("refs") or []),
                "video_coverage": len(item.get("video_ids") or []),
            })
        repair_mechanisms = []
        for item in ((selection.get("validated_review_repair_mechanisms") or {}).get("clusters") or []):
            if not isinstance(item, dict):
                continue
            repair_mechanisms.append({
                "modules": list(item.get("modules") or []),
                "repair_mechanism": str(item.get("repair_mechanism") or ""),
                "handoff_or_state_to_repair": str(item.get("handoff_or_state_to_repair") or ""),
                "falsifiable_runtime_signals": list(item.get("falsifiable_runtime_signals") or []),
                "generalization_scopes": list(item.get("generalization_scopes") or []),
                "task_coverage": len(item.get("refs") or []),
                "video_coverage": len(item.get("video_ids") or []),
            })
        weak_distribution = selection.get("weak_failure_distribution", {}) or {}
        weak_mechanisms = []
        for item in weak_distribution.get("clusters", []) or []:
            if not isinstance(item, dict):
                continue
            weak_mechanisms.append({
                "suggested_module": str(item.get("suggested_module") or "unattributed"),
                "trace_cause_kind": str(item.get("trace_cause_kind") or ""),
                "failure_stage": str(item.get("failure_stage") or ""),
                "evidence_quality": str(item.get("evidence_quality") or ""),
                "capability_signature": str(item.get("capability_signature") or ""),
                "consumer_state": str(item.get("consumer_state") or ""),
                "semantic_category": str(item.get("semantic_category") or ""),
                "effect_verdict": str(item.get("effect_verdict") or ""),
                "task_coverage": len(item.get("refs") or []),
                "video_coverage": len(item.get("video_ids") or []),
            })
        contract = {
            "modules": modules,
            "reviewed_candidate_targets": list(selection.get("reviewed_candidate_targets") or []),
            "validated_mfp_mechanism_distribution": mfp_mechanisms,
            "validated_review_repair_mechanisms": repair_mechanisms,
            "all_wrong_trace_distribution": {
                "total_task_coverage": int(weak_distribution.get("total_wrong_task_refs") or 0),
                "mechanism_clusters": weak_mechanisms,
            },
            "macro_module_hypotheses": [
                {
                    "module": str(module),
                    "level": str((details or {}).get("level") or ""),
                    # Reasons and refs remain audit-only because they can carry
                    # task-specific text. This is an aggregate hypothesis, not
                    # direct repair evidence.
                    "has_review_rationale": bool((details or {}).get("reason")),
                }
                for module, details in (
                    ((getattr(self, "evolution_review", {}).get("macro_review") or {}).get("module_responsibility") or {}).items()
                    if isinstance(getattr(self, "evolution_review", {}), dict) else []
                )
                if str(module) in {"video_structuring", "localization", "perception", "memory", "thinking"}
                and isinstance(details, dict)
            ],
            "policy": {
                "corrections_are_preserve_evidence": True,
                "regressions_are_guards_not_repair_claims": True,
            "history_is_a_guard_not_a_module_ban": True,
            "trace_distribution_selects_mechanism_not_direct_claim": True,
            "validated_mfp_is_optional_probe_witness": True,
            "validated_review_repair_mechanisms_are_design_evidence_not_target_votes": True,
                "all_model_evidence_is_task_free": True,
            },
        }
        return (
            "## Model-safe evidence summary\n"
            "This summary is derived from a machine-only ledger; task references, paths, and detailed "
            "trajectories are intentionally excluded. It records facts but does not choose target_modules. Do not reuse a previous "
            "candidate target set as a default. Compare every module in target_comparison and select one "
            "linked causal mechanism yourself. Explain why the selected module set is preferable to the concrete "
            "alternatives. Select the target from the complete trace distribution and macro hypotheses, not from the "
            "number of validated MFPs. MFP mechanisms are optional high-resolution probe witnesses only; they must "
            "not veto a broadly supported target and they must not be relabelled as global proof. Set "
            "evolution_design.evidence_strategy.selection_basis to exactly one of distribution_mechanism, "
            "narrow_mfp_repair, regression_stabilization, or bounded_exploration. Candidate corrections identify behavior worth "
            "preserving. Regression stabilization is valid only for the reviewed candidate's changed "
            "module set, and regressions remain probe guards, never repair claims. Unchanged failures must "
            "explain whether the previous mechanism was not triggered, had no effect, or was blocked "
            "upstream/downstream. Prior module outcomes are guards, never a ban on evolving that module "
            "again. If the selected module set has broad trace support, label it distribution_mechanism and give a concrete, "
            "bounded, falsifiable mechanism hypothesis. If it has neither broad trace nor regression support, label it "
            "bounded_exploration and give a concrete, bounded, falsifiable exploration "
            "rationale. In evidence_strategy.selection_rationale, explicitly state why this target is "
            "preferred over the direct-repair, stabilization, or exploration alternatives shown in the "
            "comparison table. LLM risk hypotheses are not direct evidence and must be labelled as hypotheses. "
            "evolution_design must also include "
            "audit-only evidence_strategy: target_evidence_refs, preserve_correction_refs, "
            "regression_guard_refs, unchanged_failure_refs, repair_hypothesis, correction_preservation, "
            "regression_protection, unchanged_failure_interpretation, and exploration_rationale when "
            "there is no direct repair support. Keep the final JSON concise: do not enumerate task refs, "
            "full fault lists, or per-question diagnoses. Set the four evidence-reference arrays to [] because "
            "the deterministic completion layer fills them exactly. Keep each natural-language field focused "
            "on the causal design rather than repeating review text.\n"
            "Use the complete mechanism distribution, not a single sample. Validated per-question repair mechanisms "
            "are trace-grounded design evidence: use their stated handoff and falsifiable signal when forming a causal "
            "direction, but do not turn their module labels into an automatic target vote. Validated MFP mechanisms are optional "
            "probe witnesses; the trace distribution provides broad mechanism coverage but is not a direct per-question "
            "repair claim. Include mechanism_scope in evolution_design: runtime applicability, covered "
            "failure kinds, expected producer/consumer handoff, direct-support scope, and a distinct "
            "generalization scope for probe. Include 1-3 diagnosis-owned probe_hypotheses: each states an "
            "information gap, proposed module-set intervention, expected observable evidence, preserved "
            "consumer, falsifiable success condition, and optional failure_kinds. A hypothesis may use a "
            "complementary capability rather than the capability observed in a failing trajectory, but must "
            "state the causal bridge. The design must be triggered by generic runtime state, never "
            "by a question, video, timestamp, answer, or transcript phrase.\n"
            f"{json.dumps(contract, ensure_ascii=False, indent=2)}"
        )

    def _apply_target_selection(self, report: dict) -> dict:
        """Attach deterministic evidence to the diagnosis-selected module set."""
        selection = dict(self._target_selection or {})
        decision = report.setdefault("evolution_decision", {})
        design = report.setdefault("evolution_design", {})
        algo = report.setdefault("algorithmic_evolution", {})
        raw_modules = (
            design.get("target_modules")
            or algo.get("target_modules")
            or decision.get("target_modules")
        )
        if not raw_modules:
            raise RuntimeError("Diagnosis must select at least one target module")
        try:
            from bundle_contract import normalize_modules
            modules = normalize_modules(raw_modules)
        except (ImportError, ValueError) as exc:
            raise RuntimeError(f"Diagnosis target_modules are invalid: {exc}") from exc
        design["target_modules"] = modules
        design.setdefault("initial_focus_modules", modules)
        design["runtime_capability_needs"] = _normalize_runtime_capability_needs(
            design.get("runtime_capability_needs", [])
        )
        algo["target_modules"] = modules
        decision.update({
            "target_modules": modules,
            "combo_base": "current_best",
            "base_combo_policy": "keep_current_best",
        })
        decision["probe_plan"] = decision.get("probe_plan") or {}
        decision["probe_plan"].update({
            "direct_mfp_evidence": selection.get("direct_mfp_evidence", []),
            "regression_guard": selection.get("regression_guards", []),
            "candidate_correction_evidence": selection.get("candidate_correction_evidence", []),
            "unchanged_failure_evidence": selection.get("unchanged_failure_evidence", []),
            "selection_mode": "outer_round_adaptive_module_set_fixed_inner_bundle_batch",
        })

        report["evidence_ledger"] = selection
        report["fault_distribution"] = {
            module: int(
                (selection.get("module_scores", {}).get(module, {}) or {})
                .get("eligible_ref_count", 0) or 0
            )
            for module in (
                "video_structuring", "localization", "perception", "memory", "thinking"
            )
        }
        report["fault_distribution_source"] = "deterministic_evidence_ledger"
        report["evolution_design"] = complete_evidence_strategy(
            design,
            selection,
            targets=modules,
        )
        self._refresh_machine_evaluation_contract(report, selection)

        target_text = " + ".join(modules)
        preserved = [
            module for module in (
                "video_structuring", "localization", "perception", "memory", "thinking"
            ) if module not in modules
        ]
        preservation = (
            " Preserve unaffected modules: "
            + ", ".join(f"`{module}`" for module in preserved)
            + "."
            if preserved else
            " Preserve unaffected behavior across the complete bundle."
        )
        algo["evolution_hint"] = (
            "Canonical next-round direction: evolve "
            f"`{target_text}` from the retained current-best bundle."
            + preservation
            + " Validate the affected handoff using diagnosis repair probes and "
            "protect the listed regression guards."
        )
        repair_probe = list(
            ((report.get("validation_policy") or {}).get("repair_probe") or [])
        )
        algo["target_questions"] = repair_probe
        report["target_questions"] = repair_probe
        report["diagnosis_reconciliation"] = {
            "schema_version": 1,
            "canonical_target_modules": modules,
            "evidence_modules": modules,
            "target_question_source": "validation_policy.repair_probe",
            "repair_probe_count": len(repair_probe),
            "regression_guard_source": "validation_policy.regression_guard",
            "reason": (
                "Reconciled the structured bundle direction and machine-selected "
                "validation references after diagnosis completion."
            ),
        }
        report["diagnosis_status"] = "bundle_direction_selected_by_evolution_llm"
        return report


    def _refresh_machine_evaluation_contract(self, report: dict, selection: dict) -> None:
        """Rebuild all executable probe facts after deterministic attribution.

        The LLM may describe a desired probe, but it cannot decide whether a
        candidate correction is a repair or whether a regression is a guard.
        Those roles are fixed by the paired review ledger and must be refreshed
        after `_apply_target_selection` replaces LLM diagnoses with facts.
        """
        algo = report.setdefault("algorithmic_evolution", {})
        decision = report.setdefault("evolution_decision", {})
        strategy = (report.get("evolution_design", {}) or {}).get("evidence_strategy", {}) or {}
        target_refs = list(strategy.get("target_evidence_refs") or [])
        corrections = _ledger_task_refs(selection, "candidate_correction_evidence")
        regressions = _ledger_task_refs(selection, "regression_guards")
        # A fixed review→diagnosis stage deliberately does not need a previous
        # ``previous_evolution_result`` input: the completed review is the
        # authoritative carrier of paired corrections, regressions and macro
        # constraints. Requiring both silently discarded those facts in the
        # normal staged MetaVideoAgent flow, leaving the generated probe contract
        # empty despite a valid review.
        constraints = (
            self._evolution_review_constraints()
            if getattr(self, "evolution_review", None)
            else {}
        )
        unchanged = constraints.get("unchanged_failure_summary", {}) or {}
        # An MFP is an end-to-end failure witness, not a vote for or a
        # restriction on the module selected by diagnosis.  A localization,
        # perception, or memory intervention may all improve the same
        # trajectory point.  Keep the originating module for machine-only
        # audit, but allow every validated point to be sampled by the probe.
        mfp_clusters = [
            item for item in ((selection.get("mfp_mechanism_cluster_index") or {}).get("clusters") or [])
            if isinstance(item, dict)
        ]
        hypotheses = list((report.get("evolution_design") or {}).get("probe_hypotheses") or [])
        hypothesis_witnesses = []
        witness_refs = []
        all_witness_refs = []
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            kinds = {str(value) for value in (hypothesis.get("failure_kinds") or []) if str(value)}
            matching = [
                item for item in mfp_clusters
                if not kinds or str(item.get("failure_kind") or "") in kinds
            ]
            refs = self._unique_refs([
                ref for item in matching if isinstance(item, dict)
                for ref in (item.get("refs") or [])
            ])
            hypothesis_witnesses.append({
                "hypothesis_id": str(hypothesis.get("hypothesis_id") or ""),
                "failure_kinds": sorted(kinds),
                "origin_modules": sorted({
                    str(item.get("root_module") or "") for item in matching
                    if str(item.get("root_module") or "")
                }),
                "refs": refs,
            })
            all_witness_refs.extend(refs)
            if refs:
                witness_refs.append(refs[0])
        all_weak_refs = self._unique_refs([
            ref for item in ((selection.get("weak_failure_distribution") or {}).get("clusters") or [])
            if isinstance(item, dict)
            for ref in (item.get("refs") or [])
        ])
        # Generalization is a held-out check.  It must not reuse either the
        # diagnosis-selected repair rows or optional MFP witnesses, otherwise
        # a later probe can label its repair sample as independent evidence.
        # Compare canonical forms: MFP clusters may use hash-free task refs
        # while manifests add a question hash.  A generalization pool must be
        # disjoint from every repair/witness row, not merely differently
        # serialized copies of the same training question.
        held_out_refs = {
            canonical_task_ref(ref) for ref in (
                list(target_refs) + all_witness_refs + list(corrections) + list(regressions)
            )
            if canonical_task_ref(ref)
        }
        generalization_refs = [
            ref for ref in all_weak_refs
            if canonical_task_ref(ref) not in held_out_refs
        ]
        decision["probe_plan"] = {
            "source": "diagnosis_probe_hypotheses_with_machine_selection",
            "diagnosis_probe_hypotheses": hypotheses,
            "repair_probe": target_refs,
            "diagnosis_hypothesis_witnesses": hypothesis_witnesses,
            "diagnosis_hypothesis_witness_refs": self._unique_refs(witness_refs),
            "generalization_probe_pool": generalization_refs,
            # Corrections and regressions are both preservation/risk guards.
            # Their separate buckets remain available for audit and selection.
            "regression_guard": self._unique_refs(corrections + regressions),
            "previous_candidate_corrections": corrections,
            "previous_candidate_regressions": regressions,
            "has_prior_candidate_review": self._has_prior_candidate_review(),
            "failure_clusters": unchanged.get("refs_by_category", {}),
            "rationale": (
                "Machine selects representative rows for diagnosis-owned hypotheses, then samples held-out "
                "failure-distribution strata; preserve candidate corrections and protect regressions as guards. "
                "An MFP witness is an end-to-end outcome observation. It does not prescribe the target module set, "
                "repair capability, or a hard acceptance gate."
            ),
        }
        report["probe_plan"] = decision["probe_plan"]
        report["has_prior_candidate_review"] = self._has_prior_candidate_review()
        report["validation_policy"] = self._build_validation_policy(report, constraints)
        self._attach_review_cost_contract(report, constraints)

    def _attach_review_cost_contract(self, report: dict, constraints: dict) -> None:
        """Make a completed review's cost requirement executable and auditable.

        Cost findings are aggregate review evidence, not a reason to suppress
        a valid accuracy repair.  When the review identifies systemic redundant
        work, however, every next-round bundle must carry generic runtime
        guardrails that make escalation bounded and observable.
        """
        review = getattr(self, "evolution_review", {}) or {}
        macro = review.get("macro_review", {}) or {}
        assessment = macro.get("cost_and_efficiency_assessment", {}) or {}
        audit_issues = macro.get("cost_audit_issues", []) or []
        retention = review.get("retention_review", {}) or {}
        combo_decision = retention.get("combo_decision", {}) or {}
        codes = {
            str(item.get("code") or "")
            for item in audit_issues if isinstance(item, dict)
        }
        codes.update(str(code or "") for code in (assessment.get("cost_issue_codes") or []))
        codes.update(str(code or "") for code in (combo_decision.get("cost_audit_issue_codes") or []))
        if "cost_explosion" not in codes and not assessment.get("next_round_cost_target"):
            return

        design = report.setdefault("evolution_design", {})
        guardrails = list(design.get("cost_guardrails") or design.get("cost_guards") or [])
        required = [
            "Use compact retained structure and retrieval evidence first; escalate to scoped media inspection only when the stated evidence-sufficiency condition is unmet.",
            "Deduplicate semantically equivalent retrieval/replan requests and preserve previously verified evidence instead of repeating the same inspection.",
            "Bound each replan and expensive perception escalation; emit an explicit escalation reason and stop when the bounded budget is exhausted.",
        ]
        for item in required:
            if item not in guardrails:
                guardrails.append(item)
        design["cost_guardrails"] = guardrails
        design.pop("cost_guards", None)
        report["review_cost_constraint"] = {
            "source": "teacher_evolution_macro.cost_and_efficiency_assessment",
            "enforced": True,
            "review_target": str(assessment.get("next_round_cost_target") or "preserve_accuracy_reduce_cost"),
            "issue_codes": sorted(code for code in codes if code),
            "guardrails": required,
            "policy": (
                "Preserve demonstrated corrections and regression guards while reducing redundant "
                "retrieval/inspection work through bounded, evidence-triggered escalation."
            ),
        }

    # ==================================================================
    # main entrance
    # ==================================================================

    def run(self) -> dict:
        """Run diagnostics and return a structured diagnostic report."""
        print("=" * 70)
        entry_mode = self._diagnosis_entry_mode()
        if entry_mode == "post_full_eval_review":
            mode = "Evaluation mode (recycling the results of the previous round)"
        elif entry_mode == "review_to_diagnosis":
            mode = "Review diagnosis (Consumption Teacher review)"
        elif entry_mode == "probe_rejection_rediagnosis":
            mode = "Diagnose after Probe rejection"
        elif entry_mode == "memory_augmented":
            mode = "memory enhancement diagnostics"
        else:
            mode = "first diagnosis"

        print(f"Diagnostic Agent startup [{mode}] (direct diagnosis)")
        print("=" * 70)
        print(f"   Workspace: {self.workspace_dir}")
        print(f"   Videos: {', '.join(self.video_ids) if self.video_ids else self.video_id}")
        print(f"   Model: {DIAGNOSIS_MODEL}")
        print(f"   Reference: {self.reference_label} ({self.reference_results_path or 'workspace/trajectories'})")
        print(f"   Entry mode: {entry_mode}")
        if self.evolution_review_path:
            print(f"   Teacher review input: {self.evolution_review_path}")
        if self._has_candidate_rejections():
            print(f"   Candidate rejections: {len(self.iter_context.get('candidate_rejections') or [])}")
        if self.previous_evolution_result:
            v = self.previous_evolution_result.get("verification", {})
            print(f"   Last round result: Δ={v.get('accuracy_delta', '?')}, "
                  f"repair ={len(v.get('corrections', []))}, "
                  f"regressions={len(v.get('regressions', []))}")
            if self.evolution_review_path:
                print(f"   Candidate review: {self.evolution_review_path}")
        print()

        # Review and diagnosis are separate fixed stages. Diagnosis never
        # starts review work or reads mutable review caches.
        print("[Review] The diagnosis phase only consumes the incoming review / probe rejection / current-best results.")

        selection = self._build_target_selection()
        print(
            "Evidence ledger: "
            f"repair={len(selection.get('repair_evidence') or [])} "
            f"correction={len(selection.get('candidate_correction_evidence') or [])} "
            f"regression={len(selection.get('regression_guards') or [])}"
        )

        # Diagnosis is one bounded evolution LLM design pass, not an exploratory
        # tool-using agent. All admissible evidence was loaded above.
        raw_output = self._run_direct_diagnosis()

        # Save the direct LLM exchange for audit only.
        self._save_rollout()

        # Extract diagnostic reports from LLM output
        report = getattr(
            self,
            "_accepted_diagnosis_report",
            self._parse_report(raw_output, allow_llm_repair=False),
        )
        report = self._apply_target_selection(report)

        # Complete the timestamp and save it
        report.setdefault("timestamp", int(time.time()))
        report["diagnosis_attempt"] = self.diagnosis_attempt
        if self.brief_validation_feedback:
            report["prior_brief_validation_feedback"] = list(self.brief_validation_feedback)
        report["diagnosis_entry_mode"] = entry_mode
        if self.observed_distribution_profile:
            report.setdefault(
                "observed_distribution_profile",
                self.observed_distribution_profile,
            )
            report.setdefault("distribution_leakage_policy", (
                "Diagnosis used only a label-blind observed distribution profile. "
                "Human type descriptions, if present in distribution metadata, "
                "were not prompt-visible."
            ))
        if self.reference_results_path:
            report["reference_results_path"] = os.path.abspath(self.reference_results_path)
        if self.iter_context:
            report["iter_context_path"] = os.path.abspath(self.iter_context_path) if self.iter_context_path else self.iter_context.get("path", "")
            report["iter_context"] = {
                "run_id": self.iter_context.get("run_id"),
                "iter_index": self.iter_context.get("iter_index"),
                "reference": self.iter_context.get("reference", {}),
                "current_best": self.iter_context.get("current_best", {}),
                "evolution_memory": self.iter_context.get("evolution_memory", {}),
                "previous_round": self.iter_context.get("previous_round", {}),
                "candidate_rejections": self.iter_context.get("candidate_rejections", [])[:3],
                "diagnosis_requirements": self.iter_context.get("diagnosis_requirements", {}),
            }
        report["history_input_policy"] = self._external_history_meta()
        report["reference_label"] = self.reference_label
        report["reference_policy"] = (
            "Teacher/Diagnosis trajectory tools read the explicit reference combo "
            "identified by reference_results_path."
        )
        self._save_report(report)

        return report

    def _save_rollout(self):
        """Persist the direct diagnosis LLM exchange for audit."""
        if not hasattr(self, "_rollout_trace") or not self._rollout_trace:
            return
        import time as _time
        ts = int(_time.time())
        diag_dir = self._diagnosis_dir()
        os.makedirs(diag_dir, exist_ok=True)
        suffix = f"_attempt{self.diagnosis_attempt}" if self.diagnosis_attempt > 1 else ""
        rollout_path = os.path.join(diag_dir, f"diagnosis_rollout_{ts}{suffix}.json")
        with open(rollout_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": ts,
                "video_id": self.video_id,
                "model": DIAGNOSIS_MODEL,
                "rounds": len(self._rollout_trace),
                "trace": self._rollout_trace,
            }, f, ensure_ascii=False, indent=2)
        print(f"   Rollout saved: {rollout_path}")

    # ==================================================================
    # Direct diagnosis LLM
    # ==================================================================

    def _call_diagnosis_llm_content(self, client, messages: list) -> str:
        """Call diagnosis LLM and return text content.

        The OpenAI-compatible SDK path is kept as the primary path. A raw HTTP
        fallback is used only when the SDK transport fails, which prevents large
        diagnosis prompts from being blocked by local SDK/httpx edge cases.
        """
        try:
            response = client.chat.completions.create(
                model=DIAGNOSIS_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=DIAGNOSIS_OUTPUT_MAX_TOKENS,
                extra_body=dict(EVOLUTION_LLM_REQUEST_OPTIONS),
            )
            raw_content = response.choices[0].message.content or ""
        except Exception as sdk_error:
            endpoint = EVOLUTION_LLM_BASE_URL.rstrip("/") + "/chat/completions"
            payload_data = {
                "model": DIAGNOSIS_MODEL,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": DIAGNOSIS_OUTPUT_MAX_TOKENS,
            }
            payload_data.update(EVOLUTION_LLM_REQUEST_OPTIONS)
            payload = json.dumps(payload_data).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={
                    "Authorization": f"Bearer {EVOLUTION_LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=API_TIMEOUT_SEC) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as http_error:
                body = http_error.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"SDK call failed ({sdk_error}); raw HTTP failed "
                    f"{http_error.code}: {body[:500]}"
                ) from http_error
            except Exception as http_error:
                raise RuntimeError(
                    f"SDK call failed ({sdk_error}); raw HTTP failed: {http_error}"
                ) from http_error
            try:
                raw_content = data["choices"][0]["message"].get("content") or ""
            except Exception as parse_error:
                raise RuntimeError(
                    f"Raw HTTP response missing choices[0].message.content: {data}"
                ) from parse_error

        if isinstance(raw_content, list):
            raw_content = "\n".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        return str(raw_content or "")

    def _adaptive_execution_contract_issues(self, report: dict) -> list[str]:
        """Fail closed before accepting a bundle design from diagnosis.

        Diagnosis is an upstream specification for code generation, so a
        syntactically valid model response is not sufficient.  In particular,
        its thinking fallback must preserve uncertainty as metadata while
        still requiring the execution LLM to select a concrete, parseable
        answer from the question's options.  Reuse the deterministic brief
        compiler here rather than relying on a prompt instruction that a
        model may overlook.
        """
        if not isinstance(report, dict):
            return ["diagnosis response is not a JSON object"]
        modules = (
            (report.get("evolution_design") or {}).get("target_modules")
            or (report.get("algorithmic_evolution") or {}).get("target_modules")
            or []
        )
        if not isinstance(modules, list) or not modules:
            return ["response lacks a non-empty target_modules list"]
        try:
            from diagnosis_execution_brief import (
                build_bundle_execution_brief,
                validate_bundle_execution_brief,
            )
            brief = build_bundle_execution_brief(
                report,
                diagnosis_path="<in_memory_diagnosis>",
                review=self.evolution_review,
                review_path=self.evolution_review_path,
            )
            return list(validate_bundle_execution_brief(brief))
        except Exception as exc:
            return [f"bundle execution contract rejected diagnosis: {exc}"]

    def _enforce_thinking_final_answer_contract(self, report: dict) -> dict:
        """Normalize only an invalid terminal-thinking fallback.

        The diagnosis model owns the mechanism, but it cannot alter the
        execution protocol: an evaluated task always ends in one concrete
        answer chosen by the execution LLM. If a model emits a generic
        abstention token as the terminal fallback, replace that one fallback
        with the canonical protocol and retain an auditable normalization
        record. Evidence uncertainty and bounded replanning remain part of
        the design as metadata and control flow.
        """
        if not isinstance(report, dict):
            return report
        design = report.get("evolution_design") or {}
        responsibilities = design.get("module_responsibilities") or {}
        thinking = responsibilities.get("thinking") or {}
        fallback = str(thinking.get("fallback") or "")
        output_invariant = str(thinking.get("output_invariant") or "")
        try:
            from diagnosis_execution_brief import _FORBIDDEN_THINKING_FINAL_FALLBACK_RE
            forbidden = bool(_FORBIDDEN_THINKING_FINAL_FALLBACK_RE.search(fallback)) or bool(
                _FORBIDDEN_THINKING_FINAL_FALLBACK_RE.search(output_invariant)
            )
        except Exception:
            forbidden = bool(re.search(
                r"uncertain|unanswerable|evidence[_ -]?insufficient|insufficient evidence|unanswerable|uncertain",
                fallback + "\n" + output_invariant,
                flags=re.IGNORECASE,
            ))
        probe_hypotheses = design.get("probe_hypotheses") or []
        terminal_abstention_hypotheses = [
            hypothesis for hypothesis in probe_hypotheses
            if isinstance(hypothesis, dict)
            and any(
                re.search(r"unanswerable|unable to answer", str(hypothesis.get(field) or ""), re.I)
                for field in ("expected_observable", "success_condition", "fallback")
            )
        ]
        if not forbidden and not terminal_abstention_hypotheses:
            return report
        normalized = deepcopy(report)
        if forbidden:
            normalized_thinking = (
                ((normalized.setdefault("evolution_design", {}))
                 .setdefault("module_responsibilities", {}))
                 .setdefault("thinking", {})
            )
            normalized_thinking["fallback"] = (
                "After the bounded replan cap, the thinking LLM uses the retained "
                "evidence/context, original question, and answer options to select one "
                "concrete parseable final answer. It records low confidence, unresolved "
                "evidence gaps, and replan history as metadata without replacing the answer."
            )
            normalized_thinking["output_invariant"] = (
                "When the sufficiency gate passes, finish with a concrete parseable answer; "
                "when it fails, issue a bounded replan, and when that budget is exhausted, "
                "have the LLM select one concrete parseable answer from the supplied options "
                "with low-confidence and evidence-gap metadata."
            )
            for key in ("cost_guardrails", "cost_guards"):
                values = (normalized.get("evolution_design") or {}).get(key)
                if not isinstance(values, list):
                    continue
                normalized["evolution_design"][key] = [
                    (
                        "When the bounded replan budget is exhausted, force LLM selection "
                        "of one concrete parseable answer and record uncertainty only as metadata."
                        if re.search(r"uncertain(?:ty)?|unanswerable|evidence[_ -]?insufficient", str(value), re.I)
                        else value
                    )
                    for value in values
                ]
            for need in (normalized.get("evolution_design") or {}).get("runtime_capability_needs") or []:
                if not isinstance(need, dict) or str(need.get("capability") or "") != "llm":
                    continue
                fallback_text = str(need.get("fallback") or "")
                if re.search(r"never output specific option|uncertain(?:ty)?|unanswerable|evidence[_ -]?insufficient", fallback_text, re.I):
                    need["fallback"] = (
                        "Replan conservatively while budget remains; once exhausted, the LLM selects "
                        "one concrete parseable answer from the retained evidence, question, and options, "
                        "and records limitations as metadata."
                    )
        normalized_hypotheses = (normalized.get("evolution_design") or {}).get("probe_hypotheses") or []
        for hypothesis in normalized_hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            for field in ("expected_observable", "success_condition", "fallback"):
                text = str(hypothesis.get(field) or "")
                if re.search(r"unanswerable|unable to answer", text, re.I):
                    hypothesis[field] = (
                        "After bounded replanning, the thinking LLM selects one concrete "
                        "parseable final answer from the supplied options and records low confidence "
                        "and unresolved evidence gaps only as metadata."
                    )
        normalized.setdefault("diagnosis_contract_normalizations", []).append({
            "rule": "thinking_final_answer_must_be_concrete_parseable",
            "field": "evolution_design.module_responsibilities.thinking.fallback_and_probe_hypotheses",
            "original_fallback": fallback,
            "original_output_invariant": output_invariant,
            "normalized_reason": (
                "Generic abstention cannot terminate an evaluated task; uncertainty is "
                "preserved as metadata while the LLM selects from the supplied options."
            ),
        })
        return normalized

    def _run_direct_diagnosis(self) -> str:
        """Produce one diagnosis JSON from already-loaded run evidence.

        No diagnosis tools, video inspection, Teacher calls, or review prefetches
        are available on this path.  This keeps later-round diagnosis aligned
        with the run contract: review + current-best + memory in,
        diagnosis/design out.
        """
        from openai import OpenAI

        self._rollout_trace = []
        client = OpenAI(
            api_key=EVOLUTION_LLM_API_KEY,
            base_url=EVOLUTION_LLM_BASE_URL,
            timeout=API_TIMEOUT_SEC,
        )
        system_content = self._adaptive_diagnosis_prompt()
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Return the final diagnosis JSON now."},
        ]
        last_error = ""
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                content = self._call_diagnosis_llm_content(client, messages).strip()
            except Exception as exc:
                last_error = str(exc)
                continue
            report = self._enforce_thinking_final_answer_contract(
                self._parse_report(content, allow_llm_repair=False)
            )
            modules = ((report.get("evolution_design") or {}).get("target_modules") or
                       (report.get("algorithmic_evolution") or {}).get("target_modules") or [])
            contract_issues = self._adaptive_execution_contract_issues(report)
            if isinstance(modules, list) and modules and not contract_issues:
                self._rollout_trace.append({
                    "round": attempt,
                    "type": "evolution_llm_adaptive_diagnosis",
                    "response_chars": len(content),
                    "brief_validation_feedback": list(self.brief_validation_feedback),
                })
                self._accepted_diagnosis_report = report
                return content
            last_error = "; ".join(contract_issues) or (
                "response lacks a non-empty target_modules list"
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    "Return valid JSON for the five-module bundle schema. evolution_design.target_modules "
                    "must contain 1-5 supported modules selected by the diagnosed mechanism. It must provide responsibilities "
                    "for the selected modules plus concrete boundary contracts; the deterministic compiler supplies canonical "
                    "responsibilities for untouched modules. A one-module target is valid and does not require "
                    "changing its neighboring module; its contract should still state the preserved input/output boundary. "
                    "The prior response failed this deterministic "
                    "execution contract: " + last_error + " Fix the stated fields. In particular, after the bounded "
                    "replan budget is exhausted, thinking must use the retained context, original question and its "
                    "options to have the LLM return one concrete parseable final answer. Record low confidence and "
                    "evidence gaps only as metadata; never make uncertain, unanswerable, or evidence-insufficient "
                    "the final answer."
                ),
            })
        raise RuntimeError(f"direct diagnosis failed after {LLM_MAX_RETRIES} attempts: {last_error}")

    def _adaptive_diagnosis_prompt(self) -> str:
        """Build the sole model prompt for the evolution LLM diagnosis design pass."""
        return f"""You are the diagnosis stage of a five-module MetaVideoAgent evolution run.
Return exactly one JSON object; do not edit files, call tools, inspect videos, or propose task-specific rules.

Use the supplied model-safe review, deterministic evidence distribution, current-best context and history.
Choose an initial direction containing 1-5 modules. Select one module when the failure mechanism is local; select multiple modules only when the evidence supports a joint change. This is an OUTER-ROUND direction, not an edit whitelist: later Codex probe feedback may change related modules inside the five-module bundle when an observed interface dependency requires it.

Every selected module role and boundary must be supported by a real failure-chain pattern. Do not claim all modules are at fault merely because an answer is wrong. For a module-local direction, describe its preserved input/output boundary without adding a neighboring module to target_modules unless that neighbor also needs code changes. Keep task refs, video IDs, timestamps, answers and transcript literals out of prose. Do not include example objects, colors, brands, quoted question wording, counts, or literal video events: express every trigger and intervention as a reusable runtime-state condition.

Thinking may request bounded replanning when evidence is weak. Once the bounded replan budget is exhausted, it must use the retained evidence/context, the original question, and its answer options to have the LLM select one concrete, parseable final answer. Low confidence, unresolved evidence gaps, and the replan history belong in metadata only. Do not prescribe `uncertain`, `unanswerable`, `evidence_insufficient`, or an equivalent generic failure token as the final answer.

Runtime capability inventory (exhaustive):
{json.dumps(build_runtime_capability_context(), ensure_ascii=False)}

Deterministic evidence distribution:
{self._format_target_selection_for_prompt()}

Completed review:
{self._format_evolution_review_for_prompt()}

Historical context:
{self._format_iter_context_for_prompt()}

Required JSON shape:
{{
  "summary": "...",
  "algorithmic_evolution": {{"target_modules": ["localization"], "evolution_hint": "..."}},
  "evolution_decision": {{"combo_base": "current_best", "base_combo_policy": "keep_current_best", "target_modules": ["localization"], "rationale": "..."}},
  "evolution_design": {{
    "target_modules": ["localization"],
    "initial_focus_modules": ["localization"],
    "design_summary": "...", "failure_chain": "producer → target behavior → decision", "evolution_mechanism": "...",
    "module_responsibilities": {{"localization": {{"responsibility": "...", "input_invariant": "...", "output_invariant": "...", "fallback": "..."}}}},
    "handoff_contracts": [{{"producer": "localization", "consumer": "perception", "required_fields": ["candidate_windows"], "output_protocol": "...", "consumer_behavior": "...", "fallback": "..."}}],
    "runtime_capability_needs": [{{"capability": "vlm", "profile_id": "profile from the supplied inventory", "reason": "...", "trigger": "...", "input_provenance": "active_time_window_frame", "fallback": "..."}}],
    "cost_guardrails": ["..."], "research_questions": ["..."], "probe_hypotheses": [{{"hypothesis_id": "...", "information_gap": "...", "intervention": "...", "trigger": "...", "expected_observable": "...", "consumer": "...", "success_condition": "...", "fallback": "...", "failure_kinds": ["..."]}}]
  }}
}}"""

    def _find_latest_evolution_review(self) -> str:
        """Find latest teacher_evolution_review.json if present."""
        # New path: evolution/ directory of the previous iteration
        prev = self.iter_paths.previous()
        if prev:
            candidates = glob.glob(os.path.join(prev.evolution, "**/teacher_evolution_review.json"), recursive=True)
            if candidates:
                candidates.sort(key=os.path.getmtime, reverse=True)
                return candidates[0]
        return ""

    def _load_evolution_review(self, path: str) -> dict:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _shorten(self, value, limit: int = 220) -> str:
        text = str(value or "").replace("\n", " ").strip()
        return text[:limit]

    def _has_prior_candidate_review(self) -> bool:
        """Return whether this diagnosis follows a real evolved candidate."""
        previous = getattr(self, "previous_evolution_result", None)
        if isinstance(previous, dict) and previous:
            return True
        review = getattr(self, "evolution_review", {}) or {}
        # Staged MetaVideoAgent reviews record accepted/rejected candidate lineage
        # under retention_review instead of the previous candidate_lineage key.
        # Treat that paired outcome as prior-candidate evidence so corrections
        # and regressions remain preservation/guard rows in review→diagnosis.
        retention = review.get("retention_review", {}) or {}
        post_status = str(
            retention.get("post_evolution_status")
            or ((retention.get("candidate_bundle") or {}).get("post_evolution_status"))
            or ""
        ).strip()
        if post_status in {"accepted_new_current_best", "rejected_candidate", "rejected"}:
            return True
        lineage = review.get("candidate_lineage", {}) or {}
        if not isinstance(lineage, dict):
            return False
        candidate_run = str(lineage.get("candidate_run") or "").strip()
        candidate_combo = str(lineage.get("candidate_combo_id") or "").strip()
        parent_combo = str(lineage.get("parent_reference_combo_id") or "").strip()
        return bool(candidate_run or (candidate_combo and candidate_combo != parent_combo))

    def _build_validation_policy(self, report: dict,
                                 constraints: dict = None) -> dict:
        constraints = constraints or {}
        algo = report.get("algorithmic_evolution", {}) or {}
        decision = report.get("evolution_decision", {}) or {}
        probe_plan = decision.get("probe_plan", {}) or {}
        programmable = report.get("programmable_evolution", {}) or {}
        unchanged = (
            programmable.get("unchanged_failure_analysis")
            or constraints.get("unchanged_failure_summary")
            or {}
        )
        validation_expectations = (
            (programmable.get("implementation_contract", {}) or {})
            .get("validation_expectations", {})
            or {}
        )
        repair_refs = []
        guard_refs = []

        def _ref_value(ref) -> str:
            if isinstance(ref, dict):
                return str(ref.get("task_ref") or ref.get("task_id") or ref.get("time_reference") or "")
            return str(ref or "")

        def _extend_unique(target: list, refs) -> None:
            for ref in refs or []:
                ref = _ref_value(ref)
                canonical = canonical_task_ref(ref)
                # Reviews may spell one task either with a stable hash or as
                # the equivalent video/window reference. Keep the first
                # spelling for audit, but never count it twice in a probe set.
                already_present = any(
                    (canonical and canonical_task_ref(existing) == canonical)
                    or (not canonical and existing == ref)
                    for existing in target
                )
                if ref and not already_present:
                    target.append(ref)

        _extend_unique(repair_refs, probe_plan.get("repair_probe", []))
        _extend_unique(repair_refs, validation_expectations.get("repair_probe_refs", []))
        _extend_unique(repair_refs, algo.get("target_questions", []))
        _extend_unique(repair_refs, report.get("target_questions", []))

        refs_by_category = (
            unchanged.get("refs_by_category")
            or probe_plan.get("failure_clusters")
            or validation_expectations.get("unchanged_failure_refs_by_category")
            or {}
        )
        failure_clusters = {}
        for category, refs in refs_by_category.items():
            # Incomplete Teacher reviews are retained in the review audit but
            # must never become synthetic repair targets or probe rows.
            if category == "unattributed_or_incomplete":
                continue
            refs = [ref for ref in refs or [] if ref]
            if refs:
                failure_clusters[category] = refs
                _extend_unique(repair_refs, refs[:1])

        _extend_unique(guard_refs, probe_plan.get("regression_guard", []))
        _extend_unique(guard_refs, validation_expectations.get("regression_guard_refs", []))
        _extend_unique(
            guard_refs,
            constraints.get("regression_guards", []),
        )

        has_prior_candidate_review = bool(
            report.get("has_prior_candidate_review", self._has_prior_candidate_review())
        )
        # Only a completed candidate's paired review may supply historical
        # correction/regression guards. Current repair rows are baseline-wrong
        # opportunities, never a fallback source for those roles.
        previous_corrections = list(probe_plan.get("previous_candidate_corrections") or [])
        previous_regressions = list(probe_plan.get("previous_candidate_regressions") or [])
        if not has_prior_candidate_review:
            previous_corrections = []
            previous_regressions = []
            guard_refs = []
        else:
            _extend_unique(guard_refs, previous_corrections)
        return {
            "mode": "cluster_aware_repair_plus_guard",
            "repair_probe": repair_refs,
            "regression_guard": guard_refs,
            "failure_clusters": failure_clusters,
            "previous_candidate_corrections": previous_corrections,
            "previous_candidate_regressions": previous_regressions,
            "selection_rules": [
                "cover at least one question from each unchanged failure cluster when budget allows",
                "guard previous candidate corrections when a paired review exists",
                "retain previous regressions as explicit regression guards",
                "prefer baseline-correct questions for regression guards",
            ],
        }



    def _evolution_review_constraints(self) -> dict:
        metrics = self.evolution_review.get("metrics", {})
        macro = self.evolution_review.get("macro_review", {})
        retention = self.evolution_review.get("retention_review", {})
        delta_reviews = self.evolution_review.get("delta_reviews", [])
        regressions = [
            item.get("task_id") or item.get("time_reference", "")
            for item in delta_reviews
            if item.get("change_type") == "regression" and (item.get("task_id") or item.get("time_reference"))
        ]
        corrections = [
            item.get("task_id") or item.get("time_reference", "")
            for item in delta_reviews
            if item.get("change_type") == "fix" and (item.get("task_id") or item.get("time_reference"))
        ]
        macro_unchanged = dict(macro.get("unchanged_failure_analysis") or {})
        canonical_unchanged = self.evolution_review.get("unchanged_failure_summary", {}) or {}
        # Full micro-review aggregation is factual evidence. A macro review
        # may interpret it, but may not replace its counts or references.
        if canonical_unchanged:
            macro_unchanged["taxonomy"] = canonical_unchanged.get("taxonomy", [])
            macro_unchanged["counts"] = canonical_unchanged.get("counts", {})
            macro_unchanged["refs_by_category"] = canonical_unchanged.get("refs_by_category", {})
        constraints = {
            "regression_guards": regressions,
            "candidate_correction_evidence": corrections,
            "retention_review": retention,
            "unchanged_failure_summary": self.evolution_review.get(
                "unchanged_failure_summary", {}),
            "programmable_macro_fields": {
                "unchanged_failure_analysis": macro_unchanged,
                "module_responsibility": macro.get(
                    "module_responsibility", {}),
                "diagnosis_evidence_seed": macro.get(
                    "diagnosis_evidence_seed",
                    macro.get("review_evidence_seed", {}),
                ),
                "review_evidence_seed": macro.get("review_evidence_seed", {}),
                "candidate_retention_evidence": macro.get(
                    "candidate_retention_evidence", {}),
            },
            "net_delta": metrics.get("net_delta"),
            "decision_rule": (
                "The pre-evolution current best remains the only execution baseline. "
                "When the post-evolution result was accepted, preserve it and use "
                "regressions as guards; when rejected, retain the previous current best "
                "and use the candidate only as evidence. Diagnosis selects one connected "
                "target_modules set and explains its mechanism and handoff scope "
                "in evolution_decision.rationale."
            ),
        }
        return constraints

    @staticmethod
    def _model_safe_mechanism_summary(summary: dict) -> dict:
        """Keep outer-round diagnosis aggregate and task-free.

        The review artifact is intentionally rich so it can later support
        supervised probe/Codex self-debug.  Diagnosis has a different job: it
        chooses a cross-question mechanism, not a per-question patch.  Build
        a whitelisted aggregate instead of passing the rich artifact through
        and relying on prompt prose to stop task-specific leakage.
        """
        summary = summary if isinstance(summary, dict) else {}

        def text(value: object) -> str:
            value = str(value or "")
            # Remove task/video identifiers, concrete temporal windows and
            # evidence-event IDs if a future macro template embeds them in a
            # natural-language field.
            value = re.sub(r"\b(?:BV|B1)[A-Za-z0-9]+(?:\s*::\s*\[\[.*?\]\])?(?:\s*::\s*[0-9a-f]{6,})?", "[task]", value)
            value = re.sub(r"\[\[?\s*\d+(?:\.\d+)?\s*(?:,|-|–)\s*\d+(?:\.\d+)?\s*\]?\]", "[window]", value)
            value = re.sub(r"\bev_\d+\b", "[event]", value)
            return value.strip()

        outcome = summary.get("outcome") or {}
        current = summary.get("current_best_failure_summary") or {}
        safe = {
            "review_semantics": str(summary.get("review_semantics") or ""),
            "outcome": {
                "net_delta": outcome.get("net_delta"),
                "overall_verdict": text(outcome.get("overall_verdict")),
                "verdict_reason": text(outcome.get("verdict_reason")),
            },
            "current_best_failure_summary": {
                "scope": str(current.get("scope") or ""),
                "overall_failure_chain": [text(item) for item in (current.get("overall_failure_chain") or []) if text(item)],
                "dominant_observed_defects": [text(item) for item in (current.get("dominant_observed_defects") or []) if text(item)],
            },
            "bundle_level_improvement_opportunities": [],
            "module_signals": {},
            "macro_validation": {
                "valid": bool((summary.get("macro_validation") or {}).get("valid")),
                "issues": [text(item) for item in ((summary.get("macro_validation") or {}).get("issues") or []) if text(item)],
            },
        }
        for item in summary.get("bundle_level_improvement_opportunities") or []:
            if not isinstance(item, dict):
                continue
            safe["bundle_level_improvement_opportunities"].append({
                "modules": [str(module) for module in (item.get("modules") or [])],
                "observed_pattern": text(item.get("observed_pattern")),
                "handoff_or_state_to_repair": text(item.get("handoff_or_state_to_repair")),
                "generic_repair_direction": text(item.get("generic_repair_direction")),
            })
        for module, signal in (summary.get("module_signals") or {}).items():
            if isinstance(signal, dict):
                safe["module_signals"][str(module)] = {
                    "level": str(signal.get("level") or ""),
                    "reason": text(signal.get("reason")),
                }
        return safe

    def _format_evolution_review_for_prompt(self) -> str:
        """Inject only task-free macro conclusions into outer diagnosis."""
        if not self.evolution_review:
            return ""
        review_digest = self._model_safe_mechanism_summary(
            self.evolution_review.get("mechanism_summary", {}) or {}
        )
        # Coverage is audit metadata, not target evidence.  Make omissions
        # visible so the model neither overstates review support nor treats a
        # sparse MFP subset as the full failure distribution.
        causal = self.evolution_review.get("causal_effect_coverage") or {}
        mfp = self.evolution_review.get("minimal_failure_point_coverage") or {}
        def invalid_count(payload: dict) -> int:
            invalid = payload.get("invalid", 0)
            return len(invalid) if isinstance(invalid, list) else int(invalid or 0)

        review_digest["audit_coverage"] = {
            "causal_effect_records": {
                "required": int(causal.get("required") or 0),
                "valid": int(causal.get("valid") or 0),
                "invalid": invalid_count(causal),
            },
            "minimal_failure_points": {
                "required": int(mfp.get("required") or 0),
                "valid": int(mfp.get("valid") or 0),
                "invalid": invalid_count(mfp),
            },
            "rule": (
                "Only validated causal/MFP records may support a stated mechanism or probe witness. "
                "Invalid or missing records are audit omissions, not negative evidence and not repair targets."
            ),
        }
        macro = self.evolution_review.get("macro_review", {}) or {}
        cost_assessment = macro.get("cost_and_efficiency_assessment", {}) or {}
        cost_issues = []
        for item in macro.get("cost_audit_issues", []) or []:
            if not isinstance(item, dict):
                continue
            cost_issues.append({
                "code": str(item.get("code") or ""),
                "severity": str(item.get("severity") or ""),
                "repair_goal": sanitize_tool_terms_for_evolution(
                    self._shorten(item.get("repair_goal"), 300)
                ),
            })
        if cost_assessment or cost_issues:
            review_digest["cost_constraints"] = {
                "enforced": any(item.get("code") == "cost_explosion" for item in cost_issues),
                "review_target": str(cost_assessment.get("next_round_cost_target") or ""),
                "avoid_repeating": [
                    sanitize_tool_terms_for_evolution(self._shorten(value, 300))
                    for value in (macro.get("avoid_repeating") or [])
                    if value
                ],
                "issues": cost_issues,
                "rule": (
                    "When enforced, preserve accuracy repairs and regression guards while adding "
                    "bounded, evidence-triggered escalation and duplicate-work prevention to "
                    "evolution_design.cost_guardrails."
                ),
            }
        validation_feedback = list(getattr(self, "brief_validation_feedback", []) or [])
        feedback_text = ""
        if validation_feedback:
            feedback_text = (
                "\n## Previous brief compiler feedback\n"
                "The prior diagnosis was structurally insufficient for the deterministic five-module brief. "
                "Correct these contract facts in the JSON below; do not add task-specific rules:\n- "
                + "\n- ".join(validation_feedback)
                + "\n"
            )
        return (
            "## TeacherAgent Full review after evolution \\n"
            "The following is a mechanism-level conclusion based on all topic-by-topic micro-reviews. It gives the specific cross-topic defects of the current best,"
            "Module handover status and provable improvement opportunities; it is not implementation code, nor is it a module whitelist from the inner Codex. \\n"
            "Hard constraint: target_questions should come from current-best repair/unchanged failure evidence;"
            "All regressions must enter probe_plan.regression_guard instead of pretending to be problems to be fixed;"
            "All corrections must be interpreted as retention or reuse mechanisms;"
            "evolution_decision must keep the accepted current best as the execution base and "
            "name the same non-empty target_modules set as evolution_design."
            "macro_review is review evidence, not an evolutionary plan;"
            "Every selected module must have a causal role in the mechanism rather than a task-specific patch."
            "The only initial current best review does not include pre- and post-candidate differences: no fake fixes, regressions, or last-round candidates are allowed."
            "These comparison terms should only be used when there are real candidate controls."
            "The next round must be generated independently based on the mechanism summary and the fact constraints of the local evidence ledger."
            "programmable_evolution or algorithmic_evolution field."
            "If cost_constraints.enforced of review is true, evolution_design.cost_guardrails must also"
            "Includes: compact evidence prioritization, deduplication retrieval/replanning, and justified and capped scoped awareness upgrades; may not be used as"
            "Reduce correctness or skip LLM final inference to save calls."
            "\n"
            f"{json.dumps(review_digest, ensure_ascii=False, indent=2)}"
            f"{feedback_text}"
        )

    def _load_trajectories(self) -> dict:
        """Load and cache trajectory data, grouped by task_id."""
        if self._trajectories is not None:
            return self._trajectories

        if self.reference_results_path:
            from reference_loader import load_reference_grouped
            trajs = load_reference_grouped(
                self.reference_results_path,
                reference_label=self.reference_label,
            )
        else:
            trajectory_dir = self.iter_paths.trajectories
            trajs = self._load_trajectories_from_dir(trajectory_dir)
        if self.distribution_manifest:
            allowed = set(self._load_dataset().keys())
            trajs = {
                key: value for key, value in trajs.items()
                if key in allowed
            }
        self._trajectories = trajs
        return trajs


    def _load_trajectories_from_dir(self, trajectory_dir: str) -> dict:
        """Load trajectories from the trajectory directory, grouped by task_id, with the latest record first."""
        pattern = os.path.join(trajectory_dir, "*_trajectory.jsonl")
        files = glob.glob(pattern)

        trajs = {}
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ident = identity_from_entry(entry)
                    if not ident["task_id"]:
                        continue
                    trajs.setdefault(ident["task_id"], []).append(entry)

        for entries in trajs.values():
            entries.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
        return trajs

    def _load_dataset(self) -> dict:
        """Load the dataset, indexed by task_id."""
        if self._dataset is not None:
            return self._dataset

        if self.distribution_manifest:
            from question_set_loader import questions_by_ref
            by_ref, _question_to_ref, _ids, _spec = questions_by_ref(
                self.workspace_dir,
                video_id="ALL",
                distribution_manifest=self.distribution_manifest,
                preferred_splits=("train",),
            )
            self._dataset = by_ref
            return self._dataset

        ds_dir = os.path.join(self.workspace_dir, "datasets")
        if not os.path.isdir(ds_dir):
            # Try to go up a level
            ds_dir = os.path.join(os.path.dirname(self.workspace_dir), "datasets")

        self._dataset = {}
        if os.path.isdir(ds_dir):
            for fp in glob.glob(os.path.join(ds_dir, "*.jsonl")):
                video_id = os.path.splitext(os.path.basename(fp))[0]
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = sanitize_task_for_evolution(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                        entry.setdefault("video_id", video_id)
                        task_id = make_task_id(
                            video_id,
                            entry.get("time_reference", ""),
                            entry.get("question", ""),
                        )
                        entry.setdefault("task_id", task_id)
                        self._dataset[task_id] = entry

        return self._dataset

    # ==================================================================
    # Output parsing
    # ==================================================================

    def _parse_report(self, raw_text: str, *, allow_llm_repair: bool = True) -> dict:
        """Extract JSON diagnostic reports from LLM output."""
        # Strip the <think>...</think> blocks first (part of the reasoning model will be returned)
        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', raw_text).strip()

        # Try to extract from the ```json ``` block
        m = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned)
        if m:
            parsed = self._try_parse_json(m.group(1))
            if parsed:
                return self._normalize_report(parsed)

        # Also try the original text (in case the think tag is incomplete)
        m = re.search(r'```json\s*([\s\S]*?)\s*```', raw_text)
        if m:
            parsed = self._try_parse_json(m.group(1))
            if parsed:
                return self._normalize_report(parsed)

        # Try bare JSON (found in both cleaned and raw_text)
        for source in [cleaned, raw_text]:
            m = re.search(r'\{[\s\S]*"algorithmic_evolution"[\s\S]*\}', source)
            if m:
                parsed = self._try_parse_json(m.group())
                if parsed:
                    return self._normalize_report(parsed)

        if not allow_llm_repair:
            return self._fallback_report(raw_text)

            print("   Local JSON parsing failed; requesting structured JSON repair...")

        # Only pass the json block content to format_json, not the entire text
        json_only = ""
        m_block = re.search(r'```json\s*([\s\S]*?)\s*```', raw_text)
        if m_block:
            json_only = m_block.group(1)
        else:
            m_bare = re.search(r'\{[\s\S]*"algorithmic_evolution"[\s\S]*\}', raw_text)
            if m_bare:
                json_only = m_bare.group()

        if not json_only:
            print("⚠️ Unable to extract JSON chunks from text")
            return self._fallback_report(raw_text)

            print(f"   Extracted JSON candidate ({len(json_only)} chars); preview: {json_only[:200]}")
        repaired = format_json(json_only, max_tokens=DIAGNOSIS_OUTPUT_MAX_TOKENS)
        if repaired and isinstance(repaired, dict):
            return self._normalize_report(repaired)

        # Preserve a deterministic parse-failure artifact when structured repair fails.
        print("   Unable to repair the JSON response; using a deterministic parse-failure report")
        return self._fallback_report(raw_text)

    def _try_parse_json(self, text: str) -> dict:
        """Attempts to parse JSON, including automatic fixes for common formatting issues."""
        text = text.strip()

        # Try it directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fix 1: Remove trailing commas (comma before }, ]
        fixed = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Fix 2: remove text before the first opening brace and after the last closing brace.
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace >= 0 and last_brace > first_brace:
            subset = text[first_brace:last_brace + 1]
            try:
                return json.loads(subset)
            except json.JSONDecodeError:
                pass
            # Try again to remove the trailing comma
            fixed = re.sub(r',\s*([}\]])', r'\1', subset)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        # Fix 3: Truncated JSON (try completing missing brackets)
        if text.count('{') > text.count('}'):
            diff = text.count('{') - text.count('}')
            patched = text + '}' * diff
            patched = re.sub(r',\s*([}\]])', r'\1', patched)
            try:
                return json.loads(patched)
            except json.JSONDecodeError:
                pass

        return None

    def _unique_refs(self, refs) -> list:
        seen = set()
        out = []
        for ref in refs or []:
            if ref and ref not in seen:
                seen.add(ref)
                out.append(ref)
        return out

    def _normalize_report(self, report: dict) -> dict:
        """Normalize only the five-module bundle diagnosis schema."""
        report = dict(report or {})
        report.setdefault("timestamp", int(time.time()))
        report.setdefault("total_questions", 0)
        report.setdefault("correct", 0)
        report.setdefault("incorrect", 0)
        report.setdefault("fault_distribution", {})
        report.setdefault("engineering_fixes", [])
        if self.distribution_manifest:
            report.setdefault("distribution_manifest", self.distribution_manifest)

        algo = report.setdefault("algorithmic_evolution", {})
        design = report.setdefault("evolution_design", {})
        decision = report.setdefault("evolution_decision", {})
        modules = (
            design.get("target_modules")
            or algo.get("target_modules")
            or decision.get("target_modules")
            or []
        )
        if isinstance(modules, list):
            design["target_modules"] = list(modules)
            algo["target_modules"] = list(modules)
            decision["target_modules"] = list(modules)
        algo.setdefault("evolution_hint", "")
        algo.setdefault("target_questions", [])
        report["target_questions"] = list(algo.get("target_questions") or [])
        decision["combo_base"] = "current_best"
        decision["base_combo_policy"] = "keep_current_best"
        decision.setdefault("probe_plan", {})
        decision.setdefault("rationale", "")

        if self.evolution_review:
            constraints = self._evolution_review_constraints()
            programmable = constraints.get("programmable_macro_fields") or {}
            if programmable:
                report["programmable_evolution"] = programmable

        report["base_combo_policy"] = "keep_current_best"
        report["probe_plan"] = decision.get("probe_plan") or {}
        report = sanitize_tool_terms_for_evolution(report)

        if "error" not in self._cost_baseline:
            cost_reference = {
                "per_question_avg": self._cost_baseline["per_question_avg"],
                "cost_score": self._cost_baseline["cost_score_avg"],
                "cost_budget": self._cost_baseline["cost_budget"],
                "max_cost_ratio": self._cost_baseline["max_cost_ratio"],
                "reference_label": self.reference_label,
                "reference_results_path": self.reference_results_path,
            }
            report["cost_reference"] = cost_reference
            report["cost_baseline"] = cost_reference
        report["evolution_phase_policy"] = dict(self.evolution_phase_policy)
        return report


    def _fallback_report(self, raw_text: str) -> dict:
        """Return an invalid bundle report that fails closed after parsing."""
        return {
            "timestamp": int(time.time()),
            "total_questions": 0,
            "correct": 0,
            "incorrect": 0,
            "fault_distribution": {},
            "engineering_fixes": [],
            "algorithmic_evolution": {
                "target_modules": [],
                "evolution_hint": "",
            },
            "evolution_decision": {
                "combo_base": "current_best",
                "base_combo_policy": "keep_current_best",
                "target_modules": [],
            },
            "evolution_design": {"target_modules": []},
            "target_questions": [],
            "_raw_output": raw_text[:2000],
            "_parse_error": "Unable to extract valid JSON from LLM output",
        }


    # ==================================================================
    # Auxiliary
    # ==================================================================


    def _detect_video_ids(self) -> list:
        """Automatically detect all video_ids."""
        if getattr(self, "distribution_manifest", ""):
            try:
                from question_set_loader import questions_by_ref
                _by_ref, _q_to_ref, ids, _spec = questions_by_ref(
                    self.workspace_dir,
                    video_id="ALL",
                    distribution_manifest=self.distribution_manifest,
                    preferred_splits=("train",),
                )
                if ids:
                    return ids
            except Exception:
                pass
        # Prioritize inference from trajectories of the current iteration
        ids = self.iter_paths.detect_video_ids()
        if ids:
            return ids
        # Direct workspace mode uses the canonical trajectory directory.
        trajectory_dir = os.path.join(self.workspace_dir, "trajectories")
        if os.path.isdir(trajectory_dir):
            ids = [f.replace("_trajectory.jsonl", "") for f in sorted(os.listdir(trajectory_dir))
                   if f.endswith("_trajectory.jsonl")]
        if ids:
            return ids
        # Inferred from datasets directory
        ds_dir = os.path.join(self.workspace_dir, "datasets")
        if not os.path.isdir(ds_dir):
            ds_dir = os.path.join(os.path.dirname(self.workspace_dir), "datasets")
        if os.path.isdir(ds_dir):
            for f in sorted(os.listdir(ds_dir)):
                if f.endswith(".jsonl"):
                    ids.append(os.path.splitext(f)[0])
        return ids


    def _diagnosis_dir(self) -> str:
        run_id = os.environ.get("METAVIDEOAGENT_RUN_ID", "")
        iteration = os.environ.get("METAVIDEOAGENT_ITERATION_INDEX", "")
        try:
            iteration_index = int(iteration)
        except (TypeError, ValueError):
            iteration_index = 0
        if iteration_index > 0:
            return os.path.join(
                metavideoagent_run_dir(run_id),
                f"iter_{iteration_index}",
                "diagnosis",
            )
        return os.path.join(metavideoagent_run_dir(run_id), "diagnosis")

    def _save_report(self, report: dict):
        """Save diagnostic reports to the diagnosis/ directory of the current iteration."""
        diag_dir = self._diagnosis_dir()
        os.makedirs(diag_dir, exist_ok=True)
        ts = report.get("timestamp", int(time.time()))
        suffix = f"_attempt{self.diagnosis_attempt}" if self.diagnosis_attempt > 1 else ""
        path = os.path.join(diag_dir, f"diagnosis_{ts}{suffix}.json")
        report["report_path"] = path
        capsules = report.get("diagnosis_failure_capsules") or []
        capsule_path = ""
        if capsules:
            capsule_path = os.path.join(diag_dir, f"diagnosis_failure_capsules_{ts}{suffix}.json")
            report["diagnosis_failure_capsules_path"] = capsule_path
        # Remove internal fields that do not need to be persisted
        save_report = {k: v for k, v in report.items() if not k.startswith("_")}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_report, f, ensure_ascii=False, indent=2)
        if capsules:
            with open(capsule_path, "w", encoding="utf-8") as f:
                json.dump({
                    "schema_version": 1,
                    "artifact_type": "diagnosis_failure_capsules",
                    "target_modules": list(
                        (save_report.get("evolution_decision") or {}).get("target_modules") or []
                    ),
                    "capsules": capsules,
                }, f, ensure_ascii=False, indent=2)
        print(f"\nDiagnosis report: {path}")


# =====================================================================
# CLI entry
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run MetaVideoAgent's adaptive evolution diagnosis stage."
    )
    parser.add_argument("--workspace", type=str,
                        default=default_workspace())
    parser.add_argument("--iteration", type=int, default=None,
                        help="Current iteration number (automatically detects the latest by default)")
    parser.add_argument("--previous-evolution-result", type=str, default="",
                        help="The last round of evolution conclusion JSON, used to enter evaluation mode")
    parser.add_argument("--previous-diagnosis", type=str, default="",
                        help="Last round diagnostic report JSON for evaluating schema context")
    parser.add_argument("--evolution-review", type=str, default="",
                        help="Candidate review artifact produced by the review stage")
    parser.add_argument("--distribution-profile", type=str, default="",
                        help="label-blind observed_distribution_profile.json; artificial type descriptions will not be read")
    parser.add_argument("--distribution-manifest", type=str, default="",
                        help="Distribution manifest used to limit the split and question set; semantic labels remain hidden from prompts")
    parser.add_argument("--reference-results", type=str, default="",
                        help="Explicit reference/current-best results JSONL or sandbox directory; must be passed under MetaVideoAgent or inferred from reference report")
    parser.add_argument("--reference-label", type=str, default="",
                        help="reference tag, such as initial_combo/current_best")
    parser.add_argument("--iter-context", type=str, default="",
                        help="iter_context.json produced by MetaVideoAgent orchestrator")
    parser.add_argument("--diagnosis-attempt", type=int, default=1,
                        help="The diagnosis attempt number is used for auditing; when it is greater than 1, the product is appended with the attempt suffix")
    args = parser.parse_args()
    if args.distribution_manifest:
        require_metavideoagent_runtime("diagnosis_agent --distribution-manifest")

    # Automatically load evolutionary memory. MetaVideoAgent manifest runs use run-scoped
    # iter_context/current-best ledgers instead of shared workspace history, so
    # unrelated runs cannot leak into diagnosis.
    memory_str = ""
    if not args.distribution_manifest:
        from workspace_paths import IterationPaths
        iter_paths = IterationPaths(args.workspace, iteration=args.iteration)
        memory_str = iter_paths.get_all_evolution_conclusions()

    def _load_optional_json(path: str) -> dict:
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    previous_evolution_result = _load_optional_json(args.previous_evolution_result)
    previous_diagnosis = _load_optional_json(args.previous_diagnosis)
    observed_distribution_profile = _load_optional_json(args.distribution_profile)
    if args.distribution_manifest and not observed_distribution_profile:
        try:
            from distribution_profiler import build_observed_distribution_profile
            from distribution_spec import load_distribution_spec
            spec = load_distribution_spec(args.distribution_manifest)
            out_dir = os.path.join(
                metavideoagent_run_dir(os.environ.get("METAVIDEOAGENT_RUN_ID", "")),
                "profiles",
            )
            out_path = os.path.join(
                out_dir, f"observed_distribution_profile_{int(time.time())}.json"
            )
            observed_distribution_profile = build_observed_distribution_profile(
                args.workspace,
                distribution_spec=spec,
                output_path=out_path,
                include_execution_artifacts=False,
            )
            print(f"Observed distribution profile generated: {out_path}")
        except Exception as e:
            print(f"⚠️ Could not generate the observed distribution profile; continuing without it: {e}")

    agent = DiagnosisAgent(
        args.workspace,
        evolution_memory=memory_str,
        previous_evolution_result=previous_evolution_result or None,
        previous_diagnosis=previous_diagnosis or None,
        evolution_review_path=args.evolution_review or None,
        iteration=args.iteration,
        observed_distribution_profile=observed_distribution_profile,
        distribution_manifest=args.distribution_manifest,
        reference_results_path=args.reference_results,
        reference_label=args.reference_label,
        iter_context_path=args.iter_context,
        diagnosis_attempt=args.diagnosis_attempt,
    )
    report = agent.run()
    print("\n" + "=" * 70)
    print(
        "Evolution targets: "
        + ", ".join(report.get("algorithmic_evolution", {}).get("target_modules") or [])
    )
    print(f"Evolution direction: {report['algorithmic_evolution']['evolution_hint']}")
    print(f"Engineering fixes: {len(report['engineering_fixes'])} item(s)")
    print(f"Targeted questions: {report['target_questions']}")
