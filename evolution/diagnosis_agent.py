"""Evidence-grounded diagnosis for the MetaVideoAgent evolution loop.

Diagnosis investigates completed Teacher reviews, training traces, the current
agent and previous attempts. Research and Codex retain the causal reasoning
and supporting materials. Diagnosis never reviews raw media or held-out data.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

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
    require_metavideoagent_runtime,
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
            need.setdefault("profile_id", DEFAULT_VLM_PROFILE_ID)
        elif capability == "llm":
            # Diagnosis and execution may use different configured models.
            need.setdefault("profile_id", DEFAULT_EXECUTION_LLM_PROFILE_ID)
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


def _normalize_evidence_refs(refs, *, field: str) -> list[str]:
    """Accept task-reference objects as well as strings, keeping first spelling."""
    if refs is None:
        return []
    if not isinstance(refs, list):
        raise ValueError(f"evidence_strategy.{field} must be a list of task references")
    result = []
    seen = set()
    for item in refs:
        video_id = ""
        if isinstance(item, dict):
            video_id = str(item.get("video_id") or "")
            item = item.get("task_ref") or item.get("task_id") or item.get("time_reference")
        if isinstance(item, (list, tuple)):
            value = canonical_task_ref(item, video_id=video_id)
        elif isinstance(item, str):
            value = item.strip()
        else:
            value = ""
        if not value:
            raise ValueError(f"evidence_strategy.{field} contains an empty task reference")
        canonical = canonical_task_ref(value)
        if canonical not in seen:
            seen.add(canonical)
            result.append(value)
    return result


def _normalize_evidence_strategy(strategy) -> dict:
    if strategy is None:
        return {}
    if not isinstance(strategy, dict):
        raise ValueError("evidence_strategy must be an object")
    normalized = dict(strategy)
    for key in ("target_evidence_refs", "optional_mfp_witness_refs", "preserve_correction_refs",
                "regression_guard_refs", "unchanged_failure_refs"):
        if key in normalized:
            normalized[key] = _normalize_evidence_refs(normalized[key], field=key)
    return normalized


def complete_evidence_strategy(design: dict, ledger: dict, *, targets: list[str]) -> dict:
    """Supply missing probe references without replacing the model's reasoning.

    Diagnosis owns its causal explanation and cited witnesses. The ledger adds
    defaults when omitted and ensures comparison guards are not lost downstream.
    """
    completed = dict(design or {})
    ledger = ledger if isinstance(ledger, dict) else {}
    strategy = _normalize_evidence_strategy(completed.get("evidence_strategy"))
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

    strategy.setdefault("target_evidence_refs", target_refs)
    strategy.setdefault("optional_mfp_witness_refs", direct_mfp_refs)
    strategy.setdefault("target_evidence_basis", evidence_basis)
    strategy.setdefault("selection_basis", evidence_basis)
    for key, refs in (
        ("preserve_correction_refs", corrections),
        ("regression_guard_refs", regressions),
        ("unchanged_failure_refs", unchanged),
    ):
        strategy[key] = _normalize_evidence_refs([*(strategy.get(key) or []), *refs], field=key)
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
        "Retain regression references as probe guards and assess their risk alongside corrections and full-train accuracy.",
    )
    strategy.setdefault(
        "unchanged_failure_interpretation",
        "Use unchanged failures as causal evidence of whether the selected mechanism was blocked, untriggered, or ineffective.",
    )
    strategy.setdefault("source", "diagnosis_with_ledger_defaults")
    completed["evidence_strategy"] = strategy
    return completed


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
# a cross-question causal design. Operational fields are attached to the same
# report; Research and Codex can still read its complete analysis and evidence.
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

    MAX_ROUNDS = 24

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
                 iter_context_path: str = ""):
        self.workspace_dir = workspace_dir
        from iteration_workspace import IterationWorkspace
        iter_context = iter_context or self._load_evolution_review(iter_context_path)
        root = (iter_context or {}).get("iter_dir")
        if not root:
            raise ValueError("Diagnosis requires an explicit iter_context.iter_dir")
        self.iteration = IterationWorkspace(root)
        self.iteration_index = int((iter_context or {}).get("iter_index") or iteration or 1)
        self.distribution_manifest = distribution_manifest or (
            (previous_diagnosis or {}).get("distribution_manifest", "")
            if isinstance(previous_diagnosis, dict) else ""
        )
        self.video_ids = [video_id] if video_id else self._detect_video_ids()
        self.video_id = video_id or ("multi_video" if len(self.video_ids) > 1 else (self.video_ids[0] if self.video_ids else ""))
        self.evolution_memory = evolution_memory or ""
        self.previous_evolution_result = previous_evolution_result
        self.previous_diagnosis = previous_diagnosis  # Last round of diagnostic report (my previous output)
        self.evolution_review_path = evolution_review_path or ""
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
        if not self.reference_results_path:
            raise RuntimeError(
                "MetaVideoAgent diagnosis requires explicit reference_results_path; "
                "the public workflow does not infer reference trajectories."
            )
        self.observed_distribution_profile = observed_distribution_profile or (
            (previous_diagnosis or {}).get("observed_distribution_profile", {})
            if isinstance(previous_diagnosis, dict) else {}
        )
        from cost_calculator import compute_reference_cost
        from reference_loader import load_reference_trajectories
        self._reference_cost = compute_reference_cost(
            load_reference_trajectories(
                self.reference_results_path,
                reference_label=self.reference_label,
            )
        )
        from evolution_phase_policy import for_iteration
        self.evolution_phase_policy = for_iteration(self.iteration_index)

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
        payload = {
            "reference_label": self.reference_label,
            "current_best": self.iter_context.get("current_best", {}),
            "history": self.iter_context.get("history", {}),
            "previous_round": self.iter_context.get("previous_round", {}),
            "candidate_rejections": self.iter_context.get("candidate_rejections", []),
            "diagnosis_requirements": self.iter_context.get("diagnosis_requirements", {}),
            "previous_diagnosis": self.previous_diagnosis,
            "evolution_phase_policy": self.evolution_phase_policy,
        }
        return (
            "## Iteration history and open work\n"
            "The current best is the only execution baseline. Use previous reasoning, "
            "outcomes and unresolved questions to avoid restarting the investigation. "
            "Distinguish full-evaluation evidence, probe evidence and engineering failures. "
            "A previous target is neither a required next target nor a ban on that module. "
            "Read the named materials when the supporting trace or implementation matters.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _build_target_selection(self) -> dict:
        """Build the machine-only evidence ledger before LLM diagnosis."""
        self._target_selection = build_target_selection(
            self._load_trajectories(),
            review=self.evolution_review,
            iter_context=self.iter_context,
        )
        return self._target_selection

    def _format_target_selection_for_prompt(self) -> str:
        """Expose linked evidence, not just counts with the rationale removed."""
        return (
            "## Training evidence index\n"
            "This ledger indexes evidence; it does not choose the target. Compare plausible "
            "module-level explanations against the actual review and traces. Broad trace "
            "patterns support a distribution mechanism; validated minimal failure points "
            "are optional high-resolution witnesses, not module votes or global proof. "
            "Candidate corrections are behavior to preserve. Regressions remain guards, "
            "not relabelled repair claims. Explain whether unchanged failures were blocked, "
            "untriggered or unaffected. History is evidence, not a module blacklist.\n"
            "Record the evidence references you actually used and your selection rationale "
            "in evolution_design.evidence_strategy. Label unsupported causal links as "
            "hypotheses and explain bounded exploration when direct support is absent. "
            "Describe mechanism_scope: runtime applicability, covered failure kinds, "
            "producer/consumer handoff, direct-support scope and generalization scope. "
            "Supply 1-3 falsifiable probe hypotheses covering information gaps, interventions, "
            "observable runtime signals, consumer behavior and fallbacks. A complementary "
            "capability is allowed when its causal bridge is explained. Examples may support "
            "analysis; implementation triggers must generalize beyond task IDs or answers.\n"
            + json.dumps(self._target_selection or {}, ensure_ascii=False, indent=2)
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
        algo["evolution_hint"] = algo.get("evolution_hint") or (
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
        after `_apply_target_selection` adds the training evidence ledger.
        """
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
        self._attach_review_cost_contract(report)

    def _attach_review_cost_contract(self, report: dict) -> None:
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
        self._attempt_dir = self.iteration.reserve("diagnosis")
        self.diagnosis_attempt = int(self._attempt_dir.name.removeprefix("attempt_")) + 1
        (self._attempt_dir / "request.json").write_text(json.dumps({
            "context": self.iteration.context("diagnosis"),
            "reference_results": self.reference_results_path,
            "review": self.evolution_review_path,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
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

        print(f"Diagnosis startup [{mode}] (evidence investigation)")
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

        # Investigate the supplied evidence without launching new review work.
        try:
            raw_output = self._run_direct_diagnosis()
        except Exception as exc:
            failure_path = self._attempt_dir / "failure.json"
            failure_path.write_text(json.dumps({"error": str(exc), "next_task": {
                "stage": "diagnosis", "task": "Continue the evidence investigation recorded in rollout.json.",
            }}, indent=2), encoding="utf-8")
            self.iteration.record("diagnosis", failure_path, status="pending",
                                  next_task={"stage": "diagnosis", "error": str(exc)})
            raise
        finally:
            # Failed investigations also contain useful evidence and open work.
            self._save_rollout()

        # Extract diagnostic reports from LLM output
        report = (getattr(self, "_accepted_diagnosis_report", None)
                  or self._parse_report(raw_output, allow_llm_repair=False))
        report = self._apply_target_selection(report)

        # Complete the timestamp and save it
        report.setdefault("timestamp", int(time.time()))
        report["diagnosis_attempt"] = self.diagnosis_attempt
        report["diagnosis_rollout_path"] = getattr(self, "_rollout_path", "")
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
                "iter_dir": str(self.iteration.root),
                "stage_context": self.iteration.context("diagnosis"),
                "run_id": self.iter_context.get("run_id"),
                "iter_index": self.iter_context.get("iter_index"),
                "reference": self.iter_context.get("reference", {}),
                "current_best": self.iter_context.get("current_best", {}),
                "evolution_memory": self.iter_context.get("evolution_memory", {}),
                "previous_round": self.iter_context.get("previous_round", {}),
                "candidate_rejections": self.iter_context.get("candidate_rejections", [])[:3],
                "diagnosis_requirements": self.iter_context.get("diagnosis_requirements", {}),
            }
        report["reference_label"] = self.reference_label
        report["reference_policy"] = (
            "Teacher/Diagnosis trajectory tools read the explicit reference combo "
            "identified by reference_results_path."
        )
        self._save_report(report)

        return report

    def _save_rollout(self):
        """Preserve the investigation and evidence reads, including failed attempts."""
        if not hasattr(self, "_rollout_trace") or not self._rollout_trace:
            return
        ts = int(time.time())
        diag_dir = self._diagnosis_dir()
        os.makedirs(diag_dir, exist_ok=True)
        rollout_path = os.path.join(diag_dir, "rollout.json")
        self._rollout_path = rollout_path
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
        """Require an actionable diagnosis, not an early implementation schema."""
        from bundle_contract import normalize_modules, validate_handoff_contracts
        if not isinstance(report, dict):
            return ["diagnosis response is not a JSON object"]
        design = report.get("evolution_design") or {}
        if not isinstance(design, dict):
            return ["evolution_design must be an object"]
        try:
            modules = normalize_modules(design.get("target_modules"))
        except ValueError as exc:
            return [str(exc)]
        issues = []
        for field in ("design_summary", "failure_chain", "evolution_mechanism"):
            if not str(design.get(field) or "").strip():
                issues.append(f"evolution_design.{field} must explain the causal direction")
        issues.extend(validate_handoff_contracts(design.get("handoff_contracts"), changed_modules=modules))
        try:
            design["evidence_strategy"] = _normalize_evidence_strategy(design.get("evidence_strategy"))
            ledger = getattr(self, "_target_selection", {}) or {}
            if ledger.get("target_comparison"):
                report["evolution_design"] = complete_evidence_strategy(design, ledger, targets=modules)
        except (TypeError, ValueError) as exc:
            issues.append(str(exc))
        return issues


    def _diagnosis_materials(self) -> dict:
        from evolution.working_materials import material_paths
        materials = material_paths(self.iter_context or {})
        if self.evolution_review_path:
            materials["review"] = self.evolution_review_path
        # Use the manifest-filtered trajectory loader instead of unfiltered files.
        materials.pop("current_results", None)
        return materials

    def _read_diagnosis_material(self, request: dict) -> dict:
        from evolution.working_materials import read_material
        name = str(request.get("material") or "")
        if name != "trajectories":
            return read_material(
                self._diagnosis_materials(), name,
                offset=request.get("offset", 0), limit=request.get("limit", 160),
                module=str(request.get("module") or ""),
            )
        trajectories = self._load_trajectories()
        task_id = str(request.get("task_id") or "")
        if task_id:
            if task_id not in trajectories:
                raise ValueError("Unknown task_id; read the trajectory index first")
            payload = {task_id: trajectories[task_id]}
        else:
            payload = {"task_ids": list(trajectories)}
        lines = json.dumps(payload, ensure_ascii=False, indent=2).splitlines()
        start = max(0, int(request.get("offset", 0)))
        end = min(len(lines), start + max(1, min(400, int(request.get("limit", 160)))))
        return {
            "material": name, "task_id": task_id, "content": "\n".join(lines[start:end]),
            "next_offset": end if end < len(lines) else None, "total_lines": len(lines),
        }

    def _run_direct_diagnosis(self) -> str:
        """Investigate training materials, then produce a causal design.

        Read actions use ordinary JSON messages, so providers need not implement
        a particular tool-calling API. Transport retries do not consume read steps.
        """
        from openai import OpenAI

        self._rollout_trace = []
        client = OpenAI(api_key=EVOLUTION_LLM_API_KEY,
                        base_url=EVOLUTION_LLM_BASE_URL, timeout=API_TIMEOUT_SEC)
        messages = [
            {"role": "system", "content": self._adaptive_diagnosis_prompt()},
            {"role": "user", "content": (
                "Investigate the evidence and current implementation as needed. "
                "Then return the final diagnosis with analysis, open_questions and next_task."
            )},
        ]
        last_error = ""
        for step in range(1, self.MAX_ROUNDS + 1):
            for attempt in range(LLM_MAX_RETRIES):
                try:
                    content = self._call_diagnosis_llm_content(client, messages).strip()
                    break
                except Exception:
                    if attempt + 1 == LLM_MAX_RETRIES:
                        raise
            self._rollout_trace.append({"round": step, "response": content})
            messages.append({"role": "assistant", "content": content})
            try:
                raw = content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                action = json.loads(raw)
            except (ValueError, IndexError):
                action = {}
            if isinstance(action, dict) and action.get("action") == "read":
                try:
                    result = self._read_diagnosis_material(action)
                except (OSError, ValueError, TypeError) as exc:
                    result = {
                        "error": str(exc),
                        "available_materials": list(self._diagnosis_materials()) + ["trajectories"],
                    }
                self._rollout_trace[-1]["material_result"] = result
                messages.append({"role": "user", "content": json.dumps(result, ensure_ascii=False)})
                continue
            raw_report = action.get("report") if isinstance(action, dict) and action.get("action") == "final" else None
            try:
                report = (self._normalize_report(raw_report) if isinstance(raw_report, dict)
                          else self._parse_report(content, allow_llm_repair=False))
                issues = self._adaptive_execution_contract_issues(report)
            except (TypeError, ValueError, AttributeError) as exc:
                issues = [f"diagnosis fields cannot be used: {exc}"]
            if not issues:
                report["working_materials"] = self._diagnosis_materials()
                report.setdefault("next_task", {
                    "stage": "research",
                    "objective": (report.get("evolution_design") or {}).get("design_summary", ""),
                    "questions": (report.get("evolution_design") or {}).get("research_questions", []),
                })
                self._accepted_diagnosis_report = report
                return content
            last_error = "; ".join(issues)
            self._rollout_trace[-1]["feedback"] = last_error
            messages.append({"role": "user", "content": (
                "The diagnosis is not actionable yet: " + last_error
                + ". Continue from the evidence already read; correct the causal design, "
                "not unrelated fields. Return the final diagnosis JSON when ready."
            )})
        raise RuntimeError(
            f"diagnosis exceeded {self.MAX_ROUNDS} investigation steps: "
            + (last_error or "no final diagnosis")
        )

    def _adaptive_diagnosis_prompt(self) -> str:
        """Build the sole model prompt for the evolution LLM diagnosis design pass."""
        with open(os.path.join(_CURRENT_DIR, "agent_docs", "diagnosis_prompt.md"), encoding="utf-8") as handle:
            diagnosis_instructions = handle.read()
        return f"""You are the diagnosis stage of a five-module MetaVideoAgent evolution run.
{diagnosis_instructions}

Return one JSON object per turn. Do not edit files, inspect raw media, or propose task-specific runtime rules.
You may investigate supplied artifacts with {{"action":"read","material":"name","offset":0,"limit":160}}.
The response includes next_offset for continuing long materials. Available named materials:
{json.dumps(self._diagnosis_materials(), ensure_ascii=False)}
For an agent bundle, include "module":"thinking" (or another module) to read that source directly.
Use material="trajectories" for the task index, then include task_id to read a full training trace.
When ready, return the final diagnosis JSON in the shape below, or {{"action":"final","report":{{...}}}}.
Distinguish observed failures, plausible explanations and implementation questions still needing research.

Runtime capability inventory:
{json.dumps(build_runtime_capability_context(), ensure_ascii=False)}

Training evidence index:
{self._format_target_selection_for_prompt()}

Review findings:
{self._format_evolution_review_for_prompt()}

Previous iteration findings:
{self._format_iter_context_for_prompt()}

Required JSON shape:
{{
  "summary": "...",
  "analysis": "Evidence inspected, competing explanations, causal conclusion and limitations",
  "open_questions": ["Unresolved implementation or evidence questions"],
  "next_task": {{"stage": "research", "objective": "...", "questions": ["..."]}},
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

    def _load_evolution_review(self, path: str) -> dict:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _has_prior_candidate_review(self) -> bool:
        """Return whether this diagnosis follows a real evolved candidate."""
        previous = getattr(self, "previous_evolution_result", None)
        if isinstance(previous, dict) and previous:
            return True
        review = getattr(self, "evolution_review", {}) or {}
        # A paired retention outcome supplies correction/regression evidence.
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

    def _format_evolution_review_for_prompt(self) -> str:
        """Keep Teacher conclusions and their supporting references connected."""
        if not self.evolution_review:
            return ""
        review = self.evolution_review
        overview = {
            "mechanism_summary": review.get("mechanism_summary", {}),
            "macro_review": review.get("macro_review", {}),
            "causal_effect_coverage": review.get("causal_effect_coverage", {}),
            "minimal_failure_point_coverage": review.get("minimal_failure_point_coverage", {}),
            "review_material": self.evolution_review_path,
        }
        return (
            "## Completed Teacher review\n"
            "Use these conclusions as evidence, not an implementation plan or module "
            "whitelist. Read material=review for full per-question records and use "
            "material=trajectories with task_id for the corresponding runtime trace. "
            "Teacher tools are review-only; names in a trace do not make them runtime "
            "capabilities. Keep original identifiers intact so claims can be checked.\n"
            "Use corrections/regressions only when a real candidate comparison exists. "
            "For the initial agent there is no previous-candidate improvement claim. "
            "Cost findings should lead to bounded escalation, evidence prioritization and "
            "duplicate-work prevention, not skipped reasoning or reduced correctness.\n"
            + json.dumps(overview, ensure_ascii=False, indent=2)
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
            raise ValueError("Diagnosis requires the selected reference_results_path")
        if self.distribution_manifest:
            allowed = set(self._load_dataset().keys())
            trajs = {
                key: value for key, value in trajs.items()
                if key in allowed
            }
        self._trajectories = trajs
        return trajs


    def _load_dataset(self) -> dict:
        """Load the dataset, indexed by task_id."""
        if self._dataset is not None:
            return self._dataset

        if self.distribution_manifest:
            from question_set_loader import questions_by_ref
            by_ref, _question_to_ref, _ids, _spec = questions_by_ref(
                video_id="ALL",
                distribution_manifest=self.distribution_manifest,
                preferred_splits=("train",),
            )
            self._dataset = by_ref
            return self._dataset

        raise ValueError("Diagnosis requires a distribution_manifest for its selected train tasks")

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

        if "error" not in self._reference_cost:
            report["cost_reference"] = {
                "per_question_avg": self._reference_cost["per_question_avg"],
                "cost_score": self._reference_cost["cost_score_avg"],
                "reference_label": self.reference_label,
                "reference_results_path": self.reference_results_path,
            }
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
                    video_id="ALL",
                    distribution_manifest=self.distribution_manifest,
                    preferred_splits=("train",),
                )
                if ids:
                    return ids
            except Exception:
                pass
        return []


    def _diagnosis_dir(self) -> str:
        return str(self._attempt_dir)

    def _save_report(self, report: dict):
        """Save diagnostic reports to the diagnosis/ directory of the current iteration."""
        diag_dir = self._diagnosis_dir()
        os.makedirs(diag_dir, exist_ok=True)
        attempt = self._attempt_dir
        path = str(attempt / "result.json")
        report["report_path"] = path
        capsules = report.get("diagnosis_failure_capsules") or []
        capsule_path = ""
        if capsules:
            capsule_path = str(attempt / "failure_capsules.json")
            report["diagnosis_failure_capsules_path"] = capsule_path
        # Remove internal fields that do not need to be persisted
        save_report = {k: v for k, v in report.items() if not k.startswith("_")}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_report, f, ensure_ascii=False, indent=2)
        notes = attempt / "analysis.md"
        notes.write_text(str(report.get("analysis") or report.get("summary") or ""), encoding="utf-8")
        self.iteration.record("diagnosis", path, status="complete", notes_path=notes,
                              next_task=report.get("next_task") or {})
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
                        help="Current iteration number; defaults to the explicit iteration context.")
    parser.add_argument("--previous-evolution-result", type=str, default="",
                        help="The last round of evolution conclusion JSON, used to enter evaluation mode")
    parser.add_argument("--previous-diagnosis", type=str, default="",
                        help="Last round diagnostic report JSON for evaluating schema context")
    parser.add_argument("--evolution-review", type=str, default="",
                        help="Candidate review artifact produced by the review stage")
    parser.add_argument("--distribution-profile", type=str, required=True,
                        help="label-blind observed_distribution_profile.json; artificial type descriptions will not be read")
    parser.add_argument("--distribution-manifest", type=str, required=True,
                        help="Distribution manifest used to limit the split and question set; semantic labels remain hidden from prompts")
    parser.add_argument("--reference-results", type=str, required=True,
                        help="Explicit reference/current-best results JSONL or sandbox directory; must be passed under MetaVideoAgent or inferred from reference report")
    parser.add_argument("--reference-label", type=str, default="",
                        help="reference tag, such as initial_combo/current_best")
    parser.add_argument("--iter-context", type=str, required=True,
                        help="iter_context.json produced by MetaVideoAgent orchestrator")
    args = parser.parse_args()
    if args.distribution_manifest:
        require_metavideoagent_runtime("diagnosis_agent --distribution-manifest")

    def _load_optional_json(path: str) -> dict:
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    previous_evolution_result = _load_optional_json(args.previous_evolution_result)
    previous_diagnosis = _load_optional_json(args.previous_diagnosis)
    observed_distribution_profile = _load_optional_json(args.distribution_profile)
    agent = DiagnosisAgent(
        args.workspace,
        previous_evolution_result=previous_evolution_result or None,
        previous_diagnosis=previous_diagnosis or None,
        evolution_review_path=args.evolution_review or None,
        iteration=args.iteration,
        observed_distribution_profile=observed_distribution_profile,
        distribution_manifest=args.distribution_manifest,
        reference_results_path=args.reference_results,
        reference_label=args.reference_label,
        iter_context_path=args.iter_context,
    )
    report = agent.run()
    print("\n" + "=" * 70)
    print(
        "Evolution targets: "
        + ", ".join(report.get("algorithmic_evolution", {}).get("target_modules") or [])
    )
    print(f"Evolution direction: {report['algorithmic_evolution']['evolution_hint']}")
    print(f"Targeted questions: {report['target_questions']}")
