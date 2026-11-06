"""Add operational fields to the full diagnosis without discarding reasoning.

The same report is read by Research and Codex. Probe selection remains a
separate operational sidecar because it is consumed by the evaluator.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from typing import Any, Dict

try:
    from runtime_capability_context import (
        DEFAULT_EXECUTION_LLM_PROFILE_ID,
        DEFAULT_PROFILE_IDS,
    )
    from target_selection import canonical_task_ref
except ImportError:  # pragma: no cover - package-mode fallback
    from .runtime_capability_context import (
        DEFAULT_EXECUTION_LLM_PROFILE_ID,
        DEFAULT_PROFILE_IDS,
    )
    from .target_selection import canonical_task_ref


BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MACHINE_CONTRACT_SCHEMA_VERSION = 1

# These are execution responsibilities, not an algorithm prescription.  They
# keep every slot meaningful even when the diagnosis initially focuses on only
# part of the bundle, and give static/smoke/probe the same neutral boundary.
DEFAULT_BUNDLE_RESPONSIBILITIES = {
    "video_structuring": {
        "responsibility": "persist time-bounded structure records or bounded original-media references for later retrieval",
        "input_invariant": "accept only bounded media/metadata inputs and preserve their source/time provenance",
        "output_invariant": "emit retrievable canonical records with time ranges and explicit evidence or media references",
        "fallback": "persist an explicit unavailable/no-content record without fabricating evidence",
    },
    "localization": {
        "responsibility": "turn a reasoning request into bounded structure evidence, candidate windows, or explicit unavailable state",
        "input_invariant": "consume question/context and documented structure or media references without inventing windows",
        "output_invariant": "emit a parseable localization_result with time ranges, records, or perception requests",
        "fallback": "emit localization_result status=unavailable with a reason",
    },
    "perception": {
        "responsibility": "inspect only localization-authorized bounded media and return time-linked evidence",
        "input_invariant": "consume a documented perception request and active time window before any capability call",
        "output_invariant": "emit a parseable perception_result with evidence, status, and source time ranges",
        "fallback": "emit perception_result status=unavailable or error without treating missing evidence as success",
    },
    "memory": {
        "responsibility": "retain raw request/result/decision events and provide a parseable context packet for later thinking turns",
        "input_invariant": "store documented runtime events with their producer, consumer, status, and time provenance",
        "output_invariant": "return context without dropping source events required for a later decision",
        "fallback": "return an explicit empty/limited context packet rather than fabricated summaries",
    },
    "thinking": {
        "responsibility": "use question, memory, localization, and perception state to request the next operation or return a supported answer",
        "input_invariant": "consume only documented context and evidence envelopes and preserve pending evidence requests",
        "output_invariant": "emit a parseable tool request or a final answer decision linked to available evidence",
        "fallback": "request bounded additional evidence or return a parseable final decision with explicit limitations in metadata",
    },
}

RUNTIME_CAPABILITIES = {
    "llm": "runtime_evidence.call_text_llm(messages, profile_id=..., temperature=..., max_tokens=...)",
    "vlm": "runtime_evidence.inspect_active_frames(env, capability='vlm', prompt=..., max_windows=..., frames_per_window=..., profile_id=...) or runtime_evidence.inspect_time_ranges(env, time_ranges, capability='vlm', prompt=..., max_windows=..., frames_per_window=..., profile_id=...)",
    "asr": "runtime_evidence.transcribe_active_windows(env, max_windows=..., max_seconds_per_window=..., profile_id=...) or runtime_evidence.transcribe_time_ranges(env, time_ranges, max_windows=..., max_seconds_per_window=..., profile_id=...)",
    "ocr": "runtime_evidence.inspect_active_frames(env, capability='ocr', prompt=..., max_windows=..., frames_per_window=..., profile_id=...) or runtime_evidence.inspect_time_ranges(env, time_ranges, capability='ocr', prompt=..., max_windows=..., frames_per_window=..., profile_id=...)",
    "embedding": "runtime_evidence.embed_text(input_text, profile_id=...)",
}
DEFAULT_CAPABILITY_PROFILES = {
    **DEFAULT_PROFILE_IDS,
    "llm": DEFAULT_EXECUTION_LLM_PROFILE_ID,
}

def _bundle_modules(design: Dict[str, Any]) -> list[str]:
    try:
        from bundle_contract import normalize_modules
    except ImportError:  # pragma: no cover
        from .bundle_contract import normalize_modules
    return normalize_modules((design or {}).get("target_modules"))


def validate_bundle_execution_brief(brief: Dict[str, Any]) -> list[str]:
    """Check actionable content; executable ABI/capabilities are tested on code."""
    if not isinstance(brief, dict):
        return ["diagnosis must be an object"]
    try:
        modules = _bundle_modules(brief.get("execution_policy") or brief.get("evolution_design") or {})
    except ValueError as exc:
        return [str(exc)]
    task = brief.get("implementation_task") or {}
    issues = [
        f"implementation_task.{field} is empty"
        for field in ("failure_chain", "evolution_mechanism", "goal")
        if not str(task.get(field) or "").strip()
    ]
    handoffs = task.get("handoff_contracts")
    if handoffs:
        try:
            from bundle_contract import validate_handoff_contracts
        except ImportError:
            from .bundle_contract import validate_handoff_contracts
        issues.extend(validate_handoff_contracts(handoffs, changed_modules=modules))
    return issues


def build_bundle_execution_brief(diagnosis: Dict[str, Any], *, diagnosis_path: str,
                                 review_path: str = "") -> Dict[str, Any]:
    """Expose an operational view alongside the complete causal diagnosis."""
    diagnosis = diagnosis or {}
    design = deepcopy(diagnosis.get("evolution_design", {}) or {})
    modules = _bundle_modules(design)
    decision = diagnosis.get("evolution_decision", {}) or {}
    raw_responsibilities = design.get("module_responsibilities") or {}
    responsibilities = {}
    try:
        from bundle_contract import MODULE_TYPES as all_bundle_modules
    except ImportError:  # pragma: no cover
        from .bundle_contract import MODULE_TYPES as all_bundle_modules
    for module in all_bundle_modules:
        source = raw_responsibilities.get(module) if isinstance(raw_responsibilities, dict) else {}
        source = source if isinstance(source, dict) else {}
        defaults = DEFAULT_BUNDLE_RESPONSIBILITIES[module]
        responsibilities[module] = {
            "responsibility": str(source.get("responsibility") or source.get("purpose") or defaults["responsibility"]),
            "input_invariant": str(source.get("input_invariant") or defaults["input_invariant"]),
            "output_invariant": str(source.get("output_invariant") or defaults["output_invariant"]),
            "fallback": str(source.get("fallback") or defaults["fallback"]),
        }
    handoffs = list(design.get("handoff_contracts") or [])
    capabilities = []
    for raw in design.get("runtime_capability_needs") or []:
        if not isinstance(raw, dict):
            continue
        capability = str(raw.get("capability") or "")
        if capability not in RUNTIME_CAPABILITIES:
            continue
        profile_id = str(raw.get("profile_id") or DEFAULT_CAPABILITY_PROFILES.get(capability, ""))
        capabilities.append({
            "capability": capability,
            "profile_id": profile_id,
            "runtime_call": RUNTIME_CAPABILITIES[capability],
            "input_provenance": str(raw.get("input_provenance") or ""),
            "trigger": str(raw.get("trigger") or "bounded runtime trigger"),
            "fallback": str(raw.get("fallback") or "emit structured unavailable evidence"),
        })
    task = {
        "goal": str(design.get("design_summary") or design.get("evolution_mechanism") or "Improve the diagnosed evidence-to-decision chain."),
        "failure_chain": str(design.get("failure_chain") or design.get("causal_hypothesis") or "producer evidence → target behavior → final decision"),
        "evolution_mechanism": str(design.get("evolution_mechanism") or design.get("causal_hypothesis") or "bounded evidence handling with explicit uncertainty"),
        "module_responsibilities": responsibilities,
        "handoff_contracts": handoffs,
        "runtime_capabilities": capabilities,
        "cost_guards": list(design.get("cost_guardrails") or []),
        "research_questions": list(design.get("research_questions") or []),
        "probe_hypotheses": list(design.get("probe_hypotheses") or []),
        "implementation_constraints": [
            "Codex may adapt the actual changed module set after probe feedback, but only inside the five-module bundle.",
            "Keep all five combo slots explicit and preserve documented runtime interfaces, profiles, active-window boundaries, and structured fallbacks.",
            "Do not modify orchestration, evaluators, datasets, raw media, capability profiles, or task-specific data.",
        ],
    }
    brief = {
        **deepcopy(diagnosis),
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "artifact_type": "diagnosis_execution_brief",
        "created_at": int(time.time()),
        "diagnosis_path": diagnosis_path,
        "evolution_review_path": review_path or diagnosis.get("evolution_review_path", ""),
        "execution_policy": {
            "combo_base": str(decision.get("combo_base") or "current_best"),
            "base_combo_policy": str(decision.get("base_combo_policy") or "keep_current_best"),
            "target_modules": modules,
            "initial_focus_modules": list(design.get("initial_focus_modules") or modules),
            "module_scope_policy": "codex_adaptive_within_five_module_bundle",
            "inner_loop_policy": "probe_feedback_may_change_actual_changed_modules_without_rediagnosis",
            "evolution_phase_policy": dict(diagnosis.get("evolution_phase_policy") or {}),
        },
        "current_best_contract": {
            "reference_mode": "current_best",
            "structure_artifact_policy": "reuse current-best structure unless the candidate bundle changes video_structuring",
            "five_module_bundle_required": True,
        },
        "implementation_task": task,
        "research_plan": {
            "objective": "Research the selected mechanism and affected module boundaries, not a locked per-file implementation.",
            "questions": task["research_questions"],
            "output_requirements": ["Map each recommendation to a module responsibility or handoff contract.", "Do not introduce unregistered capabilities or task-specific rules."],
        },
    }
    issues = validate_bundle_execution_brief(brief)
    if issues:
        raise RuntimeError("invalid adaptive bundle brief: " + "; ".join(issues))
    return brief


def build_bundle_machine_evaluation_contract(diagnosis: Dict[str, Any], *, brief: Dict[str, Any]) -> Dict[str, Any]:
    """Machine-only probe sidecar for an adaptive module-set round."""
    task = brief.get("implementation_task") or {}
    probe_plan = deepcopy(
        (diagnosis.get("evolution_decision") or {}).get("probe_plan")
        or diagnosis.get("probe_plan") or {}
    )
    hypotheses = [
        item for item in (task.get("probe_hypotheses") or [])
        if isinstance(item, dict) and str(item.get("hypothesis_id") or "")
    ]
    hypothesis_ids = [str(item.get("hypothesis_id")) for item in hypotheses]
    # A model selects the hypothesis and its failure-kind labels, but an exact
    # MFP label match is not guaranteed (for example, a coverage hypothesis
    # can be supported by an ``evidence_absent`` trajectory rather than a
    # literally named ``gap_miss`` event).  A probe contract with empty
    # witnesses is structurally valid yet impossible to execute.  Fill only
    # such empty slots from the deterministic diagnosis failure ledger, in a
    # stable order, and retain the source marker for audit.  This never uses
    # current-candidate corrections or regressions as repair evidence.
    existing = {
        str(item.get("hypothesis_id") or ""): dict(item)
        for item in (probe_plan.get("diagnosis_hypothesis_witnesses") or [])
        if isinstance(item, dict) and str(item.get("hypothesis_id") or "")
    }
    fallback_refs: list[str] = []

    def add_ref(value: Any) -> None:
        value = str(value or "").strip()
        if value and value not in fallback_refs:
            fallback_refs.append(value)

    for value in probe_plan.get("repair_probe") or []:
        add_ref(value)
    failure_clusters = probe_plan.get("failure_clusters") or {}
    for refs in (failure_clusters.values() if isinstance(failure_clusters, dict) else []):
        for value in refs or []:
            add_ref(value)
    ledger = diagnosis.get("evidence_ledger") or {}
    for cluster in ((ledger.get("weak_failure_distribution") or {}).get("clusters") or []):
        if isinstance(cluster, dict):
            for value in cluster.get("refs") or []:
                add_ref(value)
    for event in ledger.get("direct_mfp_evidence") or []:
        if isinstance(event, dict):
            add_ref(event.get("task_ref") or event.get("task_id"))

    used_canonical_refs = {
        canonical_task_ref(ref)
        for item in existing.values()
        for ref in (item.get("refs") or [])
        if canonical_task_ref(ref)
    }
    normalized_witnesses = []
    for hypothesis in hypotheses:
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
        witness = existing.get(hypothesis_id, {})
        refs = [str(ref) for ref in (witness.get("refs") or []) if str(ref)]
        if not refs:
            selected = next(
                (ref for ref in fallback_refs
                 if canonical_task_ref(ref) not in used_canonical_refs),
                "",
            )
            if selected:
                refs = [selected]
                used_canonical_refs.add(canonical_task_ref(selected))
        origin_modules = [str(value) for value in (witness.get("origin_modules") or []) if str(value)]
        fallback_used = bool(refs and not (witness.get("refs") or []))
        if fallback_used:
            origin_modules.append("deterministic_failure_cluster_fallback")
        normalized_witnesses.append({
            "hypothesis_id": hypothesis_id,
            "failure_kinds": list(witness.get("failure_kinds") or hypothesis.get("failure_kinds") or []),
            "origin_modules": sorted(set(origin_modules)),
            "refs": refs,
            "witness_source": (
                "deterministic_failure_cluster_fallback" if fallback_used
                else "matched_mfp_cluster"
            ),
        })
    if hypotheses:
        probe_plan["diagnosis_hypothesis_witnesses"] = normalized_witnesses
        probe_plan["diagnosis_hypothesis_witness_refs"] = [
            item["refs"][0] for item in normalized_witnesses if item.get("refs")
        ]
        held_out = {
            canonical_task_ref(ref)
            for ref in (
                list(probe_plan.get("repair_probe") or [])
                + [ref for item in normalized_witnesses for ref in (item.get("refs") or [])]
                + list(probe_plan.get("regression_guard") or [])
                + list(probe_plan.get("previous_candidate_corrections") or [])
                + list(probe_plan.get("previous_candidate_regressions") or [])
            )
            if canonical_task_ref(ref)
        }
        weak_refs = []
        for cluster in ((ledger.get("weak_failure_distribution") or {}).get("clusters") or []):
            if isinstance(cluster, dict):
                for ref in cluster.get("refs") or []:
                    ref = str(ref or "")
                    if ref and ref not in weak_refs:
                        weak_refs.append(ref)
        probe_plan["generalization_probe_pool"] = [
            ref for ref in weak_refs if canonical_task_ref(ref) not in held_out
        ]
    return {
        "schema_version": BUNDLE_MACHINE_CONTRACT_SCHEMA_VERSION,
        "artifact_type": "machine_evaluation_contract",
        "contract_mode": "adaptive_bundle",
        "target_modules": list((brief.get("execution_policy") or {}).get("target_modules") or []),
        "initial_focus_modules": list((brief.get("execution_policy") or {}).get("initial_focus_modules") or []),
        "module_scope_policy": "codex_adaptive_within_five_module_bundle",
        "handoff_contracts": task.get("handoff_contracts") or [],
        "runtime_capabilities": task.get("runtime_capabilities") or [],
        "probe_hypotheses": hypotheses,
        "probe_plan": probe_plan,
        "evidence_ledger": deepcopy(diagnosis.get("evidence_ledger") or {}),
        "mechanism_coverage_contract": {
            "schema_version": 1,
            "source": "diagnosis_probe_hypotheses",
            "diagnosis_hypothesis_ids": hypothesis_ids,
            "witness_first": True,
            "generalization_after_witness": True,
            "fixed_outer_round_probe_batch": True,
        },
    }


def validate_bundle_machine_evaluation_contract(contract: Dict[str, Any]) -> list[str]:
    """Check selected modules, probe hypotheses and executable witness references."""
    contract = contract if isinstance(contract, dict) else {}
    issues: list[str] = []
    if contract.get("artifact_type") != "machine_evaluation_contract":
        return ["artifact_type must be machine_evaluation_contract"]
    if int(contract.get("schema_version", 0) or 0) != BUNDLE_MACHINE_CONTRACT_SCHEMA_VERSION:
        issues.append("bundle machine evaluation contract has an unsupported schema_version")
    if contract.get("contract_mode") != "adaptive_bundle":
        issues.append("bundle machine evaluation contract must use contract_mode=adaptive_bundle")
    try:
        modules = _bundle_modules({"target_modules": contract.get("target_modules") or []})
    except ValueError as exc:
        issues.append(str(exc))
        modules = []
    if not modules:
        issues.append("bundle machine evaluation contract needs at least one target module")
    coverage = contract.get("mechanism_coverage_contract") or {}
    if int(coverage.get("schema_version", 0) or 0) != 1:
        issues.append("bundle machine evaluation contract requires the current mechanism coverage contract")
    elif (coverage.get("witness_first") is not True
          or coverage.get("generalization_after_witness") is not True
          or coverage.get("fixed_outer_round_probe_batch") is not True):
        issues.append("bundle machine evaluation contract has an invalid coverage policy")
    hypotheses = contract.get("probe_hypotheses") or []
    ids = [str(item.get("hypothesis_id") or "") for item in hypotheses if isinstance(item, dict)]
    expected_ids = [str(item) for item in (coverage.get("diagnosis_hypothesis_ids") or []) if str(item)]
    if ids != expected_ids:
        issues.append("bundle mechanism coverage must preserve diagnosis hypothesis ids exactly")
    plan = contract.get("probe_plan") or {}
    if not isinstance(plan, dict):
        issues.append("bundle machine evaluation contract probe_plan must be an object")
    elif expected_ids:
        witness_ids = [str(item.get("hypothesis_id") or "") for item in (plan.get("diagnosis_hypothesis_witnesses") or []) if isinstance(item, dict)]
        if set(witness_ids) != set(expected_ids):
            issues.append("bundle probe_plan does not cover exactly the diagnosis hypotheses")
        missing_witness_refs = [
            str(item.get("hypothesis_id") or "")
            for item in (plan.get("diagnosis_hypothesis_witnesses") or [])
            if isinstance(item, dict) and not list(item.get("refs") or [])
        ]
        if missing_witness_refs:
            issues.append(
                "bundle probe_plan has hypotheses without executable witness refs: "
                + ", ".join(missing_witness_refs)
            )
        if not isinstance(plan.get("generalization_probe_pool"), list):
            issues.append("bundle probe_plan is missing held-out generalization_probe_pool")
    return issues


def write_bundle_execution_brief(diagnosis: Dict[str, Any], *, diagnosis_path: str,
                                review_path: str = "") -> Dict[str, str]:
    timestamp = diagnosis.get("timestamp") or int(time.time())
    attempt = max(1, int(diagnosis.get("diagnosis_attempt", 1) or 1))
    suffix = f"_attempt{attempt}" if attempt > 1 else ""
    directory = os.path.dirname(os.path.abspath(diagnosis_path))
    brief = build_bundle_execution_brief(diagnosis, diagnosis_path=diagnosis_path, review_path=review_path)
    # One authoritative diagnosis carries analysis and operational fields.
    path = os.path.abspath(diagnosis_path)
    diagnosis.update(brief)
    diagnosis["execution_brief_path"] = path
    contract = build_bundle_machine_evaluation_contract(diagnosis, brief=brief)
    contract_issues = validate_bundle_machine_evaluation_contract(contract)
    if contract_issues:
        raise RuntimeError(
            "invalid adaptive bundle machine evaluation contract: "
            + "; ".join(contract_issues)
        )
    contract_path = os.path.join(directory, f"machine_evaluation_contract_{timestamp}{suffix}.json")
    with open(contract_path, "w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
    diagnosis["machine_evaluation_contract_path"] = contract_path
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(diagnosis, handle, ensure_ascii=False, indent=2)
    return {"execution_brief_path": path, "machine_evaluation_contract_path": contract_path}
