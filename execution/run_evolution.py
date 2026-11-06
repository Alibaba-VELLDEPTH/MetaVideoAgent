#!/usr/bin/env python3
"""Staged target-distribution evolution entrypoint.

The automatic workflow:

1. Initialize from five-frame distribution observations or a source-agent mapping.
2. Validate and run the initial combo as the first reference.
3. Diagnose the initial reference trajectories.
4. Run explicit review, diagnosis, optional research, candidate authoring,
   smoke, probe, and full-evaluation stages against the current best.

Codex performs automatic multi-round authoring and bounded repair.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
EVOLUTION_DIR = os.path.join(PROJECT_ROOT, "evolution")
METAVIDEOAGENT_RUNTIME_DIR = os.path.join(CURRENT_DIR, "action_runtime")
DEFAULT_METAVIDEOAGENT_WORKSPACE = os.path.join(CURRENT_DIR, "workspace")
ACTIVE_RUNTIME_DIR = METAVIDEOAGENT_RUNTIME_DIR
EVOLUTION_SPLIT = "train"
METAVIDEOAGENT_PYTHON = os.environ.get("METAVIDEOAGENT_PYTHON") or sys.executable
for path in (PROJECT_ROOT, EVOLUTION_DIR, ACTIVE_RUNTIME_DIR, CURRENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
if ACTIVE_RUNTIME_DIR in sys.path:
    sys.path.remove(ACTIVE_RUNTIME_DIR)
sys.path.insert(0, ACTIVE_RUNTIME_DIR)
os.environ.setdefault("METAVIDEOAGENT_RUNTIME_DIR", METAVIDEOAGENT_RUNTIME_DIR)

from runtime_config import bootstrap_cli_api_env  # noqa: E402

bootstrap_cli_api_env(sys.argv[1:])
import diagnosis_agent as diagnosis_agent_module  # noqa: E402
from channel_profile_runner import (  # noqa: E402
    build_channel_profile,
    profile_matches_manifest,
)
from codex_runtime_settings import (  # noqa: E402
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    resolve_codex_runtime_settings,
)
from deep_research_runner import run_deep_research  # noqa: E402
from diagnosis_agent import DiagnosisAgent  # noqa: E402
from diagnosis_execution_brief import (  # noqa: E402
    validate_bundle_execution_brief,
    write_bundle_execution_brief,
)
from distribution_spec import (  # noqa: E402
    load_distribution_spec,
    resolve_distribution_manifest_path,
)
from iter_context import build_iter_context, inject_iter_context  # noqa: E402
from iteration_workspace import IterationWorkspace, candidate_stage_paths  # noqa: E402
from iteration_workspace import write_json as write_workspace_json  # noqa: E402
from ledger import (  # noqa: E402
    load_evolution_memory,
    load_ledger,
    record_evolution_outcome,
    record_reference,
)
from paths import run_dir as metavideoagent_run_dir  # noqa: E402
from post_full_eval_decider import full_evaluation_is_valid  # noqa: E402
from review_contract import review_is_consumable  # noqa: E402
from runtime_paths import (  # noqa: E402
    require_metavideoagent_runtime,
    runtime_output_root,
)
from runtime_paths import (
    runtime_dir as selected_runtime_dir,
)


def _normalize_api_env(env: Dict[str, str], args: argparse.Namespace = None) -> Dict[str, str]:
    """Prepare provider and execution settings for subprocesses.

    Provider settings are injected through documented MetaVideoAgent environment variables.
    """
    if args is not None:
        explicit = {
            "METAVIDEOAGENT_API_KEY": getattr(args, "api_key", ""),
            "METAVIDEOAGENT_BASE_URL": getattr(args, "base_url", ""),
        }
        for key, value in explicit.items():
            if value:
                env[key] = value
    if args is not None:
        if getattr(args, "structure_workers", None):
            env["VIDEO_STRUCT_REBUILD_WORKERS"] = str(args.structure_workers)
            env["VIDEO_STRUCT_REBUILD_BATCH"] = str(max(1, int(args.structure_workers) * 8))
        # Candidate continuation and orchestration share one repair budget.
        if getattr(args, "probe_repair_attempts", None) is not None:
            budget = str(max(0, int(args.probe_repair_attempts or 0)))
            env["BUNDLE_FEEDBACK_BUDGET"] = budget
    return env


def _unique_run_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{os.getpid()}"


def _refresh_diagnosis_probe_context(report: Dict) -> Dict:
    """Refresh non-vetoing probe context at process boundaries.

    Target selection belongs to diagnosis over the complete distribution.  MFP
    coverage is carried forward only to select optional probe witnesses.
    """
    report = dict(report or {})
    if report.get("evidence_ledger") and report.get("algorithmic_evolution"):
        assessor = object.__new__(DiagnosisAgent)
        report["target_probe_context"] = assessor._target_probe_context(report)
    return report


def _read_json(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _probe_question_limit(args: argparse.Namespace) -> int:
    """Use the CLI probe size or the distribution manifest's probe budget."""
    requested = int(getattr(args, "first_n", 0) or 0)
    if requested > 0:
        return requested
    spec = load_distribution_spec(getattr(args, "distribution_manifest", ""))
    budgets = spec.get("budgets") if isinstance(spec.get("budgets"), dict) else {}
    configured = budgets.get("probe_first_n", 0)
    try:
        configured = int(configured or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured <= 0:
        raise RuntimeError(
            "Codex evolve requires --first-n or a positive "
            "budgets.probe_first_n in the resolved distribution manifest."
        )
    return configured


def _reference_report_path_from_initial(initial: Dict) -> str:
    report = (initial.get("initial_reference_report", {}) or initial)
    candidates = [
        report.get("report_path", ""),
        (report.get("paths", {}) or {}).get("report_path", ""),
        initial.get("report_path", ""),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return ""


def _reference_results_path_from_report(report: Dict) -> str:
    return (
        report.get("results_path", "")
        or (report.get("paths", {}) or {}).get("results_path", "")
        or report.get("full_eval_results", "")
        or report.get("sandbox_dir", "")
    )


def _same_path(left: str, right: str) -> bool:
    return bool(left and right and os.path.abspath(left) == os.path.abspath(right))


def _needs_current_best_provenance_refresh(current: Dict, report_path: str,
                                           results_path: str) -> bool:
    """Allow an idempotent registration to repair missing baseline provenance.

    This is intentionally limited to the exact report/results already named as
    current best.  It cannot promote a different reference or candidate.
    """
    current = current if isinstance(current, dict) else {}
    if not current or not (
        _same_path(report_path, current.get("report_path", ""))
        or _same_path(results_path, current.get("results_path", ""))
    ):
        return False
    artifacts = current.get("artifacts") or {}
    bundle = (artifacts.get("bundle") or {}).get("path", "")
    structure = artifacts.get("structure") or {}
    return not bool(bundle and os.path.isfile(bundle) and structure)


def _reference_label_for_report(args: argparse.Namespace, report: Dict,
                                report_path: str = "") -> str:
    """Return a truthful label for the actual reference artifact.

    Full-eval reports preserve the label used when *they* were evaluated.  A
    candidate that later becomes current best can therefore still contain
    ``initial_combo`` internally.  Later rounds must label it as current best
    when it is used as the parent reference.
    """
    ledger = load_ledger(args.run_id, args.output_root)
    current = ledger.get("current_best") or {}
    report_path = os.path.abspath(report_path) if report_path else ""
    results_path = _reference_results_path_from_report(report)
    selection = current.get("selection") or {}
    current_report_paths = (
        current.get("report_path", ""),
        selection.get("train_diagnostic_report_path", ""),
    )
    current_results_paths = (
        current.get("results_path", ""),
        selection.get("train_diagnostic_results_path", ""),
    )
    is_current_best = any(_same_path(report_path, path) for path in current_report_paths) or any(
        _same_path(results_path, path) for path in current_results_paths
    )
    combo_id = report.get("combo_id", "")
    if is_current_best or (report.get("post_full_eval_decision") or {}).get("update_current_best") is True:
        return f"current_best:{combo_id}" if combo_id else "current_best"
    if report.get("validation_scope") == "initial_reference_multi_video":
        return f"initial_combo:{combo_id}" if combo_id else "initial_combo"
    explicit = getattr(args, "reference_label", "") or ""
    if explicit and explicit not in {"initial_combo", "current_best"}:
        return explicit
    return f"reference:{combo_id}" if combo_id else (explicit or "reference")


def _resolve_iteration_index(args: argparse.Namespace) -> int:
    """Use the requested iteration or this run's persisted continuation point."""
    explicit = int(getattr(args, "iteration_index", 0) or 0)
    if explicit > 0:
        return explicit
    run_root = metavideoagent_run_dir(args.run_id, args.output_root)
    state = _read_json(os.path.join(run_root, "run.json"))
    return int(state.get("next_iteration") or 1)


def _synthetic_reference_report_from_initial(initial: Dict) -> Dict:
    report = (initial.get("initial_reference_report", {}) or initial)
    paths = report.get("paths", {}) or {}
    return {
        "results_path": report.get("results_path", "") or paths.get("results_path", ""),
        "report_path": report.get("report_path", "") or paths.get("report_path", ""),
        "combo": report.get("combo") or initial.get("combo", {}),
        "validation_scope": "initial_reference_multi_video",
        "reference_label": "initial_combo",
        "question_split": report.get("question_split", ""),
        "total_questions": report.get("total_questions", 0),
        "video_ids": report.get("video_ids", []),
    }


def _write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _stage_report_path(args: argparse.Namespace, output: Dict) -> str:
    """Keep explicit stage wrappers inside their own iteration directory."""
    context_path = output.get("iter_context_path", "") or ""
    if context_path:
        return os.path.join(
            os.path.dirname(os.path.abspath(context_path)),
            f"{args.stage}_stage_report.json",
        )
    return os.path.join(
        metavideoagent_run_dir(args.run_id, args.output_root),
        *(("checks",) if args.check_only else ()),
        "metavideoagent_evolution_report.json",
    )


def _resolve_initial_report(initial_report: str = "",
                            run_id: str = "", output_root: str = "",
                            reference_label: str = "") -> str:
    if initial_report:
        return os.path.abspath(initial_report)
    if run_id:
        ledger = _read_json(os.path.join(metavideoagent_run_dir(run_id, output_root), "current_best_ledger.json"))
        ledger_key = "finalized_combo" if reference_label == "finalized_combo" else "current_best"
        ledger_entry = ledger.get(ledger_key) or {}
        selection = ledger_entry.get("selection") or {}
        ledger_report = selection.get("train_diagnostic_report_path", "") or ledger_entry.get("report_path", "")
        if ledger_report and os.path.exists(ledger_report):
            return os.path.abspath(ledger_report)
        initial = _read_json(os.path.join(metavideoagent_run_dir(run_id, output_root),
                                         "iter_0", "metavideoagent_initial_pipeline.json"))
        report_path = (initial.get("initial_reference_report") or {}).get("report_path", "")
        if report_path and os.path.isfile(report_path) and (initial.get("initial_reference_report") or {}).get("ready"):
            return os.path.abspath(report_path)
    return ""


def _initial_bundle_path_from_report(report: Dict) -> str:
    path = (
        # A later-round reference must restore the full current-best bundle,
        # not the initial bundle from which that candidate was once derived.
        report.get("current_best_bundle_path", "")
        or report.get("candidate_full_eval_bundle_path", "")
        or ((report.get("combo_policy") or {}).get("current_best_bundle_path", ""))
        or ((report.get("combo_policy") or {}).get("effective_bundle_path", ""))
        # `run_initial_agent.py` writes this explicit snapshot field.  Prefer
        # it to the pre-repair source bundle so later rounds reproduce the
        # exact initial reference that was actually evaluated.
        or report.get("effective_initial_baseline_bundle", "")
        or report.get("initial_baseline_bundle", "")
        or ((report.get("combo_policy") or {}).get("base_bundle_path", ""))
    )
    if path:
        if os.path.exists(path):
            return os.path.abspath(path)
        report_path = report.get("report_path", "")
        if report_path:
            candidate = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(report_path)), path))
            if os.path.exists(candidate):
                return candidate
    report_path = report.get("report_path", "")
    if report_path:
        candidate = os.path.join(os.path.dirname(os.path.abspath(report_path)), "initial_baseline_bundle.json")
        if os.path.exists(candidate):
            return candidate
    for source in report.get("sources", []) or []:
        if not isinstance(source, str):
            continue
        candidate = os.path.join(os.path.dirname(os.path.abspath(source)), "initial_baseline_bundle.json")
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return ""


def _reference_structure_artifact_from_report(report: Dict) -> Dict:
    """Normalize initial-reference structure provenance for the current-best ledger."""
    artifact = report.get("structure_artifact") or {}
    if isinstance(artifact, dict) and artifact:
        return artifact
    structure_dir = str(report.get("structure_dir") or "")
    if not structure_dir or not os.path.isdir(structure_dir):
        return {}
    files = sorted(
        os.path.abspath(os.path.join(structure_dir, name))
        for name in os.listdir(structure_dir)
        if name.endswith(".jsonl") and os.path.isfile(os.path.join(structure_dir, name))
    )
    return {
        "source": str(report.get("structure_source") or "initial_reference"),
        "materialized_dir": os.path.abspath(structure_dir),
        "video_structure_files": files,
        "rebuild_structure": bool(report.get("rebuild_structure")),
    }


def _combo_from_reference_report(report: Dict) -> Dict:
    combo = report.get("combo") or report.get("agent_combo") or {}
    if isinstance(combo, dict) and combo:
        return combo
    results_path = _reference_results_path_from_report(report)
    if not results_path or not os.path.exists(results_path):
        return {}
    paths = [results_path]
    if os.path.isdir(results_path):
        paths = [
            os.path.join(results_path, name)
            for name in sorted(os.listdir(results_path))
            if name.endswith(".jsonl")
        ]
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    combo = (
                        row.get("architecture_combo")
                        or row.get("combo")
                        or row.get("agent_combo")
                        or {}
                    )
                    if isinstance(combo, dict) and combo:
                        return combo
        except OSError:
            continue
    return {}


def _profile_path(workspace: str, manifest: str, run_id: str,
                  output_root: str = "",
                  explicit_profile: str = "",
                  force: bool = False,
                  semantic_kwargs: Dict = None, check_only: bool = False) -> str:
    if explicit_profile:
        path = os.path.abspath(explicit_profile)
        if not os.path.exists(path):
            raise FileNotFoundError(f"distribution profile not found: {explicit_profile}")
        return path
    profile = build_channel_profile(
        workspace=workspace,
        distribution_manifest=manifest,
        run_id=run_id,
        output_root=output_root,
        output_path=os.path.join(metavideoagent_run_dir(run_id, output_root),
                                 *(('checks',) if check_only else ()), "iter_0", "initialization", "profile.json"),
        force=force,
        **(semantic_kwargs or {}),
    )
    return profile.get("output_path", "")


def _load_or_build_profile(workspace: str, manifest: str, run_id: str,
                           output_root: str = "",
                           explicit_profile: str = "",
                           force: bool = False,
                           semantic_kwargs: Dict = None, check_only: bool = False) -> Dict:
    path = _profile_path(
        workspace, manifest, run_id, output_root=output_root,
        explicit_profile=explicit_profile,
        force=force,
        semantic_kwargs=semantic_kwargs,
        check_only=check_only,
    )
    profile = _read_json(path)
    matches, reason = profile_matches_manifest(profile, manifest)
    from channel_profile_runner import profile_ready_for_mode
    mode = (semantic_kwargs or {}).get("semantic_observer_mode", "observe")
    if matches and profile_ready_for_mode(profile, mode):
        return profile
    if matches:
        reason = "semantic_observation_not_completed"

    # An explicit profile from a different run is never silently relabelled
    # as belonging to the current train manifest.  Rebuild a new, run-scoped
    # distribution-aware profile and retain the supplied artifact untouched for
    # audit. This also accepts profiles carrying an explicit provenance record.
    rebuilt_path = os.path.join(
        metavideoagent_run_dir(run_id, output_root),
        *(('checks',) if check_only else ()), "iter_0", "initialization", "profile.json",
    )
    rebuilt = build_channel_profile(
        workspace=workspace,
        distribution_manifest=manifest,
        output_path=rebuilt_path,
        run_id=run_id,
        output_root=output_root,
        force=True,
        **(semantic_kwargs or {}),
    )
    rebuilt["profile_rebuild"] = {
        "reason": reason,
        "supplied_profile_path": os.path.abspath(path) if path else "",
    }
    with open(rebuilt_path, "w", encoding="utf-8") as handle:
        json.dump(rebuilt, handle, ensure_ascii=False, indent=2)
    return rebuilt


def _profile_semantic_kwargs(args: argparse.Namespace) -> Dict:
    observer_mode = getattr(args, "semantic_observer_mode", "observe")
    if getattr(args, "check_only", False) and observer_mode == "observe":
        observer_mode = "plan"
    return {
        "semantic_observer_mode": observer_mode,
        "semantic_observer_dir": getattr(args, "semantic_observer_dir", ""),
        "semantic_video_budget": getattr(args, "semantic_video_budget", 0),
        "semantic_observe_workers": getattr(args, "semantic_observe_workers", 1),
        "runtime_dir": selected_runtime_dir(),
    }


def _run_deep_research_context(args: argparse.Namespace, profile: Dict,
                               run_id: str,
                               diagnosis_path: str = "",
                               review_path: str = "") -> Dict:
    if getattr(args, "check_only", False):
        return {"skipped": True, "reason": "check-only mode does not execute Codex research"}
    if getattr(args, "deep_research_report", ""):
        report = _read_json(args.deep_research_report)
        if report:
            return report
    if getattr(args, "deep_research_summary", ""):
        return {
            "summary_text": args.deep_research_summary,
            "source": "inline_argument",
            "report_path": "",
            "summary_path": "",
            "design_brief_path": "",
        }
    profile_path = profile.get("output_path", "") or getattr(args, "profile", "")
    if not profile_path:
        return {}
    iteration = IterationWorkspace(os.path.join(metavideoagent_run_dir(run_id, args.output_root), "iter_0"))
    saved = (iteration.progress().get("stages") or {}).get("initialization") or {}
    if saved.get("status") == "complete":
        report = _read_json(saved["result"])
        if report.get("formal_research") and report.get("research_complete"):
            return report
    out_dir = str(iteration.reserve("initialization"))
    result = run_deep_research(
        workspace=args.workspace,
        channel_profile_path=profile_path,
        output_dir=out_dir,
        run_id=run_id,
        output_root=args.output_root,
        diagnosis_path=diagnosis_path,
        review_path=review_path,
        max_queries=getattr(args, "deep_research_max_queries", 5),
        search_timeout=getattr(args, "research_search_timeout", 20),
        codex_cli=getattr(args, "research_codex_cli", "") or getattr(args, "codex_cli", ""),
        min_sources=getattr(args, "research_min_sources", 2),
    )
    iteration.record("initialization", result["report_path"],
                     status="complete" if result.get("research_complete") else "pending",
                     notes_path=result.get("summary_path", ""), next_task=result.get("next_task") or {})
    return result


def _run_codex_deep_research_context(args: argparse.Namespace, profile: Dict,
                                     diagnosis_path: str, review_path: str,
                                     round_dir: str) -> Dict:
    """Research the full diagnosis, evidence and unresolved implementation questions."""
    if getattr(args, "check_only", False):
        return {"skipped": True, "reason": "check-only mode does not execute Codex research"}
    if getattr(args, "probe_failed_run", "") or getattr(args, "engineering_failed_run", ""):
        return {"skipped": True, "reason": "same-candidate feedback reuses its research"}
    if getattr(args, "skip_codex_deep_research", False):
        return {"skipped": True, "reason": "--skip-iteration-research"}
    if getattr(args, "codex_deep_research_report", ""):
        report = _read_json(args.codex_deep_research_report)
        if not report or not report.get("formal_research") or not report.get("research_complete"):
            raise RuntimeError("the supplied research report is missing or incomplete")
        return report
    if getattr(args, "codex_deep_research_summary", ""):
        return {
            "summary_text": args.codex_deep_research_summary,
            "source": "inline_argument", "research_complete": True,
            "formal_research": False,
        }
    if not diagnosis_path or not _read_json(diagnosis_path):
        raise RuntimeError("research requires the completed diagnosis")
    profile_path = profile.get("output_path", "") or getattr(args, "profile", "")
    if not profile_path:
        raise RuntimeError("research requires the observed distribution profile")
    iteration = IterationWorkspace(round_dir)
    saved = (iteration.progress().get("stages") or {}).get("research") or {}
    if saved.get("status") == "complete":
        return _read_json(saved["result"])
    attempt = iteration.reserve("research")
    result = run_deep_research(
        workspace=args.workspace,
        channel_profile_path=profile_path,
        output_dir=str(attempt),
        run_id=args.run_id, output_root=args.output_root,
        diagnosis_path=diagnosis_path,
        review_path=review_path or _read_json(diagnosis_path).get("evolution_review_path", ""),
        max_queries=getattr(args, "deep_research_max_queries", 5),
        search_timeout=getattr(args, "research_search_timeout", 20),
        codex_cli=getattr(args, "research_codex_cli", "") or getattr(args, "codex_cli", ""),
        min_sources=getattr(args, "research_min_sources", 2),
    )
    result["usage"] = "diagnosis_to_automatic_code_generation"
    iteration.record("research", result["report_path"],
                     status="complete" if result.get("research_complete") else "pending",
                     notes_path=result.get("summary_path", ""), next_task=result.get("next_task") or {})
    return result


def _research_record(research: Dict) -> Dict:
    if not research:
        return {}
    design_brief = research.get("design_brief") or {}
    validation = research.get("validation") or {}
    target_modules = list(research.get("target_modules") or design_brief.get("target_modules") or [])
    return {
        "report_path": research.get("report_path", ""),
        "summary_path": research.get("summary_path", ""),
        "design_brief_path": research.get("design_brief_path", ""),
        "mode": research.get("mode", research.get("source", "")),
        "research_complete": research.get("research_complete"),
        "formal_research": research.get("formal_research"),
        "validation_passed": research.get(
            "validation_passed",
            validation.get("passed") if isinstance(validation, dict) else None,
        ),
        "target_modules": target_modules,
        "schema_type": research.get("schema_type", "adaptive_module_set_evolution" if target_modules else ""),
        "skipped": research.get("skipped", False),
        "reason": research.get("reason", ""),
        "usage": research.get("usage", ""),
    }


def _write_diagnosis_for_codex(diagnosis_path: str, research: Dict,
                               round_dir: str, iter_context: Dict = None) -> str:
    """Assemble the code-writing input without stripping upstream reasoning."""
    diagnosis = _read_json(diagnosis_path)
    if not diagnosis:
        raise RuntimeError("code generation requires the completed diagnosis")
    if research.get("skipped") is not True and research.get("research_complete") is not True:
        raise RuntimeError("research is incomplete; resolve its open questions before code generation")
    stage_context = IterationWorkspace(round_dir).context("codegen")
    diagnosis.setdefault("iter_context", {})["stage_context"] = stage_context
    diagnosis["codex_deep_research"] = {
        **_research_record(research),
        "summary_text": research.get("summary_text", ""),
        "design_brief": research.get("design_brief") or _read_json(research.get("design_brief_path", "")),
    }
    from runtime_capability_context import candidate_runtime_capabilities
    diagnosis.setdefault("implementation_task", {})["runtime_capabilities"] = candidate_runtime_capabilities(diagnosis)
    diagnosis["working_materials"] = {
        **((iter_context or {}).get("working_materials") or {}),
        **(diagnosis.get("working_materials") or {}),
        **({"research": research["report_path"]} if research.get("report_path") else {}),
    }
    from working_materials import material_paths
    diagnosis["working_materials"].update(material_paths({"stage_context": stage_context}))
    out_path = os.path.join(round_dir, "diagnosis_for_codex.json")
    _write_json(out_path, diagnosis)
    return out_path


def _load_validated_diagnosis_for_codex(path: str) -> Dict:
    """Load the diagnosis and its completed or explicitly skipped research."""
    handoff = _read_json(path)
    issues = validate_bundle_execution_brief(handoff)
    if issues:
        raise RuntimeError("saved code-writing input is incomplete: " + "; ".join(issues))
    research = handoff.get("codex_deep_research") or {}
    if research.get("skipped") is not True and research.get("research_complete") is not True:
        raise RuntimeError("saved code-writing input requires completed or explicitly skipped research")
    return handoff


def _ensure_research_ready(args: argparse.Namespace, research: Dict) -> None:
    if getattr(args, "check_only", False):
        return
    if not research or not research.get("report_path"):
        return
    if not research.get("formal_research"):
        raise RuntimeError(
            "Deep research did not complete. Rerun the automatic research stage."
        )


def _run_initial(args: argparse.Namespace, run_id: str) -> Dict:
    from run_initial_agent import run as run_initial_combo  # noqa: WPS433

    common = dict(
        workspace=args.workspace,
        distribution_manifest=args.distribution_manifest,
        profile=args.profile,
        combo_plan=args.combo_plan,
        initial_baseline_bundle=getattr(args, "initial_baseline_bundle", ""),
        initialization=getattr(args, "initialization", "observe"),
        source_agent=getattr(args, "source_agent", ""),
        out_dir=(args.out_dir or os.path.join(metavideoagent_run_dir(run_id, args.output_root), "checks", "iter_0")) if args.check_only else args.out_dir,
        run_id=run_id,
        video_id=args.video_id,
        question_split=EVOLUTION_SPLIT,
        concurrency=args.concurrency,
        first_n=args.first_n,
        rebuild_structure=args.rebuild_structure,
        structure_workers=args.structure_workers,
        combo_id=args.combo_id,
        semantic_observer_mode=getattr(args, "semantic_observer_mode", "none"),
        semantic_observer_dir=getattr(args, "semantic_observer_dir", ""),
        semantic_video_budget=getattr(args, "semantic_video_budget", 0),
        semantic_observe_workers=getattr(args, "semantic_observe_workers", 1),
        api_key=getattr(args, "api_key", ""),
        base_url=getattr(args, "base_url", ""),
        output_root=args.output_root,
        check_only=args.check_only,
        deep_research_summary=getattr(args, "deep_research_summary", ""),
        deep_research_report=getattr(args, "deep_research_report", ""),
        codex_cli=getattr(args, "initial_codex_cli", "") or getattr(args, "codex_cli", ""),
        codegen_timeout=getattr(args, "codegen_timeout", 900),
        bundle_smoke_repair_attempts=getattr(args, "repair_attempts", 0),
        adopt_smoke_report="",
        restart_from=getattr(args, "restart_from", ""),
    )
    initialization_dir = common["out_dir"] or os.path.join(metavideoagent_run_dir(run_id, args.output_root), "iter_0")
    if not common["initial_baseline_bundle"] and not os.path.isfile(os.path.join(initialization_dir, "initialization", "plan.json")):
        plan_report = run_initial_combo(SimpleNamespace(**common, stage="plan"))
        common["initial_baseline_bundle"] = plan_report.get("initial_baseline_bundle_path", "")
        common["combo_plan"] = plan_report.get("combo_plan_path", common["combo_plan"])
        if args.check_only:
            return plan_report
    smoke_report = run_initial_combo(SimpleNamespace(**common, stage="codegen_smoke"))
    if args.check_only:
        return smoke_report
    common["initial_baseline_bundle"] = smoke_report.get("effective_initial_baseline_bundle_path", common["initial_baseline_bundle"])
    return run_initial_combo(SimpleNamespace(**common, stage="run"))


def _diagnose(args: argparse.Namespace, reference_report: Dict,
              profile: Dict, iter_context: Dict = None) -> Dict:
    results_path = _reference_results_path_from_report(reference_report)
    if not results_path:
        raise RuntimeError("initial/reference report does not contain results_path")
    review_path = getattr(args, "evolution_review", "") or ""
    if review_path:
        review = _read_json(review_path)
        if not review:
            raise RuntimeError(f"evolution review is missing or unreadable: {review_path}")
        if not review_is_consumable(review):
            raise RuntimeError(
                "evolution review is missing a successful macro review; rerun the macro review before diagnosis"
            )
    reference_label = (
        args.reference_label
        or reference_report.get("reference_label")
        or (
            "initial_combo"
            if reference_report.get("validation_scope") == "initial_reference_multi_video"
            else "current_best"
        )
    )
    profile = dict(profile or {})
    combo_plan_path = reference_report.get("combo_plan", "")
    if combo_plan_path and os.path.exists(combo_plan_path):
        plan = _read_json(combo_plan_path)
        for key in ("initial_capability_blueprints", "initial_codegen_policy", "design_answers"):
            if key in plan:
                profile[key] = plan[key]
    previous_evolution_result = None
    candidate_path = ""
    if getattr(args, "full_eval_report", "") and getattr(args, "evolution_review", ""):
        full_eval_report = _read_json(args.full_eval_report)
        candidate_path = getattr(args, "candidate_bundle", "")
        candidate = (
            _read_json(candidate_path)
            if candidate_path
            else {}
        )
        if not candidate:
            candidate = full_eval_report.get("candidate", {}) or {}
        previous_evolution_result = _previous_result_from_full_eval(
            full_eval_report,
            candidate,
        )
    previous_diagnosis = _read_json(getattr(args, "previous_diagnosis", ""))
    iter_context = dict(iter_context or {})
    reference_context = iter_context.get("reference") or {}
    reference_bundle = (_initial_bundle_path_from_report(reference_report)
                        or (reference_context.get("initial_baseline_bundle") or {}).get("path", ""))
    if reference_bundle:
        iter_context["working_materials"] = {
            **(iter_context.get("working_materials") or {}),
            "current_agent": reference_bundle,
        }
    diagnosis_model = diagnosis_agent_module.DIAGNOSIS_MODEL
    agent = DiagnosisAgent(
        args.workspace,
        evolution_memory="",
        previous_evolution_result=previous_evolution_result,
        previous_diagnosis=previous_diagnosis or None,
        observed_distribution_profile=profile,
        distribution_manifest=args.distribution_manifest,
        reference_results_path=results_path,
        reference_label=reference_label,
        evolution_review_path=review_path or None,
        iter_context=iter_context,
        iter_context_path=(iter_context or {}).get("path", ""),
    )
    report = agent.run()
    report = _refresh_diagnosis_probe_context(report)
    report = inject_iter_context(report, iter_context or {})
    report["evolution_review_path"] = os.path.abspath(review_path) if review_path else ""
    report["reference_report_path"] = str(
        reference_report.get("report_path") or (reference_context.get("report") or {}).get("path") or ""
    )
    report["reference_bundle_path"] = reference_bundle
    report["reference_combo"] = reference_report.get("combo") or reference_report.get("agent_combo") or reference_context.get("combo") or {}
    report["distribution_manifest"] = args.distribution_manifest
    report["previous_diagnosis_path"] = os.path.abspath(
        getattr(args, "previous_diagnosis", "")
    ) if getattr(args, "previous_diagnosis", "") else ""
    report["diagnosis_inputs"] = {
        "reference_results_path": os.path.abspath(results_path),
        "reference_label": reference_label,
        "review_path": report["evolution_review_path"],
        "previous_diagnosis_path": report["previous_diagnosis_path"],
        "previous_full_eval_report": os.path.abspath(
            getattr(args, "full_eval_report", "")
        ) if getattr(args, "full_eval_report", "") else "",
        "previous_candidate_artifact": os.path.abspath(candidate_path) if candidate_path else "",
        "iter_context_path": (iter_context or {}).get("path", ""),
        "diagnosis_model": diagnosis_model,
        "diagnosis_attempt": report.get("diagnosis_attempt", 1),
    }
    # A current-best reference is a hard execution baseline.  Normalize the
    # LLM report before it becomes a Codex input so the accepted current best
    # remains the only executable base.
    target_modules = list((report.get("evolution_design") or {}).get("target_modules") or [])
    if not target_modules:
        raise RuntimeError(
            "diagnosis must select at least one target module; "
            "the public evolution route uses one adaptive module-set direction"
        )
    if str(reference_label).startswith("current_best"):
        decision = report.get("evolution_decision") or {}
        decision.update({
            "combo_base": "current_best",
            "base_combo_policy": "keep_current_best",
        })
        report["evolution_decision"] = decision
        algo = report.get("algorithmic_evolution") or {}
        algo["evolution_hint"] = algo.get("evolution_hint") or (
            "Preserve the accepted current-best bundle as the base. The module-set brief "
            "sets direction only; Codex may keep one target or adapt related modules inside the five-module bundle."
        )
        report["algorithmic_evolution"] = algo
    report_path = report.get("report_path", "")
    if not report_path:
        raise RuntimeError("diagnosis did not persist a report_path; refusing to compile an untracked brief")
    _write_json(report_path, report)
    execution_artifacts = write_bundle_execution_brief(
        report,
        diagnosis_path=report_path,
        review_path=review_path,
    )
    report.update(execution_artifacts)
    _write_json(report_path, report)
    return report


def _subprocess_env(args: argparse.Namespace = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["METAVIDEOAGENT_RUNTIME_DIR"] = selected_runtime_dir()
    env.setdefault("PYTHONNOUSERSITE", "1")
    if args is not None:
        settings = getattr(args, "codex_runtime_settings", {}) or {}
        if settings:
            env["CODEX_MODEL"] = settings["model"]
            env["CODEX_REASONING_EFFORT"] = settings["reasoning_effort"]
        if getattr(args, "run_id", ""):
            env["METAVIDEOAGENT_RUN_ID"] = args.run_id
        if getattr(args, "output_root", ""):
            env["METAVIDEOAGENT_OUTPUT_ROOT"] = args.output_root
        if getattr(args, "distribution_manifest", ""):
            env["METAVIDEOAGENT_DISTRIBUTION_MANIFEST"] = args.distribution_manifest
        if int(getattr(args, "iteration_index", 0) or 0) > 0:
            env["METAVIDEOAGENT_ITERATION_INDEX"] = str(args.iteration_index)
    return _normalize_api_env(env, args)


def _run_command(cmd: List[str], cwd: str, env: Dict[str, str],
                 stdout_path: str = "", stderr_path: str = "") -> int:
    if stdout_path or stderr_path:
        if stdout_path:
            os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
        if stderr_path:
            os.makedirs(os.path.dirname(stderr_path), exist_ok=True)
        with open(stdout_path or os.devnull, "w", encoding="utf-8") as out, \
                open(stderr_path or os.devnull, "w", encoding="utf-8") as err:
            return subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=err, check=False).returncode
    return subprocess.run(cmd, cwd=cwd, env=env, check=False).returncode


def _candidate_artifact_path(candidate_run: str) -> str:
    """Return the only formal candidate artifact: a complete bundle."""
    path = os.path.join(candidate_run, "candidate_bundle.json")
    return path if os.path.isfile(path) else ""


def _run_codex_subprocess(args: argparse.Namespace, diagnosis_path: str,
                          reference_report_path: str, round_dir: str,
                          machine_evaluation_contract_path: str = "") -> Dict:
    if args.check_only:
        round_dir = getattr(args, "_iteration_dir", "") or os.path.join(
            os.path.dirname(round_dir), "checks", os.path.basename(round_dir))
    iteration = IterationWorkspace(round_dir)
    saved = (iteration.progress().get("stages") or {}).get("codegen") or {}
    requested_restart = getattr(args, "restart_from", "")
    args.restart_from = ""
    repair_requested = bool(getattr(args, "probe_failed_run", "") or getattr(args, "engineering_failed_run", ""))
    resume_stage = requested_restart if not repair_requested else ""
    resume_bundle = ""
    if not repair_requested and saved.get("status") == "complete" and saved.get("result"):
        previous = os.path.dirname(saved["result"])
        gate = _probe_gate_from_candidate_run(previous)
        prior_validation = _read_json(os.path.join(previous, "validation.json"))
        if resume_stage not in {"smoke", "probe"} and gate.get("action") == "probe_missing":
            resume_stage = "probe" if (prior_validation.get("bundle_smoke") or {}).get("passed") else "smoke"
        if resume_stage not in {"smoke", "probe"} and gate.get("action") == "execution_layer_rerun":
            resume_stage = "probe" if gate.get("origin_stage") == "probe" else "smoke"
        if resume_stage in {"smoke", "probe"}:
            resume_bundle = os.path.join(previous, "candidate_bundle.json")
    if (saved.get("status") == "complete" and saved.get("result")
            and not resume_bundle and not repair_requested):
        return {"exit_code": 0, "candidate_run": os.path.dirname(saved["result"]),
                "iteration_dir": str(iteration.root), "authoring_engine": "codex", "resumed": True}
    candidate_run = str(iteration.reserve("codegen"))
    probe_first_n = _probe_question_limit(args)
    reference_report = _read_json(reference_report_path) if reference_report_path else {}
    reference_results_path = _reference_results_path_from_report(reference_report)
    if not reference_results_path:
        diagnosis = _read_json(diagnosis_path) if diagnosis_path else {}
        reference_results_path = str(diagnosis.get("reference_results_path") or "")
    cmd = [
        METAVIDEOAGENT_PYTHON,
        os.path.join(EVOLUTION_DIR, "codex_evolve.py"),
        "--workspace", args.workspace,
        "--diagnosis", diagnosis_path,
        "--distribution-manifest", args.distribution_manifest,
        "--reference-report", reference_report_path,
        "--reference-label", args.reference_label,
        "--iteration-dir", str(iteration.root),
        "--candidate-dir", candidate_run,
        "--first_n", str(probe_first_n),
        "--concurrency", str(args.concurrency),
        "--repair-attempts", str(args.repair_attempts),
        "--codex-model", args.codex_runtime_settings["model"],
        "--codex-reasoning-effort", args.codex_runtime_settings["reasoning_effort"],
    ]
    if reference_results_path:
        cmd.extend(["--reference-results", os.path.abspath(reference_results_path)])
    if machine_evaluation_contract_path:
        cmd.extend(["--machine-evaluation-contract", machine_evaluation_contract_path])
    if resume_bundle:
        cmd.extend(["--resume-stage", resume_stage, "--resume-bundle", resume_bundle])
    if getattr(args, "probe_failed_run", ""):
        cmd.extend(["--probe-failed-run", args.probe_failed_run])
    if getattr(args, "engineering_failed_run", ""):
        cmd.extend(["--engineering-failed-run", args.engineering_failed_run])
    if getattr(args, "skip_probe_verification", False):
        cmd.append("--skip_probe_verification")
    if getattr(args, "codex_cli", ""):
        cmd.extend(["--codex-cli", args.codex_cli])
    if args.run:
        cmd.append("--run")
    print("METAVIDEOAGENT_EVOLUTION_CMD")
    print(" ".join(cmd))
    if args.check_only:
        return {
            "exit_code": 0,
            "cmd": cmd,
            "iteration_dir": str(iteration.root),
            "candidate_run": candidate_run,
            "authoring_engine": "codex",
        }
    stdout_path = os.path.join(candidate_run, "stdout.txt")
    stderr_path = os.path.join(candidate_run, "stderr.txt")
    exit_code = _run_command(
        cmd, cwd=PROJECT_ROOT, env=_subprocess_env(args),
        stdout_path=stdout_path, stderr_path=stderr_path,
    )
    return {
        "exit_code": exit_code,
        "cmd": cmd,
        "iteration_dir": str(iteration.root),
        "candidate_run": candidate_run,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "authoring_engine": "codex",
    }


def _run_bundle_feedback_loop(args: argparse.Namespace, *, initial_codex: Dict,
                              diagnosis_path: str, reference_report_path: str,
                              round_dir: str, machine_evaluation_contract_path: str) -> Dict:
    """Continue one bundle through bounded same-direction Codex feedback.

    Bundle smoke repair is already internal to ``codex_evolve``.  This outer
    continuation is the matching probe layer: each child restores the previous
    candidate source, keeps the selected probe batch fixed, and gives Codex its
    complete forensic workspace.  It never returns to diagnosis/research.
    """
    current = dict(initial_codex or {})
    attempts = []
    limit = max(0, int(getattr(args, "probe_repair_attempts", 0) or 0))
    saved_probe = getattr(args, "probe_failed_run", "")
    saved_engineering = getattr(args, "engineering_failed_run", "")
    try:
        for index in range(1, limit + 1):
            candidate_run = str(current.get("candidate_run") or "")
            if current.get("exit_code") != 0 or not candidate_run:
                break
            gate = _probe_gate_from_candidate_run(candidate_run)
            action = str(gate.get("action") or "")
            if action not in {"codex_engineering_repair", "codex_same_direction_behavior_iteration"}:
                break
            if int(_read_json(os.path.join(candidate_run, "run_record.json")).get("feedback_round") or 0) >= limit:
                break
            args.probe_failed_run = ""
            args.engineering_failed_run = ""
            if action == "codex_engineering_repair" and gate.get("origin_stage") != "probe":
                args.engineering_failed_run = candidate_run
            else:
                args.probe_failed_run = candidate_run
            child = _run_codex_subprocess(
                args, diagnosis_path, reference_report_path, round_dir,
                machine_evaluation_contract_path=machine_evaluation_contract_path,
            )
            attempts.append({
                "round": index,
                "parent_candidate_run": candidate_run,
                "input_action": action,
                "input_probe_gate": gate,
                "codex": child,
            })
            current = child
    finally:
        args.probe_failed_run = saved_probe
        args.engineering_failed_run = saved_engineering
    current["bundle_feedback_attempts"] = attempts
    current["bundle_feedback_limit"] = limit
    return current


def _run_full_eval_subprocess(args: argparse.Namespace,
                              reference_report_path: str) -> int:
    if not args.candidate_run:
        raise RuntimeError("--candidate-run is required for --stage full_eval")
    candidate_record = _read_json(os.path.join(args.candidate_run, "run_record.json"))
    iteration_dir = candidate_record["iteration_dir"]
    if args.check_only:
        iteration_dir = getattr(args, "_iteration_dir", "") or os.path.join(
            os.path.dirname(iteration_dir), "checks", os.path.basename(iteration_dir))
    iteration = IterationWorkspace(iteration_dir)
    saved = (iteration.progress().get("stages") or {}).get("full_eval") or {}
    receipt = _read_json(os.path.join(args.candidate_run, "full_eval_result.json"))
    same_evaluation = bool(saved.get("result") and receipt.get("report_path") == saved["result"])
    evaluated = _read_json(saved["result"]) if same_evaluation else {}
    if full_evaluation_is_valid(evaluated) and evaluated.get("post_full_eval_decision"):
        iteration.record("full_eval", saved["result"], status="complete")
        return 0
    evaluation_dir = str(iteration.reserve("full_eval"))
    report_path = os.path.join(evaluation_dir, "full_eval_report.json")
    if not args.check_only:
        _write_json(os.path.join(args.candidate_run, "full_eval_result.json"), {"report_path": report_path})
    iteration.record("full_eval", report_path, status="running")
    cmd = [
        METAVIDEOAGENT_PYTHON,
        os.path.join(EVOLUTION_DIR, "full_eval_runner.py"),
        "--workspace", args.workspace,
        "--candidate-run", args.candidate_run,
        "--evaluation-dir", evaluation_dir,
        "--distribution-manifest", args.distribution_manifest,
        "--reference-report", reference_report_path,
        "--reference-label", args.reference_label,
        "--concurrency", str(args.concurrency),
        "--structure-workers", str(args.structure_workers),
        "--metavideoagent-run-id", args.run_id,
        "--metavideoagent-output-root", args.output_root,
    ]
    # Provider settings are passed only through the child environment. Keeping
    # credentials and endpoints out of argv prevents them from appearing in
    # command previews, process listings, CI logs, and subprocess diagnostics.
    candidate_artifact = args.candidate_bundle or _candidate_artifact_path(args.candidate_run)
    if not candidate_artifact:
        raise RuntimeError("formal MetaVideoAgent full eval requires candidate_run/candidate_bundle.json")
    cmd.extend(["--candidate-bundle", candidate_artifact])
    candidate_types = set((_read_json(candidate_artifact).get("changed_modules") or []))
    if args.rebuild_structure and "video_structuring" not in candidate_types:
        raise ValueError(
            "--rebuild-structure is only valid for a candidate changing video_structuring; "
            "other rounds must reuse the current-best structure artifact."
        )
    reference_bundle = (
        _initial_bundle_path_from_report(_read_json(reference_report_path))
        or args.initial_baseline_bundle
    )
    if reference_bundle:
        cmd.extend(["--reference-bundle", reference_bundle])
    if getattr(args, "profile", ""):
        cmd.extend(["--distribution-profile", args.profile])
    if args.diagnosis_report:
        cmd.extend(["--diagnosis-report", args.diagnosis_report])
    # An adaptive bundle can change one or more module types. Rebuild the structure
    # artifact exactly when the structuring module is among those changes.
    if "video_structuring" in candidate_types:
        cmd.append("--rebuild-structure")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.skip_post_eval_decision:
        cmd.append("--skip-post-eval-decision")
    print("METAVIDEOAGENT_FULL_EVAL_CMD")
    print(" ".join(cmd))
    if args.check_only:
        cmd.append("--check-only")
    exit_code = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, env=_subprocess_env(args)).returncode
    evaluated = _read_json(report_path)
    complete = bool(full_evaluation_is_valid(evaluated) and evaluated.get("post_full_eval_decision"))
    iteration.record("full_eval", report_path,
                     status="complete" if exit_code == 0 and complete else "pending",
                     next_task={} if exit_code == 0 and complete else {
                         "stage": "full_eval", "task": "Resolve the recorded failure and rerun full evaluation for this candidate.",
                     })
    return exit_code if args.check_only or exit_code else (0 if complete else 1)


def _run_review_subprocess(args: argparse.Namespace,
                           reference_report: Dict) -> int:
    # The first post-initial review has one executed current-best only.  It is
    # not a fabricated candidate-vs-self comparison and therefore must not
    # require candidate_bundle.  Infer this from the formal initial report so
    # the normal `--stage review --initial-report ...` command follows the
    # documented second-round entrypoint without an extra routing flag.
    initial_reference_review = bool(
        getattr(args, "initial_reference_review", False)
        or (
            reference_report.get("validation_scope") == "initial_reference_multi_video"
            and not reference_report.get("candidate_bundle")
            and not _read_json(getattr(args, "full_eval_report", "")).get("candidate_bundle")
        )
    )
    iteration = IterationWorkspace(args._iteration_dir) if getattr(args, "_iteration_dir", "") else None
    stage = "teacher" if initial_reference_review else "review"
    if iteration and not args.check_only:
        saved = (iteration.progress().get("stages") or {}).get(stage) or {}
        if saved.get("status") == "complete" and review_is_consumable(_read_json(saved["result"])):
            args.review_out_dir = os.path.dirname(saved["result"])
            return 0
        args.review_out_dir = str(iteration.reserve(stage))

    def execute_review(cmd: list[str]) -> int:
        code = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, env=_subprocess_env(args)).returncode
        if iteration:
            path = _find_review_path(args.review_out_dir)
            passed = code == 0 and bool(path) and review_is_consumable(_read_json(path))
            iteration.record(stage, path or os.path.join(args.review_out_dir, "teacher_evolution_review.json"),
                             status="complete" if passed else "pending",
                             next_task={} if passed else {"stage": stage, "task": "Complete the trajectory review."})
        return code

    if initial_reference_review:
        results_path = _reference_results_path_from_report(reference_report)
        report_path = reference_report.get("report_path", "") or args.initial_report or args.full_eval_report
        if not results_path or not report_path:
            raise RuntimeError(
                "initial-reference review requires an executed reference report with results_path; "
                "pass --initial-report"
            )
        cmd = [
            METAVIDEOAGENT_PYTHON,
            os.path.join(EVOLUTION_DIR, "evolution_review_runner.py"),
            "--workspace", args.workspace,
            "--video-id", args.video_id,
            "--evolved-results", results_path,
            "--verification-report", report_path,
            "--distribution-manifest", args.distribution_manifest,
            "--reference-results", results_path,
            "--reference-label", "initial_agent",
            "--concurrency", str(args.concurrency),
            "--initial-reference-review",
        ]
        if getattr(args, "profile", ""):
            cmd.extend(["--distribution-profile", args.profile])
        if args.review_out_dir:
            cmd.extend(["--out-dir", args.review_out_dir])
        print("METAVIDEOAGENT_INITIAL_REFERENCE_REVIEW_CMD")
        print(" ".join(cmd))
        if args.check_only:
            return 0
        return execute_review(cmd)
    results_path = args.full_eval_results or reference_report.get("results_path", "")
    report_path = args.full_eval_report or reference_report.get("report_path", "")
    verification_report = _read_json(report_path)
    reference_results = (
        verification_report.get("reference_source", "")
        or _reference_results_path_from_report(reference_report)
    )
    # The full-eval report records the reference actually used to produce the
    # candidate trajectories. It is the parent for review even if the
    # candidate was promoted to current best afterwards.
    parent_report_path = verification_report.get("reference_report_path", "")
    parent_report = _read_json(parent_report_path)
    parent_combo_id = parent_report.get("combo_id", "")
    reference_label = (
        f"pre_evolution_current_best:{parent_combo_id}"
        if parent_combo_id else "pre_evolution_current_best"
    )
    if not results_path or not report_path:
        raise RuntimeError("--full-eval-results and --full-eval-report are required for --stage review")
    cmd = [
        METAVIDEOAGENT_PYTHON,
        os.path.join(EVOLUTION_DIR, "evolution_review_runner.py"),
        "--workspace", args.workspace,
        "--video-id", args.video_id,
        "--evolved-results", results_path,
        "--verification-report", report_path,
        "--distribution-manifest", args.distribution_manifest,
        "--reference-results", reference_results,
        "--reference-label", reference_label,
        "--concurrency", str(args.concurrency),
    ]
    candidate_bundle = args.candidate_bundle or verification_report.get("candidate_bundle", "")
    if not candidate_bundle:
        raise RuntimeError("formal MetaVideoAgent review requires full-eval candidate_bundle")
    cmd.extend(["--candidate-bundle", candidate_bundle])
    if getattr(args, "profile", ""):
        cmd.extend(["--distribution-profile", args.profile])
    if args.review_out_dir:
        cmd.extend(["--out-dir", args.review_out_dir])
    if initial_reference_review:
        cmd.append("--initial-reference-review")
    print("METAVIDEOAGENT_REVIEW_CMD")
    print(" ".join(cmd))
    if args.check_only:
        return 0
    return execute_review(cmd)


def _read_candidate(candidate_run: str) -> Dict:
    path = _candidate_artifact_path(candidate_run)
    candidate = _read_json(path)
    if not candidate:
        return {}
    if candidate.get("artifact_type") != "candidate_bundle":
        raise RuntimeError(f"candidate artifact is not a candidate_bundle: {path}")
    return {
        "name": "MetaVideoAgent bundle",
        "changed_modules": candidate.get("changed_modules") or [],
        "candidate_bundle_path": path,
    }


def _find_full_eval_report(candidate_run: str) -> str:
    path = _read_json(os.path.join(candidate_run, "full_eval_result.json")).get("report_path", "")
    return path if path and os.path.isfile(path) else ""






def _find_review_path(review_dir: str) -> str:
    path = os.path.join(review_dir, "teacher_evolution_review.json")
    return path if os.path.exists(path) else ""


def _latest_valid_candidate_review(run_id: str, output_root: str = "") -> str:
    """Return the newest completed candidate review from run-scoped history."""
    memory = load_evolution_memory(run_id, output_root)
    for record in reversed(memory.get("records", []) or []):
        review_meta = ((record.get("artifacts", {}) or {}).get("review", {}))
        review_path = review_meta if isinstance(review_meta, str) else review_meta.get("path", "")
        review = _read_json(review_path)
        if review_is_consumable(review):
            return os.path.abspath(review_path)
    return ""


def _annotate_and_record_candidate_history(
    args: argparse.Namespace,
    *,
    iteration: int,
    parent_report_path: str,
    parent_report: Dict,
    candidate_run: str = "",
    candidate_bundle_path: str = "",
    diagnosis_path: str = "",
    research_report_path: str = "",
    full_eval_report_path: str = "",
    review_path: str = "",
    rejection_context_path: str = "",
    outcome_status: str = "",
) -> Dict:
    """Persist one candidate's lineage and outcome without copying trajectories."""
    parent_label = _reference_label_for_report(args, parent_report, parent_report_path)
    full_report = _read_json(full_eval_report_path)
    candidate = _read_json(candidate_bundle_path)
    if review_path and os.path.exists(review_path):
        review = _read_json(review_path)
        review["candidate_lineage"] = {
            "iteration": iteration,
            "candidate_run": os.path.abspath(candidate_run) if candidate_run else "",
            "candidate_combo_id": full_report.get("combo_id", ""),
            "target_modules": list(candidate.get("changed_modules") or []),
            "parent_reference_report": os.path.abspath(parent_report_path) if parent_report_path else "",
            "parent_reference_combo_id": parent_report.get("combo_id", ""),
            "parent_reference_label": parent_label,
            "post_full_eval_decision": full_report.get("post_full_eval_decision") or {},
            "evidence_tier": "full_eval_review",
        }
        _write_json(review_path, review)
    return record_evolution_outcome(
        args.run_id,
        iteration=iteration,
        candidate_run=candidate_run,
        candidate_bundle_path=candidate_bundle_path,
        parent_report_path=parent_report_path,
        parent_results_path=_reference_results_path_from_report(parent_report),
        parent_label=parent_label,
        diagnosis_path=diagnosis_path,
        research_report_path=research_report_path,
        full_eval_report_path=full_eval_report_path,
        review_path=review_path,
        rejection_context_path=rejection_context_path,
        outcome_status=outcome_status,
        output_root=args.output_root,
    )


def _probe_gate_from_candidate_run(candidate_run: str) -> Dict:
    """Read the recorded engineering or behavioral decision for a candidate."""
    validation = _read_json(os.path.join(candidate_run, "validation.json"))
    probe_paths = candidate_stage_paths(candidate_run, "probe")
    assembly = validation.get("assembly_validation") or {}
    action = validation.get("probe_next_action") or assembly.get("probe_next_action")
    if action:
        origin = "probe" if (
            validation.get("probe_verification")
            or os.path.isfile(probe_paths.get("result", ""))
        ) else "smoke"
        return {**action, "origin_stage": origin}
    if validation.get("probe_skipped"):
        return {"action": "probe_skipped", "reason": "Smoke completed; probe was not requested."}
    verification = validation.get("probe_verification") or _read_json(
        probe_paths.get("result", ""),
    )
    if verification:
        from codex_evolve import classify_probe_next_action
        audit = _read_json(probe_paths.get("audit", ""))
        return {**classify_probe_next_action(verification, audit), "origin_stage": "probe"}
    return {"action": "probe_missing", "reason": "The candidate has no recorded probe decision."}


def _write_candidate_rejection_context_from_run(candidate_run: str,
                                                output_path: str) -> str:
    from codex_evolve import write_candidate_rejection_context  # noqa: WPS433

    return write_candidate_rejection_context(candidate_run, output_path)


def _previous_result_from_full_eval(full_report: Dict, candidate: Dict) -> Dict:
    return {
        "target_modules": list(candidate.get("changed_modules") or []),
        "evolved_bundle": {
            "changed_modules": list(candidate.get("changed_modules") or []),
        },
        "verification": full_report,
        "verdict": (
            "success"
            if (full_report.get("final_success_criteria", {}) or {}).get("passed") else
            "rejected"
        ),
    }


def _diagnose_next_round(args: argparse.Namespace, profile: Dict,
                         reference_report: Dict, previous_diagnosis: Dict,
                         full_eval_report_path: str, candidate: Dict,
                         review_path: str,
                         iter_context: Dict = None) -> Dict:
    """Continue diagnosis with the completed candidate review and evaluation."""
    previous = {
        "evolution_review": getattr(args, "evolution_review", ""),
        "full_eval_report": getattr(args, "full_eval_report", ""),
        "candidate_bundle": getattr(args, "candidate_bundle", ""),
        "previous_diagnosis": getattr(args, "previous_diagnosis", ""),
    }
    try:
        args.evolution_review = review_path
        args.full_eval_report = full_eval_report_path
        args.candidate_bundle = os.path.join(
            candidate.get("candidate_run", ""), "candidate_bundle.json"
        ) if candidate.get("candidate_run") else getattr(args, "candidate_bundle", "")
        args.previous_diagnosis = (previous_diagnosis or {}).get("report_path", "")
        diagnosis = _diagnose(args, reference_report, profile, iter_context=iter_context)
    finally:
        args.evolution_review = previous["evolution_review"]
        args.full_eval_report = previous["full_eval_report"]
        args.candidate_bundle = previous["candidate_bundle"]
        args.previous_diagnosis = previous["previous_diagnosis"]
    return diagnosis


def _report_has_no_incorrect_trajectory(report: Dict) -> bool:
    """Return true only for a completed, non-empty, perfect evolution evaluation."""
    if not isinstance(report, dict) or report.get("evaluation_complete", True) is not True:
        return False
    total = int(report.get("total_questions") or 0)
    correct = report.get("candidate_correct")
    if correct is None:
        correct = report.get("correct")
    return total > 0 and int(correct or 0) == total


def _automatic_outcome(rounds: List[Dict], requested_rounds: int,
                       *, check_only: bool, stop_reason: str = "") -> Dict:
    """Summarize the real terminal state of an automatic multi-round run."""
    terminal_status = str((rounds[-1] if rounds else {}).get("status") or "")
    failed_statuses = {
        "reference_review_failed", "codex_failed", "unknown_probe_gate",
        "full_eval_failed", "review_failed",
    }
    incomplete_statuses = {
        "probe_execution_layer_rerun_required", "probe_requires_codex_feedback",
        "bundle_smoke_passed_probe_skipped",
    }
    if stop_reason == "no_incorrect_evolution_trajectories":
        outcome, exit_code = ("checked" if check_only else "completed"), 0
        terminal_status = stop_reason
    elif terminal_status in failed_statuses:
        outcome, exit_code = "failed", 1
    elif terminal_status in incomplete_statuses:
        outcome, exit_code = "incomplete", 2
    elif len(rounds) != requested_rounds:
        outcome, exit_code = "incomplete", 2
        terminal_status = terminal_status or "round_count_incomplete"
    else:
        outcome, exit_code = ("checked" if check_only else "completed"), 0
        terminal_status = ""
    return {
        "automatic_outcome": outcome,
        "automatic_terminal_status": terminal_status,
        "automatic_exit_code": exit_code,
    }


def _run_autonomous_loop(args: argparse.Namespace,
                         reference_report_path: str,
                         reference_report: Dict,
                         profile: Dict,
                         output: Dict) -> Dict:
    current_reference_path = reference_report_path
    current_reference_report = reference_report
    current_reference_label = _reference_label_for_report(
        args, current_reference_report, current_reference_path
    )
    current_diagnosis = None
    current_review_path = getattr(args, "evolution_review", "") or ""
    last_full_eval_report = ""
    last_candidate_run = ""
    last_candidate_bundle = ""
    rounds = []
    stop_reason = ""
    run_state_path = os.path.join(metavideoagent_run_dir(args.run_id, args.output_root), "run.json")
    run_state = _read_json(run_state_path)
    requested_budget = (args.auto_rounds if args.auto_rounds is not None
                        else run_state.get("automatic_target_iteration", 4))
    requested_rounds = max(1, int(requested_budget))
    if not args.check_only:
        run_state["automatic_target_iteration"] = requested_rounds
        write_workspace_json(run_state_path, run_state)
    start_iteration = _resolve_iteration_index(args)
    planned_rounds = max(0, requested_rounds - start_iteration + 1)
    for round_idx in range(start_iteration, requested_rounds + 1):
        # Keep automatic and staged runs in the same iter_N layout.
        round_dir = os.path.join(
            metavideoagent_run_dir(args.run_id, args.output_root),
            "checks", f"iter_{round_idx}"
        ) if args.check_only else os.path.join(
            metavideoagent_run_dir(args.run_id, args.output_root), f"iter_{round_idx}"
        )
        os.makedirs(round_dir, exist_ok=True)
        iteration = IterationWorkspace(round_dir)
        if round_idx == start_iteration and getattr(args, "restart_from", ""):
            iteration.restart(args.restart_from)
        iteration.bind_base(bundle=_initial_bundle_path_from_report(current_reference_report),
                            report=current_reference_path,
                            results=_reference_results_path_from_report(current_reference_report))
        base_reference = iteration.progress()["base_reference"]
        if base_reference["report"] != current_reference_path:
            current_reference_report = _read_json(base_reference["report"])
        current_reference_path = base_reference["report"]
        current_reference_label = _reference_label_for_report(args, current_reference_report, current_reference_path)
        args._iteration_dir = round_dir
        if _report_has_no_incorrect_trajectory(current_reference_report):
            stop_reason = "no_incorrect_evolution_trajectories"
            break
        round_record = {"round": round_idx, "round_dir": round_dir}
        if current_diagnosis is None and not current_review_path and round_idx > 1:
            current_review_path = _latest_valid_candidate_review(
                args.run_id, args.output_root
            )
            if not current_review_path and not args.check_only:
                raise RuntimeError(
                    "Cannot resume automatic evolution without the preceding candidate's valid "
                    "review. Pass --evolution-review explicitly or restore a valid "
                    "review record in evolution_memory_index.json; do not review the "
                    "current best against itself."
                )
        iter_context = build_iter_context(
            run_id=args.run_id,
            iter_index=round_idx,
            stage="automatic",
            workspace=args.workspace,
            distribution_manifest=args.distribution_manifest,
            reference_report_path=current_reference_path,
            reference_label=current_reference_label,
            output_root=args.output_root,
            profile_path=profile.get("output_path", ""),
            profile=profile,
            initial_baseline_bundle=_initial_bundle_path_from_report(current_reference_report),
            previous_diagnosis=(current_diagnosis or {}).get("report_path", "") if current_diagnosis else "",
            evolution_review=current_review_path,
            full_eval_report=last_full_eval_report,
            candidate_run=last_candidate_run,
            candidate_bundle=last_candidate_bundle,
            deep_research_summary=getattr(args, "deep_research_summary", ""),
            extra_artifacts=[getattr(args, "deep_research_report", "")],
            check_only=args.check_only,
        )
        round_record["iter_context_path"] = iter_context.get("path", "")
        saved_diagnosis = ((iteration.progress().get("stages") or {}).get("diagnosis") or {})
        if not args.check_only and saved_diagnosis.get("status") == "complete" and current_diagnosis is None:
            current_diagnosis = _read_json(saved_diagnosis["result"])

        if current_diagnosis is None:
            if args.check_only:
                review_path = os.path.join(round_dir, "review", "teacher_evolution_review.json")
                diagnosis_path = os.path.join(round_dir, "diagnosis.json")
                round_record["reference_review"] = {
                    "check_only": True,
                    "path": review_path,
                    "purpose": "reference_review_before_diagnosis",
                }
                round_record["diagnosis"] = {
                    "check_only": True,
                    "would_read_reference_results": _reference_results_path_from_report(current_reference_report),
                    "would_read_evolution_review": review_path,
                    "path": diagnosis_path,
                }
            else:
                review_path = current_review_path or getattr(args, "evolution_review", "") or ""
                if not review_path:
                    old_review_args = {
                        "full_eval_report": getattr(args, "full_eval_report", ""),
                        "full_eval_results": getattr(args, "full_eval_results", ""),
                        "review_out_dir": getattr(args, "review_out_dir", ""),
                        "candidate_bundle": getattr(args, "candidate_bundle", ""),
                    }
                    try:
                        args.full_eval_report = current_reference_path
                        args.full_eval_results = _reference_results_path_from_report(current_reference_report)
                        args.review_out_dir = os.path.join(round_dir, "teacher")
                        args.candidate_bundle = ""
                        review_exit = _run_review_subprocess(args, current_reference_report)
                        review_path = _find_review_path(args.review_out_dir)
                        round_record["review"] = {
                            "exit_code": review_exit,
                            "review_path": review_path,
                            "purpose": "reference_review_before_diagnosis",
                        }
                        if review_exit != 0 or not review_path:
                            round_record["status"] = "reference_review_failed"
                            rounds.append(round_record)
                            break
                    finally:
                        args.full_eval_report = old_review_args["full_eval_report"]
                        args.full_eval_results = old_review_args["full_eval_results"]
                        args.review_out_dir = old_review_args["review_out_dir"]
                        args.candidate_bundle = old_review_args["candidate_bundle"]
                old_evolution_review = getattr(args, "evolution_review", "")
                args.evolution_review = review_path or old_evolution_review
                current_review_path = args.evolution_review
                current_diagnosis = _diagnose(args, current_reference_report, profile, iter_context=iter_context)
                args.evolution_review = old_evolution_review
                diagnosis_path = current_diagnosis.get("report_path", "")
                round_record["diagnosis"] = {
                    "path": diagnosis_path,
                    "review_path": review_path,
                    "target_modules": list(
                        (current_diagnosis.get("algorithmic_evolution", {}) or {}).get("target_modules") or []
                    ),
                }
        else:
            diagnosis_path = current_diagnosis.get("report_path", "")
            round_record["diagnosis"] = {
                "path": diagnosis_path,
                "target_modules": list(
                    (current_diagnosis.get("algorithmic_evolution", {}) or {}).get("target_modules") or []
                ),
            }

        codex_research = (
            {"skipped": True, "reason": "check_only_planned_after_diagnosis"}
            if args.check_only else _run_codex_deep_research_context(
                args,
                profile,
                diagnosis_path,
                current_review_path,
                round_dir,
            )
        )
        if codex_research:
            _ensure_research_ready(args, codex_research)
            round_record["codex_deep_research"] = _research_record(codex_research)
        if args.check_only:
            diagnosis_for_codex = os.path.join(round_dir, "diagnosis_for_codex.json")
            machine_contract_path = os.path.join(round_dir, "probe_plan.json")
        else:
            diagnosis_for_codex = _write_diagnosis_for_codex(
                diagnosis_path,
                codex_research,
                round_dir,
                iter_context=iter_context,
            )
        if not args.check_only:
            audit = _read_json(diagnosis_path)
            machine_contract_path = str(audit.get("machine_evaluation_contract_path") or "")
            if not machine_contract_path or not os.path.isfile(machine_contract_path):
                raise RuntimeError("Codex evolve requires diagnosis.machine_evaluation_contract_path")
        codex = _run_codex_subprocess(
            args, diagnosis_for_codex, current_reference_path, round_dir,
            machine_evaluation_contract_path=machine_contract_path,
        )
        if not args.check_only:
            codex = _run_bundle_feedback_loop(
                args, initial_codex=codex, diagnosis_path=diagnosis_for_codex,
                reference_report_path=current_reference_path, round_dir=round_dir,
                machine_evaluation_contract_path=machine_contract_path,
            )
        round_record["codex"] = codex
        candidate_run = codex.get("candidate_run", "")
        if args.check_only:
            placeholder_report = os.path.join(round_dir, "full_eval", "attempt_0", "full_eval_report.json")
            placeholder_results = os.path.join(round_dir, "full_eval", "attempt_0", "full_eval_results.jsonl")
            placeholder_review_dir = os.path.join(round_dir, "review")
            round_record["full_eval"] = {
                "check_only": True,
                "candidate_run": candidate_run,
                "planned_cmd": [
                    METAVIDEOAGENT_PYTHON,
                    os.path.join(EVOLUTION_DIR, "full_eval_runner.py"),
                    "--workspace", args.workspace,
                    "--candidate-run", candidate_run,
                    "--distribution-manifest", args.distribution_manifest,
                    "--reference-report", current_reference_path,
                    "--reference-label", current_reference_label,
                    "--concurrency", str(args.concurrency),
                    "--metavideoagent-run-id", args.run_id,
                    "--metavideoagent-output-root", args.output_root,
                    "--check-only",
                ],
            }
            round_record["post_full_eval_review"] = {
                "check_only": True,
                "planned_cmd": [
                    METAVIDEOAGENT_PYTHON,
                    os.path.join(EVOLUTION_DIR, "evolution_review_runner.py"),
                    "--workspace", args.workspace,
                    "--video-id", args.video_id,
                    "--evolved-results", placeholder_results,
                    "--verification-report", placeholder_report,
                    "--distribution-manifest", args.distribution_manifest,
                    "--reference-results", _reference_results_path_from_report(current_reference_report),
                    "--reference-label", current_reference_label,
                    "--concurrency", str(args.concurrency),
                    "--out-dir", placeholder_review_dir,
                ],
            }
            rounds.append(round_record)
            continue
        if codex.get("exit_code") != 0 or not candidate_run:
            round_record["status"] = "codex_failed"
            rounds.append(round_record)
            break

        probe_gate = _probe_gate_from_candidate_run(candidate_run)
        round_record["probe_gate"] = probe_gate
        probe_action = probe_gate.get("action", "")
        if probe_action == "execution_layer_rerun":
            round_record["status"] = "probe_execution_layer_rerun_required"
            round_record["note"] = (
                "Probe observed provider/asset failure. The same candidate must be rerun "
                "after execution-layer repair; no Codex repair or diagnosis is valid."
            )
            rounds.append(round_record)
            break
        if probe_action == "probe_skipped":
            round_record["status"] = "bundle_smoke_passed_probe_skipped"
            rounds.append(round_record)
            break
        if probe_action in (
            "codex_engineering_repair",
            "codex_same_direction_behavior_iteration",
        ):
            rejection_context_path = _write_candidate_rejection_context_from_run(
                candidate_run,
                os.path.join(round_dir, "candidate_rejection_context.json"),
            )
            memory = _annotate_and_record_candidate_history(
                args,
                iteration=round_idx,
                parent_report_path=current_reference_path,
                parent_report=current_reference_report,
                candidate_run=candidate_run,
                candidate_bundle_path=_candidate_artifact_path(candidate_run),
                diagnosis_path=diagnosis_path,
                research_report_path=(codex_research or {}).get("report_path", ""),
                rejection_context_path=rejection_context_path,
                outcome_status=(
                    "engineering_invalid"
                    if probe_action == "codex_engineering_repair"
                    else "probe_same_direction_iteration_pending"
                ),
            )
            round_record["candidate_rejection_context"] = rejection_context_path
            round_record["evolution_memory_index"] = memory.get("path", "")
            round_record["status"] = "probe_requires_codex_feedback"
            resume_option = (
                "--engineering-failed-run" if probe_action == "codex_engineering_repair"
                else "--probe-failed-run"
            )
            round_record["note"] = (
                "The candidate still needs work after the configured repair budget. "
                "Current best is retained. Inspect the saved failure evidence; "
                f"candidate continuation uses {resume_option} and a larger repair budget."
            )
            rounds.append(round_record)
            break
        if probe_action != "full_eval_candidate":
            round_record["status"] = "unknown_probe_gate"
            rounds.append(round_record)
            break

        args.candidate_run = candidate_run
        args.candidate_bundle = _candidate_artifact_path(candidate_run)
        args.diagnosis_report = diagnosis_path
        args.reference_label = current_reference_label
        full_exit = _run_full_eval_subprocess(args, current_reference_path)
        full_report_path = _find_full_eval_report(candidate_run)
        round_record["full_eval"] = {
            "exit_code": full_exit,
            "report_path": full_report_path,
        }
        if full_exit != 0 or not full_report_path:
            round_record["status"] = "full_eval_failed"
            rounds.append(round_record)
            break
        full_report = _read_json(full_report_path)
        last_full_eval_report = full_report_path
        last_candidate_run = candidate_run
        last_candidate_bundle = _candidate_artifact_path(candidate_run)
        full_results = full_report.get("results_path", "")
        post_decision = full_report.get("post_full_eval_decision") or {}
        if post_decision:
            round_record["post_full_eval_decision"] = post_decision
        parent_reference_path = current_reference_path
        parent_reference_report = current_reference_report
        parent_reference_label = current_reference_label
        next_reference_path = current_reference_path
        next_reference_report = current_reference_report
        next_reference_label = current_reference_label
        if post_decision.get("update_current_best") is True:
            next_reference_path = full_report_path
            next_reference_report = full_report
            next_reference_label = _reference_label_for_report(args, full_report, full_report_path)
            round_record["current_best_updated"] = True
        selected = load_ledger(args.run_id, args.output_root).get("current_best") or {}
        selected_report = ((selected.get("selection") or {}).get("train_diagnostic_report_path")
                           or selected.get("report_path") or "")
        if selected_report:
            next_reference_path = selected_report
            next_reference_report = _read_json(selected_report)
            next_reference_label = _reference_label_for_report(args, next_reference_report, selected_report)
            round_record["current_best_updated"] = not _same_path(selected_report, parent_reference_path)

        # Promotion has already been committed by full evaluation. Reflect it
        # immediately, including when a later review fails. Paired review still
        # uses the saved pre-promotion parent, not the newly accepted candidate.
        current_reference_path = next_reference_path
        current_reference_report = next_reference_report
        current_reference_label = next_reference_label

        # Preserve every full-eval candidate even if a later Teacher review
        # crashes.  The subsequent call after review upgrades this same record
        # with the micro/macro findings.
        memory = _annotate_and_record_candidate_history(
            args,
            iteration=round_idx,
            parent_report_path=parent_reference_path,
            parent_report=parent_reference_report,
            candidate_run=candidate_run,
            candidate_bundle_path=_candidate_artifact_path(candidate_run),
            diagnosis_path=diagnosis_path,
            research_report_path=(codex_research or {}).get("report_path", ""),
            full_eval_report_path=full_report_path,
        )
        round_record["evolution_memory_index"] = memory.get("path", "")

        review_dir = os.path.join(round_dir, "review")
        args.full_eval_report = full_report_path
        args.full_eval_results = full_results
        args.candidate_bundle = _candidate_artifact_path(candidate_run)
        args.review_out_dir = review_dir
        # Review this candidate against the immutable parent reference, even
        # when post-decision has already promoted it to current best.
        args.reference_label = parent_reference_label
        review_exit = _run_review_subprocess(args, parent_reference_report)
        review_path = _find_review_path(args.review_out_dir)
        round_record["review"] = {
            "exit_code": review_exit,
            "review_path": review_path,
        }
        if review_exit != 0 or not review_path:
            round_record["status"] = "review_failed"
            rounds.append(round_record)
            break

        memory = _annotate_and_record_candidate_history(
            args,
            iteration=round_idx,
            parent_report_path=parent_reference_path,
            parent_report=parent_reference_report,
            candidate_run=candidate_run,
            candidate_bundle_path=args.candidate_bundle,
            diagnosis_path=diagnosis_path,
            research_report_path=(codex_research or {}).get("report_path", ""),
            full_eval_report_path=full_report_path,
            review_path=review_path,
        )
        round_record["evolution_memory_index"] = memory.get("path", "")

        current_review_path = review_path
        round_record["status"] = "complete"
        iteration.finish("complete")
        if (round_idx >= requested_rounds
                or _report_has_no_incorrect_trajectory(current_reference_report)):
            rounds.append(round_record)
            continue

        candidate = _read_candidate(candidate_run)
        next_iter_context = build_iter_context(
            run_id=args.run_id,
            iter_index=round_idx + 1,
            stage="next_diagnosis",
            workspace=args.workspace,
            distribution_manifest=args.distribution_manifest,
            reference_report_path=next_reference_path,
            reference_label=next_reference_label,
            output_root=args.output_root,
            profile_path=profile.get("output_path", ""),
            profile=profile,
            initial_baseline_bundle=_initial_bundle_path_from_report(next_reference_report),
            previous_diagnosis=diagnosis_path,
            evolution_review=review_path,
            full_eval_report=full_report_path,
            candidate_run=candidate_run,
            candidate_bundle=_candidate_artifact_path(candidate_run),
            deep_research_summary=getattr(args, "deep_research_summary", ""),
            extra_artifacts=[getattr(args, "deep_research_report", "")],
        )
        args.reference_label = next_reference_label
        next_diagnosis = _diagnose_next_round(
            args, profile, next_reference_report, current_diagnosis or {},
            full_report_path, candidate, review_path,
            iter_context=next_iter_context,
        )
        round_record["next_diagnosis"] = {
            "path": next_diagnosis.get("report_path", ""),
            "target_modules": list(
                (next_diagnosis.get("algorithmic_evolution", {}) or {}).get("target_modules") or []
            ),
        }
        current_diagnosis = next_diagnosis
        rounds.append(round_record)

    output["automatic_rounds"] = rounds
    output["automatic_requested_rounds"] = requested_rounds
    output["automatic_start_iteration"] = start_iteration
    output["automatic_planned_rounds"] = planned_rounds
    output["automatic_processed_rounds"] = len(rounds)
    output["automatic_stop_reason"] = stop_reason
    output["automatic_final_current_best"] = {
        "reference_label": current_reference_label,
        "report_path": current_reference_path,
        "results_path": _reference_results_path_from_report(current_reference_report),
        "bundle_path": _initial_bundle_path_from_report(current_reference_report),
    }
    output.update(_automatic_outcome(
        rounds, planned_rounds, check_only=bool(args.check_only),
        stop_reason=stop_reason,
    ))
    return output


def run(args: argparse.Namespace) -> Dict:
    runtime_dir = require_metavideoagent_runtime("run_metavideoagent_evolution")
    args.output_root = runtime_output_root(getattr(args, "output_root", ""))
    # This wrapper is the formal stage launcher.  A dry Codex invocation can
    # otherwise return successfully without a candidate and leave a report
    # that looks like a completed evolution stage.  Require one explicit
    # acknowledgement for every non-check-only stage, including review and
    # diagnosis, both of which invoke live models.
    if not getattr(args, "check_only", False) and not getattr(args, "run", False):
        raise RuntimeError(
            "pass --run to execute a formal MetaVideoAgent stage; use --check-only "
            "for a no-model preflight"
        )
    if not getattr(args, "check_only", False):
        # Provider profiles own credential and endpoint resolution. CLI values
        # are optional run-scoped overrides, which also supports local or
        # credential-free OpenAI-compatible endpoints.
        os.environ.update(_normalize_api_env({}, args))
    workspace = os.path.abspath(args.workspace)
    args.workspace = workspace
    args.distribution_manifest = resolve_distribution_manifest_path(
        args.distribution_manifest
    )
    run_id = args.run_id or _unique_run_id("metavideoagent_evolution")
    args.run_id = run_id
    os.environ["METAVIDEOAGENT_RUN_ID"] = run_id
    os.environ["METAVIDEOAGENT_OUTPUT_ROOT"] = args.output_root
    os.environ["METAVIDEOAGENT_DISTRIBUTION_MANIFEST"] = args.distribution_manifest
    reference_report_path = _resolve_initial_report(
        args.initial_report,
        args.initial_run_id or args.run_id,
        args.output_root,
        args.reference_label,
    )

    output = {
        "run_id": run_id,
        "workspace": workspace,
        "runtime_dir": runtime_dir,
        "output_root": args.output_root,
        "distribution_manifest": args.distribution_manifest,
        "stage": args.stage,
        "reference_label": args.reference_label,
        "check_only": args.check_only,
    }

    if args.stage in ("initial", "automatic") and not reference_report_path:
        mapping_initialization = getattr(args, "initialization", "observe") == "map"
        initial_profile = {} if mapping_initialization else _load_or_build_profile(
            workspace,
            args.distribution_manifest,
            run_id,
            output_root=args.output_root,
            explicit_profile=args.profile,
            force=args.refresh_profile,
            semantic_kwargs=_profile_semantic_kwargs(args),
            check_only=args.check_only,
        )
        if not args.profile:
            args.profile = initial_profile.get("output_path", "")
        if not mapping_initialization and not getattr(args, "skip_deep_research", False):
            research = _run_deep_research_context(args, initial_profile, run_id)
            _ensure_research_ready(args, research)
            output["deep_research"] = {
                "report_path": research.get("report_path", ""),
                "summary_path": research.get("summary_path", ""),
                "design_brief_path": research.get("design_brief_path", ""),
                "mode": research.get("mode", research.get("source", "")),
                "research_complete": research.get("research_complete"),
                "formal_research": research.get("formal_research"),
            }
            if not args.deep_research_summary:
                args.deep_research_summary = research.get("summary_text", "")
            if not args.deep_research_report:
                args.deep_research_report = research.get("report_path", "")
        output["initial_profile_summary"] = {
            "profile_path": initial_profile.get("output_path", ""),
            "primary_channel": (
                initial_profile.get("information_channel_hypothesis", {}) or {}
            ).get("primary_observed_channel", ""),
        }
        initial = _run_initial(args, run_id)
        output["initial"] = initial
        reference_report_path = _reference_report_path_from_initial(initial)
        output["status"] = "initial_checked" if args.check_only else "initial_completed"
        if reference_report_path:
            reference_report = _read_json(reference_report_path)
            output["reference_report_path"] = reference_report_path
            output["reference_results_path"] = _reference_results_path_from_report(reference_report)
        else:
            output["reference_report_path"] = ""
            output["reference_results_path"] = ""
        if args.check_only and not reference_report_path:
            output["status"] = "check_only_ready"
            report_path = _stage_report_path(args, output)
            output["report_path"] = report_path
            _write_json(report_path, output)
            return output
        if args.stage == "initial":
            if args.check_only and not reference_report_path:
                output["status"] = "check_only_initial_ready"
            out_dir = metavideoagent_run_dir(run_id, args.output_root)
            if output.get("reference_results_path") and not args.check_only:
                # `initial` returns before the generic reference-input path
                # below.  Seed the first executable baseline here so a fresh
                # run never starts its first evolution with an empty ledger.
                existing_ledger = load_ledger(run_id, args.output_root)
                seed_current_best = not bool(existing_ledger.get("current_best"))
                refresh_current_best = _needs_current_best_provenance_refresh(
                    existing_ledger.get("current_best") or {},
                    reference_report_path,
                    output["reference_results_path"],
                )
                ledger = record_reference(
                    run_id,
                    "current_best" if (seed_current_best or refresh_current_best) else "reference_input",
                    reference_report_path,
                    output["reference_results_path"],
                    combo=_combo_from_reference_report(reference_report),
                    metrics={
                        "total_questions": reference_report.get("total_questions"),
                        "correct": (
                            reference_report.get("candidate_correct")
                            if reference_report.get("candidate_correct") is not None
                            else reference_report.get("correct")
                        ),
                        "accuracy": reference_report.get("accuracy"),
                        "accuracy_delta": reference_report.get("accuracy_delta"),
                    },
                    output_root=args.output_root,
                    manifest_path=args.distribution_manifest,
                    split=EVOLUTION_SPLIT,
                    dataset_version=reference_report.get("dataset_version", ""),
                    bundle_path=_initial_bundle_path_from_report(reference_report),
                    structure_artifact=_reference_structure_artifact_from_report(reference_report),
                    update_current_best=(seed_current_best or refresh_current_best),
                )
                output["current_best_ledger"] = ledger.get("path", "")
            report_path = os.path.join(out_dir, "metavideoagent_evolution_report.json")
            output["report_path"] = report_path
            _write_json(report_path, output)
            return output


    if not reference_report_path:
        if args.check_only:
            output["status"] = "check_only_requires_initial_reference"
            iteration_index = int(getattr(args, "iteration_index", 0) or 0) or 1
            out_dir = os.path.join(
                metavideoagent_run_dir(run_id, args.output_root),
                "checks", f"iter_{iteration_index}",
            )
            report_path = os.path.join(out_dir, f"{args.stage}_stage_report.json")
            output["iteration_index"] = iteration_index
            _write_json(report_path, output)
            output["report_path"] = report_path
            return output
        raise RuntimeError(
            "No initial/reference report available. Run --stage initial first "
            "or pass --initial-report/--initial-run-id."
        )
    reference_report_path = os.path.abspath(reference_report_path)
    reference_report = _read_json(reference_report_path)
    if not reference_report and args.check_only and output.get("initial"):
        reference_report = _synthetic_reference_report_from_initial(output["initial"])
    inferred_reference_label = _reference_label_for_report(
        args, reference_report, reference_report_path
    )
    args.reference_label = inferred_reference_label
    iteration_index = _resolve_iteration_index(args)
    args.iteration_index = iteration_index
    args._iteration_dir = os.path.join(metavideoagent_run_dir(run_id, args.output_root),
                                     "checks", f"iter_{iteration_index}") if args.check_only else os.path.join(
                                         metavideoagent_run_dir(run_id, args.output_root), f"iter_{iteration_index}")
    if args.stage != "automatic":
        iteration = IterationWorkspace(args._iteration_dir)
        if getattr(args, "restart_from", ""):
            iteration.restart(args.restart_from)
        iteration.bind_base(bundle=_initial_bundle_path_from_report(reference_report),
                            report=reference_report_path,
                            results=_reference_results_path_from_report(reference_report))
    os.environ["METAVIDEOAGENT_ITERATION_INDEX"] = str(iteration_index)
    output["reference_label"] = inferred_reference_label
    output["iteration_index"] = iteration_index
    output["reference_report_path"] = reference_report_path
    output["reference_results_path"] = _reference_results_path_from_report(reference_report)
    needs_reference_results = args.stage in (
        "diagnose", "codex_deep_research", "evolve", "full_eval", "review", "automatic"
    )
    if needs_reference_results and not output["reference_results_path"]:
        output["status"] = "requires_executed_reference_results"
        output["error"] = (
            f"stage={args.stage} requires a real executed reference report/results. "
            "A plan/check-only pipeline report is not enough."
        )
        out_dir = metavideoagent_run_dir(run_id, args.output_root)
        report_path = os.path.join(out_dir, "metavideoagent_evolution_report.json")
        output["report_path"] = report_path
        _write_json(report_path, output)
        if args.check_only:
            return output
        raise RuntimeError(output["error"])
    if output["reference_results_path"] and not args.check_only:
        # A freshly executed initial/reference evaluation is the first legal
        # execution baseline.  Without this entry, a first candidate that has
        # no net gain leaves the run with no current-best bundle/artifact to
        # reuse.  Later stage invocations only append reference_input records
        # and must never displace the established current best.
        existing_ledger = load_ledger(run_id, args.output_root)
        seed_current_best = not bool(existing_ledger.get("current_best"))
        refresh_current_best = _needs_current_best_provenance_refresh(
            existing_ledger.get("current_best") or {},
            reference_report_path,
            output["reference_results_path"],
        )
        ledger = record_reference(
            run_id,
            "current_best" if (seed_current_best or refresh_current_best) else "reference_input",
            reference_report_path,
            output["reference_results_path"],
            combo=_combo_from_reference_report(reference_report),
            metrics={
                "total_questions": reference_report.get("total_questions"),
                "correct": (
                    reference_report.get("candidate_correct")
                    if reference_report.get("candidate_correct") is not None
                    else reference_report.get("correct")
                ),
                "accuracy": reference_report.get("accuracy"),
                "accuracy_delta": reference_report.get("accuracy_delta"),
            },
            output_root=args.output_root,
            manifest_path=args.distribution_manifest,
            split=EVOLUTION_SPLIT,
            dataset_version=reference_report.get("dataset_version", ""),
            bundle_path=_initial_bundle_path_from_report(reference_report),
            structure_artifact=_reference_structure_artifact_from_report(reference_report),
            update_current_best=(seed_current_best or refresh_current_best),
        )
        output["current_best_ledger"] = ledger.get("path", "")
    elif output["reference_results_path"]:
        output["current_best_ledger"] = os.path.join(
            metavideoagent_run_dir(run_id, args.output_root), "current_best_ledger.json"
        )

    profile_path_hint = args.profile or ""
    if not profile_path_hint and output.get("initial"):
        profile_path_hint = output["initial"].get("profile_path", "")
    profile = _load_or_build_profile(
        workspace,
        args.distribution_manifest,
        run_id,
        output_root=args.output_root,
        explicit_profile=profile_path_hint,
        force=args.refresh_profile,
        semantic_kwargs=_profile_semantic_kwargs(args),
        check_only=args.check_only,
    )
    # All later stages must consume the exact same distribution-aware profile.
    # Without this assignment Codex/review/full-eval subprocesses silently
    # rebuild separate profiles from the manifest and lose the observed-channel
    # artifact chosen for the current run.
    # ``_load_or_build_profile`` may replace an explicitly supplied previous or
    # mismatched profile with the current run's observed artifact. Propagate
    # the effective path unconditionally so later stages cannot reopen the
    # stale caller path.
    if profile.get("output_path", ""):
        args.profile = profile["output_path"]
    deep_research = {}
    # Initial deep research is only for first-bundle design. Later-round
    # diagnosis stays review-focused; diagnosis-conditioned research is run
    # separately immediately before Codex writes code.
    initial_research_stage = (
        args.stage in ("initial", "automatic")
        and getattr(args, "initialization", "observe") == "observe"
        and not args.diagnosis_report
        and not args.evolution_review
        and not args.full_eval_report
        and not getattr(args, "deep_research_report", "")
        and not getattr(args, "deep_research_summary", "")
    )
    if initial_research_stage and not getattr(args, "skip_deep_research", False):
        deep_research = _run_deep_research_context(
            args,
            profile,
            run_id,
        )
        _ensure_research_ready(args, deep_research)
        if not args.deep_research_summary:
            args.deep_research_summary = deep_research.get("summary_text", "")
    if deep_research:
        output["deep_research"] = {
            "report_path": deep_research.get("report_path", ""),
            "summary_path": deep_research.get("summary_path", ""),
            "design_brief_path": deep_research.get("design_brief_path", ""),
            "mode": deep_research.get("mode", deep_research.get("source", "")),
            "research_complete": deep_research.get("research_complete"),
            "formal_research": deep_research.get("formal_research"),
        }
    elif not initial_research_stage:
        output["deep_research"] = {
            "skipped": True,
            "reason": f"stage={args.stage} consumes existing artifacts and must not start deep research",
        }

    if args.stage == "automatic":
        output = _run_autonomous_loop(
            args,
            reference_report_path,
            reference_report,
            profile,
            output,
        )
        outcome = str(output.get("automatic_outcome") or "")
        output["status"] = {
            "checked": "automatic_checked",
            "completed": "automatic_completed",
            "incomplete": "automatic_incomplete",
            "failed": "automatic_failed",
        }.get(outcome, "automatic_failed")
        report_path = _stage_report_path(args, output)
        output["report_path"] = report_path
        _write_json(report_path, output)
        return output

    if args.stage == "diagnose":
        review_path = getattr(args, "evolution_review", "")
        if not review_path:
            raise RuntimeError(
                "stage=diagnose requires --evolution-review produced by the preceding "
                "review stage; diagnosis cannot bypass review."
            )
        if not args.check_only:
            review = _read_json(review_path)
            if not review_is_consumable(review):
                raise RuntimeError(
                    "stage=diagnose requires a consumable preceding evolution review; "
                    "rerun review before diagnosis."
                )
    output["profile_summary"] = {
        "profile_path": profile.get("output_path", ""),
        "sampling": profile.get("sampling", {}),
        "primary_channel": (
            profile.get("information_channel_hypothesis", {}) or {}
        ).get("primary_observed_channel", ""),
    }
    iter_context = build_iter_context(
        run_id=run_id,
        iter_index=iteration_index,
        stage=args.stage,
        workspace=workspace,
        distribution_manifest=args.distribution_manifest,
        reference_report_path=reference_report_path,
        reference_label=args.reference_label,
        output_root=args.output_root,
        profile_path=profile.get("output_path", ""),
        profile=profile,
        initial_baseline_bundle=_initial_bundle_path_from_report(reference_report),
        previous_diagnosis=getattr(args, "previous_diagnosis", ""),
        evolution_review=args.evolution_review,
        full_eval_report=args.full_eval_report,
        candidate_run=args.candidate_run,
        candidate_bundle=args.candidate_bundle,
        deep_research_summary=(
            getattr(args, "deep_research_summary", "")
            or (deep_research.get("summary_text", "") if deep_research else "")
        ),
        extra_artifacts=[
            deep_research.get("report_path", "") if deep_research else "",
            deep_research.get("summary_path", "") if deep_research else "",
            deep_research.get("design_brief_path", "") if deep_research else "",
        ],
        check_only=args.check_only,
    )
    output["iter_context_path"] = iter_context.get("path", "")
    if args.evolution_review:
        output["evolution_review_path"] = os.path.abspath(args.evolution_review)

    if args.stage == "diagnose":
        if args.check_only:
            output["diagnosis_check"] = {
                "would_read_reference_results": output["reference_results_path"],
                "reference_label": args.reference_label,
                "iter_context_path": iter_context.get("path", ""),
                "evolution_review_path": os.path.abspath(args.evolution_review) if args.evolution_review else "",
            }
        else:
            diagnosis = _diagnose(args, reference_report, profile, iter_context=iter_context)
            output["diagnosis"] = {
                "timestamp": diagnosis.get("timestamp"),
                "report_path": diagnosis.get("report_path", ""),
                "execution_brief_path": diagnosis.get("execution_brief_path", ""),
                "machine_evaluation_contract_path": diagnosis.get("machine_evaluation_contract_path", ""),
                "target_modules": list(
                    (diagnosis.get("algorithmic_evolution", {}) or {}).get("target_modules") or []
                ),
                "reference_results_path": diagnosis.get("reference_results_path", ""),
            }

    if args.stage == "codex_deep_research":
        diagnosis_path = args.diagnosis_report
        diagnosis_audit = _read_json(diagnosis_path)
        _refresh_diagnosis_probe_context(diagnosis_audit)
        execution_brief_path = diagnosis_audit.get("execution_brief_path", "")
        if not diagnosis_path or not execution_brief_path or not os.path.isfile(execution_brief_path):
            raise RuntimeError(
                "stage=codex_deep_research requires --diagnosis-report pointing to "
                "a diagnosis audit with an existing adaptive bundle execution_brief_path"
            )
        brief_issues = validate_bundle_execution_brief(_read_json(execution_brief_path))
        if brief_issues:
            raise RuntimeError("codex_deep_research requires a valid adaptive bundle brief: " + "; ".join(brief_issues))
        # Continue research in the iteration recorded by its diagnosis.
        diagnosis_iter_context = diagnosis_audit.get("iter_context") or {}
        round_dir = (
            diagnosis_iter_context.get("iter_dir", "")
            or iter_context.get("iter_dir", "")
            or metavideoagent_run_dir(run_id, args.output_root)
        )
        if args.check_only:
            output["codex_deep_research_check"] = {
                "diagnosis_path": os.path.abspath(diagnosis_path),
                "execution_brief_path": os.path.abspath(execution_brief_path),
                "profile_path": profile.get("output_path", ""),
                "round_dir": round_dir,
                "would_write_diagnosis_for_codex": os.path.join(round_dir, "diagnosis_for_codex.json"),
            }
        else:
            codex_research = _run_codex_deep_research_context(
                args,
                profile,
                diagnosis_path,
                args.evolution_review,
                round_dir,
            )
            _ensure_research_ready(args, codex_research)
            output["codex_deep_research"] = _research_record(codex_research)
            output["diagnosis_for_codex_path"] = _write_diagnosis_for_codex(
                diagnosis_path,
                codex_research,
                round_dir,
                iter_context=iter_context,
            )

    if args.stage == "evolve":
        diagnosis_path = args.diagnosis_report
        # Continue implementation in the diagnosis's iteration workspace.
        staged_audit = _read_json(diagnosis_path) if diagnosis_path else {}
        staged_iter_context = staged_audit.get("iter_context") or {}
        round_dir = (
            staged_iter_context.get("iter_dir", "")
            or iter_context.get("iter_dir", "")
            or metavideoagent_run_dir(run_id, args.output_root)
        )
        # Reuse the completed research handoff for this implementation stage.
        audit = _read_json(diagnosis_path)
        _refresh_diagnosis_probe_context(audit)
        execution_brief = (
            audit
            if audit.get("artifact_type") == "diagnosis_execution_brief"
            else _read_json(audit.get("execution_brief_path", ""))
        )
        if not execution_brief:
            raise RuntimeError(
                "staged evolution requires --diagnosis-report with a valid "
                "adaptive bundle execution brief"
            )
        brief_issues = validate_bundle_execution_brief(execution_brief)
        if brief_issues:
            raise RuntimeError(
                "staged evolution requires a valid adaptive bundle brief: "
                + "; ".join(brief_issues)
            )
        diagnosis_for_codex = os.path.join(round_dir, "diagnosis_for_codex.json")
        existing_handoff = _load_validated_diagnosis_for_codex(diagnosis_for_codex)
        codex_research = existing_handoff.get("codex_deep_research", {}) or {}
        output["codex_deep_research"] = _research_record(codex_research)
        output["diagnosis_for_codex_path"] = diagnosis_for_codex
        audit = _read_json(diagnosis_path)
        machine_contract_path = str(audit.get("machine_evaluation_contract_path") or "")
        if not machine_contract_path or not os.path.isfile(machine_contract_path):
            raise RuntimeError("Codex evolve requires diagnosis.machine_evaluation_contract_path")
        codex = _run_codex_subprocess(
            args,
            diagnosis_for_codex,
            reference_report_path,
            round_dir,
            machine_evaluation_contract_path=machine_contract_path,
        )
        if not args.check_only:
            codex = _run_bundle_feedback_loop(
                args,
                initial_codex=codex,
                diagnosis_path=diagnosis_for_codex,
                reference_report_path=reference_report_path,
                round_dir=round_dir,
                machine_evaluation_contract_path=machine_contract_path,
            )
        output["codex"] = codex
        output["evolution_exit_code"] = codex.get("exit_code", 1)
        if output["evolution_exit_code"] != 0:
            output["status"] = "evolve_failed"
            output["failure_stage"] = "codex_evolve"
            output["error"] = (
                "Codex evolve exited non-zero before a valid candidate/probe result. "
                "Inspect the isolated codex stdout/stderr artifacts; do not treat this as a completed evolution round."
            )
        elif codex.get("candidate_run"):
            output["candidate_run"] = codex.get("candidate_run")
            candidate_run = codex.get("candidate_run")
            # Smoke-only execution is an engineering checkpoint; behavioral
            # classification follows a subsequent probe.
            probe_gate = _probe_gate_from_candidate_run(candidate_run)
            output["probe_gate"] = probe_gate
            if probe_gate.get("action") == "probe_skipped":
                output["status"] = "bundle_smoke_passed_probe_skipped"
                output["probe_gate"] = {
                    "action": "probe_skipped",
                    "reason": (
                        "--skip-probe-verification stopped after the recorded "
                        "assembly and smoke checks; resume the automatic probe "
                        "stage for this immutable candidate before any behavioral route."
                    ),
                    "requires_probe_replay": True,
                    "codex_feedback_allowed": False,
                }
                output["note"] = (
                    "The saved candidate passed smoke. Run probe before full evaluation."
                )
                output["report_path"] = _stage_report_path(args, output)
                _write_json(output["report_path"], output)
                return output
            if probe_gate.get("action") == "execution_layer_rerun":
                output["status"] = "probe_execution_layer_rerun_required"
                output["evolution_exit_code"] = 2
                output["note"] = (
                    "Probe observed provider/asset failure. Rerun the same candidate after "
                    "execution-layer repair; do not write rejection context."
                )
                output["report_path"] = _stage_report_path(args, output)
                _write_json(output["report_path"], output)
                return output
            if probe_gate.get("action") in {
                "codex_engineering_repair",
                "codex_same_direction_behavior_iteration",
            }:
                rejection_path = _write_candidate_rejection_context_from_run(
                    candidate_run,
                    os.path.join(candidate_run, "candidate_rejection_context.json"),
                )
                memory = _annotate_and_record_candidate_history(
                    args,
                    iteration=iteration_index,
                    parent_report_path=reference_report_path,
                    parent_report=reference_report,
                    candidate_run=candidate_run,
                    candidate_bundle_path=_candidate_artifact_path(candidate_run),
                    diagnosis_path=diagnosis_path,
                    research_report_path=(codex_research or {}).get("report_path", ""),
                    rejection_context_path=rejection_path,
                    outcome_status=(
                        "engineering_invalid"
                        if probe_gate.get("action") in {
                            "codex_engineering_repair",
                        }
                        else "probe_same_direction_iteration_pending"
                    ),
                )
                output["candidate_rejection_context"] = rejection_path
                output["evolution_memory_index"] = memory.get("path", "")
                output["status"] = "probe_requires_codex_feedback"
                output["evolution_exit_code"] = 2
            elif probe_gate.get("action") != "full_eval_candidate":
                output["status"] = "unknown_probe_gate"
                output["evolution_exit_code"] = 1
        else:
            output["status"] = "evolve_failed"
            output["evolution_exit_code"] = 1
            output["error"] = "The coding stage completed without a candidate artifact."
    if args.stage == "full_eval":
        full_exit = _run_full_eval_subprocess(
            args, reference_report_path
        )
        output["full_eval_exit_code"] = full_exit
        if full_exit != 0:
            output["status"] = "full_eval_failed"
        if not args.check_only and args.candidate_run:
            full_report_path = _find_full_eval_report(args.candidate_run)
            if full_exit == 0 and not full_report_path:
                raise RuntimeError("Full evaluation completed without its results report.")
            if full_report_path:
                output["full_eval_report_path"] = full_report_path
                full_report = _read_json(full_report_path)
                output["post_full_eval_decision"] = (
                    full_report.get("post_full_eval_decision") or {}
                )
                memory = _annotate_and_record_candidate_history(
                    args,
                    iteration=iteration_index,
                    parent_report_path=reference_report_path,
                    parent_report=reference_report,
                    candidate_run=args.candidate_run,
                    candidate_bundle_path=(
                        args.candidate_bundle or full_report.get("candidate_bundle", "")
                    ),
                    diagnosis_path=args.diagnosis_report,
                    full_eval_report_path=full_report_path,
                )
                output["evolution_memory_index"] = memory.get("path", "")
    if args.stage == "review":
        output["review_exit_code"] = _run_review_subprocess(
            args, reference_report
        )
        review_path = _find_review_path(args.review_out_dir)
        if not args.check_only and output["review_exit_code"] == 0 and review_path and review_is_consumable(_read_json(review_path)):
            full_report_path = args.full_eval_report
            full_report = _read_json(full_report_path)
            candidate_run = args.candidate_run or full_report.get("candidate_run", "")
            candidate_bundle = args.candidate_bundle or full_report.get("candidate_bundle", "")
            parent_report_path = full_report.get("reference_report_path", "") or reference_report_path
            parent_report = _read_json(parent_report_path) or reference_report
            memory = _annotate_and_record_candidate_history(
                args,
                iteration=iteration_index,
                parent_report_path=parent_report_path,
                parent_report=parent_report,
                candidate_run=candidate_run,
                candidate_bundle_path=candidate_bundle,
                diagnosis_path=getattr(args, "previous_diagnosis", ""),
                full_eval_report_path=full_report_path,
                review_path=review_path,
            )
            output["review_path"] = review_path
            output["evolution_memory_index"] = memory.get("path", "")
        elif not args.check_only:
            output["status"] = "review_failed"
            output["review_exit_code"] = output["review_exit_code"] or 1
            output["review_path"] = review_path
    output.setdefault(
        "status",
        f"{args.stage}_{'checked' if args.check_only else 'completed'}",
    )
    report_path = _stage_report_path(args, output)
    output["report_path"] = report_path
    _write_json(report_path, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged MetaVideoAgent initialization and automatic evolution")
    parser.add_argument("--workspace", default=DEFAULT_METAVIDEOAGENT_WORKSPACE)
    parser.add_argument("--distribution-manifest", required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "initial", "diagnose", "codex_deep_research", "evolve",
            "full_eval", "review", "automatic",
        ),
        default="initial",
    )
    parser.add_argument("--initial-report", default="")
    parser.add_argument("--initial-run-id", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default="",
                        help="Output root; defaults to ./metavideoagent_outputs")
    parser.add_argument("--restart-from", choices=("diagnosis", "research", "codegen", "smoke", "probe", "full_eval"), default="")
    parser.add_argument("--iteration-index", type=int, default=0,
                        help=(
                            "Select iter_N explicitly; otherwise continue from run.json."
                        ))
    parser.add_argument("--reference-label", default="")
    parser.add_argument("--video-id", default="ALL")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--first-n", type=int, default=0)
    parser.add_argument("--rebuild-structure", action="store_true")
    parser.add_argument("--structure-workers", type=int, default=4)
    parser.add_argument("--combo-id", default="")
    parser.add_argument("--candidate-run", default="")
    parser.add_argument("--candidate-bundle", default="",
                        help="Current adaptive candidate bundle for diagnosis/review lineage context")
    parser.add_argument("--probe-failed-run", default="",
                        help=(
                            "Existing probe-failed candidate run for automatic repair."
                        ))
    parser.add_argument("--engineering-failed-run", default="",
                        help=(
                            "Persisted static/runtime-smoke candidate to resume with no-label "
                            "automatic engineering feedback. Used only by --stage evolve."
                        ))
    parser.add_argument("--diagnosis-report", default="")
    parser.add_argument("--previous-diagnosis", default="",
                        help=(
                            "Completed diagnosis from the preceding iteration. "
                            "Later-round diagnosis uses it as historical evidence, "
                            "not as the combo baseline."
                        ))
    parser.add_argument("--full-eval-report", default="")
    parser.add_argument("--full-eval-results", default="")
    parser.add_argument("--initial-reference-review", action="store_true",
                        help="Review one executed initial reference without a fabricated self-comparison.")
    parser.add_argument("--evolution-review", default="")
    parser.add_argument("--review-out-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-post-eval-decision", action="store_true",
                        help="Skip the optional model explanation of the train-selection result")
    parser.add_argument("--profile", default="")
    parser.add_argument("--initialization", choices=("observe", "map"), default="observe",
                        help="Initialize from distribution observations or map an existing agent; evolution is shared.")
    parser.add_argument("--source-agent", default="", help="Curated existing-agent source directory for --initialization map.")
    parser.add_argument("--refresh-profile", action="store_true",
                        help="Repeat observation and rebuild initialization/profile.json.")
    parser.add_argument("--semantic-observer-mode", choices=("none", "plan", "observe"), default="observe",
                        help="Five-frame profile observer. observe calls the configured VLM; plan only extracts frames.")
    parser.add_argument("--semantic-observer-dir", default="")
    parser.add_argument("--semantic-video-budget", type=int, default=0)
    parser.add_argument("--semantic-observe-workers", type=int, default=1)
    parser.add_argument("--api-key", default="", help="Per-run provider API key; not persisted")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--combo-plan", default="")
    parser.add_argument("--initial-baseline-bundle", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--deep-research-summary", default="")
    parser.add_argument("--deep-research-report", default="")
    parser.add_argument("--skip-deep-research", action="store_true")
    parser.add_argument("--iteration-research-summary", dest="codex_deep_research_summary", default="",
                        help="Inline diagnosis-conditioned research summary for candidate authoring")
    parser.add_argument("--iteration-research-report", dest="codex_deep_research_report", default="",
                        help="Validated diagnosis-conditioned research report")
    parser.add_argument("--skip-iteration-research", dest="skip_codex_deep_research", action="store_true",
                        help="Skip optional diagnosis-conditioned research for this iteration")
    parser.add_argument("--deep-research-max-queries", type=int, default=5)
    parser.add_argument("--research-min-sources", type=int, default=2,
                        help="Minimum verifiable sources required for deep research")
    parser.add_argument("--auto-rounds", type=int, default=None,
                        help=("Total evolution-round budget for this run; new runs default to 4. "
                              "Resuming without this option keeps the saved budget; a larger value extends it."))
    parser.add_argument("--codex-cli", default="",
                        help="Codex executable or a wrapper implementing the same command contract.")
    parser.add_argument("--research-codex-cli", default="",
                        help="Optional compatible executable used for Deep Research.")
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL,
                        help=f"Default coding model: {DEFAULT_CODEX_MODEL}.")
    parser.add_argument("--codex-reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        help=f"Default coding reasoning effort: {DEFAULT_REASONING_EFFORT}.")
    parser.add_argument("--repair-attempts", type=int, default=3,
                        help="Additional source-assembly or smoke repair attempts before probe.")
    parser.add_argument("--probe-repair-attempts", type=int, default=2,
                        help="Bounded automatic repair attempts for failed probe outcomes")
    parser.add_argument("--skip-probe-verification", action="store_true",
                        help="Stop candidate validation after assembly and smoke; do not run probe")
    parser.add_argument("--run", action="store_true",
                        help="Required acknowledgement before any non-check-only formal stage executes.")
    parser.add_argument("--check-only", action="store_true")
    parser.set_defaults(
        research_search_timeout=900,
        research_codex_cli="",
        initial_codex_cli="",
        codegen_timeout=900,
    )
    return parser


def _report_exit_code(report: Dict) -> int:
    for key in ("automatic_exit_code", "full_eval_exit_code", "evolution_exit_code", "review_exit_code"):
        code = int(report.get(key) or 0)
        if code:
            return code
    return 0


def main() -> int:
    args = build_parser().parse_args()
    codex_stages = {"initial", "evolve", "automatic"}
    if args.stage in codex_stages:
        try:
            args.codex_runtime_settings = resolve_codex_runtime_settings(
                model=args.codex_model,
                reasoning_effort=args.codex_reasoning_effort,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        os.environ["CODEX_MODEL"] = args.codex_runtime_settings["model"]
        os.environ["CODEX_REASONING_EFFORT"] = args.codex_runtime_settings["reasoning_effort"]
    else:
        args.codex_runtime_settings = {}
    from run_journal import append_operation
    started_at = time.time()
    try:
        report = run(args)
    except Exception as exc:
        append_operation(output_root=args.output_root, run_id=args.run_id, stage=args.stage,
                         args=args, started_at=started_at, status="failed", error=str(exc))
        raise
    operation_status = "checked" if args.check_only else "completed"
    exit_code = _report_exit_code(report)
    if exit_code:
        operation_status = str(report.get("automatic_outcome") or ("incomplete" if exit_code == 2 else "failed"))
    append_operation(output_root=args.output_root, run_id=args.run_id, stage=args.stage,
                     args=args, started_at=started_at,
                     status=operation_status, report=report)
    print("METAVIDEOAGENT_EVOLUTION_DONE" if exit_code == 0 else "METAVIDEOAGENT_EVOLUTION_STOPPED")
    print(f"report_path={report.get('report_path')}")
    print(f"reference_report_path={report.get('reference_report_path')}")
    print(f"reference_results_path={report.get('reference_results_path')}")
    print(f"exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
