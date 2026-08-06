#!/usr/bin/env python3
"""Staged target-distribution evolution entrypoint.

The from-scratch workflow:

1. Build five-frame, query-aware records for the target distribution.
2. Design and run an initial combo as the first reference.
3. Diagnose the initial reference trajectories.
4. Run explicit review, diagnosis, optional research, candidate authoring,
   smoke, probe, and full-evaluation stages against the current best.

Codex performs automatic multi-round authoring and bounded repair.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
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
DIAGNOSIS_BRIEF_MAX_ATTEMPTS = max(
    1, int(os.environ.get("DIAGNOSIS_BRIEF_MAX_ATTEMPTS", "3") or "3")
)
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
from capability_registry import resolve_profile  # noqa: E402
from bundle_contract import normalize_modules  # noqa: E402

DEFAULT_EXECUTION_LLM_MODEL = str(resolve_profile("llm").get("model_id") or "")
DEFAULT_EXECUTION_VLM_MODEL = str(resolve_profile("vlm").get("model_id") or "")

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
from deep_research_runner import run_deep_research, validate_design_brief  # noqa: E402
from diagnosis_agent import DiagnosisAgent, sanitize_tool_terms_for_evolution  # noqa: E402
from diagnosis_execution_brief import (  # noqa: E402
    validate_bundle_execution_brief,
    write_bundle_execution_brief,
)
from distribution_spec import (  # noqa: E402
    load_distribution_spec,
    resolve_distribution_manifest_path,
)
from evaluation_audit import build_full_eval_audit, read_jsonl, write_audit_files  # noqa: E402
from iter_context import build_iter_context, inject_iter_context  # noqa: E402
from ledger import (  # noqa: E402
    load_evolution_memory,
    load_ledger,
    record_evolution_outcome,
    record_reference,
)
from paths import run_dir as metavideoagent_run_dir  # noqa: E402
from post_full_eval_decider import current_best_update_allowed, decide_post_full_eval  # noqa: E402
from review_contract import review_is_consumable  # noqa: E402
from runtime_paths import (  # noqa: E402
    require_metavideoagent_runtime,
    runtime_output_root,
)
from runtime_paths import (
    runtime_dir as selected_runtime_dir,
)


def _normalize_api_env(env: Dict[str, str], args: argparse.Namespace = None) -> Dict[str, str]:
    """Normalize model/API env for subprocesses.

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
        if getattr(args, "exec_llm_model", ""):
            env["METAVIDEOAGENT_LLM_MODEL"] = args.exec_llm_model
            env["CODEX_SMOKE_JUDGE_MODEL"] = args.exec_llm_model
        if getattr(args, "vlm_model", ""):
            env["METAVIDEOAGENT_VLM_MODEL"] = args.vlm_model
        if getattr(args, "asr_model", ""):
            env["METAVIDEOAGENT_ASR_MODEL"] = args.asr_model
        if getattr(args, "structure_workers", None):
            env["VIDEO_STRUCT_REBUILD_WORKERS"] = str(args.structure_workers)
            env["VIDEO_STRUCT_REBUILD_BATCH"] = str(max(1, int(args.structure_workers) * 8))
        # Bundle-internal ancestry guards must use the same finite budget as
        # the staged outer feedback loop.  Otherwise a caller requesting three
        # probe repairs would be rejected by the bundle's hidden default of
        # two on the third resume.
        if getattr(args, "probe_repair_attempts", None) is not None:
            budget = str(max(1, int(args.probe_repair_attempts or 0)))
            env["BUNDLE_FEEDBACK_BUDGET"] = budget
            env["BUNDLE_ENGINEERING_FEEDBACK_BUDGET"] = budget
    return env


def _unique_run_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{os.getpid()}"


def _configure_inprocess_diagnosis_model(args: argparse.Namespace) -> str:
    """Return the configured diagnosis model, using the reproduction default."""
    del args
    return diagnosis_agent_module.DIAGNOSIS_MODEL


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
    """Return an explicit probe size without embedding a dataset-specific default.

    A caller may override the manifest with ``--first-n``.  Otherwise the
    distribution manifest is the single source of truth for bounded
    probe coverage.  MetaVideoAgent runs must not silently fall back to an
    arbitrary sample such as three questions.
    """
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


def _iter_from_path(path: str) -> int:
    if not path:
        return 0
    matches = re.findall(r"(?:^|/)iter_(\d+)(?:/|$)", os.path.abspath(path))
    return int(matches[-1]) if matches else 0


def _resolve_iteration_index(args: argparse.Namespace, reference_report_path: str = "") -> int:
    """Resolve an explicit staged iteration without silently writing ``iter_1``.

    Explicit ``--iteration-index`` wins.  Otherwise output paths identify the
    target iteration; a diagnosis that only points at the preceding full eval
    is assigned to the following iteration.  If no artifact is available, use
    the next unused iteration directory for this run.
    """
    explicit = int(getattr(args, "iteration_index", 0) or 0)
    if explicit > 0:
        return explicit
    direct_paths = (
        getattr(args, "out_dir", ""),
        getattr(args, "review_out_dir", ""),
        getattr(args, "diagnosis_report", ""),
        getattr(args, "candidate_run", ""),
        getattr(args, "candidate_bundle", ""),
        getattr(args, "evolution_review", ""),
    )
    for path in direct_paths:
        index = _iter_from_path(path)
        if index:
            return index
    prior_paths = (getattr(args, "full_eval_report", ""), reference_report_path)
    for path in prior_paths:
        index = _iter_from_path(path)
        if index:
            return index + 1 if args.stage in {"diagnose", "review"} else index
    run_root = metavideoagent_run_dir(args.run_id, args.output_root)
    existing = [
        int(match.group(1))
        for name in (os.listdir(run_root) if os.path.isdir(run_root) else [])
        if (match := re.fullmatch(r"iter_(\d+)", name))
    ]
    return max(existing, default=0) + 1



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
        "metavideoagent_evolution_report.json",
    )


def _external_history_meta(*paths: str) -> Dict:
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


def _resolve_initial_report(workspace: str, initial_report: str = "",
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
        candidates = [
            os.path.join(
                metavideoagent_run_dir(run_id, output_root),
                "initial_baseline_report.json",
            ),
            os.path.join(
                metavideoagent_run_dir(run_id, output_root),
                "initial_baselines",
                "initial_baseline_report.json",
            ),
            os.path.join(
                metavideoagent_run_dir(run_id, output_root),
                "initial_combo",
                "initial_combo_report.json",
            ),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
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


def _brief_target_modules(value: Dict) -> list[str]:
    """Return the validated adaptive target set for a later-round brief."""
    policy = value.get("execution_policy") if isinstance(value, dict) else {}
    modules = (policy or {}).get("target_modules")
    try:
        return normalize_modules(modules)
    except ValueError as exc:
        raise RuntimeError(
            "later-round evolution requires a valid non-empty "
            f"execution_policy.target_modules list: {exc}"
        ) from exc


def _profile_path(workspace: str, manifest: str, run_id: str,
                  output_root: str = "",
                  explicit_profile: str = "",
                  force: bool = False,
                  include_execution_artifacts: bool = False,
                  semantic_kwargs: Dict = None) -> str:
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
        include_execution_artifacts=include_execution_artifacts,
        force=force,
        **(semantic_kwargs or {}),
    )
    return profile.get("output_path", "")


def _load_or_build_profile(workspace: str, manifest: str, run_id: str,
                           output_root: str = "",
                           explicit_profile: str = "",
                           force: bool = False,
                           include_execution_artifacts: bool = False,
                           semantic_kwargs: Dict = None) -> Dict:
    path = _profile_path(
        workspace, manifest, run_id, output_root=output_root,
        explicit_profile=explicit_profile,
        force=force,
        include_execution_artifacts=include_execution_artifacts,
        semantic_kwargs=semantic_kwargs,
    )
    profile = _read_json(path)
    matches, reason = profile_matches_manifest(profile, manifest)
    if matches:
        return profile

    # An explicit profile from a different run is never silently relabelled
    # as belonging to the current train manifest.  Rebuild a new, run-scoped
    # distribution-aware profile and retain the supplied artifact untouched for
    # audit. This also accepts profiles carrying an explicit provenance record.
    rebuilt_path = os.path.join(
        metavideoagent_run_dir(run_id, output_root),
        "profiles",
        "channel_profile_manifest_attested.json",
    )
    rebuilt = build_channel_profile(
        workspace=workspace,
        distribution_manifest=manifest,
        output_path=rebuilt_path,
        run_id=run_id,
        output_root=output_root,
        include_execution_artifacts=include_execution_artifacts,
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
    # Both public workflows expose the selected visual model as ``--vlm-model``.
    # Normalize once so a semantic profile
    # cannot accidentally inherit an unrelated shell/default model.
    formal_vlm_model = getattr(args, "vlm_model", "") or DEFAULT_EXECUTION_VLM_MODEL
    observer_mode = getattr(args, "semantic_observer_mode", "observe")
    if getattr(args, "check_only", False) and observer_mode == "observe":
        observer_mode = "plan"
    return {
        "semantic_observer_mode": observer_mode,
        "semantic_observer_dir": getattr(args, "semantic_observer_dir", ""),
        "semantic_video_budget": getattr(args, "semantic_video_budget", 0),
        "semantic_observe_workers": getattr(args, "semantic_observe_workers", 1),
        "runtime_dir": selected_runtime_dir(),
        "vlm_model": formal_vlm_model,
    }


def _run_deep_research_context(args: argparse.Namespace, profile: Dict,
                               run_id: str,
                               diagnosis_path: str = "",
                               review_path: str = "",
                               output_subdir: str = "deep_research") -> Dict:
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
    out_dir = os.path.join(metavideoagent_run_dir(run_id, args.output_root), output_subdir)
    return run_deep_research(
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


def _run_codex_deep_research_context(args: argparse.Namespace, profile: Dict,
                                     diagnosis_path: str, review_path: str,
                                     round_dir: str) -> Dict:
    """Run diagnosis-conditioned research immediately before automatic codegen."""
    if getattr(args, "check_only", False):
        return {"skipped": True, "reason": "check-only mode does not execute Codex research"}
    if getattr(args, "probe_failed_run", "") or getattr(args, "engineering_failed_run", ""):
        return {
            "skipped": True,
            "reason": "same-candidate feedback reuses the existing diagnosis/research handoff",
        }
    if getattr(args, "skip_codex_deep_research", False):
        return {"skipped": True, "reason": "--skip-codex-deep-research"}
    if getattr(args, "codex_deep_research_report", ""):
        if not diagnosis_path:
            raise RuntimeError("cannot resume research without the diagnosis execution brief")
        resume_diagnosis = _read_json(diagnosis_path)
        resume_execution_brief_path = str(resume_diagnosis.get("execution_brief_path") or "")
        execution_brief = _read_json(resume_execution_brief_path)
        if not execution_brief or validate_bundle_execution_brief(execution_brief):
            raise RuntimeError("cannot resume Codex research with an invalid diagnosis execution brief")
        report_path = os.path.abspath(args.codex_deep_research_report)
        report = _read_json(report_path)
        if report:
            # An explicit report can be resumed only after it is revalidated
            # against the *current* execution brief.  This supports recovery
            # from a repaired validator without re-running an already
            # completed online Codex search, while preserving the original
            # failed report as immutable audit evidence.
            design_brief_path = str(report.get("design_brief_path") or "")
            design_brief = _read_json(design_brief_path)
            allowed_capabilities = (
                (execution_brief.get("implementation_task", {}) or {})
                .get("runtime_capabilities", [])
            )
            validation = validate_design_brief(
                design_brief,
                min_sources=max(1, int(getattr(args, "research_min_sources", 2) or 2)),
                verify_source_urls=False,
                expected_target_modules=_brief_target_modules(execution_brief),
                allowed_runtime_capabilities=allowed_capabilities,
                required_bundle_contract=execution_brief,
            )
            previous_validation = report.get("validation") or {}
            prior_source_verified = (
                previous_validation.get("source_url_verification") is True
                and not list(previous_validation.get("unreachable_sources") or [])
                and int(previous_validation.get("valid_source_count", 0) or 0)
                >= max(1, int(getattr(args, "research_min_sources", 2) or 2))
            )
            if not validation.get("passed") or not prior_source_verified:
                raise RuntimeError(
                    "explicit deep-research report is not resumable under the current "
                    "execution brief: "
                    + "; ".join(validation.get("issues") or ["source verification missing"])
                )
            resumed = dict(report)
            resumed.update({
                "formal_research": True,
                "research_complete": True,
                "validation_passed": True,
                "target_modules": _brief_target_modules(execution_brief),
                "validation": {
                    **validation,
                    "source_url_verification": True,
                    "resumed_from_report": report_path,
                    "resumed_after_validator_revalidation": True,
                },
                "resumed_from_report": report_path,
            })
            resume_dir = os.path.join(
                round_dir, "codex_deep_research", _unique_run_id("revalidated")
            )
            resumed_path = os.path.join(resume_dir, "deep_research_report.json")
            resumed["report_path"] = resumed_path
            _write_json(resumed_path, resumed)
            return resumed
    if getattr(args, "codex_deep_research_summary", ""):
        return {
            "summary_text": args.codex_deep_research_summary,
            "source": "inline_argument",
            "report_path": "",
            "summary_path": "",
            "design_brief_path": "",
            "research_complete": True,
            "formal_research": False,
        }
    if not diagnosis_path:
        return {"skipped": True, "reason": "missing diagnosis_path"}
    diagnosis = _read_json(diagnosis_path)
    execution_brief_path = diagnosis.get("execution_brief_path", "")
    if not execution_brief_path or not os.path.exists(execution_brief_path):
        raise RuntimeError(
            "diagnosis is missing execution_brief_path; rerun diagnosis so deep research "
            "does not consume the full audit report"
        )
    execution_brief = _read_json(execution_brief_path)
    _brief_target_modules(execution_brief)
    from diagnosis_execution_brief import validate_bundle_machine_evaluation_contract
    contract_path = str(diagnosis.get("machine_evaluation_contract_path") or "")
    contract = _read_json(contract_path)
    issues = validate_bundle_machine_evaluation_contract(contract)
    if issues:
        raise RuntimeError(
            "diagnosis-conditioned deep research requires a valid machine evaluation contract: "
            + "; ".join(issues)
        )
    profile_path = profile.get("output_path", "") or getattr(args, "profile", "")
    if not profile_path:
        return {"skipped": True, "reason": "missing channel profile"}
    out_dir = os.path.join(round_dir, "codex_deep_research", _unique_run_id("attempt"))
    result = run_deep_research(
        workspace=args.workspace,
        channel_profile_path=profile_path,
        output_dir=out_dir,
        run_id=args.run_id,
        output_root=args.output_root,
        diagnosis_path=execution_brief_path,
        review_path="",
        max_queries=getattr(args, "deep_research_max_queries", 5),
        search_timeout=getattr(args, "research_search_timeout", 20),
        codex_cli=getattr(args, "research_codex_cli", "") or getattr(args, "codex_cli", ""),
        min_sources=getattr(args, "research_min_sources", 2),
    )
    result["usage"] = "diagnosis_to_automatic_code_generation"
    result["full_diagnosis_path"] = os.path.abspath(diagnosis_path)
    result["execution_brief_path"] = os.path.abspath(execution_brief_path)
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
                               round_dir: str,
                               iter_context: Dict = None) -> str:
    if not diagnosis_path or not research or research.get("skipped"):
        return diagnosis_path
    source_diagnosis = _read_json(diagnosis_path)
    if not source_diagnosis:
        return diagnosis_path
    diagnosis = source_diagnosis
    if diagnosis.get("artifact_type") != "diagnosis_execution_brief":
        execution_brief_path = diagnosis.get("execution_brief_path", "")
        execution_brief = _read_json(execution_brief_path)
        if not execution_brief or execution_brief.get("artifact_type") != "diagnosis_execution_brief":
            raise RuntimeError(
                "Codex handoff requires diagnosis_execution_brief; rerun diagnosis "
                "instead of passing the full audit report to code generation"
            )
        diagnosis = execution_brief
    expected_targets = _brief_target_modules(diagnosis)
    validation = research.get("validation", {}) or {}
    validation_passed = research.get(
        "validation_passed",
        validation.get("passed") if isinstance(validation, dict) else None,
    )
    design_brief_path = str(research.get("design_brief_path") or "")
    if not (
        research.get("formal_research") is True
        and research.get("research_complete") is True
        and validation_passed is True
        and list(research.get("target_modules") or []) == expected_targets
        and design_brief_path
        and os.path.isfile(design_brief_path)
    ):
        raise RuntimeError(
            "refusing to write diagnosis_for_codex without a validated same-direction "
            "formal research record and design brief"
        )
    # The compact execution brief intentionally omits audit prose, but bundle
    # assembly/runtime need an executable current-best provenance record.  It
    # must survive audit -> brief -> Codex handoff rather than be reconstructed
    # from a mutable workspace after Codex has already edited source files.
    source_context = source_diagnosis.get("iter_context") or iter_context or {}
    # Explicit staged diagnoses may have been compiled from a checked
    # provenance attestation instead of the normal iter_context injector.
    # Treat it as an equally auditable source, but never discover paths by
    # scanning mutable output trees.
    attested_runtime = source_diagnosis.get("runtime_provenance") or {}
    source_reference = source_context.get("reference") or {}
    source_best = source_context.get("current_best") or {}
    source_artifacts = source_best.get("artifacts") or {}
    bundle_info = source_artifacts.get("bundle") or source_reference.get("initial_baseline_bundle") or {}
    # A staged later-round diagnosis can intentionally run without a
    # mutable current-best ledger.  In that case ``iter_context.reference`` is
    # the historical initial reference, while ``diagnosis_inputs`` carries the
    # immutable full-eval report of the candidate that the just-completed
    # review diagnosed.  The latter is the executable parent for the next
    # bundle; silently retaining the initial bundle would select the wrong base.
    diagnosis_inputs = source_diagnosis.get("diagnosis_inputs") or {}
    previous_round = source_context.get("previous_round") or {}
    previous_full_eval_path = str(
        diagnosis_inputs.get("previous_full_eval_report")
        or (previous_round.get("full_eval_report") or {}).get("path")
        or ""
    )
    previous_full_eval = _read_json(previous_full_eval_path)
    previous_candidate_bundle = _initial_bundle_path_from_report(previous_full_eval)
    previous_candidate_results = _reference_results_path_from_report(previous_full_eval)
    previous_candidate_combo = previous_full_eval.get("combo") or previous_full_eval.get("agent_combo") or {}
    use_staged_candidate_parent = bool(
        previous_candidate_bundle
        and os.path.isfile(previous_candidate_bundle)
        and previous_candidate_results
        and os.path.isfile(previous_candidate_results)
        and isinstance(previous_candidate_combo, dict)
        and previous_candidate_combo
        and not source_best
        and not source_diagnosis.get("reference_bundle_path")
        and not attested_runtime.get("reference_bundle_path")
    )
    runtime_provenance = {
        "reference_report_path": str(
            previous_full_eval_path if use_staged_candidate_parent else
            source_diagnosis.get("reference_report_path")
            or attested_runtime.get("reference_report_path")
            or (source_reference.get("report") or {}).get("path") or ""
        ),
        "reference_results_path": str(
            previous_candidate_results if use_staged_candidate_parent else
            source_diagnosis.get("reference_results_path")
            or attested_runtime.get("reference_results_path")
            or diagnosis_inputs.get("reference_results_path")
            or (source_reference.get("results") or {}).get("path") or ""
        ),
        "reference_bundle_path": str(
            previous_candidate_bundle if use_staged_candidate_parent else
            source_diagnosis.get("reference_bundle_path")
            or attested_runtime.get("reference_bundle_path")
            or bundle_info.get("path") or ""
        ),
        "reference_combo": dict(
            previous_candidate_combo if use_staged_candidate_parent else
            source_diagnosis.get("reference_combo")
            or attested_runtime.get("reference_combo")
            or source_best.get("combo") or source_reference.get("combo") or {}
        ),
        "distribution_manifest": str(
            source_diagnosis.get("distribution_manifest")
            or attested_runtime.get("distribution_manifest")
            or source_context.get("distribution_manifest") or ""
        ),
    }
    missing_runtime = [
        name for name in ("reference_report_path", "reference_results_path", "reference_bundle_path", "distribution_manifest")
        if not runtime_provenance[name] or not os.path.isfile(runtime_provenance[name])
    ]
    if missing_runtime or not runtime_provenance["reference_combo"]:
        raise RuntimeError(
            "refusing Codex bundle handoff without immutable runtime provenance: "
            + ", ".join(missing_runtime or ["reference_combo"])
        )
    diagnosis["runtime_provenance"] = runtime_provenance
    diagnosis.update(runtime_provenance)
    diagnosis["codex_deep_research"] = {
        **_research_record(research),
        "summary_text": (research.get("summary_text") or "")[:20000],
        "design_brief": research.get("design_brief", {}),
    }
    diagnosis = sanitize_tool_terms_for_evolution(diagnosis)
    out_path = os.path.join(round_dir, "diagnosis_for_codex.json")
    _write_json(out_path, diagnosis)
    return out_path


def _load_validated_diagnosis_for_codex(path: str, execution_brief: Dict) -> Dict:
    """Load a persisted same-direction research handoff for staged evolution."""
    handoff = _read_json(path)
    if handoff.get("artifact_type") != "diagnosis_execution_brief":
        raise RuntimeError("staged evolution requires an existing diagnosis_for_codex execution brief")
    issues = validate_bundle_execution_brief(handoff)
    if issues:
        raise RuntimeError("existing diagnosis_for_codex is invalid: " + "; ".join(issues))
    handoff_task = handoff.get("implementation_task", {}) or {}
    brief_task = execution_brief.get("implementation_task", {}) or {}
    _brief_target_modules(execution_brief)
    if handoff_task != brief_task:
        raise RuntimeError(
            "existing diagnosis_for_codex does not match the supplied diagnosis execution brief"
        )
    research = handoff.get("codex_deep_research", {}) or {}
    validation = research.get("validation", {}) or {}
    validation_passed = research.get(
        "validation_passed",
        validation.get("passed") if isinstance(validation, dict) else None,
    )
    expected_targets = _brief_target_modules(execution_brief)
    design_brief_path = str(research.get("design_brief_path") or "")
    if not (
        research.get("formal_research") is True
        and research.get("research_complete") is True
        and validation_passed is True
        and list(research.get("target_modules") or []) == expected_targets
        and design_brief_path
        and os.path.isfile(design_brief_path)
    ):
        raise RuntimeError(
            "existing diagnosis_for_codex lacks a validated same-direction formal research record"
        )
    return handoff


def _validate_existing_diagnosis_for_codex(path: str, execution_brief: Dict) -> None:
    """Validate a persisted handoff before a no-write evolve preflight."""
    _load_validated_diagnosis_for_codex(path, execution_brief)


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
        out_dir=args.out_dir,
        run_id=run_id,
        video_id=args.video_id,
        question_split=EVOLUTION_SPLIT,
        concurrency=args.concurrency,
        first_n=args.first_n,
        llm_model=args.exec_llm_model,
        rebuild_structure=args.rebuild_structure,
        structure_workers=args.structure_workers,
        combo_id=args.combo_id,
        # The staged orchestrator has already built and attested this exact
        # profile before calling the initial plan/codegen stages. Reuse it
        # verbatim so codegen cannot silently create a second profile.
        skip_profile=bool(args.profile) or args.skip_profile,
        semantic_observer_mode=getattr(args, "semantic_observer_mode", "none"),
        semantic_observer_dir=getattr(args, "semantic_observer_dir", ""),
        semantic_video_budget=getattr(args, "semantic_video_budget", 0),
        semantic_observe_workers=getattr(args, "semantic_observe_workers", 1),
        vlm_model=(getattr(args, "vlm_model", "") or DEFAULT_EXECUTION_VLM_MODEL),
        asr_model=getattr(args, "asr_model", ""),
        api_key=getattr(args, "api_key", ""),
        base_url=getattr(args, "base_url", ""),
        output_root=args.output_root,
        check_only=args.check_only,
        deep_research_summary=getattr(args, "deep_research_summary", ""),
        deep_research_report=getattr(args, "deep_research_report", ""),
        codex_cli=getattr(args, "initial_codex_cli", "") or getattr(args, "codex_cli", ""),
        codegen_timeout=getattr(args, "codegen_timeout", 900),
        skip_bundle_smoke_test=False,
        smoke_only=False,
        bundle_smoke_repair_attempts=getattr(args, "repair_attempts", 0),
        adopt_smoke_report="",
    )
    if not common["initial_baseline_bundle"]:
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
    diagnosis_model = _configure_inprocess_diagnosis_model(args)
    diagnosis_attempt = max(1, int(getattr(args, "_diagnosis_attempt", 1) or 1))
    prior_brief_feedback = list(getattr(args, "_brief_validation_feedback", []) or [])
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
        brief_validation_feedback=prior_brief_feedback,
        diagnosis_attempt=diagnosis_attempt,
    )
    report = agent.run()
    report = _refresh_diagnosis_probe_context(report)
    report = inject_iter_context(report, iter_context or {})
    report["evolution_review_path"] = os.path.abspath(review_path) if review_path else ""
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
        "exec_llm_model": diagnosis_model,
        "diagnosis_attempt": diagnosis_attempt,
        "prior_brief_validation_feedback": prior_brief_feedback,
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
        algo["evolution_hint"] = (
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
        review=_read_json(review_path) if review_path else {},
        review_path=review_path,
    )
    execution_brief_path = str(execution_artifacts.get("execution_brief_path") or "")
    execution_brief = _read_json(execution_brief_path)
    brief_issues = validate_bundle_execution_brief(execution_brief)
    if brief_issues:
        failure = {
            "attempt": diagnosis_attempt,
            "issues": list(brief_issues),
            "execution_brief_path": execution_brief_path,
            "action": "retry_evolution_llm_diagnosis" if diagnosis_attempt < DIAGNOSIS_BRIEF_MAX_ATTEMPTS else "fail_closed",
        }
        report["brief_compilation_failure"] = failure
        _write_json(report_path, report)
        if diagnosis_attempt < DIAGNOSIS_BRIEF_MAX_ATTEMPTS:
            retry_args = copy.copy(args)
            retry_args._diagnosis_attempt = diagnosis_attempt + 1
            retry_args._brief_validation_feedback = list(brief_issues)
            return _diagnose(retry_args, reference_report, profile, iter_context=iter_context)
        raise RuntimeError(
            "diagnosis did not produce a valid adaptive bundle execution brief after "
            f"{diagnosis_attempt} evolution LLM attempts: " + "; ".join(brief_issues)
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


def _latest_candidate_run(codex_root: str, started_at: float = 0) -> str:
    root = os.path.join(codex_root, "codex_candidates")
    if not os.path.isdir(root):
        return ""
    candidates = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        artifact_path = os.path.join(path, "candidate_bundle.json")
        # A Codex attempt creates its run directory before it can guarantee a
        # serializable bundle.  Codegen/extraction failures therefore leave a
        # perfectly legitimate directory without this artifact.  It is not a
        # candidate and must not turn discovery of the *next* candidate into a
        # FileNotFoundError.
        if not os.path.isdir(path) or not os.path.isfile(artifact_path):
            continue
        mtime = os.path.getmtime(artifact_path)
        if started_at and mtime + 1 < started_at:
            continue
        candidates.append((mtime, path))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


def _candidate_artifact_path(candidate_run: str) -> str:
    """Return the only formal candidate artifact: a complete bundle."""
    path = os.path.join(candidate_run, "candidate_bundle.json")
    return path if os.path.isfile(path) else ""



def _run_codex_subprocess(args: argparse.Namespace, diagnosis_path: str,
                          reference_report_path: str, round_dir: str,
                          machine_evaluation_contract_path: str = "",
                          iter_context_path: str = "") -> Dict:
    codex_root = os.path.join(round_dir, "codex")
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
        "--output-root", codex_root,
        "--first_n", str(probe_first_n),
        "--concurrency", str(args.concurrency),
        "--repair-attempts", str(args.repair_attempts),
        "--probe-repair-attempts", str(args.probe_repair_attempts),
        "--candidate-rounds", str(args.candidate_rounds),
        "--codex-model", args.codex_runtime_settings["model"],
        "--codex-reasoning-effort", args.codex_runtime_settings["reasoning_effort"],
    ]
    if reference_results_path:
        cmd.extend(["--reference-results", os.path.abspath(reference_results_path)])
    if machine_evaluation_contract_path:
        cmd.extend(["--machine-evaluation-contract", machine_evaluation_contract_path])
    if args.probe_feedback_rounds is not None:
        cmd.extend(["--probe-feedback-rounds", str(args.probe_feedback_rounds)])
    if args.profile:
        cmd.extend(["--distribution-profile", args.profile])
    if iter_context_path:
        cmd.extend(["--iter-context", iter_context_path])
    if getattr(args, "probe_failed_run", ""):
        cmd.extend(["--probe-failed-run", args.probe_failed_run])
        cmd.extend(["--probe-feedback-round", str(args.probe_feedback_round)])
    if getattr(args, "engineering_failed_run", ""):
        cmd.extend(["--engineering-failed-run", args.engineering_failed_run])
    if getattr(args, "candidate_rejection_run", ""):
        cmd.extend(["--candidate-rejection-run", args.candidate_rejection_run])
    if getattr(args, "candidate_rejection_context", ""):
        cmd.extend(["--candidate-rejection-context", args.candidate_rejection_context])
    if getattr(args, "write_candidate_rejection_context", ""):
        cmd.extend([
            "--write-candidate-rejection-context",
            args.write_candidate_rejection_context,
        ])
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
            "codex_root": codex_root,
            "candidate_run": os.path.join(codex_root, "codex_candidates", "<candidate_run>"),
            "authoring_engine": "codex",
        }
    started_at = time.time()
    stdout_path = os.path.join(round_dir, "codex_stdout.txt")
    stderr_path = os.path.join(round_dir, "codex_stderr.txt")
    exit_code = _run_command(
        cmd, cwd=PROJECT_ROOT, env=_subprocess_env(args),
        stdout_path=stdout_path, stderr_path=stderr_path,
    )
    candidate_run = _latest_candidate_run(codex_root, started_at=started_at)
    return {
        "exit_code": exit_code,
        "cmd": cmd,
        "codex_root": codex_root,
        "candidate_run": candidate_run,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "authoring_engine": "codex",
    }


def _run_bundle_feedback_loop(args: argparse.Namespace, *, initial_codex: Dict,
                              diagnosis_path: str, reference_report_path: str,
                              round_dir: str, machine_evaluation_contract_path: str,
                              iter_context_path: str = "") -> Dict:
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
    saved_round = getattr(args, "probe_feedback_round", 1)
    try:
        for index in range(1, limit + 1):
            candidate_run = str(current.get("candidate_run") or "")
            if current.get("exit_code") != 0 or not candidate_run:
                break
            gate = _probe_gate_from_candidate_run(candidate_run)
            action = str(gate.get("action") or "")
            if action not in {"codex_engineering_repair", "codex_same_direction_behavior_iteration"}:
                break
            args.probe_failed_run = ""
            args.engineering_failed_run = ""
            if action == "codex_engineering_repair":
                args.engineering_failed_run = candidate_run
            else:
                args.probe_failed_run = candidate_run
                args.probe_feedback_round = index
            child_dir = os.path.join(round_dir, "bundle_feedback", f"attempt_{index}")
            child = _run_codex_subprocess(
                args, diagnosis_path, reference_report_path, child_dir,
                machine_evaluation_contract_path=machine_evaluation_contract_path,
                iter_context_path=iter_context_path,
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
        args.probe_feedback_round = saved_round
    current["bundle_feedback_attempts"] = attempts
    current["bundle_feedback_limit"] = limit
    return current


def _resolve_candidate_rejection_context(args: argparse.Namespace, run_id: str) -> str:
    """Materialize candidate rejection context for diagnosis/iteration prompts."""
    if getattr(args, "candidate_rejection_context", ""):
        context_path = os.path.abspath(args.candidate_rejection_context)
        if not _rejection_context_allows_rediagnosis(context_path):
            raise RuntimeError(
                "--candidate-rejection-context identifies an implementation or "
                "dataflow contract failure. Use --probe-failed-run for same-direction "
                "Codex repair instead of entering diagnosis."
            )
        return context_path
    rejection_run = getattr(args, "candidate_rejection_run", "")
    if not rejection_run:
        # Explicit stage transitions should have the same evidence flow as the
        # auto-loop.  Reuse the newest run-scoped rejected probe context rather
        # than requiring a caller to re-enter its path explicitly.
        memory = load_evolution_memory(run_id, args.output_root)
        for record in reversed(memory.get("records", []) or []):
            outcome = record.get("outcome", {}) if isinstance(record, dict) else {}
            if outcome.get("status") != "probe_rejected":
                continue
            artifacts = record.get("artifacts", {}) if isinstance(record, dict) else {}
            context = artifacts.get("candidate_rejection_context", {}) if isinstance(artifacts, dict) else {}
            context_path = context if isinstance(context, str) else context.get("path", "")
            if (
                context_path
                and os.path.exists(context_path)
                and _rejection_context_allows_rediagnosis(context_path)
            ):
                return os.path.abspath(context_path)
        return ""
    output_path = getattr(args, "write_candidate_rejection_context", "") or os.path.join(
        metavideoagent_run_dir(run_id, args.output_root),
        "candidate_rejection_context.json",
    )
    try:
        from codex_evolve import write_candidate_rejection_context  # noqa: WPS433

        context_path = write_candidate_rejection_context(rejection_run, output_path)
        context = _read_json(context_path)
        decision = context.get("decision", {}) if isinstance(context, dict) else {}
        next_action = decision.get("probe_next_action", {}) or {}
        if (
            decision.get("candidate_should_enter_full_eval")
            or next_action.get("action") == "full_eval_candidate"
        ):
            raise RuntimeError(
                "--candidate-rejection-run points to a probe result that should "
                "enter full evaluation, not round-level rediagnosis. Run full_eval "
                "or pass a genuinely rejected/no-gain probe candidate."
            )
        if next_action.get("action") == "execution_layer_rerun":
            raise RuntimeError(
                "--candidate-rejection-run points to a provider/asset execution failure. "
                "Repair the execution environment and rerun the same candidate; do not create diagnosis input."
            )
        if next_action.get("action") != "round_level_rediagnosis":
            raise RuntimeError(
                "--candidate-rejection-run identifies an implementation or dataflow "
                "contract failure. Use --probe-failed-run for same-direction Codex repair."
            )
        return context_path
    except Exception as exc:
        if "should enter full evaluation" in str(exc):
            raise
        if getattr(args, "check_only", False):
            return os.path.abspath(output_path)
        raise RuntimeError(
            f"Failed to build candidate rejection context from {rejection_run}: {exc}"
        ) from exc


def _run_full_eval_subprocess(args: argparse.Namespace,
                              reference_report_path: str) -> int:
    if not args.candidate_run:
        raise RuntimeError("--candidate-run is required for --stage full_eval")
    cmd = [
        METAVIDEOAGENT_PYTHON,
        os.path.join(EVOLUTION_DIR, "full_eval_runner.py"),
        "--workspace", args.workspace,
        "--candidate-run", args.candidate_run,
        "--distribution-manifest", args.distribution_manifest,
        "--reference-report", reference_report_path,
        "--reference-label", args.reference_label,
        "--concurrency", str(args.concurrency),
        "--llm-model", args.exec_llm_model,
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
        args.initial_baseline_bundle
        or _initial_bundle_path_from_report(_read_json(reference_report_path))
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
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, env=_subprocess_env(args)).returncode


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
            and not getattr(args, "full_eval_report", "")
        )
    )
    if initial_reference_review:
        results_path = _reference_results_path_from_report(reference_report)
        report_path = args.initial_report or reference_report.get("report_path", "")
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
            "--reference-label", "initial_existing_agent",
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
        return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, env=_subprocess_env(args)).returncode
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
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, env=_subprocess_env(args)).returncode


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
        "bundle_fingerprint": candidate.get("bundle_fingerprint", ""),
        "candidate_bundle_path": path,
    }


def _find_full_eval_report(candidate_run: str) -> str:
    candidates = [
        os.path.join(candidate_run, "full_eval_report.json"),
        os.path.join(candidate_run, "full_eval_report_retry.json"),
    ]
    import glob
    candidates.extend(glob.glob(os.path.join(candidate_run, "full_eval*_report.json")))
    candidates = [path for path in dict.fromkeys(candidates) if os.path.exists(path)]
    if not candidates:
        return ""
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _find_eval_audit_path(full_report_path: str, full_report: Dict = None) -> str:
    full_report = full_report or _read_json(full_report_path)
    path = (
        full_report.get("evaluation_audit_path", "")
        or ((full_report.get("evaluation_audit") or {}).get("path", ""))
    )
    if path and os.path.exists(path):
        return path
    candidate = os.path.join(
        os.path.dirname(os.path.abspath(full_report_path)),
        "evaluation_audit.json",
    )
    return candidate if os.path.exists(candidate) else ""


def _run_post_decision_stage(args: argparse.Namespace,
                             run_id: str = "",
                             full_report_path: str = "") -> Dict:
    full_report_path = full_report_path or args.full_eval_report
    if not full_report_path and args.candidate_run:
        full_report_path = _find_full_eval_report(args.candidate_run)
    if not full_report_path:
        raise RuntimeError("--full-eval-report or --candidate-run is required for post_decision")
    full_report = _read_json(full_report_path)
    # Persist the same deterministic objective policy that diagnosis/brief use,
    # so the executable current-best decision cannot silently revert to a
    # cost-sensitive default.
    from evolution_phase_policy import for_iteration
    phase_index = int(getattr(args, "iteration_index", 0) or 0)
    full_report.setdefault("evolution_phase_policy", for_iteration(phase_index))
    decision_dir = (
        os.path.join(
            metavideoagent_run_dir(run_id or args.run_id, args.output_root),
            "checks",
            f"iter_{int(getattr(args, 'iteration_index', 0) or 0) or 1}",
            "post_decision",
        )
        if args.check_only
        else
        os.path.dirname(os.path.abspath(full_report_path))
    )
    audit_path = _find_eval_audit_path(full_report_path, full_report)
    if not audit_path:
        audit_path = os.path.join(decision_dir, "evaluation_audit.json")
        audit = build_full_eval_audit(
            full_report,
            read_jsonl(full_report.get("results_path", "")),
            read_jsonl(full_report.get("reference_source", "")),
            full_report.get("sandbox_dir", ""),
        )
        write_audit_files(
            audit,
            audit_path,
            os.path.join(os.path.dirname(audit_path), "evaluation_audit_summary.md"),
        )
    else:
        audit = _read_json(audit_path)
    if full_report.get("evaluation_complete", True) is not True:
        raise RuntimeError(
            "post_decision requires a completed full eval; rerun the same candidate instead"
        )
    ledger_path = os.path.join(
        metavideoagent_run_dir(run_id or args.run_id, args.output_root),
        "current_best_ledger.json",
    )
    decision = decide_post_full_eval(
        full_report,
        audit,
        _read_json(ledger_path),
        call_llm=(not args.skip_post_eval_decision and not args.check_only),
    )
    out_path = os.path.join(
        decision_dir,
        "post_full_eval_decision.json",
    )
    _write_json(out_path, decision)
    can_update_best = current_best_update_allowed(full_report, decision)
    if not args.check_only:
        full_report["evaluation_audit_path"] = audit_path
        full_report["post_full_eval_decision_path"] = out_path
        full_report["evaluation_audit"] = {
            "path": audit_path,
            "summary_path": os.path.join(os.path.dirname(audit_path), "evaluation_audit_summary.md"),
            "cost_ratio_vs_reference": audit.get("cost_ratio_vs_reference"),
            "recommended_next_focus": (
                audit.get("decision_guidance", {}) or {}
            ).get("recommended_next_focus", ""),
            "issue_summary": audit.get("issue_summary", []),
        }
        if can_update_best:
            bundle_source = (
                full_report.get("current_best_bundle_path", "")
                or full_report.get("candidate_full_eval_bundle_path", "")
                or (full_report.get("combo_policy", {}) or {}).get("effective_bundle_path", "")
            )
            if not bundle_source or not os.path.isfile(bundle_source):
                raise RuntimeError(
                    "accepted current-best candidate is missing its complete effective bundle: "
                    f"{bundle_source}"
                )
            current_best_bundle = os.path.join(
                os.path.dirname(os.path.abspath(full_report_path)),
                "current_best_bundle.json",
            )
            if os.path.abspath(bundle_source) != os.path.abspath(current_best_bundle):
                shutil.copy2(bundle_source, current_best_bundle)
            full_report["current_best_bundle_path"] = current_best_bundle
            full_report.setdefault("combo_policy", {})["current_best_bundle_path"] = current_best_bundle
        full_report["post_full_eval_decision"] = decision
        _write_json(full_report_path, full_report)
    if (not args.check_only) and full_report.get("evaluation_complete", True) is True:
        should_update_best = can_update_best
        record_reference(
            run_id or args.run_id,
            "current_best" if should_update_best else "candidate_full_eval",
            full_report.get("report_path", full_report_path),
            full_report.get("results_path", ""),
            combo=full_report.get("combo", {}),
            metrics={
                "total_questions": full_report.get("total_questions"),
                "correct": full_report.get("candidate_correct"),
                "accuracy_delta": full_report.get("accuracy_delta", full_report.get("accuracy_delta_vs_reference")),
                "corrections": len(full_report.get("corrections", []) or []),
                "regressions": len(full_report.get("regressions", []) or []),
                "cost_ratio_vs_reference": audit.get("cost_ratio_vs_reference"),
                "api_cost_per_question": (
                    (audit.get("candidate_cost", {}) or {}).get("per_question_avg", {})
                ),
            },
            output_root=args.output_root,
            manifest_path=args.distribution_manifest,
            split=full_report.get("question_split", EVOLUTION_SPLIT),
            dataset_version=full_report.get("dataset_version", ""),
            update_current_best=should_update_best,
            decision=decision,
            decision_path=out_path,
            bundle_path=(
                full_report.get("current_best_bundle_path", "")
                or full_report.get("candidate_full_eval_bundle_path", "")
                or (full_report.get("combo_policy", {}) or {}).get("effective_bundle_path", "")
            ),
            candidate_bundle_path=full_report.get("candidate_bundle", ""),
            module_artifacts=full_report.get("module_artifacts", {}),
            structure_artifact=full_report.get("structure_artifact", {}),
            selection_split=(full_report.get("current_best_selection") or {}).get("this_eval_split", ""),
            train_diagnostic_report_path=(
                (full_report.get("current_best_selection") or {}).get("train_diagnostic_report_path", "")
            ),
            train_diagnostic_results_path=(
                (full_report.get("current_best_selection") or {}).get("train_diagnostic_results_path", "")
            ),
        )
    return {
        "decision_path": out_path,
        "decision": decision,
        "full_eval_report": full_report_path,
        "evaluation_audit": audit_path,
        "current_best_ledger": ledger_path,
    }


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
            "candidate_bundle_fingerprint": candidate.get("bundle_fingerprint", ""),
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
    """Read the Codex probe decision for a candidate run.

    `codex_evolve.py` writes the behavioral gate into validation.json.  The
    autonomous MetaVideoAgent loop must honor that gate before spending full-eval
    cost: probe net-positive candidates advance, while every other runnable
    candidate returns to bounded same-direction Codex feedback. A new diagnosis
    is only for an exhausted mechanism budget or an explicitly unclassifiable
    audit, never the default no-gain route.
    """
    if not candidate_run:
        return {"action": "", "reason": "missing candidate_run"}
    validation = _read_json(os.path.join(candidate_run, "validation.json"))
    if not validation:
        return {"action": "", "reason": "missing validation.json"}
    # Bundle assembly and runtime-smoke failures deliberately keep the gate
    # inside ``assembly_validation``: no probe has happened yet, so presenting
    # it as a top-level behavioral result would be misleading.  The staged
    # launcher nevertheless has to honor that engineering gate before it
    # considers previous/probe fallbacks.  Read both layouts so an incomplete
    # bundle resumes with --engineering-failed-run rather than being sent to
    # the supervised --probe-failed-run path (which correctly requires probe
    # artifacts that a smoke failure cannot have produced).
    assembly = validation.get("assembly_validation") or {}
    action = validation.get("probe_next_action") or (
        assembly.get("probe_next_action") if isinstance(assembly, dict) else {}
    ) or {}
    if action:
        return _reclassify_probe_gate_from_artifacts(candidate_run, action)
    if validation.get("probe_skipped"):
        return _reclassify_probe_gate_from_artifacts(candidate_run, {
            "action": "full_eval_candidate",
            "reason": "probe verification was explicitly skipped after smoke",
            "requires_full_eval": True,
        })
    sandbox = validation.get("sandbox") or {}
    status = sandbox.get("probe_status", "")
    if status == "promising":
        return _reclassify_probe_gate_from_artifacts(candidate_run, {
            "action": "full_eval_candidate",
            "reason": "previous promising probe status",
            "requires_full_eval": True,
        })
    if status in ("invalid", "unsafe"):
        return _reclassify_probe_gate_from_artifacts(candidate_run, {
            "action": "codex_engineering_repair",
            "reason": f"previous probe status={status}",
            "codex_feedback_allowed": True,
        })
    return _reclassify_probe_gate_from_artifacts(candidate_run, {
        "action": "codex_same_direction_behavior_iteration",
        "reason": f"previous runnable probe needs same-direction feedback; probe_status={status or 'unknown'}",
        "codex_feedback_allowed": True,
    })


def _reclassify_probe_gate_from_artifacts(candidate_run: str,
                                          stored_action: Dict) -> Dict:
    """Upgrade stale gates when probe artifacts reveal a contract failure.

    Older candidate directories may carry `round_level_rediagnosis` from a
    weaker routing policy.  The persisted audit is authoritative for the
    behavior actually observed, so reclassify stale diagnosis gates and leave
    a full-eval or already-feedback gate untouched.
    """
    stored_action = dict(stored_action or {})
    if stored_action.get("action") not in ("", "round_level_rediagnosis"):
        return stored_action
    verification = _read_json(os.path.join(candidate_run, "probe_verification.json"))
    audit = _read_json(os.path.join(candidate_run, "probe_audit.json"))
    if not verification and not audit:
        return stored_action
    try:
        from codex_evolve import classify_probe_next_action  # noqa: WPS433

        current_action = classify_probe_next_action(verification, audit)
    except Exception:
        return stored_action
    if current_action.get("action") not in {
        "codex_engineering_repair",
        "codex_same_direction_behavior_iteration",
    }:
        return stored_action
    return {
        **current_action,
        "reclassified_from": stored_action,
        "reason": (
            "stored probe gate was reclassified from round-level diagnosis using "
            "the persisted probe audit and current same-direction feedback policy"
        ),
    }


def _rejection_context_allows_rediagnosis(context_path: str) -> bool:
    """Return whether a persisted rejection context is behavioral, not repairable."""
    context = _read_json(context_path)
    source_run = str((context or {}).get("source_probe_run") or "")
    if not source_run or not os.path.isdir(source_run):
        # Previous/missing artifacts cannot be reclassified. Preserve the old
        # behavior rather than silently discarding the caller's explicit input.
        return True
    return _probe_gate_from_candidate_run(source_run).get("action") == "round_level_rediagnosis"


def _write_candidate_rejection_context_from_run(args: argparse.Namespace,
                                                candidate_run: str,
                                                output_path: str) -> str:
    from codex_evolve import write_candidate_rejection_context  # noqa: WPS433

    return write_candidate_rejection_context(candidate_run, output_path)


def _previous_result_from_full_eval(full_report: Dict, candidate: Dict) -> Dict:
    return {
        "target_modules": list(candidate.get("changed_modules") or []),
        "evolved_bundle": {
            "bundle_fingerprint": candidate.get("bundle_fingerprint", ""),
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
                         round_dir: str, iter_context: Dict = None) -> Dict:
    """Use the same diagnosis-to-brief path as explicit staged iterations.

    Auto-loop must not bypass `_diagnose`: doing so leaves the next round
    without the bounded execution handoff and makes later research/codegen read
    an inconsistent audit report.
    """
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
    snapshot = os.path.join(round_dir, "next_diagnosis.json")
    _write_json(snapshot, diagnosis)
    diagnosis["snapshot_path"] = snapshot
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
    requested_rounds = max(1, int(args.auto_rounds or 1))
    start_iteration = max(1, int(getattr(args, "iteration_index", 0) or 1))
    for round_idx in range(
        start_iteration,
        start_iteration + requested_rounds,
    ):
        if _report_has_no_incorrect_trajectory(current_reference_report):
            stop_reason = "no_incorrect_evolution_trajectories"
            break
        # Use the same iter_N layout as staged execution. This makes a
        # resumed automatic loop append to the run history instead of
        # overwriting a previous round_1 directory.
        round_dir = os.path.join(
            metavideoagent_run_dir(args.run_id, args.output_root),
            "checks", f"iter_{round_idx}"
        ) if args.check_only else os.path.join(
            metavideoagent_run_dir(args.run_id, args.output_root), f"iter_{round_idx}"
        )
        os.makedirs(round_dir, exist_ok=True)
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
        prior_candidate_rejection_run = getattr(args, "candidate_rejection_run", "")
        prior_candidate_rejection_context = getattr(args, "candidate_rejection_context", "")
        prior_write_candidate_rejection_context = getattr(args, "write_candidate_rejection_context", "")
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
            candidate_rejection_context=_resolve_candidate_rejection_context(args, args.run_id),
            deep_research_summary=getattr(args, "deep_research_summary", ""),
            extra_artifacts=[getattr(args, "deep_research_report", "")],
            check_only=args.check_only,
        )
        round_record["iter_context_path"] = iter_context.get("path", "")

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
                        args.review_out_dir = os.path.join(round_dir, "review")
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
            diagnosis_path = current_diagnosis.get("report_path") or current_diagnosis.get("snapshot_path", "")
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
            audit = _read_json(diagnosis_path)
            execution_brief = _read_json(audit.get("execution_brief_path", ""))
            diagnosis_for_codex = os.path.join(round_dir, "diagnosis_for_codex.json")
            _validate_existing_diagnosis_for_codex(diagnosis_for_codex, execution_brief)
        else:
            diagnosis_for_codex = _write_diagnosis_for_codex(
                diagnosis_path,
                codex_research,
                round_dir,
                iter_context=iter_context,
            )
        audit = _read_json(diagnosis_path)
        machine_contract_path = str(audit.get("machine_evaluation_contract_path") or "")
        if not machine_contract_path or not os.path.isfile(machine_contract_path):
            raise RuntimeError("Codex evolve requires diagnosis.machine_evaluation_contract_path")
        codex = _run_codex_subprocess(
            args, diagnosis_for_codex, current_reference_path, round_dir,
            machine_evaluation_contract_path=machine_contract_path,
            iter_context_path=iter_context.get("path", ""),
        )
        round_record["codex"] = codex
        candidate_run = codex.get("candidate_run", "")
        if args.check_only:
            placeholder_report = os.path.join(candidate_run, "full_eval_report.json")
            placeholder_results = os.path.join(candidate_run, "full_eval_results.jsonl")
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
                    "--llm-model", args.exec_llm_model,
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
        if probe_action == "round_level_rediagnosis":
            rejection_context_path = os.path.join(
                round_dir,
                "candidate_rejection_context.json",
            )
            rejection_context_path = _write_candidate_rejection_context_from_run(
                args,
                candidate_run,
                rejection_context_path,
            )
            round_record["candidate_rejection_context"] = rejection_context_path
            rejection_memory = _annotate_and_record_candidate_history(
                args,
                iteration=round_idx,
                parent_report_path=current_reference_path,
                parent_report=current_reference_report,
                candidate_run=candidate_run,
                candidate_bundle_path=_candidate_artifact_path(candidate_run),
                diagnosis_path=diagnosis_path,
                research_report_path=(codex_research or {}).get("report_path", ""),
                rejection_context_path=rejection_context_path,
                outcome_status="probe_rejected",
            )
            round_record["evolution_memory_index"] = rejection_memory.get("path", "")
            args.candidate_rejection_run = candidate_run
            args.candidate_rejection_context = rejection_context_path
            args.write_candidate_rejection_context = rejection_context_path
            next_iter_context = build_iter_context(
                run_id=args.run_id,
                iter_index=round_idx + 1,
                stage="probe_rejection_rediagnosis",
                workspace=args.workspace,
                distribution_manifest=args.distribution_manifest,
                reference_report_path=current_reference_path,
                reference_label=current_reference_label,
                output_root=args.output_root,
                profile_path=profile.get("output_path", ""),
                profile=profile,
                initial_baseline_bundle=_initial_bundle_path_from_report(current_reference_report),
                previous_diagnosis=diagnosis_path,
                evolution_review=current_review_path,
                candidate_run=candidate_run,
                candidate_bundle=_candidate_artifact_path(candidate_run),
                candidate_rejection_context=rejection_context_path,
                deep_research_summary=getattr(args, "deep_research_summary", ""),
                extra_artifacts=[getattr(args, "deep_research_report", "")],
            )
            current_diagnosis = _diagnose(
                args,
                current_reference_report,
                profile,
                iter_context=next_iter_context,
            )
            round_record["next_diagnosis"] = {
                "path": current_diagnosis.get("report_path", ""),
                "target_modules": list(
                    (current_diagnosis.get("algorithmic_evolution", {}) or {}).get("target_modules") or []
                ),
                "reason": "probe round_level_rediagnosis",
            }
            args.candidate_rejection_run = prior_candidate_rejection_run
            args.candidate_rejection_context = prior_candidate_rejection_context
            args.write_candidate_rejection_context = prior_write_candidate_rejection_context
            rounds.append(round_record)
            continue
        if probe_action in (
            "codex_engineering_repair",
            "codex_same_direction_behavior_iteration",
        ):
            rejection_context_path = _write_candidate_rejection_context_from_run(
                args,
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
            round_record["note"] = (
                "codex_evolve did not produce a full-eval-ready candidate after "
                "the configured bounded same-direction feedback attempts; resume it "
                "with --probe-failed-run, or start a new diagnosis only after the "
                "per-mechanism feedback budget is exhausted"
            )
            rounds.append(round_record)
            break
        if probe_action and probe_action != "full_eval_candidate":
            round_record["status"] = "unknown_probe_gate"
            rounds.append(round_record)
            break

        args.candidate_run = candidate_run
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
        next_reference_path = current_reference_path
        next_reference_report = current_reference_report
        next_reference_label = current_reference_label
        if post_decision.get("update_current_best") is True:
            next_reference_path = full_report_path
            next_reference_report = full_report
            next_reference_label = _reference_label_for_report(args, full_report, full_report_path)
            round_record["current_best_updated"] = True

        # Preserve every full-eval candidate even if a later Teacher review
        # crashes.  The subsequent call after review upgrades this same record
        # with the micro/macro findings.
        memory = _annotate_and_record_candidate_history(
            args,
            iteration=round_idx,
            parent_report_path=current_reference_path,
            parent_report=current_reference_report,
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
        args.reference_label = current_reference_label
        review_exit = _run_review_subprocess(args, current_reference_report)
        review_path = _find_review_path(review_dir)
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
            parent_report_path=current_reference_path,
            parent_report=current_reference_report,
            candidate_run=candidate_run,
            candidate_bundle_path=args.candidate_bundle,
            diagnosis_path=diagnosis_path,
            research_report_path=(codex_research or {}).get("report_path", ""),
            full_eval_report_path=full_report_path,
            review_path=review_path,
        )
        round_record["evolution_memory_index"] = memory.get("path", "")

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
            candidate_rejection_context=_resolve_candidate_rejection_context(args, args.run_id),
            deep_research_summary=getattr(args, "deep_research_summary", ""),
            extra_artifacts=[getattr(args, "deep_research_report", "")],
        )
        args.reference_label = next_reference_label
        next_diagnosis = _diagnose_next_round(
            args, profile, next_reference_report, current_diagnosis or {},
            full_report_path, candidate, review_path, round_dir,
            iter_context=next_iter_context,
        )
        round_record["next_diagnosis"] = {
            "path": next_diagnosis.get("report_path", ""),
            "snapshot_path": next_diagnosis.get("snapshot_path", ""),
            "target_modules": list(
                (next_diagnosis.get("algorithmic_evolution", {}) or {}).get("target_modules") or []
            ),
        }
        current_diagnosis = next_diagnosis
        current_reference_path = next_reference_path
        current_reference_report = next_reference_report
        current_reference_label = next_reference_label
        current_review_path = review_path
        rounds.append(round_record)

    output["automatic_rounds"] = rounds
    output["automatic_requested_rounds"] = requested_rounds
    output["automatic_processed_rounds"] = len(rounds)
    output["automatic_stop_reason"] = stop_reason
    output["automatic_final_current_best"] = {
        "reference_label": current_reference_label,
        "report_path": current_reference_path,
        "results_path": _reference_results_path_from_report(current_reference_report),
        "bundle_path": _initial_bundle_path_from_report(current_reference_report),
    }
    output.update(_automatic_outcome(
        rounds, requested_rounds, check_only=bool(args.check_only),
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
    if (getattr(args, "probe_failed_run", "") or getattr(args, "engineering_failed_run", "")) and (
        getattr(args, "candidate_rejection_run", "")
        or getattr(args, "candidate_rejection_context", "")
        or getattr(args, "write_candidate_rejection_context", "")
    ):
        raise RuntimeError(
            "same-candidate Codex feedback is incompatible with "
            "--candidate-rejection-run/context, which is for round-level "
            "rediagnosis/new-candidate context."
        )
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
        workspace,
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

    if args.stage in ("initial", "all", "automatic") and not reference_report_path:
        initial_profile = _load_or_build_profile(
            workspace,
            args.distribution_manifest,
            run_id,
            output_root=args.output_root,
            explicit_profile=args.profile,
            force=args.refresh_profile,
            include_execution_artifacts=False,
            semantic_kwargs=_profile_semantic_kwargs(args),
        )
        if not args.profile:
            args.profile = initial_profile.get("output_path", "")
        if not getattr(args, "skip_deep_research", False):
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
            out_dir = metavideoagent_run_dir(run_id, args.output_root)
            report_path = os.path.join(out_dir, "metavideoagent_evolution_report.json")
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

    if args.stage == "post_decision":
        post_decision = _run_post_decision_stage(args, run_id=run_id)
        output["post_full_eval_decision"] = post_decision
        output["post_decision"] = post_decision
        output["status"] = "post_decision_checked" if args.check_only else "post_decision_completed"
        out_dir = (
            os.path.join(
                metavideoagent_run_dir(run_id, args.output_root),
                "checks",
                f"iter_{int(getattr(args, 'iteration_index', 0) or 0) or 1}",
                "post_decision",
            )
            if args.check_only
            else metavideoagent_run_dir(run_id, args.output_root)
        )
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
    iteration_index = _resolve_iteration_index(args, reference_report_path)
    args.iteration_index = iteration_index
    os.environ["METAVIDEOAGENT_ITERATION_INDEX"] = str(iteration_index)
    output["reference_label"] = inferred_reference_label
    output["iteration_index"] = iteration_index
    output["reference_report_path"] = reference_report_path
    output["reference_results_path"] = _reference_results_path_from_report(reference_report)
    output["history_input_policy"] = _external_history_meta(
        reference_report_path,
        output["reference_results_path"],
    )
    needs_reference_results = args.stage in (
        "diagnose", "codex_deep_research", "evolve", "full_eval", "review", "all", "automatic"
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
        include_execution_artifacts=False,
        semantic_kwargs=_profile_semantic_kwargs(args),
    )
    # All later stages must consume the exact same distribution-aware profile.
    # Without this assignment Codex/review/full-eval subprocesses silently
    # rebuild separate profiles from the manifest and lose the observed-channel
    # artifact chosen for the current run.
    # ``_load_or_build_profile`` may replace an explicitly supplied previous or
    # mismatched profile with a newly attested run-scoped artifact.  Propagate
    # the effective path unconditionally so later stages cannot reopen the
    # stale caller path.
    if profile.get("output_path", ""):
        args.profile = profile["output_path"]
    deep_research = {}
    # Initial deep research is only for first-bundle design. Later-round
    # diagnosis stays review-focused; diagnosis-conditioned research is run
    # separately immediately before Codex writes code.
    initial_research_stage = (
        args.stage in ("initial", "all", "automatic")
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

    # A diagnosis is a decision over an executed review, never a direct
    # interpretation of raw full-eval trajectories.  The first evolution round
    # has only the initial baseline, so create the explicit single-baseline
    # review here when the convenience ``all`` entrypoint is used.  Staged
    # diagnosis, by contrast, must be given its already completed review: this
    # prevents a caller from silently skipping the documented
    # review -> diagnosis handoff.
    if args.stage == "all" and not args.evolution_review:
        review_dir = os.path.join(
            metavideoagent_run_dir(run_id, args.output_root),
            f"iter_{iteration_index}",
            "review",
        )
        previous_review_out_dir = args.review_out_dir
        try:
            args.review_out_dir = review_dir
            review_exit = _run_review_subprocess(args, reference_report)
        finally:
            args.review_out_dir = previous_review_out_dir
        review_path = _find_review_path(review_dir)
        output["initial_reference_review"] = {
            "exit_code": review_exit,
            "review_path": review_path,
            "purpose": "required_review_before_first_diagnosis",
        }
        if args.check_only:
            # Check-only has no subprocess output to discover.  Keep the
            # planned path in the report without pretending it is consumable.
            args.evolution_review = review_path or os.path.join(
                review_dir, "teacher_evolution_review.json"
            )
        elif review_exit != 0 or not review_path or not review_is_consumable(_read_json(review_path)):
            raise RuntimeError(
                "The required initial-reference review failed or is not consumable; "
                "do not run diagnosis until review is repaired."
            )
        else:
            args.evolution_review = review_path

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
        candidate_rejection_context=_resolve_candidate_rejection_context(args, run_id),
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
        attempt_tag=(
            getattr(args, "attempt_tag", "")
            # Any explicitly resumed evolution needs an isolated output tree.
            # The formal diagnosis_for_codex handoff remains in iter_N/, while
            # context, candidates and stage report are written under this tag.
            if args.stage == "evolve"
            else ""
        ),
    )
    output["iter_context_path"] = iter_context.get("path", "")
    if args.evolution_review:
        output["evolution_review_path"] = os.path.abspath(args.evolution_review)

    if args.stage in ("diagnose", "all"):
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
        # In an explicitly resumed stage, the supplied diagnosis audit is the
        # authoritative lineage record.  Prefer its explicit round directory
        # so a historical run_id (or a relocated runtime root) cannot create a
        # parallel iter_N tree merely to write the Codex handoff.
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

    # A no-model ``all`` preflight intentionally does not materialize a
    # diagnosis or a Codex handoff.  Do not fall through and treat the absent
    # machine-evaluation contract as a real pipeline error; report the planned
    # research/evolve boundary instead.  Real ``all`` execution always writes
    # the diagnosis above and continues through the normal gates.
    if args.stage == "all" and args.check_only:
        planned_round_dir = iter_context.get("iter_dir", "") or metavideoagent_run_dir(
            run_id, args.output_root
        )
        output["codex_deep_research_check"] = {
            "requires": "diagnosis.execution_brief_path from the preceding real diagnosis stage",
            "planned_output_dir": os.path.join(planned_round_dir, "codex_deep_research"),
        }
        output["evolve_check"] = {
            "requires": [
                "validated diagnosis_execution_brief",
                "validated diagnosis.machine_evaluation_contract_path",
                "validated diagnosis-conditioned Codex deep-research handoff",
            ],
            "planned_candidate_root": os.path.join(planned_round_dir, "codex", "codex_candidates"),
            "smoke_then_probe": True,
        }
        output["status"] = "all_checked"
        report_path = _stage_report_path(args, output)
        output["report_path"] = report_path
        _write_json(report_path, output)
        return output

    if args.stage in ("evolve", "all"):
        diagnosis_path = (
            (output.get("diagnosis") or {}).get("report_path", "")
            or args.diagnosis_report
        )
        # Staged evolution resumes an attested diagnosis. Its lineage
        # owns the formal iter_N directory, even when the historical run_id
        # was created under a different output-root layout.
        staged_audit = _read_json(diagnosis_path) if diagnosis_path else {}
        staged_iter_context = staged_audit.get("iter_context") or {}
        round_dir = (
            staged_iter_context.get("iter_dir", "")
            or iter_context.get("iter_dir", "")
            or metavideoagent_run_dir(run_id, args.output_root)
        )
        if args.stage == "evolve":
            # A staged evolution resume follows an already completed explicit
            # codex_deep_research stage. Re-running that stage here would spend
            # Codex quota and could replace the audited research handoff.
            audit = _read_json(diagnosis_path)
            _refresh_diagnosis_probe_context(audit)
            # A staged diagnosis argument may be either a wrapper that points
            # at its compiled brief or the adaptive bundle brief itself. The latter
            # is the persisted handoff copied beside a candidate/probe and is
            # the normal input for same-direction feedback; requiring a redundant
            # self-reference made this formal entrypoint reject a valid brief.
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
            formal_round_dir = round_dir
            diagnosis_for_codex = os.path.join(formal_round_dir, "diagnosis_for_codex.json")
            existing_handoff = _load_validated_diagnosis_for_codex(
                diagnosis_for_codex, execution_brief,
            )
            codex_research = existing_handoff.get("codex_deep_research", {}) or {}
            output["codex_deep_research"] = _research_record(codex_research)
            output["diagnosis_for_codex_path"] = diagnosis_for_codex
        else:
            codex_research = _run_codex_deep_research_context(
                args,
                profile,
                diagnosis_path,
                args.evolution_review,
                round_dir,
            )
            if codex_research:
                _ensure_research_ready(args, codex_research)
                output["codex_deep_research"] = _research_record(codex_research)
            diagnosis_for_codex = _write_diagnosis_for_codex(
                diagnosis_path,
                codex_research,
                round_dir,
                iter_context=iter_context,
            )
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
            iter_context_path=(
                str(audit.get("iter_context_path") or "")
                or iter_context.get("path", "")
            ),
        )
        if not args.check_only:
            codex = _run_bundle_feedback_loop(
                args,
                initial_codex=codex,
                diagnosis_path=diagnosis_for_codex,
                reference_report_path=reference_report_path,
                round_dir=round_dir,
                machine_evaluation_contract_path=machine_contract_path,
                iter_context_path=iter_context.get("path", ""),
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
            # A smoke-only supervised apply is deliberately a checkpoint, not
            # a behavioral judgement.  There is no probe evidence to classify
            # and therefore no valid rejection/repair context to materialize.
            # Leaving this to the previous fallback below fabricated an
            # ``probe_status=unknown`` same-direction repair route.
            if getattr(args, "skip_probe_verification", False):
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
                    "Smoke passed and was retained as an authoritative artifact. "
                    "No candidate rejection context or repair route was created "
                    "because no behavioral probe has run."
                )
                return output
            probe_gate = _probe_gate_from_candidate_run(candidate_run)
            output["probe_gate"] = probe_gate
            if probe_gate.get("action") == "execution_layer_rerun":
                output["status"] = "probe_execution_layer_rerun_required"
                output["note"] = (
                    "Probe observed provider/asset failure. Rerun the same candidate after "
                    "execution-layer repair; do not write rejection context."
                )
                return output
            if probe_gate.get("action") == "round_level_rediagnosis":
                rejection_path = _write_candidate_rejection_context_from_run(
                    args,
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
                    outcome_status="probe_rejected",
                )
                output["candidate_rejection_context"] = rejection_path
                output["evolution_memory_index"] = memory.get("path", "")
            elif probe_gate.get("action") in {
                "codex_engineering_repair",
                "codex_same_direction_behavior_iteration",
            }:
                rejection_path = _write_candidate_rejection_context_from_run(
                    args,
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
    # ``all`` is the one-round convenience route.  Once its candidate passes
    # the real smoke/probe gate, it must execute the same full evaluation and
    # current-best decision as the explicit ``full_eval`` stage.  Previously
    # it stopped at a successful probe, leaving a seemingly complete round
    # without the mandatory initial/candidate full-eval artifact.
    if (
        args.stage == "all"
        and output.get("candidate_run")
        and (output.get("probe_gate") or {}).get("action") == "full_eval_candidate"
    ):
        saved_candidate_run = getattr(args, "candidate_run", "")
        saved_candidate_bundle = getattr(args, "candidate_bundle", "")
        saved_diagnosis_report = getattr(args, "diagnosis_report", "")
        try:
            args.candidate_run = output["candidate_run"]
            args.candidate_bundle = _candidate_artifact_path(args.candidate_run)
            args.diagnosis_report = diagnosis_path
            full_exit = _run_full_eval_subprocess(args, reference_report_path)
        finally:
            args.candidate_run = saved_candidate_run
            args.candidate_bundle = saved_candidate_bundle
            args.diagnosis_report = saved_diagnosis_report
        output["full_eval_exit_code"] = full_exit
        if full_exit != 0:
            output["status"] = "full_eval_failed"
            output["failure_stage"] = "full_eval"
            output["error"] = (
                "Candidate passed smoke/probe but its mandatory full eval failed; "
                "repair or rerun this same candidate before another evolution round."
            )
        else:
            full_report_path = _find_full_eval_report(output["candidate_run"])
            if not full_report_path:
                raise RuntimeError(
                    "full eval exited successfully without a persisted full_eval_report.json"
                )
            full_report = _read_json(full_report_path)
            output["full_eval_report_path"] = full_report_path
            output["post_full_eval_decision"] = full_report.get("post_full_eval_decision") or {}
            memory = _annotate_and_record_candidate_history(
                args,
                iteration=iteration_index,
                parent_report_path=reference_report_path,
                parent_report=reference_report,
                candidate_run=output["candidate_run"],
                candidate_bundle_path=_candidate_artifact_path(output["candidate_run"]),
                diagnosis_path=diagnosis_path,
                research_report_path=(codex_research or {}).get("report_path", ""),
                full_eval_report_path=full_report_path,
            )
            output["evolution_memory_index"] = memory.get("path", "")
    if args.stage == "full_eval":
        full_exit = _run_full_eval_subprocess(
            args, reference_report_path
        )
        output["full_eval_exit_code"] = full_exit
        if not args.check_only and args.candidate_run:
            full_report_path = _find_full_eval_report(args.candidate_run)
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
        elif output["review_exit_code"] != 0 or (review_path and not review_is_consumable(_read_json(review_path))):
            output["status"] = "review_failed"
            output["review_path"] = review_path
    output.setdefault(
        "status",
        f"{args.stage}_{'checked' if args.check_only else 'completed'}",
    )
    report_path = _stage_report_path(args, output)
    output["report_path"] = report_path
    _write_json(report_path, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run staged, from-scratch MetaVideoAgent evolution")
    parser.add_argument("--workspace", default=DEFAULT_METAVIDEOAGENT_WORKSPACE)
    parser.add_argument("--distribution-manifest", required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "initial", "diagnose", "codex_deep_research", "evolve",
            "full_eval", "post_decision", "review", "all", "automatic",
        ),
        default="initial",
    )
    parser.add_argument("--initial-report", default="")
    parser.add_argument("--initial-run-id", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default="",
                        help="Output root; defaults to ./metavideoagent_outputs")
    parser.add_argument("--iteration-index", type=int, default=0,
                        help=(
                            "Explicit evolution iteration for staged recovery runs. "
                            "Required when paths cannot identify the intended iter_N; "
                            "prevents silent writes to iter_1."
                        ))
    parser.add_argument("--reference-label", default="")
    parser.add_argument("--video-id", default="ALL")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--first-n", type=int, default=0)
    parser.add_argument("--exec-llm-model", default=DEFAULT_EXECUTION_LLM_MODEL)
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
    parser.add_argument("--probe-feedback-round", type=int, default=1)
    parser.add_argument("--attempt-tag", default="",
                        help=(
                            "Optional isolated subdirectory for a staged "
                            "--probe-failed-run evolve attempt. It preserves the "
                            "formal iter_N handoff while isolating new context, "
                            "authoring output, and stage report."
                        ))
    parser.add_argument("--candidate-rejection-run", default="",
                        help=(
                            "Runnable candidate/probe run rejected by probe behavior. "
                            "Used by --stage evolve as negative next-round context, "
                            "not as same-candidate repair input."
                        ))
    parser.add_argument("--candidate-rejection-context", default="",
                        help="Prebuilt candidate_rejection_context.json passed to candidate evolution")
    parser.add_argument("--write-candidate-rejection-context", default="",
                        help="Optional output path for context built from --candidate-rejection-run")
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
                        help="Use heuristic post-full-eval decision instead of calling the LLM judge")
    parser.add_argument("--profile", default="")
    parser.add_argument("--refresh-profile", action="store_true",
                        help="Rebuild channel_profile.json even if it already exists")
    parser.add_argument("--semantic-observer-mode", choices=("none", "plan", "observe"), default="observe",
                        help="Five-frame profile observer. observe calls the configured VLM; plan only extracts frames.")
    parser.add_argument("--semantic-observer-dir", default="")
    parser.add_argument("--semantic-video-budget", type=int, default=0)
    parser.add_argument("--semantic-observe-workers", type=int, default=1)
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_EXECUTION_VLM_MODEL,
        help="Execution VLM model; defaults to the paper-reproduction model.",
    )
    parser.add_argument("--asr-model", default="")
    parser.add_argument("--api-key", default="", help="Per-run provider API key; not persisted")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--combo-plan", default="")
    parser.add_argument("--initial-baseline-bundle", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--skip-profile", action="store_true")
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
    parser.add_argument("--auto-rounds", type=int, default=4,
                        help="Number of evolution rounds for --stage automatic.")
    parser.add_argument("--codex-cli", default="",
                        help="Codex executable or a wrapper implementing the same command contract.")
    parser.add_argument("--research-codex-cli", default="",
                        help="Optional compatible executable used for Deep Research.")
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL,
                        help=f"Codex model; reproduction default: {DEFAULT_CODEX_MODEL}.")
    parser.add_argument("--codex-reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        help=f"Codex reasoning effort; reproduction default: {DEFAULT_REASONING_EFFORT}.")
    parser.add_argument("--repair-attempts", type=int, default=3,
                        help="Maximum automatic generate/repair attempts before probe")
    parser.add_argument("--probe-repair-attempts", type=int, default=2,
                        help="Bounded automatic repair attempts for failed probe outcomes")
    parser.add_argument("--probe-feedback-rounds", type=int, default=None,
                        help=(
                            "Preferred name for probe-failed outer feedback rounds. "
                            "Each round feeds the failed candidate and probe audit "
                            "back to the coding-agent adapter for repair or regeneration."
                        ))
    parser.add_argument("--candidate-rounds", type=int, default=0,
                        help=(
                            "Maximum outer candidate+probe rounds. N implies at "
                            "least N-1 probe feedback rounds unless explicitly "
                            "overridden by --probe-feedback-rounds."
                        ))
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
    args = parser.parse_args()
    codex_stages = {"initial", "evolve", "automatic", "all"}
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
    if args.probe_feedback_rounds is not None:
        args.probe_repair_attempts = args.probe_feedback_rounds
    if args.candidate_rounds and args.probe_feedback_rounds is None:
        args.probe_repair_attempts = max(
            int(args.probe_repair_attempts or 0),
            max(0, int(args.candidate_rounds) - 1),
        )

    from run_journal import append_operation
    started_at = time.time()
    try:
        report = run(args)
    except Exception as exc:
        append_operation(output_root=args.output_root, run_id=args.run_id, stage=args.stage,
                         args=args, started_at=started_at, status="failed", error=str(exc))
        raise
    operation_status = "checked" if args.check_only else "completed"
    if int(report.get("automatic_exit_code") or 0) != 0:
        operation_status = str(report.get("automatic_outcome") or "failed")
    append_operation(output_root=args.output_root, run_id=args.run_id, stage=args.stage,
                     args=args, started_at=started_at,
                     status=operation_status, report=report)
    exit_code = 0
    exit_label = ""
    if "full_eval_exit_code" in report and int(report.get("full_eval_exit_code") or 0) != 0:
        exit_code = int(report.get("full_eval_exit_code") or 1)
        exit_label = "full_eval_exit_code"
    elif "evolution_exit_code" in report:
        exit_code = int(report.get("evolution_exit_code") or 0)
        exit_label = "evolution_exit_code"
    elif "automatic_exit_code" in report:
        exit_code = int(report.get("automatic_exit_code") or 0)
        exit_label = "automatic_exit_code"
    print("METAVIDEOAGENT_EVOLUTION_DONE" if exit_code == 0 else "METAVIDEOAGENT_EVOLUTION_STOPPED")
    print(f"report_path={report.get('report_path')}")
    print(f"reference_report_path={report.get('reference_report_path')}")
    print(f"reference_results_path={report.get('reference_results_path')}")
    if exit_label:
        print(f"{exit_label}={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
