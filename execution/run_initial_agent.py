#!/usr/bin/env python3
"""Observation or source-mapping initialization with a common verification flow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
EVOLUTION_DIR = os.path.join(PROJECT_ROOT, "evolution")
METAVIDEOAGENT_RUNTIME_DIR = os.path.join(CURRENT_DIR, "action_runtime")
DEFAULT_METAVIDEOAGENT_WORKSPACE = os.path.join(CURRENT_DIR, "workspace")
METAVIDEOAGENT_PYTHON = os.environ.get("METAVIDEOAGENT_PYTHON") or sys.executable
for path in (PROJECT_ROOT, EVOLUTION_DIR, METAVIDEOAGENT_RUNTIME_DIR, CURRENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("METAVIDEOAGENT_RUNTIME_DIR", METAVIDEOAGENT_RUNTIME_DIR)

from runtime_config import bootstrap_cli_api_env, nonsecret_runtime_audit  # noqa: E402

bootstrap_cli_api_env(sys.argv[1:])

from agent_sources import collect_agent_source, materialize_agent  # noqa: E402
from capability_registry import DEFAULT_PROFILE_IDS, resolve_profile  # noqa: E402
from channel_profile_runner import build_channel_profile  # noqa: E402
from initial_baseline_agent import (  # noqa: E402
    _normalize_codegen_bundle,
    build_initial_baseline_bundle,
    repair_initial_baseline_bundle_with_smoke_feedback,
)
from initial_combo_designer import design_initial_combo  # noqa: E402
from initial_combo_runner import run as run_initial_reference  # noqa: E402
from iteration_workspace import IterationWorkspace  # noqa: E402
from paths import ensure_dir  # noqa: E402
from paths import output_root as metavideoagent_output_root
from paths import run_dir as metavideoagent_run_dir
from runtime_paths import require_metavideoagent_runtime  # noqa: E402

DEFAULT_LLM_MODEL = str(resolve_profile("llm").get("model_id") or "")


def _execution_vlm_concurrency(value: object) -> int:
    """Permit a larger in-flight pool only with an explicit start-rate gate."""
    try:
        starts_per_second = int(os.environ.get("METAVIDEOAGENT_VLM_STARTS_PER_SECOND", "0") or 0)
    except ValueError:
        starts_per_second = 0
    ceiling = 128 if starts_per_second > 0 else 32
    try:
        requested = int(value or 32)
    except (TypeError, ValueError):
        requested = 32
    return max(1, min(ceiling, requested))


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)




def _smoke_pass_record_path(out_dir: str) -> str:
    return os.path.join(out_dir, "initial_bundle_smoke_pass.json")


def _load_verified_smoke_pass_record(out_dir: str, manifest: str) -> dict:
    """Load the executable captured by a successful smoke, not current source."""
    record_path = _smoke_pass_record_path(out_dir)
    if not os.path.isfile(record_path):
        return {}
    try:
        with open(record_path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"initial smoke-pass record is unreadable: {record_path}: {exc}") from exc
    bundle_path = os.path.abspath(str(record.get("effective_bundle_path") or ""))
    if not bundle_path or not os.path.isfile(bundle_path):
        raise RuntimeError("initial smoke-pass record points to a missing effective bundle")
    if os.path.abspath(str(record.get("distribution_manifest") or "")) != os.path.abspath(manifest):
        raise RuntimeError("initial smoke-pass record belongs to a different distribution manifest")
    smoke = _read_json_if_exists(str(record.get("smoke_results_path") or ""))
    if smoke.get("passed") is not True or smoke.get("stage") not in {
        "bundle_smoke_suite", "reassessed_existing_bundle_smoke",
    }:
        raise RuntimeError("initial smoke-pass record does not link to a passed runtime smoke")
    tested_bundle = smoke.get("tested_bundle")
    if not isinstance(tested_bundle, dict) or not tested_bundle.get("modules") or not tested_bundle.get("combo"):
        raise RuntimeError("initial smoke result has no captured executable; rerun codegen_smoke")
    return {**record, "tested_bundle": tested_bundle}


def _materialize_verified_bundle(out_dir: str, tested_bundle: dict) -> str:
    """Give evaluation its own copy of the executable carried by smoke."""
    path = os.path.join(out_dir, "execution_bundle.json")
    _write_json(path, tested_bundle)
    return path


def _adopt_verified_smoke_report(out_dir: str, manifest: str, smoke_report_path: str) -> dict:
    """Persist a full-eval admission record from a separately run real smoke.

    Repair smoke is a standalone entrypoint. This bridge requires its persisted
    suite result, captured executable and distribution manifest. These are
    trusted local run artifacts, not a security boundary against their editor.
    """
    report_path = os.path.abspath(str(smoke_report_path or ""))
    if not report_path or not os.path.isfile(report_path):
        raise RuntimeError("--adopt-smoke-report must name an existing smoke report")
    report = _read_json_if_exists(report_path)
    if report.get("ready") is not True or report.get("status") not in {
        "bundle_smoke_passed", "bundle_smoke_reassessed_passed",
    }:
        raise RuntimeError("adopted smoke report is not a passed real bundle smoke")
    report_manifest = os.path.abspath(str(report.get("distribution_manifest") or ""))
    if report_manifest != os.path.abspath(manifest):
        raise RuntimeError("adopted smoke report belongs to a different distribution manifest")
    smoke_results_path = os.path.abspath(str(report.get("smoke_results_path") or ""))
    smoke_result = _read_json_if_exists(smoke_results_path)
    if smoke_result.get("passed") is not True or smoke_result.get("stage") not in {
        "bundle_smoke_suite", "reassessed_existing_bundle_smoke",
    }:
        raise RuntimeError("adopted smoke report does not link to a passed real smoke result")
    tested_bundle = smoke_result.get("tested_bundle")
    if not isinstance(tested_bundle, dict) or not tested_bundle.get("modules") or not tested_bundle.get("combo"):
        raise RuntimeError("adopted smoke result has no captured executable; rerun smoke")
    effective_bundle = os.path.abspath(str(report.get("effective_initial_baseline_bundle") or ""))
    if not effective_bundle or not os.path.isfile(effective_bundle):
        raise RuntimeError("adopted smoke report has no effective bundle artifact")
    record = {
        "schema_version": 1,
        "run_id": str(report.get("run_id") or ""),
        "distribution_manifest": os.path.abspath(manifest),
        "effective_bundle_path": effective_bundle,
        "smoke_report_path": report_path,
        "smoke_results_path": smoke_results_path,
        "admission_source": "verified_standalone_repair_smoke",
        "created_at": int(time.time()),
    }
    _write_json(_smoke_pass_record_path(out_dir), record)
    return record


def _read_text_or_literal(value: str) -> str:
    if not value:
        return ""
    if "\n" in value or "\r" in value:
        return value
    stripped = value.lstrip()
    if stripped.startswith("#") or stripped.startswith("- ") or stripped.startswith("* "):
        return value
    path = os.path.abspath(value)
    if os.path.exists(path) and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    looks_like_path = (
        os.path.sep in value
        or value.endswith((".md", ".txt", ".json", ".jsonl"))
    )
    if looks_like_path:
        raise FileNotFoundError(f"Text input path does not exist: {value}")
    return value


def _set_env_if_value(key: str, value: str) -> None:
    if value:
        os.environ[key] = value


def _apply_api_env(args: argparse.Namespace) -> None:
    """Apply per-run API configuration without touching user shell config."""
    _set_env_if_value("METAVIDEOAGENT_API_KEY", getattr(args, "api_key", ""))
    _set_env_if_value("METAVIDEOAGENT_BASE_URL", getattr(args, "base_url", ""))


def _default_run_dir(run_id: str, output_root: str = "", *, check_only: bool = False) -> str:
    return os.path.join(metavideoagent_run_dir(run_id, output_root),
                        *(("checks",) if check_only else ()), "iter_0")


def _read_json_if_exists(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _validate_codegen_research_handoff(report: dict,
                                       summary_text: str,
                                       *, require_formal: bool = True) -> str:
    """Carry completed research findings into initial implementation."""
    if require_formal and (not report.get("formal_research") or not report.get("research_complete")):
        raise RuntimeError("initial codegen requires completed source-grounded research")
    brief = report.get("design_brief") or _read_json_if_exists(str(report.get("design_brief_path") or ""))
    report_summary = str(report.get("summary_text") or "")
    if brief:
        return ("## Complete research findings and implementation handoff\n"
                + json.dumps(brief, ensure_ascii=False, indent=2))
    return report_summary or summary_text


def _failed_smoke_artifact(attempt_dir: str) -> tuple[str, dict]:
    """Return the persisted failure object produced by the real smoke runner."""
    for name in (
        "bundle_smoke_result.json",
        "initial_bundle_static_failure.json",
        "initial_baseline_report.json",
    ):
        path = os.path.join(attempt_dir, name)
        payload = _read_json_if_exists(path)
        if payload and payload.get("passed") is not True:
            return path, payload
    return "", {}


def run(args: argparse.Namespace) -> dict:
    runtime_dir = require_metavideoagent_runtime("run_initial_agent")
    args.output_root = metavideoagent_output_root(getattr(args, "output_root", ""))
    os.environ["METAVIDEOAGENT_OUTPUT_ROOT"] = args.output_root
    workspace = os.path.abspath(args.workspace)
    manifest = os.path.abspath(args.distribution_manifest) if args.distribution_manifest else ""
    run_id = args.run_id or f"metavideoagent_initial_{int(time.time())}"
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else _default_run_dir(
        run_id, args.output_root, check_only=args.check_only
    )
    ensure_dir(out_dir)
    iteration = IterationWorkspace(out_dir)
    restart = getattr(args, "restart_from", "")
    if args.stage == "codegen_smoke" and restart in {"codegen", "smoke"}:
        iteration.restart(restart)
    saved_stages = iteration.progress().get("stages") or {}
    saved_pipeline = _read_json_if_exists(os.path.join(out_dir, "metavideoagent_initial_pipeline.json"))
    if not args.check_only and args.stage == "codegen_smoke" and (saved_stages.get("smoke") or {}).get("status") == "complete":
        smoke_record = _load_verified_smoke_pass_record(out_dir, manifest)
        if smoke_record:
            return {
                **saved_pipeline,
                "status": "codegen_smoke_passed",
                "effective_initial_baseline_bundle_path": smoke_record["effective_bundle_path"],
            }
    if not args.check_only and args.stage == "run" and (saved_stages.get("full_eval") or {}).get("status") == "complete":
        if saved_pipeline.get("status") == "completed":
            _load_verified_smoke_pass_record(out_dir, manifest)
            return saved_pipeline
    initialization = getattr(args, "initialization", "observe")
    mapping = initialization == "map"
    source_agent = getattr(args, "source_agent", "")
    has_draft = (iteration.root / "agent.json").exists() or (iteration.root / "agent").exists()
    continue_draft = (
        args.stage == "codegen_smoke" and has_draft and restart != "codegen" and not args.check_only
        and ((saved_stages.get("codegen") or {}).get("status") != "failed" or restart == "smoke")
    )
    if mapping and not source_agent and not has_draft and not args.initial_baseline_bundle and args.stage != "run":
        raise ValueError("--initialization map requires --source-agent")
    _apply_api_env(args)
    profile_path = args.profile or os.path.join(out_dir, "initialization", "profile.json")
    plan_path = args.combo_plan or os.path.join(out_dir, "initialization", "plan.json")
    supplied_bundle_path = bool(args.initial_baseline_bundle)
    bundle_path = args.initial_baseline_bundle or os.path.join(out_dir, "initial_baseline_bundle.json")
    smoke_pass_record = {}
    evaluation_dir = ""
    if args.stage == "run":
        if args.adopt_smoke_report:
            _adopt_verified_smoke_report(out_dir, manifest, args.adopt_smoke_report)
        smoke_pass_record = _load_verified_smoke_pass_record(out_dir, manifest)
        if not smoke_pass_record:
            raise RuntimeError(
                "formal initial stage=run requires initial_bundle_smoke_pass.json from stage=codegen_smoke"
            )
        bundle_path = str(smoke_pass_record["effective_bundle_path"])
        if supplied_bundle_path and os.path.abspath(args.initial_baseline_bundle) != os.path.abspath(bundle_path):
            raise RuntimeError(
                "stage=run was given an initial bundle different from the persisted smoke-pass bundle; "
                "refusing to evaluate a stale or unverified initial candidate"
            )
        if not args.check_only:
            evaluation_dir = str(iteration.reserve("full_eval"))
            bundle_path = _materialize_verified_bundle(evaluation_dir, smoke_pass_record["tested_bundle"])
            iteration.record("full_eval", os.path.join(evaluation_dir, "initial_baseline_report.json"), status="running")
    bundle_prompt_path = os.path.join(out_dir, "initial_baseline_prompt.txt")
    deep_research_summary = _read_text_or_literal(args.deep_research_summary)
    deep_research_report = _read_json_if_exists(getattr(args, "deep_research_report", ""))
    deep_research_brief = deep_research_report.get("design_brief") or {}
    if not deep_research_brief:
        deep_research_brief = _read_json_if_exists(
            str(deep_research_report.get("design_brief_path") or "")
        )
    if (not mapping and not supplied_bundle_path and not continue_draft and not args.check_only) and args.stage in {"plan", "codegen_smoke"}:
        if not deep_research_report or not deep_research_report.get("formal_research"):
            raise RuntimeError(
                f"formal initial stage={args.stage} requires a formal --deep-research-report; "
                "supply or automatically generate a validated research report before authoring"
            )

    profile = {}
    if mapping or (supplied_bundle_path and args.stage == "codegen_smoke"):
        profile = {}
    elif args.stage == "run":
        if os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
    elif continue_draft:
        profile = _read_json_if_exists(profile_path)
    else:
        profile = build_channel_profile(
            workspace,
            output_path=profile_path,
            distribution_manifest=manifest,
            semantic_observer_mode=(
                "plan"
                if args.check_only and args.semantic_observer_mode == "observe"
                else args.semantic_observer_mode
            ),
            semantic_observer_dir=args.semantic_observer_dir,
            semantic_video_budget=args.semantic_video_budget,
            semantic_observe_workers=args.semantic_observe_workers,
            runtime_dir=METAVIDEOAGENT_RUNTIME_DIR,
        )

    plan = {}
    bundle = {}
    if args.stage == "profile":
        plan_path = ""
    elif args.stage == "plan" and mapping:
        from existing_agent_initializer import write_mapping_plan
        plan = write_mapping_plan(source_agent=source_agent, output_dir=out_dir, output_path=plan_path)
    elif args.stage == "plan":
        plan = design_initial_combo(
            profile,
            manifest_path=manifest,
            output_path=plan_path,
            deep_research_brief=deep_research_brief,
        )
    elif args.stage == "codegen_smoke":
        if os.path.isfile(plan_path):
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
        elif not mapping and not supplied_bundle_path and not continue_draft:
            plan = design_initial_combo(
                profile,
                manifest_path=manifest,
                output_path=plan_path,
                deep_research_brief=deep_research_brief,
            )
        if not mapping and not supplied_bundle_path and not continue_draft:
            deep_research_summary = _validate_codegen_research_handoff(
                deep_research_report, deep_research_summary,
                require_formal=not args.check_only,
            )
    elif args.stage == "run":
        if args.combo_plan and os.path.exists(args.combo_plan):
            with open(args.combo_plan, "r", encoding="utf-8") as f:
                plan = json.load(f)
        elif os.path.exists(plan_path):
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
        if not os.path.exists(bundle_path):
            raise RuntimeError(
                "stage=run requires an existing --initial-baseline-bundle "
                "or a bundle generated by stage=codegen_smoke; refusing to regenerate it."
            )
        with open(bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

    if args.stage == "codegen_smoke" and supplied_bundle_path:
        bundle = _read_json_if_exists(bundle_path)
        if not bundle and not has_draft:
            raise ValueError(f"Initial bundle could not be loaded: {bundle_path}")
        if not has_draft:
            materialize_agent(bundle, iteration.root)
        bundle.setdefault("codegen", {})["iteration_dir"] = str(iteration.root)
    elif continue_draft:
        bundle = _read_json_if_exists(bundle_path)
    elif args.stage == "codegen_smoke" and mapping:
        from existing_agent_initializer import build_mapped_initial_bundle
        bundle, bundle_prompt = build_mapped_initial_bundle(
            source_agent=source_agent, workspace=workspace, distribution_manifest=manifest,
            output_dir=out_dir, codex_cli=getattr(args, "codex_cli", ""),
            timeout=getattr(args, "codegen_timeout", 900), execute_codegen=not args.check_only,
        )
        bundle_path = os.path.join(out_dir, "initial_baseline_bundle.json") if bundle else ""
        with open(bundle_prompt_path, "w", encoding="utf-8") as handle:
            handle.write(bundle_prompt)
    elif args.stage == "codegen_smoke":
        bundle, bundle_prompt = build_initial_baseline_bundle(
            workspace=workspace,
            distribution_manifest=manifest,
            profile=profile,
            deep_research_summary=deep_research_summary,
            codex_cli=getattr(args, "codex_cli", ""),
            codegen_timeout=getattr(args, "codegen_timeout", 900),
            codegen_output_dir=out_dir,
            execute_codegen=not args.check_only,
        )
        bundle_path = (
            os.path.join(out_dir, "initial_baseline_bundle.json") if bundle else ""
        )
        if bundle:
            _write_json(bundle_path, bundle)
        with open(bundle_prompt_path, "w", encoding="utf-8") as handle:
            handle.write(bundle_prompt)
        if bundle:
            bundle["prompt_path"] = os.path.abspath(bundle_prompt_path)
            _write_json(bundle_path, bundle)

    if args.stage == "codegen_smoke" and not args.check_only and (continue_draft or supplied_bundle_path):
        metadata = bundle
        codegen = {**metadata.get("codegen", {}), "iteration_dir": str(iteration.root)}
        try:
            draft = collect_agent_source(iteration.root)
            codegen["assembly_error"] = ""
        except RuntimeError as exc:
            draft = {}
            codegen["assembly_error"] = str(exc)
        bundle = _normalize_codegen_bundle(
            draft, workspace=workspace, distribution_manifest=manifest,
            generation_mode=metadata.get("generation_mode") or ("existing_agent_mapping" if mapping else "coding_agent"),
            codegen_result=codegen,
        )
        bundle["initialization"] = initialization
        bundle["codegen"]["context_path"] = str(iteration.root / "initialization/context.json")
        if mapping:
            bundle["runtime_policy"] = {"capability_selection": "source_agent_registered_profiles"}

    report_combo = bundle.get("combo", {})
    report_combo_source = "initial_baseline_bundle" if report_combo else ""
    pipeline_report = {
        "run_id": run_id,
        "workspace": workspace,
        "runtime_dir": runtime_dir,
        "output_root": metavideoagent_output_root(getattr(args, "output_root", "")),
        "distribution_manifest": manifest,
        "profile_path": profile_path if os.path.isfile(profile_path) else "",
        "combo_plan_path": plan_path,
        "initial_baseline_bundle_path": bundle_path if bundle else "",
        "initial_smoke_pass_record": _smoke_pass_record_path(out_dir) if smoke_pass_record else "",
        "initial_baseline_prompt_path": (
            bundle.get("prompt_path") or bundle_prompt_path if bundle else ""
        ),
        "initial_baseline_bundle_validation": bundle.get("validation", {}) if bundle else {},
        "combo": report_combo,
        "combo_source_for_report": report_combo_source,
        "stage": args.stage,
        "initialization": initialization,
        "authoring_engine": "codex",
        "provider_configuration": {
            **nonsecret_runtime_audit(),
            "source": "capability_profiles_with_optional_cli_overrides",
        },
        "runtime_models": {
            "vlm_profile_id": DEFAULT_PROFILE_IDS["vlm"],
            "llm_profile_id": DEFAULT_PROFILE_IDS["llm"],
            "llm_model": DEFAULT_LLM_MODEL,
            "asr_profile_id": DEFAULT_PROFILE_IDS["asr"],
            "embedding_profile_id": DEFAULT_PROFILE_IDS["embedding"],
        },
        "execution_capability_limits": {
            "vlm_max_concurrency": _execution_vlm_concurrency(
                getattr(args, "vlm_max_concurrency", 32)
            ),
            "llm_max_concurrency": max(
                1, min(16, int(getattr(args, "llm_max_concurrency", 8) or 8))
            ),
            "asr_max_concurrency": 16,
            "embedding_max_concurrency": 16,
        },
        "provider_start_rate_limits": {
            "vlm_starts_per_second": max(
                0, int(os.environ.get("METAVIDEOAGENT_VLM_STARTS_PER_SECOND", "0") or 0)
            ),
            "asr_starts_per_minute": max(
                0, int(os.environ.get("METAVIDEOAGENT_ASR_STARTS_PER_MINUTE", "0") or 0)
            ),
        },
        "leakage_policy": (
            bundle.get("leakage_policy")
            or plan.get("leakage_policy", "")
        ),
    }
    pipeline_report_path = os.path.join(out_dir, "metavideoagent_initial_pipeline.json")
    pipeline_report["pipeline_report_path"] = pipeline_report_path

    if args.stage == "profile":
        pipeline_report["status"] = "profiled"
        pipeline_report["next_step"] = "Run stage=plan to produce initialization/plan.json."
        _write_json(pipeline_report_path, pipeline_report)
        return pipeline_report

    if args.stage == "plan":
        pipeline_report["status"] = "planned"
        pipeline_report["next_step"] = (
            "Run stage=codegen_smoke to generate the initial bundle and execute "
            "the automatic authoring-to-smoke repair loop."
        )
        _write_json(pipeline_report_path, pipeline_report)
        return pipeline_report

    codegen_smoke_stage = args.stage == "codegen_smoke"
    smoke_loop = []
    if codegen_smoke_stage:
        pipeline_report["codegen_smoke_loop"] = {
            "enabled": True,
            "max_automatic_repairs": max(
                0, int(getattr(args, "bundle_smoke_repair_attempts", 0) or 0)
            ),
            "scope": "real_runtime_preflight_then_deterministic_single_case_smoke",
            "full_eval": False,
        }

    def runner_args_for(bundle_for_run: str, output_for_run: str, *, smoke_only: bool,
                        check_only: bool = False) -> argparse.Namespace:
        return argparse.Namespace(
            workspace=workspace,
            combo_plan=plan_path,
            initial_baseline_bundle=bundle_for_run,
            distribution_manifest=manifest,
            video_id=args.video_id,
            question_split=args.question_split,
            concurrency=1 if smoke_only else args.concurrency,
            vlm_max_concurrency=_execution_vlm_concurrency(
                getattr(args, "vlm_max_concurrency", 32)
            ),
            llm_max_concurrency=max(1, min(16, int(getattr(args, "llm_max_concurrency", 8) or 8))),
            provider_timeout_sec=max(1, int(getattr(args, "provider_timeout_sec", 180) or 180)),
            first_n=0 if smoke_only else args.first_n,
            rebuild_structure=True if (smoke_only or args.stage == "run") else args.rebuild_structure,
            structure_workers=1 if smoke_only else args.structure_workers,
            combo_id=args.combo_id,
            run_id=run_id,
            output_root=output_for_run,
            check_only=check_only,
            skip_bundle_smoke_test=(False if smoke_only else args.stage == "run"),
            smoke_only=smoke_only,
            smoke_time_reference_only=True if smoke_only else False,
        )

    if codegen_smoke_stage and not args.check_only:
        current_bundle_path = os.path.abspath(bundle_path)
        max_repairs = max(
            0, int(getattr(args, "bundle_smoke_repair_attempts", 0) or 0)
        )
        initial_report = {}
        for smoke_attempt in range(max_repairs + 1):
            attempt_index = smoke_attempt
            attempt_dir = str(iteration.reserve("smoke"))
            current_bundle_path = os.path.join(attempt_dir, "execution_bundle.json")
            _write_json(current_bundle_path, bundle)
            try:
                initial_report = run_initial_reference(
                    runner_args_for(current_bundle_path, attempt_dir, smoke_only=True)
                )
                smoke_loop.append({
                    "attempt": attempt_index,
                    "bundle_path": current_bundle_path,
                    "smoke_dir": attempt_dir,
                    "status": "passed",
                    "smoke_report_path": initial_report.get("report_path", ""),
                })
                break
            except RuntimeError as exc:
                smoke_result_path, smoke_failure = _failed_smoke_artifact(attempt_dir)
                bundle_snapshot = os.path.join(attempt_dir, "initial_baseline_bundle.json")
                attempt_record = {
                    "attempt": attempt_index,
                    "bundle_path": current_bundle_path,
                    "bundle_snapshot": bundle_snapshot,
                    "smoke_dir": attempt_dir,
                    "smoke_result_path": smoke_result_path,
                    "status": "failed",
                    "error": str(exc),
                }
                if not smoke_result_path or not smoke_failure:
                    smoke_loop.append(attempt_record)
                    raise RuntimeError(
                        "initial smoke failed without a persisted failure artifact; "
                        f"attempt={attempt_index}, error={exc}"
                    ) from exc
                if (smoke_failure.get("failure_kind") == "execution_layer"
                        or smoke_failure.get("execution_layer_rerun_required")
                        or smoke_failure.get("repair_route") == "rerun_same_bundle_after_execution_layer_fix"):
                    attempt_record["next_action"] = "execution_layer_rerun"
                    smoke_loop.append(attempt_record)
                    _write_json(os.path.join(out_dir, "initial_codegen_smoke_loop.json"), {
                        "schema_version": 1, "passed": False,
                        "effective_bundle_path": current_bundle_path, "attempts": smoke_loop,
                    })
                    raise RuntimeError(
                        f"Initial smoke requires an execution-layer fix; retain {current_bundle_path} "
                        f"and rerun it after resolving the failure recorded in {smoke_result_path}."
                    ) from exc
                if smoke_attempt >= max_repairs:
                    attempt_record["next_action"] = "repair_budget_exhausted"
                    smoke_loop.append(attempt_record)
                    _write_json(os.path.join(out_dir, "initial_codegen_smoke_loop.json"), {
                        "schema_version": 1, "passed": False, "attempts": smoke_loop,
                    })
                    raise RuntimeError(
                        f"initial bundle smoke failed after {smoke_attempt + 1} attempt(s): {exc}"
                    ) from exc
                repair_dir = attempt_dir
                repaired, repair = repair_initial_baseline_bundle_with_smoke_feedback(
                    _read_json_if_exists(
                        bundle_snapshot if os.path.isfile(bundle_snapshot) else current_bundle_path
                    ),
                    smoke_failure,
                    output_dir=repair_dir,
                    iteration_dir=str(iteration.root),
                    attempt=attempt_index + 1,
                    codex_cli=getattr(args, "codex_cli", ""),
                    timeout=getattr(args, "codegen_timeout", 900),
                )
                attempt_record["repair"] = repair
                smoke_loop.append(attempt_record)
                if (repair.get("codegen") or {}).get("returncode") != 0:
                    raise RuntimeError(
                        "initial bundle repair command failed; inspect its execution artifacts: "
                        + str(repair.get("repaired_bundle_path") or repair_dir)
                    )
                current_bundle_path = os.path.abspath(
                    str(repair.get("repaired_bundle_path") or "")
                )
                if not current_bundle_path or not os.path.isfile(current_bundle_path):
                    raise RuntimeError("automatic repair succeeded without a bundle artifact")
                bundle = repaired
        else:
            raise RuntimeError("initial codegen smoke loop ended without a smoke result")
        if not initial_report.get("ready"):
            raise RuntimeError("initial codegen smoke loop completed without a smoke-pass report")
        bundle_path = current_bundle_path
        bundle = _read_json_if_exists(bundle_path)
        pipeline_report["codegen_smoke_attempts"] = smoke_loop
        _write_json(os.path.join(out_dir, "initial_codegen_smoke_loop.json"), {
            "schema_version": 1,
            "passed": True,
            "effective_bundle_path": bundle_path,
            "attempts": smoke_loop,
        })
        iteration.record("smoke", os.path.join(out_dir, "initial_codegen_smoke_loop.json"), status="complete")
    elif args.stage == "run" and not args.check_only:
        evaluation_args = runner_args_for(bundle_path, evaluation_dir, smoke_only=False)
        initial_report = run_initial_reference(evaluation_args)
    else:
        initial_report = run_initial_reference(
            runner_args_for(bundle_path, out_dir, smoke_only=bool(codegen_smoke_stage), check_only=args.check_only)
        )
    if args.check_only:
        pipeline_report["status"] = "checked"
    elif codegen_smoke_stage:
        pipeline_report["status"] = "codegen_smoke_passed"
        pipeline_report["next_step"] = (
            "The generated initial baseline bundle passed real-runtime smoke. "
            "Run stage=run with the effective bundle path for full initial reference evaluation."
        )
        effective_bundle = str(
            initial_report.get("effective_initial_baseline_bundle")
            or initial_report.get("initial_baseline_bundle")
            or ""
        )
        if not effective_bundle or not os.path.isfile(effective_bundle):
            raise RuntimeError("codegen smoke passed without an effective bundle artifact")
        smoke_pass_record = {
            "schema_version": 1,
            "run_id": run_id,
            "distribution_manifest": manifest,
            "effective_bundle_path": os.path.abspath(effective_bundle),
            "smoke_report_path": initial_report.get("report_path", ""),
            "smoke_results_path": initial_report.get("smoke_results_path", ""),
            "created_at": int(time.time()),
        }
        _write_json(_smoke_pass_record_path(out_dir), smoke_pass_record)
        _load_verified_smoke_pass_record(out_dir, manifest)
        pipeline_report["initial_smoke_pass_record"] = _smoke_pass_record_path(out_dir)
    else:
        pipeline_report["status"] = "completed" if initial_report.get("ready") else "full_eval_incomplete"
    pipeline_report["initial_reference_report"] = initial_report
    pipeline_report["effective_initial_baseline_bundle_path"] = (
        initial_report.get("effective_initial_baseline_bundle")
        or initial_report.get("initial_baseline_bundle")
        or bundle_path
    )
    if codegen_smoke_stage:
        pipeline_report["initial_baseline_bundle_path"] = bundle_path
        pipeline_report["initial_baseline_bundle_validation"] = bundle.get("validation", {}) if bundle else {}
        pipeline_report["combo"] = bundle.get("combo", {}) if bundle else pipeline_report.get("combo", {})
        pipeline_report["combo_source_for_report"] = "initial_baseline_bundle"
    _write_json(pipeline_report_path, pipeline_report)
    if args.stage == "run" and initial_report.get("ready"):
        iteration.record("full_eval", initial_report.get("report_path") or pipeline_report_path, status="complete")
        iteration.finish("complete")
    elif args.stage == "run" and not args.check_only:
        iteration.record("full_eval", initial_report.get("report_path") or pipeline_report_path, status="pending",
                         next_task={"stage": "full_eval", "task": "Resolve the recorded execution failures and complete initial train evaluation."})
        raise RuntimeError(f"Initial train evaluation is incomplete; inspect {initial_report.get('report_path') or pipeline_report_path}")
    return pipeline_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize and validate a MetaVideoAgent")
    parser.add_argument("--workspace", default=DEFAULT_METAVIDEOAGENT_WORKSPACE)
    parser.add_argument("--distribution-manifest", default="", required=True)
    parser.add_argument("--stage", choices=("profile", "plan", "codegen_smoke", "run"), default="plan")
    parser.add_argument("--profile", default="")
    parser.add_argument("--combo-plan", default="")
    parser.add_argument("--initial-baseline-bundle", default="")
    parser.add_argument("--initialization", choices=("observe", "map"), default="observe")
    parser.add_argument("--source-agent", default="", help="Curated source package for mapping initialization.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--output-root", default="",
                        help="Output root; defaults to ./metavideoagent_outputs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--video-id", default="ALL")
    parser.add_argument("--question-split", default="train")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--vlm-max-concurrency", type=int, default=32,
                        help="VLM in-flight ceiling (1-32 normally; up to 128 only with METAVIDEOAGENT_VLM_STARTS_PER_SECOND).")
    parser.add_argument("--llm-max-concurrency", type=int, default=8,
                        help="Text-LLM request concurrency ceiling for formal runtime stages (1-16).")
    parser.add_argument("--provider-timeout-sec", type=int, default=180,
                        help="Provider request deadline for smoke and full evaluation, in seconds (default: 180).")
    parser.add_argument("--first-n", type=int, default=0)
    parser.add_argument("--rebuild-structure", action="store_true")
    parser.add_argument("--structure-workers", type=int, default=4)
    parser.add_argument("--combo-id", default="")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--semantic-observer-mode", choices=("none", "plan", "observe"), default="observe",
                        help="Five-frame initial profile observer. observe calls the configured VLM; plan only extracts frames.")
    parser.add_argument("--semantic-observer-dir", default="")
    parser.add_argument("--semantic-video-budget", type=int, default=0)
    parser.add_argument("--semantic-observe-workers", type=int, default=1)
    parser.add_argument("--api-key", default="", help="Per-run provider API key; not persisted")
    parser.add_argument("--base-url", default="", help="Provider API base URL")
    parser.add_argument("--deep-research-summary", default="")
    parser.add_argument("--deep-research-report", default="",
                        help="Validated initial deep-research report consumed by stage=plan")
    parser.add_argument("--codex-cli", default="",
                        help="Codex executable or a wrapper implementing the same command contract.")
    parser.add_argument("--codegen-timeout", type=int, default=900)
    parser.add_argument("--bundle-smoke-repair-attempts", type=int, default=2)
    parser.add_argument("--restart-from", choices=("codegen", "smoke"), default="",
                        help="Repeat this initialization stage while retaining the current Agent draft.")
    parser.add_argument("--adopt-smoke-report", default="",
                        help="Verify and adopt a separately run real repair-smoke report before stage=run")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from run_journal import append_operation
    started_at = time.time()
    try:
        report = run(args)
    except Exception as exc:
        append_operation(output_root=args.output_root, run_id=args.run_id, stage=f"initial:{args.stage}",
                         args=args, started_at=started_at, status="failed", error=str(exc))
        raise
    append_operation(output_root=args.output_root, run_id=args.run_id, stage=f"initial:{args.stage}",
                     args=args, started_at=started_at,
                     status="checked" if args.check_only else "completed", report=report)
    print("METAVIDEOAGENT_INITIAL_PIPELINE")
    print(f"status={report.get('status')}")
    print(f"profile_path={report.get('profile_path')}")
    print(f"combo_plan_path={report.get('combo_plan_path')}")
    print(f"initial_baseline_bundle_path={report.get('initial_baseline_bundle_path')}")
    if report.get("effective_initial_baseline_bundle_path"):
        print(f"effective_initial_baseline_bundle_path={report.get('effective_initial_baseline_bundle_path')}")
    if report.get("initial_reference_report"):
        initial = report["initial_reference_report"]
        print(f"initial_report_path={initial.get('report_path')}")
        print(f"initial_results_path={initial.get('results_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
