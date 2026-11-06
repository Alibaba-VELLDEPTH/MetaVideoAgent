"""Execute an initial bundle and retain trajectories for Teacher and Diagnosis."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, Iterable, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
from runtime_config import bootstrap_cli_api_env

bootstrap_cli_api_env(sys.argv[1:])
from runtime_paths import (
    default_workspace,
    runtime_id,
    runtime_output_root,
    use_runtime_path,
)

RUNTIME_DIR = use_runtime_path()
from capability_registry import DEFAULT_PROFILE_IDS, resolve_profile  # noqa: E402
from combo_contract import require_complete_agent_combo  # noqa: E402

DEFAULT_LLM_MODEL = str(resolve_profile("llm").get("model_id") or "")

from question_set_loader import (  # noqa: E402
    load_questions as load_question_set,
)
from question_set_loader import (
    select_questions_streaming,
)
from sandbox_evaluator import (
    _inspect_smoke_structure_quality,
    build_reference_lookup,
    compare_with_reference,
    detect_runtime_engineering_issues,
    evaluate_combo_multi_video,
    inject_module_bundle,
    prebuild_isolated_video_structures,
    preflight_bundle_runtime,
    single_question_bundle_smoke_test,
)


def _execution_vlm_concurrency(value: object) -> int:
    """Keep normal runs at 32; start-rate-gated full eval may retain 128 in flight."""
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


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _structure_artifact(structure_dir: str, combo: Dict, video_ids: List[str],
                        rebuilt: bool) -> Dict:
    """Persist explicit, run-scoped structure provenance for later reuse."""
    materialized_dir = os.path.abspath(structure_dir) if structure_dir else ""
    files = []
    if materialized_dir and os.path.isdir(materialized_dir):
        for name in sorted(os.listdir(materialized_dir)):
            path = os.path.join(materialized_dir, name)
            if not name.endswith(".jsonl") or not os.path.isfile(path):
                continue
            files.append({
                "name": name,
                "path": path,
                "size": os.path.getsize(path),
            })
    return {
        "artifact_type": "metavideoagent_structure_artifact",
        "schema_version": 1,
        "source": "rebuilt_isolated_initial_bundle" if rebuilt else "explicit_reuse",
        "materialized_dir": materialized_dir,
        "source_dir": materialized_dir,
        "video_structure_files": files,
        "combo_video_structuring": str((combo or {}).get("video_structuring") or ""),
        "video_ids": list(video_ids or []),
        "rebuild_structure": bool(rebuilt),
    }


def _paths(run_id: str, output_root: str = "") -> Dict[str, str]:
    run_dir = (
        os.path.abspath(output_root)
        if output_root else os.path.join(runtime_output_root(), "initial_combos", run_id)
    )
    return {
        "run_dir": run_dir,
        "sandbox_dir": os.path.join(run_dir, "sandbox"),
        "results_path": os.path.join(run_dir, "initial_baseline_results.jsonl"),
        "report_path": os.path.join(run_dir, "initial_baseline_report.json"),
        "plan_snapshot": os.path.join(run_dir, "initial_combo_plan.json"),
        "bundle_snapshot": os.path.join(run_dir, "initial_baseline_bundle.json"),
        "summary_path": os.path.join(run_dir, "initial_baseline_summary.md"),
    }


def _dedupe_results(results: List[dict], questions: List[dict]) -> List[dict]:
    by_task = {row.get("task_id", ""): row for row in results if row.get("task_id")}
    ordered = []
    for question in questions:
        task_id = question.get("task_id", "")
        if task_id in by_task:
            ordered.append(by_task[task_id])
    seen = {row.get("task_id", "") for row in ordered}
    ordered.extend(row for row in results if row.get("task_id", "") not in seen)
    return ordered


def _time_reference_bounds(question: Dict) -> tuple:
    time_reference = question.get("time_reference") or []
    starts = []
    ends = []
    if isinstance(time_reference, (int, float)):
        starts.append(float(time_reference))
        ends.append(float(time_reference))
    elif isinstance(time_reference, str):
        import re
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", time_reference)]
        if nums:
            starts.append(nums[0])
            ends.append(nums[-1])
    elif isinstance(time_reference, list):
        for item in time_reference:
            if isinstance(item, (int, float)):
                starts.append(float(item))
                ends.append(float(item))
            elif isinstance(item, list) and item:
                starts.append(float(item[0]))
                ends.append(float(item[-1]))
    if not starts or not ends:
        return 1e12, 1e12
    return min(starts), max(ends)


def _smoke_question_key(question: Dict) -> tuple:
    """Prefer low-cost questions for engineering smoke, not hard/evidence-long ones."""
    start, end = _time_reference_bounds(question)
    duration = max(0.0, end - start) if end < 1e12 and start < 1e12 else 1e12
    return (end, duration, start)


def _smoke_case_limit() -> int:
    try:
        return max(1, int(os.environ.get("INITIAL_BUNDLE_SMOKE_CASES", "1") or 1))
    except ValueError:
        return 1


def _smoke_suite_questions(questions: List[dict], *, require_time_reference: bool = False) -> List[dict]:
    """Choose a deterministic no-label engineering suite across videos.

    Smoke is intentionally one cheap, deterministic engineering case. It is
    stable for every repair attempt and never uses labels or correctness.
    """
    limit = _smoke_case_limit()
    candidates = list(questions or [])
    if require_time_reference:
        candidates = [
            item for item in candidates
            if (lambda bounds: bounds[0] < bounds[1] < 1e12)(_time_reference_bounds(item))
        ]
        if not candidates:
            raise RuntimeError(
                "strict time-reference smoke requires at least one training task with a valid time_reference interval"
            )
    ordered = sorted(candidates, key=lambda item: (_smoke_question_key(item), str(item.get("task_id") or "")))
    selected, videos = [], set()
    for question in ordered:
        video_id = str(question.get("video_id") or "")
        if video_id and video_id in videos and len(videos) < min(limit, len({str(q.get("video_id") or "") for q in ordered})):
            continue
        selected.append(question)
        videos.add(video_id)
        if len(selected) >= limit:
            return selected
    for question in ordered:
        if question in selected:
            continue
        selected.append(question)
        if len(selected) >= limit:
            break
    return selected


def _set_env_temporarily(updates: Dict[str, str]) -> Dict[str, str]:
    old_values = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        os.environ[key] = value
    return old_values


def _restore_env(old_values: Dict[str, str]) -> None:
    for key, value in old_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _reassess_existing_bundle_smoke(smoke_result_path: str) -> Dict:
    """Recheck captured structure evidence while retaining each case's runtime outcomes."""
    source_path = os.path.abspath(smoke_result_path or "")
    if not source_path or not os.path.isfile(source_path):
        raise RuntimeError(f"reusable smoke result does not exist: {smoke_result_path}")
    source = _read_json(source_path)
    source_bundle = source.get("tested_bundle")
    if not isinstance(source_bundle, dict) or not source_bundle.get("modules") or not source_bundle.get("combo"):
        raise RuntimeError("reusable smoke has no captured executable; rerun smoke")

    cases = source.get("smoke_case_results") or []
    if not isinstance(cases, list) or not cases or any(not isinstance(case, dict) for case in cases):
        raise RuntimeError("reusable smoke must contain structured smoke cases")
    structure_quality_issue_types = {
        "structure_artifact_missing",
        "structure_jsonl_missing",
        "structure_jsonl_empty",
        "structure_canonical_record_missing",
        "structure_required_evidence_channel_absent",
    }
    reassessed_cases = []
    for case in cases:
        sandbox_dir = str(case.get("sandbox_dir") or "")
        video_id = str(case.get("sample_video_id") or "")
        if not sandbox_dir or not video_id:
            raise RuntimeError("reusable smoke case is missing its sandbox directory or video identity")
        structure_quality = _inspect_smoke_structure_quality(
            sandbox_dir, video_id, source_bundle, case.get("combo") or source_bundle["combo"],
        )
        issues = [
            issue for issue in (case.get("runtime_engineering_issues") or [])
            if issue.get("issue_type") not in structure_quality_issue_types
        ] + list(structure_quality.get("issues") or [])
        passed = bool(
            (case.get("validation") or {}).get("passed")
            and (case.get("behavioral_smoke") or {}).get("passed")
            and not issues
        )
        reassessed = {
            **case,
            "passed": passed,
            "engineering_invalid": not passed,
            "structure_quality": structure_quality,
            "artifact_quality": structure_quality,
            "runtime_engineering_issues": issues,
        }
        if passed:
            reassessed.pop("failure_contract", None)
            reassessed.pop("repair_instruction", None)
        reassessed_cases.append(reassessed)
    passed = all(case["passed"] for case in reassessed_cases)
    first_case = reassessed_cases[0]
    return {
        "passed": passed,
        "stage": "reassessed_existing_bundle_smoke",
        "reassessment_policy": (
            "structure evidence recheck; original validation and behavioral outcomes retained"
        ),
        "reassessment_source": source_path,
        "tested_bundle": source_bundle,
        "smoke_case_results": reassessed_cases,
        "smoke_suite_task_ids": [case.get("sample_task_id", "") for case in reassessed_cases],
        "sample_task_id": first_case.get("sample_task_id", ""),
        "sample_video_id": first_case["sample_video_id"],
        "sandbox_dir": first_case["sandbox_dir"],
        "runtime_engineering_issues": [
            issue for case in reassessed_cases for issue in case["runtime_engineering_issues"]
        ],
        "engineering_invalid": not passed,
    }


def _full_eval_admission(results: List[dict], questions: List[dict]) -> Dict:
    """Gate reference eligibility only after every scheduled task has returned.

    Error rows and missing task IDs remain available for diagnosing the stage
    before a complete evaluation rerun.
    """
    expected_ids = [str(question.get("task_id") or "") for question in questions]
    result_ids = [str(row.get("task_id") or "") for row in results]
    expected_set = set(expected_ids)
    result_set = set(result_ids)
    duplicate_result_ids = sorted({item for item in result_ids if item and result_ids.count(item) > 1})
    missing_task_ids = sorted(expected_set - result_set)
    unexpected_task_ids = sorted(item for item in result_set - expected_set if item)
    error_rows = []
    for row in results:
        answer = str(row.get("answer") or "")
        if row.get("error") or answer.strip().startswith("ERROR:"):
            error_rows.append({
                "task_id": str(row.get("task_id") or ""),
                "video_id": str(row.get("video_id") or ""),
                "error": str(row.get("error") or answer)[:1000],
            })
    runtime_issues = detect_runtime_engineering_issues(results)
    execution_complete = (
        len(results) == len(questions)
        and len(result_ids) == len(set(result_ids))
        and not missing_task_ids
        and not unexpected_task_ids
        and not any(not item for item in result_ids)
    )
    passed = execution_complete and not error_rows and not runtime_issues
    return {
        "artifact_type": "initial_full_eval_admission", "schema_version": 1,
        "retry_policy": "rerun the complete evaluation after resolving recorded execution failures",
        "expected_task_count": len(questions),
        "returned_row_count": len(results),
        "execution_complete": execution_complete,
        "missing_task_ids": missing_task_ids,
        "duplicate_result_task_ids": duplicate_result_ids,
        "unexpected_result_task_ids": unexpected_task_ids,
        "error_task_count": len(error_rows),
        "error_tasks": error_rows,
        "runtime_engineering_issues": runtime_issues,
        "passed": passed,
    }


def _write_summary(path: str, report: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Initial Combo Reference\n\n")
        f.write(f"- run_id: `{report.get('run_id')}`\n")
        f.write(f"- combo_id: `{report.get('combo_id')}`\n")
        f.write(f"- question_split: `{report.get('question_split')}`\n")
        f.write(f"- total_questions: `{report.get('total_questions')}`\n")
        f.write(f"- correct: `{report.get('candidate_correct')}`\n")
        results_path = report.get("results_path") or report.get("smoke_results_path") or ""
        f.write(f"- results_path: `{results_path}`\n")
        f.write(f"- sandbox_dir: `{report.get('sandbox_dir')}`\n")
        f.write("\n## Combo\n\n")
        f.write("```json\n")
        f.write(json.dumps(report.get("combo", {}), ensure_ascii=False, indent=2))
        f.write("\n```\n")


def _module_map_refs() -> Dict[str, dict]:
    import module_map
    return {
        "structuring": module_map.STRUCTURING_MAP,
        "thinking": module_map.THINKING_MAP,
        "memory": module_map.WORK_MEMORY_MAP,
        "localization": module_map.LOCALIZATION_MAP,
        "perception": module_map.PERCEPTION_MAP,
    }


def _backup_maps(module_map_refs: Dict[str, dict]) -> Dict[str, dict]:
    return {key: dict(value) for key, value in module_map_refs.items()}


def _restore_maps(module_map_refs: Dict[str, dict], backups: Dict[str, dict]) -> None:
    for key, backup in backups.items():
        module_map_refs[key].clear()
        module_map_refs[key].update(backup)


def _execution_layer_smoke_failure(smoke_result: Dict) -> str:
    """Return a rerun-only code for provider/asset failures, if present.

    Codex cannot repair credentials, provider request transport, or raw-media
    availability.  Keep those failures out of the generated-code repair loop
    even when a downstream artifact check also detects missing evidence.
    """
    if smoke_result.get("execution_layer_rerun_required") or smoke_result.get("failure_kind") == "execution_layer":
        return str(smoke_result.get("error_code") or "runtime_provider_or_asset_failure")
    for issue in smoke_result.get("runtime_engineering_issues") or []:
        if not isinstance(issue, dict):
            continue
        kind = str(issue.get("issue_type") or "")
        if kind == "structured_capability_provider_failure":
            return "capability_provider_failure"
        if kind in {"raw_video_missing", "structure_artifact_missing", "dataset_input_missing"}:
            return kind
    return ""


def _initial_static_failure_contract(bundle: Dict, validation: Dict) -> Dict:
    """Create actionable real-runtime preflight feedback before dataset loading."""
    issues = [str(item) for item in (validation.get("issues") or []) if str(item).strip()]
    return {
        "artifact_type": "initial_bundle_static_contract", "schema_version": 1,
        "failure_kind": "engineering_contract",
        "producer_state": {"producer": "initial_codegen", "state": "five_module_bundle_emitted"},
        "consumer_state": {"consumer": "initial_bundle_preflight", "state": "bundle_rejected_before_runtime_smoke"},
        "missing_handoff": {
            "kind": "static_contract_violation",
            "observed_issues": issues,
            "bundle_module_types": [
                str(item.get("module_type") or "")
                for item in (bundle.get("modules") or []) if isinstance(item, dict)
            ],
        },
        "capability_requirement": {
            "required_vlm_profile_id": DEFAULT_PROFILE_IDS["vlm"],
            "required_adapter": "runtime_evidence",
        },
        "falsifiable_success": [
            "preflight_bundle_runtime(bundle).passed is true",
            "all five module sources compile and are injected into the real module maps",
            "the real execution-layer constructors accept the bundle's five modules",
        ],
    }


def run(args: argparse.Namespace) -> Dict:
    workspace = os.path.abspath(args.workspace)
    bundle_path = os.path.abspath(args.initial_baseline_bundle) if getattr(args, "initial_baseline_bundle", "") else ""
    plan_path = os.path.abspath(args.combo_plan) if getattr(args, "combo_plan", "") else ""
    if not bundle_path:
        raise RuntimeError(
            "MetaVideoAgent initial reference execution requires --initial-baseline-bundle; "
            "initial_combo_plan is design context only and is not an executable baseline."
        )

    reusable_smoke_path = str(getattr(args, "reuse_smoke_result", "") or "")
    if reusable_smoke_path:
        bundle = _read_json(reusable_smoke_path).get("tested_bundle")
        if not isinstance(bundle, dict) or not bundle.get("modules") or not bundle.get("combo"):
            raise RuntimeError("reusable smoke has no captured executable; rerun smoke")
    else:
        bundle = _read_json(bundle_path)
    plan = _read_json(plan_path) if plan_path and os.path.exists(plan_path) else {}
    combo_source = "initial_baseline_bundle"
    bundle_validation = {"validation_mode": "not_run"}
    combo = require_complete_agent_combo(bundle.get("combo"), "initial bundle")
    reuse_structure_raw = str(getattr(args, "reuse_structure_dir", "") or "").strip()
    reuse_structure_dir = os.path.abspath(reuse_structure_raw) if reuse_structure_raw else ""
    if reuse_structure_dir and not os.path.isdir(reuse_structure_dir):
        raise RuntimeError(f"--reuse-structure-dir is not a directory: {reuse_structure_dir}")
    # An explicitly supplied library can serve the same bundle's evaluation;
    # otherwise initialization builds the library from its structuring module.
    structure_rebuild = bool(args.rebuild_structure or not reuse_structure_dir)

    run_id = args.run_id or f"initial_combo_{int(time.time())}"
    paths = _paths(run_id, getattr(args, "output_root", ""))

    # Check runtime construction before loading the evaluation data.
    if not args.check_only:
        os.makedirs(paths["run_dir"], exist_ok=True)
        if plan:
            _write_json(paths["plan_snapshot"], plan)
        _write_json(paths["bundle_snapshot"], bundle)
        bundle_validation = preflight_bundle_runtime(bundle)
        if not bundle_validation.get("passed"):
            contract = _initial_static_failure_contract(bundle, bundle_validation)
            failure = {
                "ready": False,
                "status": "initial_bundle_preflight_failed",
                "workspace": workspace,
                "combo_source": combo_source,
                "initial_baseline_bundle": bundle_path,
                "bundle_validation": bundle_validation,
                "failure_contract": contract,
                "error": "initial generated bundle failed real-runtime preflight",
                "passed": False,
                "repair_route": "automatic_initial_bundle_repair",
                "repair_required": {
                    "bundle": paths["bundle_snapshot"],
                    "smoke_result": os.path.join(paths["run_dir"], "initial_bundle_static_failure.json"),
                    "next_action": "Repair this exact bundle, then rerun the real smoke.",
                },
            }
            _write_json(os.path.join(paths["run_dir"], "initial_bundle_static_failure.json"), failure)
            _write_json(paths["report_path"], failure)
            raise RuntimeError(
                f"initial baseline bundle runtime preflight failed: {bundle_validation.get('issues')}"
            )

    split = args.question_split
    training_slice_preflight = bool(getattr(args, "training_slice_preflight", False))
    if getattr(args, "smoke_only", False):
        # Stream only the configured no-label smoke cases into memory.
        questions, video_ids, split_spec = select_questions_streaming(
            video_id=args.video_id,
            distribution_manifest=args.distribution_manifest,
            preferred_splits=(split,),
            max_questions=_smoke_case_limit(),
            selection_key=_smoke_question_key,
        )
    else:
        questions, video_ids, split_spec = load_question_set(
            video_id=args.video_id,
            distribution_manifest=args.distribution_manifest,
            preferred_splits=(split,),
        )
        if args.first_n:
            questions = questions[:args.first_n]
    if not questions:
        raise RuntimeError("no questions found for initial combo run")
    if reuse_structure_dir:
        struct_name = str(combo.get("video_structuring") or "")
        missing_structures = [
            video_id for video_id in video_ids
            if not os.path.isfile(os.path.join(reuse_structure_dir, f"{video_id}_{struct_name}.jsonl"))
        ]
        if missing_structures:
            raise RuntimeError(
                "--reuse-structure-dir lacks required split-local structures: "
                f"{missing_structures}"
            )

    scheduled_questions = list(questions)
    readiness = {
        "ready": True,
        "workspace": workspace,
        "runtime_id": runtime_id(),
        "combo_source": combo_source,
        "combo_plan": plan_path,
        "initial_baseline_bundle": bundle_path,
        "bundle_validation": bundle_validation,
        "distribution_manifest": os.path.abspath(args.distribution_manifest) if args.distribution_manifest else "",
        "question_split": split,
        "total_questions": len(scheduled_questions),
        "scheduled_question_count": len(questions),
        "video_ids": video_ids,
        "combo": combo,
        "paths": paths,
        "results_path": paths["results_path"],
        "report_path": paths["report_path"],
        "sandbox_dir": paths["sandbox_dir"],
    }
    if args.check_only:
        return readiness

    os.makedirs(paths["run_dir"], exist_ok=True)
    if plan:
        _write_json(paths["plan_snapshot"], plan)
    _write_json(paths["bundle_snapshot"], bundle)
    readiness["bundle_validation"] = bundle_validation
    os.environ.setdefault("VIDEO_STRUCT_REBUILD_WORKERS", str(args.structure_workers))
    if structure_rebuild:
        os.environ["SANDBOX_DISABLE_FULL_STRUCTURE_BUILD"] = "0"
        os.environ["SANDBOX_ALLOW_FULL_STRUCTURE_REBUILD"] = "1"
        os.environ["SANDBOX_STRUCTURE_CHROMA_PER_VIDEO"] = "1"
    else:
        os.environ.setdefault("SANDBOX_DISABLE_FULL_STRUCTURE_BUILD", "1")
        os.environ.setdefault("SANDBOX_ALLOW_FULL_STRUCTURE_REBUILD", "0")
        os.environ.setdefault("SANDBOX_STRUCTURE_CHROMA_PER_VIDEO", "0")

    combo_id = args.combo_id or f"InitialCombo_{int(time.time())}"
    smoke_result = {}
    if reusable_smoke_path:
        smoke_result = _reassess_existing_bundle_smoke(reusable_smoke_path)
        reassessment_path = os.path.join(paths["run_dir"], "bundle_smoke_reassessment.json")
        _write_json(reassessment_path, smoke_result)
        if not smoke_result.get("passed"):
            failure = {
                **readiness,
                "ready": False,
                "status": "reused_bundle_smoke_reassessment_failed",
                "bundle_smoke_result": smoke_result,
                "effective_initial_baseline_bundle": paths["bundle_snapshot"],
                "engineering_invalid": True,
                "error": "existing smoke did not pass deterministic reassessment",
            }
            _write_json(paths["report_path"], failure)
            raise RuntimeError(failure["error"])
        if getattr(args, "smoke_only", False):
            report = {
                **readiness,
                "ready": True,
                "status": "bundle_smoke_reassessed_passed",
                "run_id": run_id,
                "combo_id": combo_id,
                "bundle_smoke_result": smoke_result,
                "effective_initial_baseline_bundle": paths["bundle_snapshot"],
                "results_path": "",
                "report_path": paths["report_path"],
                "summary_path": paths["summary_path"],
                "sandbox_dir": paths["sandbox_dir"],
                "smoke_results_path": reassessment_path,
                "smoke_sandbox_dir": smoke_result.get("sandbox_dir", ""),
                "total_questions": len(smoke_result["smoke_case_results"]),
                "full_eval_total_questions_pending": len(questions),
                "note": "Reassessed the captured smoke executable and evidence; no model calls were made.",
            }
            _write_json(paths["report_path"], report)
            _write_summary(paths["summary_path"], report)
            return report
    if not reusable_smoke_path and not getattr(args, "skip_bundle_smoke_test", False):
        strict_time_reference_smoke = bool(
            getattr(args, "smoke_time_reference_only", False)
        )
        smoke_questions = _smoke_suite_questions(
            questions, require_time_reference=strict_time_reference_smoke,
        )
        suite_results = []
        smoke_case_results = []
        trace_rows = []
        failed_result = None
        for case_index, smoke_question in enumerate(smoke_questions, start=1):
            smoke_dir = os.path.join(
                paths["run_dir"], f"bundle_smoke_sandbox_case_{case_index}"
            )
            smoke_env_updates = {}
            if strict_time_reference_smoke:
                # Total-attempt limits of one disable provider retries for
                # the bounded engineering smoke.
                smoke_env_updates.update({
                    "LLM_MAX_RETRIES": "1",
                    "VLM_MAX_RETRIES": "1",
                    "EMBEDDING_MAX_RETRIES": "1",
                    "OCR_MAX_RETRIES": "1",
                    # Use the configured smoke request deadline.
                    "METAVIDEOAGENT_PROVIDER_TIMEOUT_SEC": str(max(
                        1, int(getattr(args, "provider_timeout_sec", 180) or 180)
                    )),
                    "RUNTIME_PROVIDER_AUDIT_PATH": os.path.join(paths["run_dir"], "provider_calls.jsonl"),
                })
            if structure_rebuild:
                smoke_env_updates.update({
                    "SANDBOX_STRUCTURE_TIME_REF_ONLY": "1",
                    "VIDEO_STRUCT_REBUILD_WORKERS": "1",
                })
            old_smoke_env = _set_env_temporarily(smoke_env_updates)
            try:
                case_result = single_question_bundle_smoke_test(
                    bundle,
                    smoke_question,
                    workspace,
                    smoke_dir,
                    combo_id=f"{combo_id}_smoke_case_{case_index}",
                    force_rebuild=structure_rebuild,
                    isolate_structure=True,
                    runtime_time_reference_only=strict_time_reference_smoke,
                    reference_structure_dir=reuse_structure_dir,
                )
            finally:
                _restore_env(old_smoke_env)
            suite_results.append({
                "case_index": case_index,
                "task_id": str(smoke_question.get("task_id") or ""),
                "video_id": str(smoke_question.get("video_id") or ""),
                "time_reference": smoke_question.get("time_reference"),
                "passed": bool(case_result.get("passed")),
                "failure_kind": case_result.get("failure_kind", ""),
                "sandbox_dir": case_result.get("sandbox_dir", ""),
            })
            case_record = dict(case_result)
            case_trace = case_record.pop("engineering_execution_trace", None)
            if isinstance(case_trace, dict):
                trace_rows.extend([
                    row for row in (case_trace.get("rows") or []) if isinstance(row, dict)
                ])
            smoke_case_results.append(case_record)
            if not case_result.get("passed") and failed_result is None:
                failed_result = case_result
        if failed_result is None:
            smoke_result = {
                "passed": True,
                "stage": "bundle_smoke_suite",
                "smoke_suite": suite_results,
                "smoke_suite_task_ids": [item["task_id"] for item in suite_results],
            }
        else:
            smoke_result = dict(failed_result)
            smoke_result["smoke_suite"] = suite_results
            smoke_result["smoke_suite_task_ids"] = [item["task_id"] for item in suite_results]
        smoke_result["smoke_case_results"] = smoke_case_results
        # Carry the executable that actually ran, including failed
        # attempts eligible for a later execution-checker reassessment.
        smoke_result["tested_bundle"] = copy.deepcopy(bundle)
        if trace_rows:
            smoke_result["engineering_execution_trace"] = {
                "artifact_type": "engineering_smoke_execution_trace", "schema_version": 1,
                "trace_policy": (
                    "complete no-label trajectories for every selected smoke case; "
                    "not an accuracy or correctness artifact"
                ),
                "rows": trace_rows,
            }
        engineering_trace = smoke_result.pop("engineering_execution_trace", None)
        if isinstance(engineering_trace, dict) and engineering_trace:
            trace_path = os.path.join(
                paths["run_dir"], "bundle_smoke_execution_trace.json",
            )
            _write_json(trace_path, engineering_trace)
            smoke_result["engineering_execution_trace_path"] = trace_path
        execution_error = _execution_layer_smoke_failure(smoke_result)
        if execution_error:
            smoke_result["failure_kind"] = "execution_layer"
            smoke_result["error_code"] = execution_error
            smoke_result["repair_route"] = "rerun_same_bundle_after_execution_layer_fix"
        elif not smoke_result.get("passed"):
            smoke_result["repair_route"] = "automatic_initial_bundle_repair"
            smoke_result["repair_required"] = {
                "bundle": paths["bundle_snapshot"],
                "smoke_result": os.path.join(paths["run_dir"], "bundle_smoke_result.json"),
                "next_action": "Repair the bundle using the recorded smoke failure, then rerun smoke.",
            }
        _write_json(os.path.join(paths["run_dir"], "bundle_smoke_result.json"), smoke_result)
        if not smoke_result.get("passed"):
            failure = {
                **readiness,
                "ready": False,
                "status": "bundle_smoke_failed",
                "bundle_smoke_result": smoke_result,
                "effective_initial_baseline_bundle": paths["bundle_snapshot"],
                "engineering_invalid": bool(smoke_result.get("engineering_invalid")),
                "error": smoke_result.get("error", "bundle smoke test failed"),
            }
            _write_json(paths["report_path"], failure)
            raise RuntimeError(f"initial baseline bundle smoke failed: {failure['error']}")
        if getattr(args, "smoke_only", False):
            report = {
                **readiness,
                "ready": True,
                "status": "bundle_smoke_passed",
                "run_id": run_id,
                "combo_id": combo_id,
                "bundle_smoke_result": smoke_result,
                "effective_initial_baseline_bundle": paths["bundle_snapshot"],
                "results_path": "",
                "report_path": paths["report_path"],
                "summary_path": paths["summary_path"],
                "sandbox_dir": paths["sandbox_dir"],
                "smoke_results_path": os.path.join(paths["run_dir"], "bundle_smoke_result.json"),
                "smoke_sandbox_dir": smoke_result.get("sandbox_dir", ""),
                "total_questions": len(smoke_questions),
                "full_eval_total_questions_pending": len(questions),
                "note": "Stopped after real bundle smoke by --smoke-only; no full initial reference evaluation was run.",
            }
            _write_json(paths["report_path"], report)
            _write_summary(paths["summary_path"], report)
            return report

    module_map_refs = _module_map_refs()
    map_backups = _backup_maps(module_map_refs)
    injected_validation = {}
    try:
        ok, combo, custom_configs, injected_validation = inject_module_bundle(
            bundle, module_map_refs,
        )
        if not ok:
            raise RuntimeError(
                "initial baseline bundle injection failed: "
                f"{injected_validation.get('issues')}"
            )
        struct_name = combo.get("video_structuring")
        if structure_rebuild and struct_name and getattr(args, "structure_workers", 0):
            struct_cfg = dict(custom_configs.get(struct_name) or {})
            physics = dict(struct_cfg.get("physics") or {})
            physics["parallel_workers"] = int(args.structure_workers)
            physics.setdefault("parallel_batch_delay", 0.2)
            struct_cfg["physics"] = physics
            custom_configs[struct_name] = struct_cfg
        # Evaluate the complete split once and retain each task's outcome,
        # including execution failures, for the stage report.
        try:
            structure_vlm_start_rate_limit = max(
                0, int(os.environ.get("METAVIDEOAGENT_VLM_STARTS_PER_SECOND", "0") or 0)
            )
        except ValueError:
            structure_vlm_start_rate_limit = 0
        full_eval_env_backup = _set_env_temporarily({
            "SANDBOX_QUESTION_MAX_RETRIES": "1",
            "LLM_MAX_RETRIES": "1",
            "VLM_MAX_RETRIES": "1",
            # ASR uses a separately paced RPM queue.  Retain one jittered
            # retry for a transient provider rejection instead of persisting
            # an incomplete audio evidence record on the first 429.
            "ASR_MAX_RETRIES": "2",
            "EMBEDDING_MAX_RETRIES": "1",
            "OCR_MAX_RETRIES": "1",
            "METAVIDEOAGENT_VLM_MAX_CONCURRENCY": str(_execution_vlm_concurrency(
                getattr(args, "vlm_max_concurrency", 32)
            )),
            "METAVIDEOAGENT_LLM_MAX_CONCURRENCY": str(max(
                1, min(16, int(getattr(args, "llm_max_concurrency", 8) or 8))
            )),
            # Record the transport deadline and request-level execution audit.
            "METAVIDEOAGENT_PROVIDER_TIMEOUT_SEC": str(max(
                1, int(getattr(args, "provider_timeout_sec", 180) or 180)
            )),
            "RUNTIME_PROVIDER_AUDIT_PATH": os.path.join(paths["run_dir"], "provider_calls.jsonl"),
        })
        try:
            # Build this evaluation's video libraries before answering questions.
            structure_prebuild = []
            question_video_ids = sorted({
                str(item.get("video_id") or "") for item in questions
                if str(item.get("video_id") or "")
            })
            if structure_rebuild and question_video_ids:
                structure_prebuild = prebuild_isolated_video_structures(
                    workspace,
                    question_video_ids,
                    combo,
                    paths["sandbox_dir"],
                    custom_configs=custom_configs,
                )
            # The optional fixed-window VLM start gate is a structure-build
            # throughput policy.  Once every video library is complete,
            # answer workers follow their own global question concurrency and
            # must not inherit a build-only per-second throttle.
            os.environ["METAVIDEOAGENT_VLM_STARTS_PER_SECOND"] = "0"
            results = evaluate_combo_multi_video(
                workspace,
                questions,
                combo,
                combo_id,
                paths["sandbox_dir"],
                custom_configs=custom_configs,
                # The explicit prebuild above owns structure construction;
                # answer workers only reuse those run-local artifacts.
                force_rebuild=False,
                isolate_structure=True,
                concurrency=args.concurrency,
                global_question_scheduler=True,
                reference_structure_dir=reuse_structure_dir,
            ) if questions else []
        finally:
            _restore_env(full_eval_env_backup)
    finally:
        _restore_maps(module_map_refs, map_backups)
    full_eval_admission = _full_eval_admission(results, scheduled_questions)
    ordered = _dedupe_results(results, scheduled_questions)
    _write_jsonl(paths["results_path"], ordered)
    # Self-reference gives a zero-delta report with strict correctness counts.
    reference_label = "initial_baseline"
    reference_lookup = build_reference_lookup(ordered, reference_label=reference_label)
    report = compare_with_reference(
        ordered,
        reference_lookup,
        reference_label=reference_label,
        reference_source=paths["results_path"],
    )
    reference_eligible = bool(full_eval_admission.get("passed")) and not training_slice_preflight
    report.update({
        "run_id": run_id,
        "runtime_id": runtime_id(),
        "combo_id": combo_id,
        "combo": combo,
        "combo_source": combo_source,
        "combo_plan": plan_path,
        "initial_baseline_bundle": bundle_path,
        "effective_initial_baseline_bundle": paths["bundle_snapshot"],
        "bundle_validation": injected_validation or bundle_validation,
        "bundle_smoke_result": smoke_result,
        "validation_scope": (
            "training_slice_preflight"
            if training_slice_preflight else "initial_reference_multi_video"
        ),
        "training_slice_preflight": training_slice_preflight,
        "reference_label": reference_label,
        "reference_source": paths["results_path"],
        "question_split": split,
        "question_split_spec": {
            "manifest_path": split_spec.get("manifest_path", "") if isinstance(split_spec, dict) else "",
            "split_names": sorted((split_spec.get("splits", {}) or {}).keys()) if isinstance(split_spec, dict) else [],
        },
        "video_ids": video_ids,
        "expected_total_questions": len(scheduled_questions),
        "evaluation_complete": bool(full_eval_admission.get("execution_complete")),
        "full_eval_admission": full_eval_admission,
        "engineering_invalid": not bool(full_eval_admission.get("passed")),
        "reference_eligible": reference_eligible,
        "ready": reference_eligible,
        "full_eval_retry_policy": full_eval_admission.get("retry_policy"),
        "concurrency": args.concurrency,
        "llm_model": DEFAULT_LLM_MODEL,
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
            "llm_max_concurrency": args.llm_max_concurrency,
            "asr_max_concurrency": 16,
            "embedding_max_concurrency": 16,
        },
        "provider_transport_policy": {
            "provider_timeout_sec": max(1, int(getattr(args, "provider_timeout_sec", 180) or 180)),
            "initial_full_eval_attempts_per_provider_call": 1,
            "evaluation_retry_policy": "whole_stage_after_execution_fix",
            "provider_call_audit_path": os.path.join(paths["run_dir"], "provider_calls.jsonl"),
            "vlm_start_rate_limit_per_second": max(
                0, int(os.environ.get("METAVIDEOAGENT_VLM_STARTS_PER_SECOND", "0") or 0)
            ),
            "structure_build_vlm_start_rate_limit_per_second": (
                structure_vlm_start_rate_limit
            ),
            "structure_build_asr_start_rate_limit_per_minute": max(
                0, int(os.environ.get("METAVIDEOAGENT_ASR_STARTS_PER_MINUTE", "0") or 0)
            ),
        },
        "rebuild_structure": structure_rebuild,
        "structure_source": "rebuilt_isolated_initial_bundle" if structure_rebuild else "explicit_reuse",
        "reused_structure_dir": reuse_structure_dir,
        "structure_dir": os.path.join(paths["sandbox_dir"], "video_structure"),
        "structure_artifact": _structure_artifact(
            os.path.join(paths["sandbox_dir"], "video_structure"),
            combo,
            video_ids,
            rebuilt=structure_rebuild,
        ),
        "structure_workers": args.structure_workers,
        "structure_prebuild": structure_prebuild if 'structure_prebuild' in locals() else [],
        "sandbox_dir": paths["sandbox_dir"],
        "results_path": paths["results_path"],
        "report_path": paths["report_path"],
        "summary_path": paths["summary_path"],
        "ready_for_first_diagnosis": reference_eligible,
        "next_step": (
            "Run Teacher/Diagnosis with this report/results as the first "
            "reference combo, not fixed trajectories baseline."
            if reference_eligible else
            "Training-slice preflight completed; inspect its artifacts, then run a complete "
            "evolution-split full evaluation before entering evolution."
            if training_slice_preflight else
            "Resolve the recorded execution failures and rerun the complete "
            "initial evaluation before entering evolution."
        ),
    })
    _write_json(paths["report_path"], report)
    _write_summary(paths["summary_path"], report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an admitted initial bundle as the reference agent")
    parser.add_argument("--workspace", default=default_workspace())
    parser.add_argument("--combo-plan", default="")
    parser.add_argument("--initial-baseline-bundle", default="")
    parser.add_argument("--distribution-manifest", default="")
    parser.add_argument("--video-id", default="ALL")
    parser.add_argument("--question-split", default="train")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--vlm-max-concurrency", type=int, default=32,
                        help="VLM in-flight ceiling (1-32 normally; up to 128 with explicit start-rate gate).")
    parser.add_argument("--provider-timeout-sec", type=int, default=180,
                        help="Provider request deadline in seconds (default: 180).")
    parser.add_argument("--first-n", type=int, default=0)
    parser.add_argument("--llm-max-concurrency", type=int, default=8,
                        help="Process-wide text-LLM HTTP ceiling (1-16; default 8; thinking disabled).")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--rebuild-structure", action="store_true")
    parser.add_argument("--reuse-structure-dir", default="",
                        help=("Use this verified split-local structure artifact for answer-only evaluation; "
                              "no full-video structure rebuild is allowed."))
    parser.add_argument("--structure-workers", type=int, default=4)
    parser.add_argument("--combo-id", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default="",
                        help="Directory for initial-agent outputs")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--reuse-smoke-result", default="",
                        help="Reassess a recorded smoke and its captured executable without model calls.")
    parser.add_argument("--training-slice-preflight", action="store_true",
                        help="Run a training-only subset as a non-reference preflight; it can never enter evolution.")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run real bundle smoke only and stop before full initial reference evaluation")
    parser.add_argument("--smoke-time-reference-only", action="store_true",
                        help="Require every smoke case and media tool call to stay inside time_reference")
    args = parser.parse_args()
    vlm_ceiling = 128 if int(os.environ.get("METAVIDEOAGENT_VLM_STARTS_PER_SECOND", "0") or 0) > 0 else 32
    if args.vlm_max_concurrency < 1 or args.vlm_max_concurrency > vlm_ceiling:
        raise RuntimeError(f"--vlm-max-concurrency must be in [1, {vlm_ceiling}]")
    if args.llm_max_concurrency < 1 or args.llm_max_concurrency > 16:
        raise RuntimeError("--llm-max-concurrency must be in [1, 16]")

    report = run(args)
    if args.check_only:
        print("INITIAL_COMBO_CHECK")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ready") else 1

    print("INITIAL_COMBO_DONE")
    for key in ("report_path", "results_path", "sandbox_dir", "candidate_correct", "total_questions"):
        print(f"{key}={report.get(key)}")
    if report.get("training_slice_preflight") and report.get("full_eval_admission", {}).get("passed"):
        return 0
    return 0 if report.get("ready_for_first_diagnosis", report.get("ready", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
