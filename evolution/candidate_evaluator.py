"""Candidate probe and evaluation orchestration for MetaVideoAgent.

This component executes an already-authored candidate bundle, monitors runtime
failures, and compares its results with the current reference. It does not edit
source code or invoke a coding agent. Codex authoring and bounded repair
are handled by the outer workflow.
"""

import glob
import json
import os
import re
import sys
import time

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)
from runtime_paths import use_runtime_path

_RUNTIME_DIR = use_runtime_path()

from agent_sources import module_records, module_resource_inputs
from answer_normalizer import extract_gold_answer
from bundle_contract import normalize_modules
from combo_contract import require_complete_agent_combo
from iteration_workspace import write_json
from question_set_loader import load_questions as load_question_set
from reference_loader import combo_from_reference
from structure_artifacts import reference_structure_artifact_dir
from target_selection import canonical_task_ref
from task_identity import make_task_id

# =====================================================================
# Candidate evaluator
# =====================================================================

def finalize_probe_comparison(verification: dict, conclusion: dict) -> None:
    """Publish one probe verdict from the latest paired execution evidence."""
    corrections = verification.get("corrections") or []
    regressions = verification.get("regressions") or []
    invalid = (
        verification.get("engineering_invalid")
        or verification.get("execution_layer_rerun_required")
        or not verification.get("comparison_valid", False)
        or verification.get("evaluation_complete") is not True
    )
    status = "invalid" if invalid else CandidateEvaluator._classify_probe_status(
        len(corrections), len(regressions),
    )
    verification["probe_status"] = status
    verification["status"] = "promising_unverified" if status == "promising" else status
    verification["accepted"] = False
    verification["full_verification"] = {
        "status": "deferred", "reason": "formal full evaluation is owned by full_eval_runner",
    }
    verification["final_success_criteria"] = {
        "passed": False, "requires_full_eval": True,
        "reason": "Probe feedback does not promote a candidate; full train evaluation is required.",
    }
    mechanism = verification.get("mechanism_generalization") or {}
    corrected = {str(row.get("task_id") or "") for row in corrections}
    witnesses = set(mechanism.get("diagnosis_hypothesis_selected") or [])
    generalization = set(mechanism.get("generalization_selected") or [])
    mechanism.update({
        "diagnosis_hypothesis_corrected": sorted(witnesses & corrected),
        "generalization_corrected": sorted(generalization & corrected),
        "status": "not_configured" if not witnesses else (
            "hypothesis_and_generalization_improved" if generalization & corrected else
            "hypothesis_observation_improved" if witnesses & corrected else "hypothesis_not_observed_improved"
        ),
    })
    verification["mechanism_generalization"] = mechanism
    verdicts = {
        "promising": ("probe_promising_pending_full_eval", "The paired probe improved; full train evaluation is pending."),
        "risk_detected": ("risk_detected", "The paired probe contains regressions; inspect the complete trajectories."),
        "inconclusive_stable": ("inconclusive_stable", "The paired probe demonstrated no correction or regression."),
        "invalid": ("engineering_blocked", "The paired probe is incomplete or has an execution failure."),
    }
    conclusion["verification"] = verification
    conclusion["verdict"], conclusion["verdict_reason"] = verdicts[status]


class CandidateEvaluator:
    """Execute and compare a candidate produced by the automatic workflow."""

    def __init__(self, diagnosis_report: dict, workspace_dir: str,
                 first_n: int = None, concurrency: int = 3, dry_run: bool = False,
                 sandbox_dir: str = None):
        self.report = diagnosis_report
        self.workspace_dir = workspace_dir
        self.first_n = first_n
        self.concurrency = concurrency
        self.dry_run = dry_run
        self.sandbox_dir = sandbox_dir or os.path.join(workspace_dir, "sandbox")
        self._last_probe_results = []

        # Extract key information from diagnostic reports
        if self.report.get("artifact_type") != "diagnosis_execution_brief":
            raise ValueError("CandidateEvaluator requires a diagnosis_execution_brief artifact")
        # Candidate continuation reuses its recorded batch and selection groups.
        # New outer iterations can select a fresh representative batch.
        self.fixed_probe_task_ids = list(self.report.get("fixed_probe_task_ids", []) or [])
        self.fixed_probe_plan = dict(self.report.get("fixed_probe_plan") or {})
        task = self.report.get("implementation_task", {}) or {}
        evaluation = self.report.get("_machine_evaluation_contract", {}) or {}
        if not evaluation:
            raise RuntimeError(
                "execution brief requires an explicit machine_evaluation_contract sidecar; "
                "do not embed probe evidence in the model brief"
            )
        policy = self.report.get("execution_policy", {}) or {}
        try:
            bundle_targets = normalize_modules(policy.get("target_modules", []) or [])
        except ValueError as exc:
            raise RuntimeError(f"CandidateEvaluator received invalid target_modules: {exc}") from exc
        probe_plan = evaluation.get("probe_plan", {}) or {}
        target_questions = evaluation.get("target_question_refs", []) or []
        if not target_questions and isinstance(probe_plan, dict):
            for group in ("repair_probe", "representative_rows", "held_out_generalization", "regression_guard"):
                for ref in probe_plan.get(group, []) or []:
                    if ref and ref not in target_questions:
                        target_questions.append(ref)
        self.algo_evo = {
            "target_modules": bundle_targets,
            "target_questions": target_questions,
            "error_diagnoses": evaluation.get("error_diagnoses", []) or [],
            "evolution_hint": (
                task.get("evolution_mechanism") or task.get("goal")
                or task.get("codex_task", "")
            ),
        }
        self.target_modules = self.algo_evo["target_modules"]
        self.evolution_decision = policy
        self.target_questions = self.algo_evo["target_questions"]
        self.error_diagnoses = self.algo_evo["error_diagnoses"]
        self.evolution_hint = self.algo_evo["evolution_hint"]
        self.evaluation_policy = evaluation.get("validation_policy", {}) or {}
        self.target_probe_context = evaluation.get("target_probe_context", {}) or {}
        self.machine_probe_plan = evaluation.get("probe_plan", {}) or {}
        self.probe_feedback_excluded_refs = list(
            evaluation.get("probe_feedback_excluded_refs", []) or []
        )
        self.observed_distribution_profile = (
            self.report.get("observed_distribution_profile", {}) or {}
        )
        profile_spec = (
            self.observed_distribution_profile.get("distribution_spec", {})
            if isinstance(self.observed_distribution_profile, dict) else {}
        )
        self.distribution_manifest = (
            self.report.get("distribution_manifest", "")
            or profile_spec.get("manifest_path", "")
        )
        self.reference_results_path = (
            self.report.get("reference_results_path", "")
            or self.evolution_decision.get("reference_results_path", "")
        )
        self.reference_report_path = (
            self.report.get("reference_report_path", "")
            or self.evolution_decision.get("reference_report_path", "")
            or ((self.report.get("iter_context", {}) or {}).get("reference", {}) or {}).get("report", {}).get("path", "")
        )
        self.reference_label = (
            self.report.get("reference_label", "")
            or self.evolution_decision.get("reference_label", "")
            or "reference"
        )
        if not self.reference_results_path:
            raise RuntimeError(
                "MetaVideoAgent CandidateEvaluator requires explicit reference_results_path in diagnosis. "
                "The public evaluator does not infer reference trajectories."
            )

        # Evolutionary results
        self.evolved_bundle = None
        self.verification_report = None

        # Runtime monitoring status
        self._runtime_issues = []

    def _baseline_combo_config(self) -> dict:
        source = self.report.get("reference_combo") or {}
        if not source and self.reference_results_path:
            resolved = self._resolve_result_path(
                self.reference_results_path, self.workspace_dir, os.getcwd()
            )
            source = combo_from_reference(resolved)
        return require_complete_agent_combo(source, "CandidateEvaluator reference combo")


    def _read_json_optional(self, path: str) -> dict:
        path = self._resolve_result_path(path, self.workspace_dir, os.getcwd())
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _reference_bundle_candidate_paths(self) -> list:
        """Resolve the current-best source bundle used by paired evaluation."""
        paths = []
        report_path = self._resolve_result_path(
            self.reference_report_path, self.workspace_dir, os.getcwd()
        )
        report = self._read_json_optional(report_path)

        for container in (self.report, report):
            if not isinstance(container, dict):
                continue
            for key in (
                "reference_bundle",
                "reference_bundle_path",
                "initial_baseline_bundle",
                "bundle_path",
                "current_best_bundle_path",
            ):
                value = container.get(key)
                if isinstance(value, str) and value:
                    paths.append(value)
                elif isinstance(value, dict):
                    for sub_key in ("path", "bundle_path", "source_path"):
                        sub_value = value.get(sub_key)
                        if isinstance(sub_value, str) and sub_value:
                            paths.append(sub_value)

        iter_context = self.report.get("iter_context", {}) or {}
        for dotted in (
            ("initial_baseline", "bundle", "path"),
            ("reference", "initial_baseline_bundle", "path"),
            ("current_best", "bundle", "path"),
        ):
            cursor = iter_context
            for key in dotted:
                cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
            if isinstance(cursor, str) and cursor:
                paths.append(cursor)

        resolved = []
        seen = set()
        for path in paths:
            resolved_path = self._resolve_result_path(
                path,
                os.path.dirname(os.path.abspath(report_path)) if report_path else "",
                self.workspace_dir,
                os.getcwd(),
            )
            if resolved_path and resolved_path not in seen and os.path.exists(resolved_path):
                seen.add(resolved_path)
                resolved.append(resolved_path)
        return resolved

    def _load_reference_bundle_for_combo(self, combo_config: dict) -> tuple:
        from sandbox_evaluator import validate_bundle_assembly

        expected_agent_combo = {
            "video_structuring": combo_config.get("video_structuring", ""),
            "thinking": combo_config.get("thinking", ""),
            "memory": combo_config.get("memory", ""),
            "localization": combo_config.get("localization", ""),
            "perception": combo_config.get("perception", ""),
        }
        for path in self._reference_bundle_candidate_paths():
            bundle = self._read_json_optional(path)
            if not bundle:
                continue
            bundle_combo = bundle.get("combo") or {}
            if bundle_combo != expected_agent_combo:
                continue
            validation = validate_bundle_assembly(bundle)
            if not validation.get("passed"):
                continue
            return bundle, path, validation
        return {}, "", {
            "passed": False,
            "issues": [
                "No explicit reference bundle matching the reference combo was found. "
                "MetaVideoAgent verification refuses builtin alias fallback."
            ],
        }

    def _inject_candidate(self, module_map_refs: dict) -> tuple:
        """Load candidate code once; retain reference metadata for comparison."""
        from sandbox_evaluator import inject_module_bundle

        reference_bundle, bundle_path, reference_validation = self._load_reference_bundle_for_combo(
            self._baseline_combo_config(),
        )
        if not reference_bundle:
            raise RuntimeError(f"Verification requires a complete reference bundle: {reference_validation}")
        ok, combo, configs, validation = inject_module_bundle(self.evolved_bundle, module_map_refs)
        if not ok:
            raise RuntimeError(f"Candidate bundle injection failed: {validation}")
        return combo, configs, {
            "reference_bundle_path": bundle_path,
            "reference_bundle_validation": reference_validation,
            "reference_bundle": reference_bundle,
        }

    def _structure_reuse_decision(self, reference_injection: dict,
                                  candidate_combo: dict) -> dict:
        """Decide structure reuse from the actual reference/candidate bundles.

        A complete combo always has a struct slot, so target module names and
        slot presence are not valid rebuild signals. Rebuild only when the
        injected candidate's struct implementation differs from the explicit
        current-best reference bundle.
        """
        reference_bundle = (reference_injection or {}).get("reference_bundle") or {}
        reference_combo = reference_bundle.get("combo") or {}
        reference_struct = reference_combo.get("video_structuring", "")
        candidate_struct = (candidate_combo or {}).get("video_structuring", "")
        reference_module = module_records(reference_bundle).get("video_structuring") or {}
        candidate_module = module_records(self.evolved_bundle).get("video_structuring") or {}
        reference_code = reference_module.get("code", "")
        candidate_code = candidate_module.get("code", "")

        if not reference_struct or not candidate_struct:
            raise RuntimeError(
                "Cannot determine structure reuse: reference and candidate combos "
                f"must both define struct (reference={reference_struct!r}, "
                f"candidate={candidate_struct!r})."
            )
        if candidate_struct != reference_struct:
            reason = "struct_module_name_changed"
            rebuild = True
        elif self.evolved_bundle and (
            module_resource_inputs(self.evolved_bundle, "video_structuring")
            != module_resource_inputs(reference_bundle, "video_structuring")
        ):
            reason, rebuild = "struct_resources_or_config_changed", True
        elif candidate_module.get("source_file") != reference_module.get("source_file"):
            reason, rebuild = "struct_source_location_changed", True
        elif candidate_code and reference_code and candidate_code != reference_code:
            reason = "struct_module_code_changed"
            rebuild = True
        elif candidate_code and not reference_code:
            reason = "reference_struct_code_unavailable"
            rebuild = True
        else:
            reason = "struct_implementation_matches_current_best"
            rebuild = False
        return {
            "rebuild_structure": rebuild,
            "reason": reason,
            "reference_struct": reference_struct,
            "candidate_struct": candidate_struct,
            "reference_bundle_path": (reference_injection or {}).get("reference_bundle_path", ""),
        }

    def _reference_structure_artifact_dir(self) -> str:
        """Resolve the reference report's structure artifact for isolated probe use."""
        report_paths = [self.reference_report_path, self.report.get("reference_report_path", "")]
        for raw_path in report_paths:
            report_path = self._resolve_result_path(raw_path, self.workspace_dir, os.getcwd())
            if not report_path or not os.path.isfile(report_path):
                continue
            resolved = reference_structure_artifact_dir(
                self._read_json_optional(report_path), report_path,
            )
            if resolved:
                return resolved
        return ""

    def _read_result_rows(self, path: str) -> list:
        rows = []
        if not path or not os.path.exists(path):
            return rows
        paths = [path]
        if os.path.isdir(path):
            paths = sorted(glob.glob(os.path.join(path, "*_sandbox.jsonl")))
        for item in paths:
            with open(item, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    def _resolve_result_path(self, path: str, *bases: str) -> str:
        if not path:
            return ""
        candidates = [path]
        if not os.path.isabs(path):
            candidates.extend(
                os.path.normpath(os.path.join(base, path))
                for base in bases
                if base
            )
            candidates.append(os.path.abspath(path))
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return path

    def _reference_lookup(self) -> tuple:
        """Load the explicit reference rows attested by the diagnosis brief."""
        from sandbox_evaluator import build_reference_lookup

        source = self._resolve_result_path(
            self.reference_results_path, self.workspace_dir, os.getcwd()
        )
        rows = self._read_result_rows(source)
        if not rows:
            raise RuntimeError(
                "MetaVideoAgent CandidateEvaluator could not read explicit "
                f"reference rows: {source}"
            )
        return build_reference_lookup(
            rows, reference_label=self.reference_label
        ), self.reference_label, source


    def _runtime_monitor_callback(self, question_idx: int, result: dict,
                                   all_results: list) -> bool:
        """Stop a probe when repeated runtime failures indicate a systemic issue.

        Detect whether there are systemic engineering issues after each question is completed.
        If a critical/high severity system problem is detected, abort immediately.

        Args:
            question_idx: current question index (0-based)
            result: the execution result of the current question
            all_results: all results so far

        Returns:
            True to continue, or False to abort.
        """
        from sandbox_evaluator import detect_runtime_engineering_issues

        # Complete at least 1 question before making a judgment (a single question may be an accidental error)
        if question_idx < 1:
            return True

        issues = detect_runtime_engineering_issues(all_results)
        high_issues = [i for i in issues if i["severity"] in ("critical", "high")]

        if high_issues:
            print(f"\nRuntime monitor detected {len(high_issues)} systemic issue(s):")
            for iss in high_issues:
                print(
                    f"      ⚠️ [{iss['issue_type']}] {iss['description']} "
                    f"({iss['total_occurrences']} occurrence(s), affecting "
                    f"{iss['affected_questions']}/{iss['total_questions']} question(s))"
                )
            self._runtime_issues = high_issues
            return False  # ABORT — stop for repairs

        return True  # CONTINUE

    def _step4_verify(self):
        """Execute the candidate, monitor runtime events and compare with current best."""
        print(f"\n{'─' * 60}")
        print("Step 4: Candidate verification")
        print(f"{'─' * 60}")

        if not self.evolved_bundle:
            print("   ⏭️ No candidate bundle; verification skipped")
            return

        from sandbox_evaluator import (
            _ensure_runtime_path_for_workspace,
            compare_with_reference,
            evaluate_combo_multi_video,
        )

        # ─── Loading infrastructure ───────────────────────────────────
        reference_lookup, reference_label, reference_source = self._reference_lookup()
        print(f"   Reference: {reference_label} ({reference_source})")

        # Load the training split declared by the distribution manifest.
        all_questions, question_video_ids, _split_spec = load_question_set(
            video_id="ALL",
            distribution_manifest=self.distribution_manifest,
            preferred_splits=("train",),
        )

        targeted, probe_plan = self._select_probe_questions(
            all_questions, reference_lookup
        )
        probe_plan = dict(probe_plan or {})
        probe_plan["time_reference_bounded_probe"] = True
        probe_plan["probe_execution_policy"] = (
            "per_question_time_reference_rebuild_for_all_target_modules"
        )
        probe_plan["selection_note"] = (
            str(probe_plan.get("selection_note", "")).rstrip()
            + " Probe execution is intentionally bounded: each selected "
              "question is evaluated in an isolated sandbox structure DB "
              "built only from that question's authoritative time_reference "
              "window. This keeps probe as a lightweight runnable-candidate "
              "screen instead of a full-video localization benchmark."
        ).strip()
        write_json(os.path.join(self.sandbox_dir, "probe_plan.json"), probe_plan)

        print(f"   Targeted verification: {len(targeted)} question(s)")
        print(f"      mode: {probe_plan.get('mode')}")
        print(f"      target_questions: {len(probe_plan.get('target_question_refs') or [])} refs")
        if probe_plan.get("regression_guard_refs"):
            print(f"      regression_guard: {probe_plan['regression_guard_refs']}")
        if not targeted:
            print("   ❌ No questions to verify")
            return

        # ─── Monitored sandbox execution ───────────────────────────
        results = None

        self._runtime_issues = []  # Reset runtime monitoring status

        combo_id = f"Evolved_bundle_{int(time.time())}"
        isolate_structure = True

        _ensure_runtime_path_for_workspace(self.workspace_dir)
        import module_map
        original_maps = {
            "structuring": dict(module_map.STRUCTURING_MAP),
            "thinking": dict(module_map.THINKING_MAP),
            "memory": dict(module_map.WORK_MEMORY_MAP),
            "localization": dict(module_map.LOCALIZATION_MAP),
            "perception": dict(module_map.PERCEPTION_MAP),
        }
        module_map_refs = {
            "structuring": module_map.STRUCTURING_MAP,
            "thinking": module_map.THINKING_MAP,
            "memory": module_map.WORK_MEMORY_MAP,
            "localization": module_map.LOCALIZATION_MAP,
            "perception": module_map.PERCEPTION_MAP,
        }
        try:
            agent_combo, custom_configs, reference_injection = self._inject_candidate(module_map_refs)
            structure_reuse_decision = self._structure_reuse_decision(
                reference_injection, agent_combo,
            )
            evolved_uses_structuring = structure_reuse_decision["rebuild_structure"]
            # Structuring candidates build the selected slice; other candidates
            # receive the current-best structure artifact filtered to that slice.
            time_reference_bounded_probe = (
                os.environ.get("SANDBOX_TIME_REFERENCE_BOUNDED_PROBE", "1").lower()
                in ("1", "true", "yes")
            )
            reference_structure_dir = ""
            if not evolved_uses_structuring:
                reference_structure_dir = self._reference_structure_artifact_dir()
                if not reference_structure_dir:
                    raise RuntimeError(
                        "Current-best structure_artifact directory is required "
                        "when the candidate reuses its structuring implementation."
                    )
            if isolate_structure:
                struct_name = agent_combo["video_structuring"]
                if struct_name:
                    struct_cfg = custom_configs.setdefault(struct_name, {})
                    physics = struct_cfg.setdefault("physics", {})
                    workers = int(os.environ.get("VIDEO_STRUCT_REBUILD_WORKERS", "8") or "8")
                    physics.setdefault("parallel_workers", workers)
                    physics.setdefault("parallel_batch_delay", 0.2)
                    api_tools = struct_cfg.setdefault("api_tools", {})
                    # Preserve generated class-specific prompt schemas while
                    # applying the configured VLM profile to the generic builder.
                    api_cfg = api_tools.setdefault("segment_text", {})
                    from capability_registry import resolve_profile
                    vlm_model = str(resolve_profile("vlm").get("model_id") or "")
                    api_cfg["active_model"] = vlm_model
                    api_cfg["model"] = vlm_model

            sandbox_dir = self.sandbox_dir
            print(f"\nRunning targeted verification: {combo_id} ({len(targeted)} question(s))")
            if reference_injection.get("reference_bundle_path"):
                print(f"      reference_bundle: {reference_injection['reference_bundle_path']}")

            old_env = {}
            if evolved_uses_structuring:
                sandbox_env = {
                    "SANDBOX_ALLOW_FULL_STRUCTURE_REBUILD": "1",
                    "SANDBOX_DISABLE_FULL_STRUCTURE_BUILD": "0",
                }
                if time_reference_bounded_probe:
                    sandbox_env.update({
                        "SANDBOX_STRUCTURE_TIME_REF_ONLY": "1",
                        "SANDBOX_STRUCTURE_TIME_REF_PADDING": os.environ.get(
                            "SANDBOX_STRUCTURE_TIME_REF_PADDING", "0"
                        ),
                        # A task-local media scope permits independent
                        # per-question structure builds in parallel.
                        "SANDBOX_PER_QUESTION_STRUCTURE_REBUILD": "0",
                        "SANDBOX_MERGE_TIME_REF_WINDOWS": "0",
                        "SANDBOX_QUESTION_MAX_RETRIES": "1",
                    })
            else:
                sandbox_env = {
                    "SANDBOX_REFERENCE_STRUCTURE_DIR": reference_structure_dir,
                    "SANDBOX_DISABLE_FULL_STRUCTURE_BUILD": "1",
                }
            # Apply the selected structure policy for this evaluation.
            for key, value in sandbox_env.items():
                old_env[key] = os.environ.get(key)
                os.environ[key] = value
            try:
                results = evaluate_combo_multi_video(
                    self.workspace_dir, targeted, agent_combo, combo_id,
                    sandbox_dir, custom_configs=custom_configs,
                    on_question_done=self._runtime_monitor_callback,
                    force_rebuild=evolved_uses_structuring,
                    isolate_structure=isolate_structure,
                    concurrency=self.concurrency,
                    # Non-structuring candidates must receive the immutable
                    # current-best artifact explicitly, not only through a
                    # process-global environment variable.
                    reference_structure_dir=reference_structure_dir,
                    runtime_time_reference_only=time_reference_bounded_probe,
                    # Probe questions are independent, each with its own
                    # task-local media scope and isolated structure path.
                    # Use the global scheduler so the requested budget is
                    # actual question concurrency rather than per-video
                    # batching followed by an idle tail.
                    global_question_scheduler=True,
                )
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        finally:
            # Restore MAP
            for key, orig in original_maps.items():
                map_ref = module_map_refs[key]
                map_ref.clear()
                map_ref.update(orig)

        # Runtime failures are evidence for the outer authoring workflow.
        # This evaluator never edits code or retries a modified candidate.
        self._last_probe_results = list(results or [])
        execution_metadata = {
            "sandbox_dir": sandbox_dir, "probe_plan": probe_plan,
            "combo_id": combo_id, "runtime_issues": list(self._runtime_issues),
            "target_modules": list(self.target_modules),
        }

        # ─── Phase C: Normal path — comparison results ───────────────────
        if not results:
            self.verification_report = {
                **execution_metadata,
                "status": "no_results",
                "probe_status": "invalid",
                "engineering_invalid": True,
                "execution_layer_rerun_required": True,
                "accepted": False,
            }
            return

        self.verification_report = compare_with_reference(
            results,
            reference_lookup,
            reference_label=reference_label,
            reference_source=reference_source,
        )
        self.verification_report.update(execution_metadata)
        from sandbox_evaluator import detect_runtime_engineering_issues
        probe_runtime_issues = detect_runtime_engineering_issues(results)
        if probe_runtime_issues:
            self.verification_report["runtime_engineering_issues"] = probe_runtime_issues
            self.verification_report["engineering_invalid"] = True
            self.verification_report["engineering_invalid_reason"] = (
                "probe contains runtime/API/structure engineering errors; "
                "do not count these rows as regressions"
            )
            # Provider and media-asset availability are execution-layer
            # conditions.  They must rerun the same immutable candidate after
            # the environment is restored, never be misrouted as a Codex
            # source repair merely because every result row carries an error.
            execution_layer_issue_types = {
                "structured_capability_provider_failure",
                "video_asset_missing",
                "embedding_api_failure",
                "api_auth_401",
                "api_connection_refused",
                "llm_api_connection_error",
                "api_rate_limit",
                "api_timeout",
                "embedding_endpoint_mismatch",
            }
            observed_issue_types = {
                str(item.get("issue_type") or "")
                for item in probe_runtime_issues if isinstance(item, dict)
            }
            if observed_issue_types & execution_layer_issue_types:
                self.verification_report["execution_layer_rerun_required"] = True
                self.verification_report["execution_error_code"] = (
                    "runtime_provider_or_asset_failure"
                )
        self.verification_report["combo_id"] = combo_id
        self.verification_report["target_modules"] = list(self.target_modules)
        self.verification_report["reference_label"] = reference_label
        self.verification_report["reference_source"] = reference_source
        self.verification_report["evolution_decision"] = self.evolution_decision
        self.verification_report["evolved_combo_config"] = agent_combo
        self.verification_report["structure_reuse_decision"] = structure_reuse_decision
        self.verification_report["sandbox_dir"] = sandbox_dir
        self.verification_report["probe_plan"] = probe_plan
        self.verification_report["verification_scope"] = (
            "full" if len(targeted) == len(all_questions) else "probe"
        )
        expected_probe_ids = [str(item.get("task_id") or "") for item in targeted]
        observed_probe_ids = [str(item.get("task_id") or "") for item in results]
        expected_set = {item for item in expected_probe_ids if item}
        observed_set = {item for item in observed_probe_ids if item}
        missing_probe_ids = sorted(expected_set - observed_set)
        unexpected_probe_ids = sorted(observed_set - expected_set)
        duplicate_probe_ids = sorted({
            item for item in observed_set if observed_probe_ids.count(item) > 1
        })
        probe_execution_complete = bool(
            expected_set
            and not missing_probe_ids
            and not unexpected_probe_ids
            and not duplicate_probe_ids
            and len(observed_probe_ids) == len(expected_probe_ids)
        )
        self.verification_report.update({
            "expected_total_questions": len(expected_probe_ids),
            "actual_total_questions": len(observed_probe_ids),
            "evaluation_complete": probe_execution_complete,
            "probe_execution_complete": probe_execution_complete,
            "missing_result_count": len(missing_probe_ids),
            "duplicate_result_count": len(duplicate_probe_ids),
            # IDs remain machine-only audit facts.  They are never included in
            # the Codex-safe probe feedback packet.
            "probe_execution_accounting": {
                "missing_task_ids": missing_probe_ids,
                "unexpected_task_ids": unexpected_probe_ids,
                "duplicate_task_ids": duplicate_probe_ids,
            },
        })
        if not probe_execution_complete:
            self.verification_report["comparison_valid"] = False
            if not self.verification_report.get("engineering_invalid"):
                self.verification_report["execution_layer_rerun_required"] = True
            self.verification_report["engineering_invalid"] = True
            self.verification_report["execution_error_code"] = "incomplete_or_misaligned_probe_results"

        corrections = len(self.verification_report.get("corrections", []))
        regressions = len(self.verification_report.get("regressions", []))
        correction_refs = {
            str(item.get("task_id") or "")
            for item in (self.verification_report.get("corrections") or [])
            if isinstance(item, dict)
        }
        witness_refs = set(
            probe_plan.get("selected_diagnosis_hypothesis_refs")
            or probe_plan.get("selected_mfp_witness_refs")
            or []
        )
        generalization_refs = set(probe_plan.get("selected_generalization_refs") or [])
        witness_corrections = sorted(correction_refs & witness_refs)
        generalization_corrections = sorted(correction_refs & generalization_refs)
        self.verification_report["mechanism_generalization"] = {
            "diagnosis_hypothesis_selected": sorted(witness_refs),
            "diagnosis_hypothesis_corrected": witness_corrections,
            "generalization_selected": sorted(generalization_refs),
            "generalization_corrected": generalization_corrections,
            "status": "not_configured" if not witness_refs else (
                "hypothesis_and_generalization_improved" if generalization_corrections else
                "hypothesis_observation_improved" if witness_corrections else
                "hypothesis_not_observed_improved"
            ),
        }
        if self.verification_report.get("engineering_invalid"):
            probe_status = "invalid"
        elif not self.verification_report.get("comparison_valid", True):
            probe_status = "invalid"
        else:
            probe_status = self._classify_probe_status(corrections, regressions)
        self.verification_report["probe_status"] = probe_status

        print("\nTarget verification results:")
        print(f"      Reference: {self.verification_report['reference_correct']}"
              f"/{self.verification_report['total_questions']}")
        print(f"      Evolved:  {self.verification_report['candidate_correct']}"
              f"/{self.verification_report['total_questions']}")
        print(f"Corrections: {corrections}; regressions: {regressions}")
        print(f"      Δaccuracy: {self.verification_report.get('accuracy_delta', 0):+d}")

        if probe_status == "promising":
            self.verification_report["accepted"] = False
            self.verification_report["full_verification"] = {
                "status": "deferred",
                "reason": "formal full evaluation is owned by full_eval_runner",
            }
            self._apply_final_success_criteria()
            self.verification_report["accepted"] = False
            self.verification_report["status"] = "promising_unverified"
            self.verification_report["final_success_criteria"]["passed"] = False
            self.verification_report["final_success_criteria"]["requires_full_eval"] = True
            print("   ✅ Targeted probe status: promising; awaiting formal full evaluation")
        elif probe_status == "risk_detected":
            self.verification_report["accepted"] = False
            print(
                "   ⚠️ Targeted probe status: risk_detected; "
                "this is screening feedback, not a full-evaluation verdict."
            )
        elif probe_status == "unsafe":
            self.verification_report["accepted"] = False
            print("   ❌ Targeted probe status: unsafe; a serious runtime or regression risk blocks promotion")
        elif probe_status == "mechanism_validated_but_not_generalized":
            self.verification_report["accepted"] = False
            print(
                "   ⏸️ Targeted probe status: mechanism_validated_but_not_generalized; "
                "only one minimal-failure witness improved, so cross-sample generalization is unproven"
            )
        else:
            self.verification_report["accepted"] = False
            print("   ⏸️ Targeted probe status: inconclusive_stable; no regression and no demonstrated correction")

        print("   Reporting: results recorded in the candidate evaluation report")

    def _select_probe_questions(self, all_questions: list,
                                reference_lookup: dict) -> tuple:
        """Select questions for targeted verification.

        If the diagnosis provides target_questions, prioritize them.  A
        target ref can name a concrete task_id or a video time_reference; time
        refs intentionally expand to every dataset question in that interval.
        `first_n` limits a new batch; a saved batch keeps its task IDs.
        """
        id_to_q = {q.get("task_id", ""): q for q in all_questions}
        fixed_refs = [
            str(value) for value in (getattr(self, "fixed_probe_task_ids", []) or []) if str(value)
        ]
        if fixed_refs:
            missing = [ref for ref in fixed_refs if ref not in id_to_q]
            if missing:
                raise RuntimeError(
                    "saved probe batch no longer matches the active train manifest; "
                    f"missing task_ids={missing}"
                )
            # Preserve the original order and duplicates check here instead of
            # deduplicating through the normal targeting heuristics.  A broken
            # fixed batch is an artifact error, not permission to replace it.
            if len(set(fixed_refs)) != len(fixed_refs):
                raise RuntimeError("saved probe batch contains duplicate task IDs")
            targeted = [id_to_q[ref] for ref in fixed_refs]
            return targeted, {
                **getattr(self, "fixed_probe_plan", {}),
                "mode": "fixed_probe_batch",
                "requested_budget": len(fixed_refs),
                "effective_budget": len(fixed_refs),
                "selected_refs": fixed_refs,
                "fixed_probe_task_ids": fixed_refs,
                "selection_note": (
                    "Reuse the saved task IDs in their original order for this candidate continuation."
                ),
            }
        ref_to_ids = {}
        for q in all_questions:
            task_id = q.get("task_id", "")
            time_ref = q.get("time_reference", "")
            time_key = self._probe_ref_key(time_ref)
            video_ref = f"{q.get('video_id', '')}::{time_key}" if time_key else ""
            if time_key:
                ref_to_ids.setdefault(time_key, []).append(task_id)
                compact_time_key = self._compact_probe_ref_key(time_key)
                if compact_time_key and compact_time_key != time_key:
                    ref_to_ids.setdefault(compact_time_key, []).append(task_id)
            if video_ref:
                ref_to_ids.setdefault(video_ref, []).append(task_id)
                compact_video_ref = self._compact_probe_ref_key(video_ref)
                if compact_video_ref and compact_video_ref != video_ref:
                    ref_to_ids.setdefault(compact_video_ref, []).append(task_id)
            # Diagnosis contracts intentionally use hash-free refs, while the
            # dataset task id adds a question hash.  Store one shared canonical
            # form for both so whitespace/serialization variants cannot erase a
            # valid MFP witness during probe selection.
            for canonical in {
                canonical_task_ref(task_id, video_id=str(q.get("video_id") or "")),
                canonical_task_ref(time_ref, video_id=str(q.get("video_id") or "")),
            }:
                if canonical:
                    ref_to_ids.setdefault(canonical, []).append(task_id)
        all_refs = [q.get("task_id", "") for q in all_questions]

        target_refs = self._normalize_probe_refs(self.target_questions, id_to_q, ref_to_ids)
        validation_policy = self.evaluation_policy
        policy_repair_refs = self._normalize_probe_refs(
            validation_policy.get("repair_probe", []), id_to_q, ref_to_ids
        )
        policy_guard_refs = self._normalize_probe_refs(
            validation_policy.get("regression_guard", []), id_to_q, ref_to_ids
        )
        previous_correction_refs = self._normalize_probe_refs(
            validation_policy.get("previous_candidate_corrections", []),
            id_to_q,
            ref_to_ids,
        )
        previous_regression_refs = self._normalize_probe_refs(
            validation_policy.get("previous_candidate_regressions", []),
            id_to_q,
            ref_to_ids,
        )
        if not target_refs:
            target_refs = policy_repair_refs
        # Candidate corrections are preservation guards, not repair evidence
        # for the newly selected module.  Mixing them into target_refs makes a
        # probe report claim that a regression-protection question proves
        # repair benefit.
        if not target_refs:
            diagnosis_refs = [
                d.get("task_id") or d.get("time_reference", "")
                for d in self.error_diagnoses
            ]
            target_refs = self._normalize_probe_refs(diagnosis_refs, id_to_q, ref_to_ids)
        target_refs = [ref for ref in target_refs if ref in id_to_q]
        excluded_feedback_refs = set(self._normalize_probe_refs(
            getattr(self, "probe_feedback_excluded_refs", []), id_to_q, ref_to_ids,
        ))
        if excluded_feedback_refs:
            target_refs = [ref for ref in target_refs if ref not in excluded_feedback_refs]

        fault_module_by_ref = self._diagnosis_fault_modules_by_ref(id_to_q, ref_to_ids)
        target_probe_context = self.target_probe_context
        target_refs = self._target_aware_ref_order(
            self._ordered_unique_refs(target_refs),
            fault_module_by_ref,
        )
        direct_target_refs = list(target_refs)
        reference_incorrect_refs = self._reference_incorrect_candidates(
            all_questions, reference_lookup
        )
        reference_incorrect_refs = self._target_aware_ref_order(
            reference_incorrect_refs,
            fault_module_by_ref,
        )
        reference_incorrect_refs = [
            ref for ref in reference_incorrect_refs if ref not in excluded_feedback_refs
        ]
        # A probe without current-best/reference wrong questions is only a
        # regression guard.  It can expose risk, but it cannot show repair
        # benefit.  Always give repair opportunities priority when available.
        if reference_incorrect_refs:
            target_refs = self._ordered_unique_refs(reference_incorrect_refs + target_refs)
        cluster_seed_refs, failure_clusters = self._failure_cluster_probe_refs(
            id_to_q, ref_to_ids
        )
        cluster_seed_refs = [ref for ref in cluster_seed_refs if ref not in excluded_feedback_refs]
        selected_refs = self._ordered_unique_refs(target_refs + cluster_seed_refs)
        guard_refs = self._ordered_unique_refs([
            ref for ref in (
                policy_guard_refs + previous_correction_refs + previous_regression_refs
            )
            if ref in id_to_q
        ])
        guard_refs = [ref for ref in guard_refs if ref not in excluded_feedback_refs]
        requested_budget = self.first_n if self.first_n and self.first_n > 0 else None
        baseline_guard_refs = self._regression_guard_candidates(
            all_questions, reference_lookup, selected_refs
        )
        guard_pool = self._ordered_unique_refs(guard_refs + baseline_guard_refs)
        guard_pool = [ref for ref in guard_pool if ref not in excluded_feedback_refs]
        # Select representative rows for the diagnosed hypotheses.
        machine_clusters = self.machine_probe_plan.get("diagnosis_hypothesis_witnesses") or []
        witness_clusters = []
        witness_refs = []
        for cluster in machine_clusters:
            if not isinstance(cluster, dict):
                continue
            refs = self._normalize_probe_refs(cluster.get("refs", []), id_to_q, ref_to_ids)
            raw_refs = list(cluster.get("refs", []) or [])
            if raw_refs and not refs:
                raise RuntimeError(
                    "machine contract MFP/hypothesis witness refs do not resolve "
                    "to the active train manifest; rebuild diagnosis/brief with "
                    "canonical task references instead of silently dropping them"
                )
            refs = [ref for ref in refs if ref not in excluded_feedback_refs]
            if not refs:
                continue
            seed = refs[0]
            witness_clusters.append({
                "hypothesis_id": str(cluster.get("hypothesis_id") or ""),
                "failure_kinds": list(cluster.get("failure_kinds") or []),
                "selected_ref": seed,
                "available_refs": refs,
            })
            witness_refs.append(seed)
        machine_generalization_refs = self._normalize_probe_refs(
            self.machine_probe_plan.get("generalization_probe_pool", []), id_to_q, ref_to_ids
        )
        protected_probe_refs = set(witness_refs) | excluded_feedback_refs
        overlap = set(machine_generalization_refs) & set(witness_refs)
        if overlap:
            raise RuntimeError(
                "machine contract generalization pool overlaps diagnosis witness refs; "
                "recompile diagnosis before probe rather than treating repair rows as generalization"
            )
        machine_generalization_refs = [
            ref for ref in machine_generalization_refs if ref not in protected_probe_refs
        ]
        if witness_clusters:
            # Prefer one representative row per diagnosis hypothesis when the
            # budget permits. A small probe is evidence, not a hard proof: an
            # unobserved hypothesis is recorded rather than treated as an
            # engineering failure or a reason to reject a net-positive result.
            budget = requested_budget or len(all_questions)
            reserved_guard = 1 if guard_pool else 0
            selected_refs = self._ordered_unique_refs(witness_refs[:max(0, budget - reserved_guard)])
            generalization_pool = self._ordered_unique_refs(
                machine_generalization_refs + [
                    ref for ref in reference_incorrect_refs
                    if ref not in set(direct_target_refs) and ref not in set(witness_refs)
                ]
            )
            for ref in generalization_pool:
                if len(selected_refs) >= max(0, budget - reserved_guard):
                    break
                if ref not in selected_refs:
                    selected_refs.append(ref)
            for ref in direct_target_refs + guard_pool:
                if len(selected_refs) >= budget:
                    break
                if ref not in selected_refs:
                    selected_refs.append(ref)
        elif requested_budget:
            repair_pool = self._ordered_unique_refs(selected_refs)
            repair_min = min(len(reference_incorrect_refs), max(1, requested_budget - 2))
            selected_refs = repair_pool[:requested_budget]
            if repair_min:
                selected_refs = self._ordered_unique_refs(
                    reference_incorrect_refs[:repair_min] + selected_refs
                )[:requested_budget]
            if guard_pool and not any(ref in selected_refs for ref in guard_pool):
                first_guard = guard_pool[0]
                if len(selected_refs) >= requested_budget and selected_refs:
                    selected_refs[-1] = first_guard
                    selected_refs = self._ordered_unique_refs(selected_refs)
                elif len(selected_refs) < requested_budget:
                    selected_refs.append(first_guard)
            for ref in guard_pool:
                if len(selected_refs) >= requested_budget:
                    break
                if ref not in selected_refs:
                    selected_refs.append(ref)
        else:
            for ref in guard_pool:
                if ref not in selected_refs:
                    selected_refs.append(ref)
        if not selected_refs:
            budget = requested_budget or len(all_questions)
            selected_refs = [ref for ref in all_refs if ref not in excluded_feedback_refs][:budget]

        targeted = [id_to_q[ref] for ref in selected_refs if ref in id_to_q]
        probe_plan = {
            "mode": "all_target_questions" if target_refs else "fallback_first_n",
            "requested_budget": self.first_n,
            "effective_budget": len(selected_refs),
            "repair_probe_refs": direct_target_refs if witness_clusters else target_refs,
            "reference_incorrect_repair_refs": reference_incorrect_refs,
            "module_aligned_repair_refs": [
                ref for ref in (direct_target_refs if witness_clusters else target_refs)
                if self._ref_matches_target_modules(ref, fault_module_by_ref)
            ],
            "cross_module_repair_refs": [
                {
                    "ref": ref,
                    "diagnosed_modules": sorted(fault_module_by_ref.get(ref, [])),
                }
                for ref in (direct_target_refs if witness_clusters else target_refs)
                if (
                    fault_module_by_ref.get(ref)
                    and not self._ref_matches_target_modules(ref, fault_module_by_ref)
                )
            ],
            "target_probe_status": target_probe_context.get("status", ""),
            "selected_repair_refs": [
                ref for ref in selected_refs
                if ref in (direct_target_refs if witness_clusters else target_refs)
                or ref in cluster_seed_refs
                or ref in previous_correction_refs
                or ref in reference_incorrect_refs
            ],
            "selected_repair_opportunity_refs": [
                ref for ref in selected_refs if ref in reference_incorrect_refs
            ],
            "selected_guard_refs": [ref for ref in selected_refs if ref in guard_pool],
            "regression_guard_refs": guard_refs,
            "reference_correct_guard_refs": baseline_guard_refs,
            "failure_cluster_refs": failure_clusters,
            "cluster_seed_refs": cluster_seed_refs,
            "diagnosis_hypothesis_witnesses": witness_clusters,
            "diagnosis_hypothesis_witness_refs": witness_refs,
            "selected_diagnosis_hypothesis_refs": [
                ref for ref in selected_refs if ref in set(witness_refs)
            ],
            "generalization_probe_pool": machine_generalization_refs,
            "selected_generalization_refs": [
                ref for ref in selected_refs
                if ref in set(machine_generalization_refs)
                and ref not in set(witness_refs)
            ],
            "probe_feedback_excluded_refs": sorted(excluded_feedback_refs),
            "previous_candidate_corrections": validation_policy.get(
                "previous_candidate_corrections", []),
            "previous_candidate_regressions": validation_policy.get(
                "previous_candidate_regressions", []),
            "previous_correction_probe_refs": previous_correction_refs,
            "previous_regression_guard_refs": previous_regression_refs,
            "selected_refs": selected_refs,
            "target_question_refs": direct_target_refs if witness_clusters else target_refs,
            "truncated_target_refs": (
                [ref for ref in (direct_target_refs if witness_clusters else target_refs) if ref not in selected_refs]
                if requested_budget else []
            ),
            "selection_note": (
                "Probe selection is deterministic. Machine-contract diagnosis-hypothesis "
                "representatives are preferred before broad current-best wrong questions. Remaining "
                "budget is stratified for held-out generalization opportunities and regression guards. "
                "Probe behavior, not a per-cluster hard gate, determines whether a candidate advances."
            ),
        }
        return targeted, probe_plan

    def _diagnosis_fault_modules_by_ref(self, id_to_q: dict, ref_to_ids: dict) -> dict:
        """Map concrete probe task refs to diagnosed fault modules."""
        mapping = {}
        for diag in self.error_diagnoses or []:
            if not isinstance(diag, dict):
                continue
            module = (
                diag.get("primary_fault")
                or diag.get("fault_module")
                or diag.get("improvement_target_module")
                or diag.get("target_module")
                or ""
            )
            module = self._normalize_module_name(module)
            if not module:
                continue
            raw_ref = diag.get("task_id") or diag.get("time_reference", "")
            refs = self._normalize_probe_refs([raw_ref], id_to_q, ref_to_ids)
            for ref in refs:
                mapping.setdefault(ref, set()).add(module)

        return mapping

    def _normalize_module_name(self, module: str) -> str:
        module = str(module or "").strip()
        aliases = {
            "video_structuring": "video_structuring",
            "structuring": "video_structuring",
            "memory": "memory",
            "planning": "thinking",
            "reasoning": "thinking",
        }
        module = aliases.get(module, module)
        if module in ("thinking", "memory", "video_structuring", "localization", "perception"):
            return module
        return ""

    def _ref_matches_target_modules(self, ref: str, fault_module_by_ref: dict) -> bool:
        modules = fault_module_by_ref.get(ref) or set()
        if not modules:
            return True
        return bool(set(self.target_modules) & modules)

    def _target_aware_ref_order(self, refs: list, fault_module_by_ref: dict) -> list:
        """Order refs so linked-target repair opportunities are probed first."""
        refs = self._ordered_unique_refs(refs)
        aligned, unknown, cross = [], [], []
        targets = set(self.target_modules)
        for ref in refs:
            modules = fault_module_by_ref.get(ref) or set()
            if not modules:
                unknown.append(ref)
            elif targets & modules:
                aligned.append(ref)
            else:
                cross.append(ref)
        return self._ordered_unique_refs(aligned + unknown + cross)

    def _failure_cluster_probe_refs(self, id_to_q: dict, ref_to_ids: dict) -> tuple:
        """Pick one deterministic repair probe from each diagnosed failure cluster."""
        clusters = {}
        for diag in self.error_diagnoses or []:
            if not isinstance(diag, dict):
                continue
            raw_ref = diag.get("task_id") or diag.get("time_reference", "")
            refs = self._normalize_probe_refs([raw_ref], id_to_q, ref_to_ids)
            if not refs:
                continue
            key_parts = [
                str(diag.get("fault_module") or diag.get("primary_fault") or "").strip(),
                str(diag.get("root_cause") or diag.get("error_type") or "").strip()[:80],
            ]
            key = " | ".join(part for part in key_parts if part) or "unclustered"
            clusters.setdefault(key, [])
            for ref in refs:
                if ref not in clusters[key]:
                    clusters[key].append(ref)
        cluster_seed_refs = []
        for key in sorted(clusters):
            refs = clusters[key]
            if refs:
                cluster_seed_refs.append(refs[0])
        return self._ordered_unique_refs(cluster_seed_refs), clusters

    def _normalize_probe_refs(self, refs, id_to_q: dict, ref_to_ids: dict) -> list:
        normalized = []
        for item in refs or []:
            if isinstance(item, dict):
                ref = item.get("task_id")
                if not ref:
                    ref = make_task_id(
                        item.get("video_id", ""),
                        item.get("time_reference", ""),
                        item.get("question", ""),
                    )
                candidates = [ref]
            else:
                ref = self._probe_ref_key(item)
                candidates = [ref] if ref in id_to_q else ref_to_ids.get(ref, [])
                if not candidates:
                    compact_ref = self._compact_probe_ref_key(ref)
                    if compact_ref:
                        candidates = ref_to_ids.get(compact_ref, [])
                if not candidates:
                    canonical_ref = canonical_task_ref(ref)
                    if canonical_ref:
                        candidates = ref_to_ids.get(canonical_ref, [])
            for candidate in candidates:
                if candidate and candidate in id_to_q and candidate not in normalized:
                    normalized.append(candidate)
        return normalized

    def _probe_ref_key(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)

    def _compact_probe_ref_key(self, value) -> str:
        """Canonicalize refs across diagnosis strings and dataset task_ids."""
        ref = self._probe_ref_key(value)
        if not ref:
            return ""
        parts = ref.split("::")
        if len(parts) >= 3 and re.fullmatch(r"[0-9a-fA-F]{6,}", parts[-1] or ""):
            ref = "::".join(parts[:-1])
        return re.sub(r"\s+", "", ref)

    def _ordered_unique_refs(self, refs) -> list:
        seen = set()
        ordered = []
        for ref in refs or []:
            if not ref or ref in seen:
                continue
            seen.add(ref)
            ordered.append(ref)
        return ordered

    def _reference_correct_for_question(self, question: dict,
                                       reference_lookup: dict) -> bool:
        key = question.get("question", "")
        item = reference_lookup.get(question.get("task_id", "")) or reference_lookup.get(key, {})
        return bool(item.get("is_correct"))

    def _regression_guard_candidates(self, all_questions: list,
                                     reference_lookup: dict,
                                     exclude_refs: list) -> list:
        excluded = set(exclude_refs)
        by_ref = {q.get("task_id", ""): q for q in all_questions}
        ref_to_ids = {}
        for q in all_questions:
            task_id = q.get("task_id", "")
            time_ref = q.get("time_reference", "")
            time_key = self._probe_ref_key(time_ref)
            video_ref = f"{q.get('video_id', '')}::{time_key}" if time_key else ""
            if time_key:
                ref_to_ids.setdefault(time_key, []).append(task_id)
                compact_time_key = self._compact_probe_ref_key(time_key)
                if compact_time_key and compact_time_key != time_key:
                    ref_to_ids.setdefault(compact_time_key, []).append(task_id)
            if video_ref:
                ref_to_ids.setdefault(video_ref, []).append(task_id)
                compact_video_ref = self._compact_probe_ref_key(video_ref)
                if compact_video_ref and compact_video_ref != video_ref:
                    ref_to_ids.setdefault(compact_video_ref, []).append(task_id)
        guard_refs = []

        validation_policy = self.evaluation_policy
        previous_correction_set = set(
            self._normalize_probe_refs(
                validation_policy.get("previous_candidate_corrections", []),
                by_ref,
                ref_to_ids,
            )
        )
        for ref in self._normalize_probe_refs(
            validation_policy.get("regression_guard", []),
            by_ref,
            ref_to_ids,
        ):
            q = by_ref.get(ref)
            if (
                q and ref not in excluded
                and (
                    self._reference_correct_for_question(q, reference_lookup)
                    or ref in previous_correction_set
                )
            ):
                guard_refs.append(ref)

        for ref in self._normalize_probe_refs(
            self.algo_evo.get("regression_guard_questions", []),
            by_ref,
            ref_to_ids,
        ):
            q = by_ref.get(ref)
            if q and ref not in excluded and ref not in guard_refs and self._reference_correct_for_question(q, reference_lookup):
                guard_refs.append(ref)

        audit = self.report.get("post_evolution_review_audit", {}) or {}
        for ref in self._normalize_probe_refs(
            audit.get("regressions_required_in_target_questions", []),
            by_ref,
            ref_to_ids,
        ):
            q = by_ref.get(ref)
            if q and ref not in excluded and self._reference_correct_for_question(q, reference_lookup):
                guard_refs.append(ref)

        for ref, q in by_ref.items():
            if ref in excluded or ref in guard_refs:
                continue
            if self._reference_correct_for_question(q, reference_lookup):
                guard_refs.append(ref)
        return guard_refs

    def _reference_incorrect_candidates(self, all_questions: list,
                                        reference_lookup: dict) -> list:
        """Questions where the explicit reference/current-best was wrong.

        These are the only small-sample rows that can demonstrate a repair
        opportunity.  Correct reference rows are valuable as regression guards
        but should not dominate the entire probe budget.
        """
        refs = []
        for q in all_questions or []:
            ref = q.get("task_id", "")
            if not ref:
                continue
            if not self._reference_correct_for_question(q, reference_lookup):
                refs.append(ref)
        return self._ordered_unique_refs(refs)

    @staticmethod
    def _classify_probe_status(corrections: int, regressions: int) -> str:
        if corrections >= 1 and corrections > regressions:
            return "promising"
        if regressions >= 1:
            return "risk_detected"
        return "inconclusive_stable"

    def _apply_final_success_criteria(self):
        """Apply the shared net-positive effectiveness policy."""
        if not self.verification_report:
            return
        from sandbox_evaluator import apply_effectiveness_policy
        full_v = self.verification_report.get("full_verification", {}) or {}
        expected = (
            full_v.get("full_total_questions")
            or self.verification_report.get("expected_total_questions")
            or self.verification_report.get("total_questions")
            or 0
        )
        scope = self.verification_report.get("verification_scope", "")
        require_complete = scope == "full"
        apply_effectiveness_policy(
            self.verification_report,
            expected_total_questions=expected,
            require_complete=require_complete,
            scope=scope or "probe",
        )

    # ==================================================================
    # Step 5: Final conclusion
    # ==================================================================

    def _build_conclusion(self) -> dict:
        """Construct the final conclusion (including feedback information for closed-loop reuse)"""
        conclusion = {
            "timestamp": int(time.time()),
            "target_modules": list(self.target_modules),
            "observed_distribution_profile": self.observed_distribution_profile,
            "candidate_evolution": {
                "success": self.evolved_bundle is not None,
                "changed_modules": list((self.evolved_bundle or {}).get("changed_modules") or []),
            },
            "verification": self.verification_report or {"status": "not_run"},
            "dry_run": self.dry_run,
            # Feedback information: for closed-loop reuse of diagnostic module
            "feedback": {
                "evolution_hint_used": self.evolution_hint,
                "verification_trajectories": self._collect_verification_trajectories(),
            },
        }

        # final judgment
        v = self.verification_report or {}
        v_status = v.get("status", "")

        if self.dry_run:
            conclusion["verdict"] = "dry_run"
            conclusion["verdict_reason"] = "Dry-run mode; no candidate execution was performed."

        elif v_status == "engineering_abort":
            conclusion["verdict"] = "engineering_blocked"
            runtime_issues = v.get("runtime_issues", [])
            issue_names = ", ".join(i.get("issue_type", "?") for i in runtime_issues)
            conclusion["verdict_reason"] = (
                f"Runtime issues blocked verification: {issue_names}."
            )

        elif self.evolved_bundle and not self.verification_report:
            conclusion["verdict"] = "unverified"
            conclusion["verdict_reason"] = "A candidate was generated but has not been verified."

        elif (
            v.get("probe_status") == "promising"
            and v.get("evaluation_complete") is True
            and not v.get("engineering_invalid")
            and (v.get("full_verification") or {}).get("status") == "deferred"
        ):
            # A bounded probe can show a net-positive, complete engineering
            # result while deliberately withholding full evaluation.  This is
            # neither acceptance nor rejection: calling it ``rejected`` loses
            # the outcome semantics and can misroute the next evolution step.
            conclusion["verdict"] = "probe_promising_pending_full_eval"
            conclusion["verdict_reason"] = (
                "The probe completed with a positive signal "
                f"(Δ={v.get('accuracy_delta', 0):+d}), but full evaluation has not run; "
                "the candidate remains pending."
            )

        elif v and not v.get("accepted"):
            full_v = v.get("full_verification", {})
            probe_status = v.get("probe_status", "")
            if probe_status == "inconclusive_stable":
                conclusion["verdict"] = "inconclusive_stable"
                conclusion["verdict_reason"] = (
                    "The targeted probe was stable but demonstrated no correction "
                    f"(Δ={v.get('accuracy_delta', 0):+d})."
                )
            elif probe_status == "risk_detected":
                regs = len(v.get("regressions", []))
                conclusion["verdict"] = "risk_detected"
                conclusion["verdict_reason"] = (
                    f"The targeted probe detected {regs} regression(s). This screening result "
                    "is feedback evidence, not a full-evaluation verdict."
                )
            elif probe_status == "unsafe":
                regs = len(v.get("regressions", []))
                conclusion["verdict"] = "rejected"
                conclusion["verdict_reason"] = (
                    f"The targeted probe found serious runtime risk and {regs} regression(s); the candidate is unsafe."
                )
            elif probe_status == "invalid" or full_v.get("status") == "invalid":
                conclusion["verdict"] = "rejected"
                conclusion["verdict_reason"] = (
                    "Verification failed or produced incomplete results: "
                    f"{full_v.get('reason', v_status or 'invalid')}"
                )
            elif full_v.get("status") in ("regressed", "net_positive_with_regressions"):
                regs = full_v.get("remaining_regressions", 0)
                conclusion["verdict"] = "rejected"
                conclusion["verdict_reason"] = (
                    f"Full verification did not meet acceptance criteria: {regs} remaining regression(s), "
                    f"final Δ={v.get('accuracy_delta', 0):+d}."
                )
            else:
                conclusion["verdict"] = "rejected"
                conclusion["verdict_reason"] = (
                    f"The candidate did not meet acceptance criteria (Δ={v.get('accuracy_delta', 0):+d})."
                )

        else:
            conclusion["verdict"] = "failed"
            conclusion["verdict_reason"] = "Evolution process failed"

        return conclusion


    def _collect_verification_trajectories(self) -> list:
        """Collect trajectories generated during Step 4 verification (for feedback closed-loop reuse)"""
        trajectories = []
        if not self.verification_report:
            return trajectories

        # Read the verification trace from the current round's sandbox.
        sandbox = self.verification_report.get("sandbox_dir")
        if not sandbox:
            return trajectories
        if not os.path.isdir(sandbox):
            return trajectories

        for root, _dirs, files in os.walk(sandbox):
            for f in sorted(files):
                if not f.endswith("_sandbox.jsonl"):
                    continue
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, "r", encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                entry = json.loads(line)
                                question = entry.get("question", "")
                                time_reference = entry.get("time_reference", "")
                                ans = (
                                    entry.get("answer")
                                    or entry.get("candidate_answer")
                                    or ""
                                )
                                gt = extract_gold_answer(entry)
                                from sandbox_evaluator import _rule_based_judge
                                is_correct = entry.get("is_correct")
                                if not isinstance(is_correct, bool):
                                    is_correct = _rule_based_judge(ans, entry)
                                trajectory = entry.get("trajectory") or []
                                num_steps = (
                                    len(trajectory) if isinstance(trajectory, list)
                                    else int(entry.get("steps") or 0)
                                )
                                trajectories.append({
                                    "task_id": entry.get("task_id", ""),
                                    "question": str(question)[:80],
                                    "time_reference": time_reference,
                                    "pred_answer": ans[:60],
                                    "gt_answer": gt,
                                    "is_correct": is_correct,
                                    "num_steps": num_steps,
                                    "architecture": entry.get("combo") or entry.get("architecture_combo", {}),
                                    "source_result_path": filepath,
                                })
                except (json.JSONDecodeError, OSError):
                    continue

        # Global-scheduler probes write aggregate per-video files and isolated
        # ``question_runs`` files.  They represent the same task outcomes;
        # deduplicate by stable task id before publishing feedback.
        deduplicated = {}
        for item in trajectories:
            key = str(item.get("task_id") or "")
            if not key:
                key = json.dumps(
                    [item.get("question", ""), item.get("time_reference", "")],
                    ensure_ascii=False, sort_keys=True,
                )
            existing = deduplicated.get(key)
            if existing is None or "/question_runs/" in str(item.get("source_result_path", "")):
                deduplicated[key] = item
        return list(deduplicated.values())
