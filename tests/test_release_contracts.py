from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evolution.coding_agent_runtime import (
    CODING_AGENT_PROTOCOL,
    coding_agent_environment,
    resolve_coding_agent_executable,
)
from evolution.video_distribution_observer import train_video_queries_from_manifest
from execution import run_evolution
from execution.action_runtime.evidence_contract import (
    canonicalize_structure_input,
    validate_canonical_structure_record,
)
from execution.paths import output_root
from execution.run_journal import _safe_args, _safe_error, append_operation


def test_run_journal_redacts_credentials() -> None:
    safe = _safe_args(
        argparse.Namespace(
            api_key="example-api-key",
            access_token="example-access-token",
            base_url="https://private-provider.example/signed?token=secret",
            workspace="workspace",
        )
    )

    assert safe["api_key"] == "<redacted>"
    assert safe["access_token"] == "<redacted>"
    assert safe["base_url"] == "<redacted>"
    assert safe["workspace"] == "workspace"

    persisted_error = _safe_error(
        "request to https://private-provider.example/signed?token=secret "
        "failed with example-api-key",
        argparse.Namespace(
            api_key="example-api-key",
            base_url="https://private-provider.example/signed?token=secret",
        ),
    )
    assert "example-api-key" not in persisted_error
    assert "private-provider.example" not in persisted_error


def test_run_journal_uses_canonical_output_root(
    tmp_path: Path, monkeypatch
) -> None:
    configured = tmp_path / "configured-output"
    monkeypatch.setenv("METAVIDEOAGENT_OUTPUT_ROOT", str(configured))
    path = append_operation(
        output_root="",
        run_id="run-1",
        stage="check",
        args=argparse.Namespace(),
        started_at=0.0,
        status="checked",
    )
    assert path == str(configured / "run-1" / "operations.jsonl")


def test_coding_agent_adapter_is_single_protocol() -> None:
    assert CODING_AGENT_PROTOCOL == "codex_exec_compatible"
    assert resolve_coding_agent_executable("/bin/true") == "/bin/true"


def test_coding_agent_environment_excludes_runtime_provider_secrets(
    monkeypatch,
) -> None:
    monkeypatch.setenv("METAVIDEOAGENT_API_KEY", "runtime-secret")
    monkeypatch.setenv("METAVIDEOAGENT_BASE_URL", "https://private-runtime.example")
    monkeypatch.setenv("CODEX_HOME", "/tmp/codex-home")
    env = coding_agent_environment()
    assert "METAVIDEOAGENT_API_KEY" not in env
    assert "METAVIDEOAGENT_BASE_URL" not in env
    assert env["CODEX_HOME"] == "/tmp/codex-home"


def test_codex_invocations_keep_workspace_write_sandbox(
    tmp_path: Path,
) -> None:
    codex_evolve = importlib.import_module("codex_evolve")
    deep_research_runner = importlib.import_module("deep_research_runner")

    try:
        codex_evolve.run_codex("test prompt", "/bin/true", 30)
    except ValueError as exc:
        assert "explicit isolated working directory" in str(exc)
    else:
        raise AssertionError("run_codex accepted an implicit project-root workspace")

    completed = SimpleNamespace(returncode=1, stdout="", stderr="expected test failure")
    with patch.object(codex_evolve.subprocess, "run", return_value=completed) as run_call:
        codex_evolve.run_codex(
            "test prompt",
            "/bin/true",
            30,
            workdir=str(tmp_path),
        )
    command = run_call.call_args.args[0]
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert "--dangerously-bypass-approvals-and-sandbox" not in command

    research_dir = tmp_path / "research"
    research_dir.mkdir()
    with patch.object(
        deep_research_runner.subprocess,
        "run",
        return_value=completed,
    ) as research_call:
        deep_research_runner._run_codex_research(
            {},
            {},
            {},
            [],
            output_dir=str(research_dir),
            codex_cli="/bin/true",
            min_sources=0,
        )
    research_command = research_call.call_args.args[0]
    assert research_command[research_command.index("--sandbox") + 1] == "workspace-write"
    assert "--dangerously-bypass-approvals-and-sandbox" not in research_command
    assert research_call.call_args.kwargs["cwd"] == str(research_dir)


def test_default_output_root_is_outside_the_installed_package(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert output_root() == str(tmp_path / "metavideoagent_outputs")


def test_flat_time_reference_pair_is_one_smoke_interval() -> None:
    from evolution.smoke_test_runner import _single_time_reference_bounds

    assert _single_time_reference_bounds([0.0, 30.0]) == (0.0, 30.0)
    assert _single_time_reference_bounds([[0.0, 30.0]]) == (0.0, 30.0)


def test_full_eval_keeps_provider_configuration_out_of_argv(
    tmp_path: Path, capsys
) -> None:
    candidate_bundle = tmp_path / "candidate_bundle.json"
    candidate_bundle.write_text(
        json.dumps({"changed_modules": []}),
        encoding="utf-8",
    )
    secret = "example-api-key"
    base_url = "https://model-provider.example/v1"
    args = SimpleNamespace(
        candidate_run=str(tmp_path / "candidate_run"),
        candidate_bundle=str(candidate_bundle),
        workspace=str(tmp_path / "workspace"),
        distribution_manifest=str(tmp_path / "manifest.json"),
        reference_label="current_best",
        concurrency=1,
        exec_llm_model="test-model",
        structure_workers=1,
        run_id="test-run",
        output_root=str(tmp_path / "outputs"),
        api_key=secret,
        base_url=base_url,
        initial_baseline_bundle=str(candidate_bundle),
        rebuild_structure=False,
        profile="",
        diagnosis_report="",
        overwrite=False,
        skip_post_eval_decision=True,
        check_only=True,
        codex_runtime_settings={},
        iteration_index=0,
        vlm_model="",
        asr_model="",
        probe_repair_attempts=None,
    )

    with patch.object(
        run_evolution.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0),
    ) as subprocess_run:
        assert run_evolution._run_full_eval_subprocess(args, "reference.json") == 0

    command = subprocess_run.call_args.args[0]
    child_env = subprocess_run.call_args.kwargs["env"]
    captured = capsys.readouterr().out
    assert secret not in command
    assert base_url not in command
    assert secret not in captured
    assert base_url not in captured
    assert child_env["METAVIDEOAGENT_API_KEY"] == secret
    assert child_env["METAVIDEOAGENT_BASE_URL"] == base_url


def test_observer_normalizes_options_to_choices(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps(
            {
                "video_id": "video-1",
                "question": "What is visible?",
                "options": ["A", "B"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"splits": {"train": "train.jsonl"}}),
        encoding="utf-8",
    )

    video_ids, queries = train_video_queries_from_manifest(str(manifest))

    assert video_ids == ["video-1"]
    assert queries["video-1"] == [
        {"question": "What is visible?", "choices": ["A", "B"]}
    ]


def test_automatic_terminal_state_never_reports_partial_success() -> None:
    failed = run_evolution._automatic_outcome(
        [{"status": "codex_failed"}], 4, check_only=False
    )
    incomplete = run_evolution._automatic_outcome(
        [{"status": "complete"}], 4, check_only=False
    )
    complete = run_evolution._automatic_outcome(
        [{"status": "complete"}] * 4, 4, check_only=False
    )

    assert (failed["automatic_outcome"], failed["automatic_exit_code"]) == (
        "failed",
        1,
    )
    assert (incomplete["automatic_outcome"], incomplete["automatic_exit_code"]) == (
        "incomplete",
        2,
    )
    assert (complete["automatic_outcome"], complete["automatic_exit_code"]) == (
        "completed",
        0,
    )


def test_evolution_accepts_single_or_multiple_target_modules(tmp_path: Path) -> None:
    """One adaptive bundle route must cover both local and joint changes."""
    from evolution.candidate_evaluator import CandidateEvaluator
    from evolution.bundle_contract import (
        MODULE_TYPES,
        make_candidate_bundle,
        validate_candidate_bundle,
    )
    from evolution.deep_research_runner import _target_modules
    from evolution.diagnosis_execution_brief import (
        build_bundle_execution_brief,
        build_bundle_machine_evaluation_contract,
        validate_bundle_execution_brief,
        validate_bundle_machine_evaluation_contract,
    )

    for targets in (["thinking"], ["memory", "thinking"]):
        diagnosis = {
            "evolution_design": {
                "target_modules": targets,
                "initial_focus_modules": targets,
                "design_summary": "Improve the evidence-based decision mechanism.",
                "failure_chain": "retained context -> target behavior -> final decision",
                "evolution_mechanism": "Use bounded evidence and preserve the module protocol.",
                "handoff_contracts": [{
                    "producer": "memory",
                    "consumer": "thinking",
                    "required_fields": ["context"],
                    "output_protocol": "structured context packet",
                    "consumer_behavior": "consume the packet without dropping evidence",
                    "fallback": "use an explicit limited-context packet",
                }],
            },
            "evolution_decision": {
                "combo_base": "current_best",
                "base_combo_policy": "keep_current_best",
            },
        }
        brief = build_bundle_execution_brief(diagnosis, diagnosis_path="<test>")
        contract = build_bundle_machine_evaluation_contract(diagnosis, brief=brief)

        assert validate_bundle_execution_brief(brief) == []
        assert validate_bundle_machine_evaluation_contract(contract) == []
        assert _target_modules(brief) == targets
        assert run_evolution._brief_target_modules(brief) == targets

        modules = {
            module: {
                "module_type": module,
                "name": f"Test{module.title().replace('_', '')}",
                "source_file": f"{module}_agent.py",
                "code": f"class Test{module.title().replace('_', '')}:\n    pass\n",
            }
            for module in MODULE_TYPES
        }
        combo = {module: modules[module]["name"] for module in MODULE_TYPES}
        candidate = make_candidate_bundle(
            combo=combo,
            modules=modules,
            changed_modules=targets,
            handoff_contracts=brief["implementation_task"]["handoff_contracts"],
            implementation_spec={
                "contract_mode": "adaptive_bundle",
                "module_protocol_version": "metavideoagent_module_protocol",
                "module_responsibilities": brief["implementation_task"]["module_responsibilities"],
            },
        )
        assert validate_candidate_bundle(candidate) == []

        evaluator_input = dict(brief)
        evaluator_input["_machine_evaluation_contract"] = contract
        evaluator_input["reference_results_path"] = "<test-reference>"
        evaluator = CandidateEvaluator(
            evaluator_input,
            str(tmp_path / ("-".join(targets))),
            dry_run=True,
        )
        assert evaluator.target_modules == targets


def test_canonical_structure_contract_rejects_incomplete_handoffs() -> None:
    from evolution.sandbox_evaluator import (
        _candidate_consumable_structure_text,
        _summarize_smoke_structure_artifacts,
    )

    canonical = canonicalize_structure_input({
        "start_sec": 1.0,
        "end_sec": 3.0,
        "multimodal_narration": "A person handles an object.",
    })
    assert validate_canonical_structure_record(canonical) == []

    direct_only = {
        "start_sec": 1.0,
        "end_sec": 3.0,
        "multimodal_narration": "A person handles an object.",
        "retrieval_document": "A person handles an object.",
    }
    issues = validate_canonical_structure_record(direct_only)
    assert "canonical_evidence is missing" in issues
    assert _candidate_consumable_structure_text(direct_only) == ""
    summary = _summarize_smoke_structure_artifacts([], [direct_only], [])
    assert summary["canonical_record_missing_count"] == 1
    assert summary["canonical_contract_issues"] == {
        "canonical_evidence is missing": 1,
    }
