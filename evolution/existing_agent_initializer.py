"""Source-preserving capability mapping into the common initial-agent pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from initial_baseline_agent import (
    _normalize_codegen_bundle,
    _render_embedded_artifacts,
    _run_codegen_command,
    build_initial_codegen_context,
    write_json,
)


def prepare_source_agent(source: str, output_dir: str) -> Path:
    """Snapshot the curated source package once for a resumable initialization."""
    source = Path(source).resolve()
    destination = Path(output_dir, "initialization", "original_agent")
    if not destination.exists():
        if not source.is_dir():
            raise ValueError("--source-agent must point to a directory of agent source and required resources")
        if destination.resolve().is_relative_to(source):
            raise ValueError("Keep initialization outputs outside the source-agent directory.")
        excluded = shutil.ignore_patterns(
            ".git", ".env", ".env.*", "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache",
            ".venv", "venv", "*.pem", "*.key", "AGENTS.md",
        )

        def ignore(directory, names):
            return set(excluded(directory, names)) | {name for name in names if Path(directory, name).is_symlink()}

        shutil.copytree(source, destination, ignore=ignore)
    return destination


def build_mapped_initial_bundle(*, source_agent: str, workspace: str,
                                 distribution_manifest: str, output_dir: str,
                                 codex_cli: str = "", timeout: int = 900,
                                 execute_codegen: bool = True) -> tuple[dict, str]:
    root = Path(output_dir).resolve()
    source = prepare_source_agent(source_agent, str(root))
    context = build_initial_codegen_context({}, {}, "", str(root))
    runtime_ids = {
        "five_module_interface_contract", "generated_runtime_abi_witness", "runtime_module_protocol",
        "runtime_capability_manual", "runtime_capability_profiles",
    }
    artifacts = [item for item in context["artifacts"] if item["artifact_id"] in runtime_ids]
    output_path = str(root / "initial_baseline_bundle.json")
    instructions = Path(__file__).with_name("agent_docs").joinpath("map_existing_agent_prompt.md").read_text(encoding="utf-8")
    artifacts.insert(0, {
        "artifact_id": "existing_agent_mapping_rules", "title": "Source-agent mapping rules",
        "purpose": "Preserve the original algorithm while adapting its MVA interfaces.",
        "content_kind": "text", "content": instructions,
    })
    manifest_path = str(root / "agent.json")
    prompt = f"requested_output_path: {manifest_path}\n\n" + _render_embedded_artifacts(artifacts)
    write_json(str(root / "initialization/context.json"), {
        "artifact_type": "initial_codegen_context", "artifacts": artifacts,
        "initialization": "map", "source_snapshot": str(source),
    })
    if not execute_codegen:
        return {}, prompt
    result = _run_codegen_command(
        prompt, output_path=manifest_path, iteration_dir=output_dir, codex_cli=codex_cli, timeout=timeout,
        artifact_label="initial_agent_mapping", source_agent_dir=str(source),
    )
    bundle = _normalize_codegen_bundle(
        result.get("bundle") or {}, workspace=workspace,
        distribution_manifest=distribution_manifest,
        generation_mode="existing_agent_mapping", codegen_result=result,
    )
    bundle["initialization"] = "map"
    bundle["codegen"]["context_path"] = str(root / "initialization/context.json")
    bundle["codegen"]["iteration_dir"] = str(root.resolve())
    bundle["runtime_policy"] = {"capability_selection": "source_agent_registered_profiles"}
    write_json(output_path, bundle)
    if result.get("returncode") != 0:
        raise RuntimeError("The mapping command failed; inspect the initial_agent_mapping execution artifacts before retrying.")
    # Shape and real runtime errors enter the same smoke/repair loop as a
    # generated initial agent. No score-based claim of equivalence is made here.
    return bundle, prompt


def write_mapping_plan(*, source_agent: str, output_dir: str, output_path: str) -> dict:
    source = prepare_source_agent(source_agent, output_dir)
    plan = {
        "initialization": "map", "source_snapshot": str(source),
        "next_task": "Inspect the source execution path and map its existing capabilities to the five MVA interfaces.",
    }
    Path(output_path).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan
