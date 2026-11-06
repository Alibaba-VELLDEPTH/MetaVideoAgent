"""Initial bundle generation for MetaVideoAgent.

This module creates the first five-module bundle for a MetaVideoAgent run.
It does not execute the bundle; execution and injection belong to the initial
combo runner and sandbox layer. Codex performs automatic authoring and bounded
repair, and every generated bundle passes through the same runtime gates.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
METAVIDEOAGENT_RUNTIME_DIR = os.path.join(PROJECT_ROOT, "execution", "action_runtime")
if os.path.isdir(METAVIDEOAGENT_RUNTIME_DIR) and METAVIDEOAGENT_RUNTIME_DIR not in sys.path:
    sys.path.insert(0, METAVIDEOAGENT_RUNTIME_DIR)

from agent_sources import collect_agent_source, copy_agent_source  # noqa: E402
from authoring_workspace import create_authoring_workspace  # noqa: E402
from capability_registry import (  # noqa: E402
    DEFAULT_PROFILE_IDS,
    PROFILE_MANUAL_PATH,
)
from codex_runtime_settings import (  # noqa: E402
    codex_exec_model_args,
    resolve_codex_runtime_settings,
)
from coding_agent_runtime import (  # noqa: E402
    CODING_AGENT_PROTOCOL,
    coding_agent_environment,
    resolve_coding_agent_executable,
)
from distribution_spec import load_distribution_spec, prompt_safe_spec  # noqa: E402
from iteration_workspace import IterationWorkspace  # noqa: E402

MODULE_ORDER = [
    "video_structuring",
    "localization",
    "perception",
    "memory",
    "thinking",
]

SOURCE_FILES = {
    module: f"agent/{module}.py" for module in MODULE_ORDER
}
INITIAL_VLM_PROFILE = DEFAULT_PROFILE_IDS["vlm"]
INITIAL_ASR_PROFILE = DEFAULT_PROFILE_IDS["asr"]


def read_json(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_prompt() -> str:
    path = os.path.join(CURRENT_DIR, "agent_docs", "initial_baseline_prompt.md")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _channel_hypothesis(profile: Dict[str, Any]) -> Dict[str, Any]:
    value = profile.get("information_channel_hypothesis") or {}
    return value if isinstance(value, dict) else {}


def observed_channel_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    hypothesis = _channel_hypothesis(profile)
    raw = profile.get("raw_video_signal_profile") or {}
    schema = profile.get("information_channel_schema") or {}
    semantic = profile.get("semantic_channel_observation_profile") or {}
    return {
        "primary_observed_channel": hypothesis.get("primary_observed_channel", "mixed_or_unknown"),
        "secondary_observed_channels": hypothesis.get("secondary_observed_channels", []),
        "channel_scores": hypothesis.get("channel_scores", {}),
        "information_channel_schema": schema,
        "raw_video_summary": raw.get("summary", {}) if isinstance(raw, dict) else {},
        "semantic_observation_summary": {
            "sampled_observations": semantic.get("sampled_observations", 0),
            "source_available": semantic.get("source_available", False),
            "density": semantic.get("density", {}),
        },
    }


def prompt_safe_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    summary = observed_channel_summary(profile)
    channel_summary = dict(profile.get("channel_profile_summary") or summary)
    for key in list(channel_summary):
        if key == "profile_path" or key.endswith("_path"):
            channel_summary.pop(key, None)
    sampling = profile.get("sampling") if isinstance(profile.get("sampling"), dict) else {}
    safe_sampling = {
        "video_ids_seen": sampling.get("video_ids_seen"),
        "video_split_source": sampling.get("video_split_source"),
        "video_ids_profiled": sampling.get("video_ids_profiled"),
        "question_rows_profiled": sampling.get("question_rows_profiled"),
        "raw_videos_profiled": sampling.get("raw_videos_profiled"),
        "unique_videos_profiled": sampling.get("unique_videos_profiled"),
        "supervision_used": sampling.get("supervision_used"),
        "forbidden_inputs": sampling.get("forbidden_inputs", []),
    }
    return {
        "profile_source": "prompt_safe_five_frame_query_aware_distribution",
        "channel_profile_summary": channel_summary,
        "observed_channel_summary": summary,
        "distribution_records": profile.get("distribution_records", []),
        "sampling": safe_sampling,
        "leakage_policy": profile.get("leakage_policy", ""),
    }






def _context_artifact(artifact_id: str, title: str, purpose: str, content: str,
                      content_kind: str) -> Dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "title": title,
        "purpose": purpose,
        "content_kind": content_kind,
        "content": content,
    }


def _context_json_artifact(artifact_id: str, title: str, purpose: str, value: Any) -> Dict[str, Any]:
    return _context_artifact(
        artifact_id, title, purpose, json.dumps(value, ensure_ascii=False, indent=2), "json",
    )


def _read_optional_context_artifact(artifact_id: str, title: str, purpose: str,
                                    path: str) -> Dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    return _context_artifact(artifact_id, title, purpose, content, "text")


def _read_context_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _render_embedded_artifacts(artifacts: List[Dict[str, Any]]) -> str:
    rendered = []
    for artifact in artifacts:
        artifact_id = str(artifact.get("artifact_id") or "unnamed_artifact")
        rendered.append(
            f"### {artifact_id}: {artifact.get('title', '')}\n"
            f"Purpose: {artifact.get('purpose', '')}\n"
            f"Content type: {artifact.get('content_kind', '')}\n"
            f"<<<BEGIN {artifact_id}>>>\n{artifact.get('content', '')}\n<<<END {artifact_id}>>>"
        )
    return "\n\n".join(rendered)


def build_initial_codegen_context(profile: Dict[str, Any], manifest_spec: Dict[str, Any],
                                  deep_research_summary: str, output_dir: str) -> Dict[str, Any]:
    """Gather complete initialization inputs for codegen and repair."""
    docs = os.path.join(CURRENT_DIR, "agent_docs")
    artifacts: List[Dict[str, Any]] = [
        _context_artifact(
            "initial_codegen_rules", "Initial codegen rules",
            "Distribution observations, module-design requirements and bundle output fields.",
            load_prompt(),
            "text",
        ),
        _context_json_artifact(
            "initial_execution_policy", "Initial execution policy",
            "Initial capability defaults and execution-owned build lifecycle.",
            {
                "execution_llm_profile_id": DEFAULT_PROFILE_IDS["llm"],
                "asr_profile_id": INITIAL_ASR_PROFILE,
                "vlm_profile_id": INITIAL_VLM_PROFILE,
                "structure_build_lifecycle": (
                    "The execution-layer structure builder owns bounded media sampling and model calls. "
                    "The generated video_structuring module persists, indexes, and retrieves supplied records; "
                    "its addStructure must not make another model call. An optional buildStructure(env) "
                    "hook may own a custom bounded build using registered adapters; it must persist canonical records."
                ),
            },
        ),
        _context_artifact(
            "five_module_interface_contract", "Five-module interface contract",
            "Responsibilities and handoff contracts for the five modules.",
            _read_context_file(os.path.join(docs, "module_interface_contract.md")), "text",
        ),
        _context_artifact(
            "generated_runtime_abi_witness", "Generated runtime ABI witness",
            "Exact base-method signatures, return shapes and minimal calls.",
            _read_context_file(os.path.join(docs, "generated_runtime_abi_witness.json")), "json",
        ),
        _context_artifact(
            "runtime_module_protocol", "Execution-layer module protocol",
            "Actual neutral runtime protocol that generated modules must obey.",
            _read_context_file(os.path.join(PROJECT_ROOT, "execution", "action_runtime", "module_protocol.py")), "text",
        ),
        _context_artifact(
            "runtime_capability_manual", "Runtime capability manual",
            "Model/media adapter APIs and their failure semantics.",
            _read_context_file(os.path.join(docs, "runtime_capability_manual.md")), "text",
        ),
        _context_artifact(
            "runtime_capability_profiles", "Registered runtime capability profiles",
            "Registered profiles, transports, provenance and selected model defaults.",
            _read_context_file(str(PROFILE_MANUAL_PATH)), "json",
        ),
        _context_json_artifact(
            "prompt_safe_distribution_manifest", "Prompt-safe distribution manifest",
            "Dataset split/distribution specification available to initial codegen.",
            prompt_safe_spec(manifest_spec),
        ),
        _context_json_artifact(
            "distribution_aware_profile", "Distribution-aware initialization profile",
            "Five-frame per-video observations with associated questions/options and no answer or evidence supervision.",
            prompt_safe_profile(profile),
        ),
    ]
    if deep_research_summary:
        artifacts.append(_context_artifact(
            "deep_research_summary", "Deep-research summary",
            "Research summary supplied to initial codegen.", deep_research_summary, "text",
        ))
    plan = _read_optional_context_artifact(
        "initial_combo_plan", "Initial combo plan",
        "Run-specific initial design/assembly plan.", os.path.join(output_dir, "initialization", "plan.json"),
    )
    if plan:
        artifacts.append(plan)
    return {
        "artifact_type": "initial_codegen_context", "schema_version": 1,
        "purpose": "distribution observations, research and runtime interfaces for initial authoring and repair",
        "artifacts": artifacts,
    }


def build_prompt(codegen_context: Dict[str, Any], output_path: str = "") -> str:
    """Render the first-codegen prompt from the canonical shared context."""
    artifacts = codegen_context.get("artifacts") if isinstance(codegen_context.get("artifacts"), list) else []
    return (
        "Author the initial executable five-module MetaVideoAgent from the supplied distribution observations and research. "
        "Use the embedded module responsibilities, ABI witness, return protocol and configured capability profiles. "
        "The workspace contains the execution SDK for inspection, offline imports and focused interface checks. "
        "Write complete source and resource files under agent/ and select their entrypoints in agent.json. "
        "Record the design and checks in implementation.md. The host then runs runtime preflight and smoke.\n\n"
        f"requested_output_path: {output_path}\n"
        "## Embedded Initial Codegen Context\n\n"
        f"{_render_embedded_artifacts(artifacts)}\n"
    )


def validate_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Validate canonical bundle shape without mutating Codex output.

    The single real compile/inject/construct preflight is executed by the
    initial smoke runner, where any failure is persisted as repair feedback.
    Keeping this function manifest-only avoids executing every generated bundle
    twice before its first real smoke.
    """
    try:
        from sandbox_evaluator import validate_bundle_assembly
        validation = validate_bundle_assembly(bundle)
    except Exception as exc:
        return {
            "passed": False,
            "validation_mode": "initial_bundle_manifest",
            "issues": [f"bundle manifest validation could not start: {type(exc).__name__}: {exc}"],
        }
    issues = list(validation.get("issues") or [])
    if bundle.get("artifact_type") != "metavideoagent_bundle":
        issues.append("artifact_type must be 'metavideoagent_bundle'")
    if bundle.get("schema_version") != 1:
        issues.append("bundle must use the current schema")
    if str(bundle.get("mode") or "") != "initial_baseline":
        issues.append("mode must be 'initial_baseline'")
    for field in ("design_answers", "observed_channel_summary", "deep_research_summary"):
        if field not in bundle:
            issues.append(f"initial bundle is missing required field: {field}")
    validation["issues"] = issues
    validation["passed"] = bool(validation.get("passed")) and not issues
    return validation


def _create_codegen_staging_workspace(parent_dir: str, artifact_label: str) -> str:
    """Create the SDK workspace for a from-scratch initial bundle."""
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact_label or "codegen")
    return create_authoring_workspace(parent_dir, f".codex_stage_{safe_label}_", initial=True)


def _run_codegen_command(prompt: str,
                         *,
                         output_path: str,
                         iteration_dir: str,
                         codex_cli: str = "",
                         timeout: int = 900,
                         artifact_label: str = "initial_baseline_codegen",
                         source_agent_dir: str = "") -> Dict[str, Any]:
    """Run the workflow's coding-agent adapter in an isolated workspace."""
    settings = resolve_codex_runtime_settings()
    iteration = IterationWorkspace(iteration_dir)
    attempt = iteration.reserve("codegen")
    context = iteration.context("codegen")
    if context["stage_conclusions"] or context["pending_task"]:
        prompt += "\n\n## Current iteration work\n" + json.dumps(context, ensure_ascii=False, indent=2)
    executable = resolve_coding_agent_executable(codex_cli)
    prompt_template_path = str(attempt / "prompt_template.txt")
    write_text(prompt_template_path, prompt)
    stage = _create_codegen_staging_workspace(str(attempt), artifact_label)
    copy_agent_source(iteration.root, stage)
    staged_output = os.path.join(stage, "agent.json")
    staged_last_message = os.path.join(stage, "last_message.json")
    prompt_for_codex = prompt.replace(os.path.abspath(output_path), staged_output)
    prompt_path = str(attempt / "prompt.txt")
    write_text(prompt_path, prompt_for_codex)
    started = time.time()
    cmd = [
        executable,
        "exec",
        *codex_exec_model_args(settings),
        "-C",
        stage,
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "--output-last-message",
        staged_last_message,
        "-",
    ]
    env = coding_agent_environment(workspace=stage)
    env["INITIAL_BASELINE_PROMPT_PATH"] = prompt_path
    env["INITIAL_AGENT_MANIFEST"] = staged_output
    env["CODEX_MODEL"] = settings["model"]
    env["CODEX_REASONING_EFFORT"] = settings["reasoning_effort"]
    returncode = 0
    stdout = stderr = command_error = ""
    timed_out = False
    bundle = {}
    assembly_error = ""
    write_json(str(attempt / "request.json"), {
        "stage": "codegen", "instructions": prompt,
        "context": context, "source_agent": source_agent_dir,
    })
    try:
        if source_agent_dir:
            shutil.copytree(source_agent_dir, os.path.join(stage, "original_agent"))
        try:
            result = subprocess.run(
                cmd,
                cwd=stage,
                input=prompt_for_codex,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            returncode, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            returncode, timed_out = 124, True
            stdout, stderr = exc.stdout or "", exc.stderr or ""
            command_error = f"Coding command exceeded its {timeout}-second deadline."
        except OSError as exc:
            returncode = 1
            command_error = str(exc)
        try:
            bundle = collect_agent_source(stage)
        except RuntimeError as exc:
            assembly_error = str(exc)
        if os.path.exists(staged_last_message):
            shutil.copy2(staged_last_message, attempt / "last_message.json")
    finally:
        copy_agent_source(stage, iteration.root)
        shutil.rmtree(stage, ignore_errors=True)
    output_path = str(attempt / "bundle.json")
    if returncode and not command_error:
        command_error = f"Coding command exited with status {returncode}."
    next_task = {
        "stage": "codegen" if returncode else "smoke",
        "task": ("Continue the current draft after resolving the recorded authoring failure."
                 if returncode else "Run preflight and runtime smoke on the authored draft."),
        "error": command_error or assembly_error,
    }
    response = {
        "ok": bool(returncode == 0 and bundle),
        "returncode": returncode,
        "stdout": stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout,
        "stderr": stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr,
        "timed_out": timed_out,
        "command_error": command_error,
        "summary": command_error or assembly_error or "Initial agent source authored.",
        "next_task": next_task,
        "elapsed_sec": round(time.time() - started, 1),
        "prompt_path": prompt_path,
        "prompt_template_path": prompt_template_path,
        "output_path": output_path,
        "bundle": bundle,
        "assembly_error": assembly_error,
        "iteration_dir": str(iteration.root),
        "cmd": cmd,
        "codex_runtime_settings": settings,
        "coding_agent_protocol": CODING_AGENT_PROTOCOL,
        "execution_scope": "isolated_codegen_staging_workspace",
    }
    write_json(output_path, bundle)
    write_json(str(attempt / "response.json"), {key: value for key, value in response.items() if key != "bundle"})
    write_text(str(attempt / "stdout.txt"), response["stdout"])
    write_text(str(attempt / "stderr.txt"), response["stderr"])
    iteration.record("codegen", attempt / "response.json", status="failed" if returncode else "complete",
                     notes_path=iteration.root / "implementation.md",
                     next_task=next_task)
    return response


def _normalize_codegen_bundle(bundle: Dict[str, Any],
                              *,
                              workspace: str,
                              distribution_manifest: str,
                              generation_mode: str,
                              codegen_result: Dict[str, Any]) -> Dict[str, Any]:
    bundle = dict(bundle or {})
    raw_modules = bundle.get("modules") or []
    modules = list(raw_modules.values()) if isinstance(raw_modules, dict) else raw_modules
    for module in modules:
        if not isinstance(module, dict):
            continue
        module_type = module.get("module_type")
        module.setdefault("source_file", SOURCE_FILES.get(module_type, ""))
    bundle.update({
        # Do not repair semantic bundle fields on Codex's behalf.  Missing
        # schema/combo/design fields are actionable codegen failures and must
        # return through the same repair loop rather than being guessed here.
        "schema_version": bundle.get("schema_version", ""),
        "mode": bundle.get("mode", ""),
        "generation_mode": generation_mode,
        "created_at": bundle.get("created_at") or int(time.time()),
        "workspace": os.path.abspath(workspace),
        "distribution_manifest": os.path.abspath(distribution_manifest) if distribution_manifest else "",
        "combo": bundle.get("combo"),
        "modules": modules,
        "runtime_policy": {"required_vlm_profile_id": INITIAL_VLM_PROFILE},
        "codegen": {
            "ok": bool(codegen_result.get("ok")),
            "returncode": codegen_result.get("returncode"),
            "elapsed_sec": codegen_result.get("elapsed_sec"),
            "prompt_path": codegen_result.get("prompt_path", ""),
            "output_path": codegen_result.get("output_path", ""),
            "execution_scope": codegen_result.get("execution_scope", ""),
            "iteration_dir": codegen_result.get("iteration_dir", ""),
            "assembly_error": codegen_result.get("assembly_error", ""),
        },
    })
    bundle["validation"] = validate_bundle(bundle)
    return bundle


def build_initial_baseline_bundle(
    workspace: str,
    distribution_manifest: str,
    profile: Dict[str, Any],
    deep_research_summary: str = "",
    codex_cli: str = "",
    codegen_timeout: int = 900,
    codegen_output_dir: str = "",
    execute_codegen: bool = True,
) -> Tuple[Dict[str, Any], str]:
    manifest_spec = load_distribution_spec(distribution_manifest) if distribution_manifest else {}
    # Initial codegen is strictly distribution-profile, research, and plan driven.
    output_dir = os.path.abspath(codegen_output_dir or os.getcwd())
    # Export the authored source for the initial execution pipeline.
    codegen_bundle_path = os.path.join(output_dir, "initial_baseline_bundle.json")
    codegen_context = build_initial_codegen_context(
        profile, manifest_spec, deep_research_summary, output_dir,
    )
    codegen_context_path = os.path.join(output_dir, "initialization", "context.json")
    write_json(codegen_context_path, codegen_context)
    prompt = build_prompt(
        codegen_context,
        output_path=os.path.join(output_dir, "agent.json") if execute_codegen else "",
    )
    if not execute_codegen:
        return {}, prompt
    result = _run_codegen_command(
        prompt,
        output_path=os.path.join(output_dir, "agent.json"), iteration_dir=output_dir,
        codex_cli=codex_cli,
        timeout=codegen_timeout,
    )
    bundle = _normalize_codegen_bundle(
        result.get("bundle") or {},
        workspace=workspace,
        distribution_manifest=distribution_manifest,
        generation_mode="coding_agent",
        codegen_result=result,
    )
    bundle.setdefault("codegen", {})["context_path"] = codegen_context_path
    if not result.get("ok"):
        bundle.setdefault("validation", {})["passed"] = False
        bundle.setdefault("validation", {}).setdefault("issues", []).append(
            result.get("assembly_error") or "codegen command did not produce an executable Agent"
        )
    write_json(codegen_bundle_path, bundle)
    if result.get("returncode") != 0:
        raise RuntimeError("Initial authoring command failed; inspect its codegen attempt before retrying.")
    return bundle, prompt


def _repair_modules(failure_contract: Dict[str, Any]) -> List[str]:
    modules = []
    for value in (
        failure_contract.get("target_module"),
        (failure_contract.get("producer") or {}).get("module"),
        (failure_contract.get("consumer") or {}).get("module"),
    ):
        module = str(value or "")
        if module in MODULE_ORDER and module not in modules:
            modules.append(module)
    return modules or list(MODULE_ORDER)


# Smoke repair is a new Codex invocation.  It has no inherited conversation or
# filesystem-reading guarantee, so every input it needs is attached to its
# prompt. Long traces remain available in full; a size threshold must not
# replace evidence with an administrative failure.
_NO_LABEL_KEYS = {
    "answer", "answers", "gold", "gold_answer", "gt_answer", "ground_truth", "groundtruth",
    "candidate_answer", "candidate_final_answer", "final_answer", "final_agent_answer",
    "correct", "is_correct", "correctness", "reward",
}


def _read_artifact_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _redact_no_label(value: Any) -> Any:
    """Preserve runtime facts while excluding answer/correctness supervision."""
    if isinstance(value, dict):
        return {
            str(item_key): _redact_no_label(item_value)
            for item_key, item_value in value.items()
            if str(item_key).strip().lower() not in _NO_LABEL_KEYS
        }
    if isinstance(value, list):
        return [_redact_no_label(item) for item in value]
    return value


def _artifact(artifact_id: str, title: str, purpose: str, content: str,
              content_kind: str) -> Dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "title": title,
        "purpose": purpose,
        "content_kind": content_kind,
        "content": content,
    }


def _json_artifact(artifact_id: str, title: str, purpose: str, value: Any) -> Dict[str, Any]:
    return _artifact(
        artifact_id, title, purpose,
        json.dumps(value, ensure_ascii=False, indent=2), "json",
    )


def _candidate_for_repair(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """The current code once, without recursively attaching past feedback."""
    keep = {
        key: value for key, value in (bundle or {}).items()
        if key not in {"smoke_repair", "codegen", "validation", "prompt_path"}
    }
    return keep


def _existing_text_artifact(artifact_id: str, title: str, purpose: str,
                            path: str, *, required: bool = False) -> Dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        if required:
            raise RuntimeError(f"required repair-context artifact missing: {artifact_id}")
        return None
    return _artifact(artifact_id, title, purpose, _read_artifact_text(path), "text")


def _base_codegen_artifacts(bundle: Dict[str, Any], output_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Gather initial codegen inputs for the next implementation attempt."""
    codegen = bundle.get("codegen") if isinstance(bundle.get("codegen"), dict) else {}
    root = output_dir or str(codegen.get("iteration_dir") or "")
    context_path = str(codegen.get("context_path") or os.path.join(root, "initialization", "context.json"))
    context = read_json(context_path) if os.path.isfile(context_path) else {}
    artifacts = context.get("artifacts") or []
    if context.get("artifact_type") == "initial_codegen_context" and artifacts:
        return list(artifacts), {
            "context_path": os.path.abspath(context_path),
            "source_snapshot": str(context.get("source_snapshot") or ""),
        }
    docs = os.path.join(CURRENT_DIR, "agent_docs")
    runtime_protocol = os.path.join(PROJECT_ROOT, "execution", "action_runtime", "module_protocol.py")
    specs = [
        ("initial_codegen_rules", "Initial bundle generation rules", "Read-only initial-codegen requirements and output contract.", os.path.join(docs, "initial_baseline_prompt.md"), True),
        ("five_module_interface_contract", "Five-module interface contract", "Responsibilities and handoff protocol for video_structuring, localization, perception, memory, and thinking.", os.path.join(docs, "module_interface_contract.md"), True),
        ("generated_runtime_abi_witness", "Generated runtime ABI witness", "Exact live signatures and minimal dynamic call witnesses; overrides must preserve these facts.", os.path.join(docs, "generated_runtime_abi_witness.json"), True),
        ("runtime_capability_manual", "Runtime capability manual", "Only supported model/tool adapter APIs and their contracts.", os.path.join(docs, "runtime_capability_manual.md"), True),
        ("runtime_capability_profiles", "Registered capability profiles", "Registered profile IDs and transport/provenance contracts.", str(PROFILE_MANUAL_PATH), True),
        ("runtime_module_protocol", "Execution-layer module protocol", "The actual base protocol that injected modules must obey.", runtime_protocol, True),
    ]
    artifacts = []
    for artifact_id, title, purpose, path, required in specs:
        item = _existing_text_artifact(artifact_id, title, purpose, path, required=required)
        if item:
            artifacts.append(item)
    return artifacts, {}


def _smoke_build_output_artifacts(smoke_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    sandboxes = [str(smoke_result.get("sandbox_dir") or "")]
    for case in smoke_result.get("smoke_case_results") or []:
        if isinstance(case, dict):
            sandboxes.append(str(case.get("sandbox_dir") or ""))
    structure_dirs = []
    for sandbox in sandboxes:
        structure_dir = os.path.join(sandbox, "video_structure")
        if os.path.isdir(structure_dir) and structure_dir not in structure_dirs:
            structure_dirs.append(structure_dir)
    if not structure_dirs:
        return []
    artifacts = []
    paths = sorted(
        os.path.join(directory, name)
        for structure_dir in structure_dirs
        for directory, _, names in os.walk(structure_dir)
        for name in names if name.endswith(".jsonl")
    )
    for index, path in enumerate(paths, start=1):
        raw = _read_artifact_text(path)
        sanitized_lines = []
        for line in raw.splitlines():
            try:
                sanitized_lines.append(json.dumps(_redact_no_label(json.loads(line)), ensure_ascii=False))
            except json.JSONDecodeError:
                sanitized_lines.append(line)
        artifacts.append(_artifact(
            f"smoke_build_output_{index}", "Smoke structure/API output",
            "Complete build-time structure and capability output produced by this candidate during smoke; use it to trace evidence loss.",
            "\n".join(sanitized_lines) + ("\n" if raw.endswith("\n") else ""), "jsonl",
        ))
    return artifacts


def build_smoke_repair_packet(bundle: Dict[str, Any], smoke_result: Dict[str, Any],
                              *, output_dir: str = "") -> Dict[str, Any]:
    """Build the complete, annotated, no-label repair attachment set.

    The repair context contains complete file contents. Codex receives it
    inline and is never asked to recover evidence from an unpublished path.
    """
    failure_contract = dict(smoke_result.get("failure_contract") or {})
    repair_modules = _repair_modules(failure_contract)
    behavior = smoke_result.get("behavioral_smoke") if isinstance(smoke_result.get("behavioral_smoke"), dict) else {}
    failure_boundary = {
        "smoke_stage": smoke_result.get("stage", ""),
        "smoke_error": smoke_result.get("error", ""),
        "repair_instruction": smoke_result.get("repair_instruction", ""),
        "failure_contract": failure_contract,
        "runtime_engineering_issues": smoke_result.get("runtime_engineering_issues", []),
        "behavioral_verdict": behavior,
        "repair_modules_are_starting_points_not_a_limit": repair_modules,
    }
    artifacts, initial_context = _base_codegen_artifacts(bundle, output_dir)
    artifacts.append(_json_artifact(
        "current_candidate_bundle", "Current candidate bundle and five module sources",
        "Current five-module implementation and its resources for this repair.",
        _candidate_for_repair(bundle),
    ))
    artifacts.append(_json_artifact(
        "smoke_failure_boundary", "Deterministic smoke failure boundary",
        "Machine-derived failure contract, engineering issues, and falsifiable success conditions. This is the acceptance boundary for this repair.",
        _redact_no_label(failure_boundary),
    ))
    trace_path = str(smoke_result.get("engineering_execution_trace_path") or "")
    if trace_path and os.path.isfile(trace_path):
        try:
            trace = json.loads(_read_artifact_text(trace_path))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"complete_smoke_execution_trace is invalid JSON: {exc}") from exc
        artifacts.append(_json_artifact(
            "complete_smoke_execution_trace", "Complete no-label smoke task trajectory",
            "Complete task request, module trajectory, localized windows, observations, and capability-event timestamps for this smoke execution. Diagnose across fields, not from a summary.",
            _redact_no_label(trace),
        ))
    runtime_output = dict(smoke_result)
    for key in ("failure_contract", "runtime_engineering_issues", "behavioral_smoke", "repair_instruction", "engineering_execution_trace_path"):
        runtime_output.pop(key, None)
    artifacts.append(_json_artifact(
        "smoke_runtime_output", "Smoke runtime output",
        "Complete smoke result and runtime-produced artifact-quality output, excluding the separately attached failure boundary and no-label supervision.",
        _redact_no_label(runtime_output),
    ))
    artifacts.extend(_smoke_build_output_artifacts(smoke_result))
    context_path = initial_context.get("context_path", "")
    source_snapshot = initial_context.get("source_snapshot", "")
    if bundle.get("initialization") == "map" and not os.path.isdir(source_snapshot):
        raise RuntimeError("Mapped initialization repair requires this run's original_agent source snapshot.")
    return {
        "artifact_type": "initial_smoke_repair_context", "schema_version": 1,
        "purpose": "minimal complete engineering repair context: current code, exact ABI, and this smoke failure",
        "repair_modules": repair_modules,
        "failure_contract": failure_contract,
        "context_root_found": bool(context_path),
        "initial_codegen_context_path": context_path,
        "source_snapshot": source_snapshot,
        "artifacts": artifacts,
    }


def build_smoke_repair_prompt(bundle: Dict[str, Any],
                              smoke_result: Dict[str, Any],
                              output_path: str,
                              repair_packet: Dict[str, Any] | None = None) -> str:
    # A new Codex invocation cannot inherit a previous codegen conversation or
    # assume it can read a path.  The packet therefore embeds all input files.
    repair_packet = repair_packet or build_smoke_repair_packet(bundle or {}, smoke_result)
    embedded = []
    for artifact in repair_packet.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("artifact_id") or "unnamed_artifact")
        embedded.append(
            f"### {artifact_id}: {artifact.get('title', '')}\n"
            f"Purpose: {artifact.get('purpose', '')}\n"
            f"Content type: {artifact.get('content_kind', '')}\n"
            f"<<<BEGIN {artifact_id}>>>\n{artifact.get('content', '')}\n<<<END {artifact_id}>>>"
        )
    return (
        "You are the automatic code-authoring agent repairing a generated initial baseline bundle for "
        "MetaVideoAgent.\n\n"
        "This is a fresh repair invocation. Do not rely on prior conversation "
        "state. Runtime references and current candidate inputs are embedded "
        "below and annotated with their roles. For mapped initialization, the "
        "original source snapshot is also available in `original_agent/`. The packet includes the "
        "original design, shared runtime interfaces and current implementation.\n\n"
        + "The smoke stage is for engineering correctness, not answer accuracy. "
        "Read the current candidate source, complete no-label trajectory, runtime "
        "output, and build/API output together. Locate the earliest real "
        "dataflow, control-flow, or interface break and repair all implicated "
        "modules; the listed target modules are starting points, not a restriction. "
        "Use the failure boundary's observable success conditions to verify the repair.\n\n"
        "At `force_finish=True`, return `action == \"finish\"` with one concrete, "
        "parseable `payload.answer`. "
        "After a successful perception result, do not repeat the same verification "
        "without state-changing evidence.\n\n"
        "Continue editing the current agent/ source files and agent.json. Keep five concrete "
        "entrypoints and update their manifest entries when the implementation changes. Preserve "
        "resources, configuration, design_answers, observed_channel_summary and deep_research_summary. "
        "Keep code and resource contents in real files; the host exports the execution bundle. "
        "Record integration findings and next checks in implementation.md.\n\n"
        "The embedded initial-codegen rules, five-module ABI, runtime protocol, "
        "capability manual, and registered profiles are binding. Use only their "
        "documented `runtime_evidence` adapters and preserve the candidate's selected capability profiles. "
        "For a mapped source agent, preserve its prompts, tool/message protocol, model parameters and algorithm; "
        "repair the broken integration without replacing that behavior.\n\n"
        "If `requested_output_path` is provided, write the updated agent.json there "
        "as UTF-8 JSON and return only a short completion status. The host will run real-runtime "
        "preflight and smoke after completion; do not inspect unlisted workspace files.\n\n"
        f"requested_output_path: {output_path}\n\n"
        "## Embedded Repair Input Files\n\n"
        + "\n\n".join(embedded)
        + "\n"
    )


def repair_initial_baseline_bundle_with_smoke_feedback(
    bundle: Dict[str, Any],
    smoke_result: Dict[str, Any],
    *,
    output_dir: str,
    iteration_dir: str = "",
    attempt: int = 1,
    codex_cli: str = "",
    timeout: int = 900,
    artifact_tag: str = "smoke",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if (smoke_result.get("failure_kind") == "execution_layer"
            or smoke_result.get("execution_layer_rerun_required")
            or smoke_result.get("repair_route") == "rerun_same_bundle_after_execution_layer_fix"):
        raise RuntimeError("Resolve the execution-layer failure and rerun the same bundle; code repair was not requested.")
    os.makedirs(output_dir, exist_ok=True)
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(artifact_tag or "smoke"))
    iteration_dir = iteration_dir or str((bundle.get("codegen") or {}).get("iteration_dir") or output_dir)
    repaired_path = os.path.join(iteration_dir, "agent.json")
    repair_packet_path = os.path.join(
        output_dir, f"initial_smoke_repair_context_{safe_tag}_attempt_{attempt}.json"
    )
    repair_packet = build_smoke_repair_packet(bundle, smoke_result, output_dir=iteration_dir)
    write_json(repair_packet_path, repair_packet)
    prompt = build_smoke_repair_prompt(
        bundle, smoke_result, repaired_path,
        repair_packet=repair_packet,
    )
    result = _run_codegen_command(
        prompt,
        output_path=repaired_path, iteration_dir=iteration_dir,
        codex_cli=codex_cli,
        timeout=timeout,
        artifact_label=f"bundle_smoke_repair_{safe_tag}_attempt_{attempt}",
        source_agent_dir=repair_packet["source_snapshot"],
    )
    repaired_path = result["output_path"]
    prompt_path = str(result.get("prompt_path") or "")
    repaired = dict(result.get("bundle") or {})
    # Export only what the current draft declares; preflight receives any
    # missing manifest fields as actionable repair evidence.
    repaired["generation_mode"] = "coding_agent_smoke_repair"
    repaired["smoke_repair"] = {
        "attempt": attempt,
        "prompt_path": prompt_path,
        "codegen_ok": bool(result.get("ok")),
        "repair_context_path": repair_packet_path,
        "repair_packet_path": repair_packet_path,
        "engineering_execution_trace_path": str(
            smoke_result.get("engineering_execution_trace_path") or ""
        ),
    }
    repaired["runtime_policy"] = dict(bundle.get("runtime_policy") or {})
    repaired["initialization"] = bundle.get("initialization", "observe")
    repaired["codegen"] = {"context_path": repair_packet["initial_codegen_context_path"],
                           "iteration_dir": iteration_dir, "assembly_error": result.get("assembly_error", "")}
    repaired["validation"] = validate_bundle(repaired)
    write_json(repaired_path, repaired)
    return repaired, {
        "attempt": attempt,
        "ok": (
            bool(result.get("ok"))
            and repaired.get("validation", {}).get("passed", False)
        ),
        "prompt_path": prompt_path,
        "repaired_bundle_path": repaired_path,
        "codegen": {
            "returncode": result.get("returncode"),
            "elapsed_sec": result.get("elapsed_sec"),
            "cmd": result.get("cmd"),
        },
        "validation": repaired.get("validation", {}),
        "repair_context_path": repair_packet_path,
        "repair_packet_path": repair_packet_path,
        "repair_context_schema_version": repair_packet.get("schema_version", ""),
    }
