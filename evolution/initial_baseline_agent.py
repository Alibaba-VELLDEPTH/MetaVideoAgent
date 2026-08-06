"""Initial bundle generation for MetaVideoAgent.

This module creates the first five-module bundle for a MetaVideoAgent run.
It does not execute the bundle; execution and injection belong to the initial
combo runner and sandbox layer. Codex performs automatic authoring and bounded
repair, and every generated bundle passes through the same runtime gates.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
METAVIDEOAGENT_RUNTIME_DIR = os.path.join(PROJECT_ROOT, "execution", "action_runtime")
if os.path.isdir(METAVIDEOAGENT_RUNTIME_DIR) and METAVIDEOAGENT_RUNTIME_DIR not in sys.path:
    sys.path.insert(0, METAVIDEOAGENT_RUNTIME_DIR)

from capability_registry import (  # noqa: E402
    DEFAULT_PROFILE_IDS,
    LLM_ROLE_PROFILE_IDS,
    PROFILE_MANUAL_PATH,
    resolve_profile,
)
from codex_runtime_settings import (  # noqa: E402
    codex_exec_model_args,
    resolve_codex_runtime_settings,
)
from coding_agent_runtime import (  # noqa: E402
    CODING_AGENT_PROTOCOL,
    resolve_coding_agent_executable,
)
from distribution_spec import load_distribution_spec, prompt_safe_spec  # noqa: E402
from runtime_capability_context import build_runtime_capability_context  # noqa: E402

MODULE_ORDER = [
    "video_structuring",
    "localization",
    "perception",
    "memory",
    "thinking",
]

CLASS_NAMES = {
    "video_structuring": "InitialBaselineStructuring",
    "localization": "InitialBaselineLocalization",
    "perception": "InitialBaselinePerception",
    "memory": "InitialBaselineMemory",
    "thinking": "InitialBaselineThinking",
}

SOURCE_FILES = {
    "video_structuring": "execution/action_runtime/video_structuring_modules.py",
    "localization": "execution/action_runtime/localization_modules.py",
    "perception": "execution/action_runtime/perception_modules.py",
    "memory": "execution/action_runtime/memory_modules.py",
    "thinking": "execution/action_runtime/thinking_module.py",
}
INITIAL_VLM_PROFILE = DEFAULT_PROFILE_IDS["vlm"]
INITIAL_ASR_PROFILE = DEFAULT_PROFILE_IDS["asr"]
INITIAL_VLM_PROFILE_DATA = resolve_profile("vlm", INITIAL_VLM_PROFILE)


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        "observed_information_channels": profile.get("observed_information_channels", {}),
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
        "baseline_trajectories_profiled": sampling.get("baseline_trajectories_profiled"),
        "structure_segments_profiled": sampling.get("structure_segments_profiled"),
        "raw_videos_profiled": sampling.get("raw_videos_profiled"),
        "unique_videos_profiled": sampling.get("unique_videos_profiled"),
        "include_execution_artifacts": sampling.get("include_execution_artifacts"),
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


def _initial_runtime_capability_context() -> Dict[str, Any]:
    """Return prompt-safe capabilities with the MetaVideoAgent VLM policy pinned."""
    context = build_runtime_capability_context()
    functions = context.get("available_runtime_functions")
    if isinstance(functions, dict) and isinstance(functions.get("vlm"), dict):
        functions["vlm"] = dict(functions["vlm"])
        functions["vlm"]["default_profile"] = INITIAL_VLM_PROFILE
    policy = list(context.get("profile_selection_policy") or [])
    policy.append(
        "For this MetaVideoAgent initial baseline, every VLM adapter call MUST use "
        f"the registered profile_id `{INITIAL_VLM_PROFILE}`; do not omit profile_id."
    )
    context["profile_selection_policy"] = policy
    context["initial_vlm_policy"] = {
        "required_profile_id": INITIAL_VLM_PROFILE,
        "provider": INITIAL_VLM_PROFILE_DATA.get("provider", ""),
        "model_id": INITIAL_VLM_PROFILE_DATA.get("model_id", ""),
    }
    return context


def _build_previous_codegen_contract(profile: Dict[str, Any],
                                  manifest_spec: Dict[str, Any],
                                  task_sample: List[dict],
                                  deep_research_summary: str = "",
                                  output_path: str = "") -> str:
    base = load_prompt()
    payload = {
        "distribution_aware_profile": prompt_safe_profile(profile),
        "prompt_safe_manifest": prompt_safe_spec(manifest_spec),
        "task_sample": None,
        "task_supervision_policy": (
            "Associated questions and answer options are available only inside the "
            "five-frame per-video profile records. Answer labels, evidence windows, "
            "task categories, and task identifiers are unavailable."
        ),
        "deep_research_summary": deep_research_summary,
        "runtime_capability_manual": _initial_runtime_capability_context(),
        "requested_output_path": output_path,
    }
    return base + "\n\n## Runtime Input JSON\n\n```json\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ) + "\n```\n" + (
        "\n## Output Contract\n\n"
        "Return exactly one JSON object for `initial_baseline_bundle`. Do not "
        "return Markdown prose around it. The object must include `combo` and "
        "five executable module records under `modules`, one for each of: "
        "`video_structuring`, `localization`, `perception`, `memory`, "
        "`thinking`.\n"
        "If `requested_output_path` is non-empty, write the canonical JSON object only to "
        "that UTF-8 path. Return a short completion status, never a second bundle channel. "
        "The host will run real-runtime preflight and smoke after completion; "
        "do not inspect unlisted workspace files.\n"
        "\n## Mandatory Runtime Interface Contract\n\n"
        "The generated classes are dynamically injected into the existing "
        "execution layer. They must match these constructor/call signatures "
        "exactly or accept compatible `**kwargs`:\n"
        "- video_structuring: `__init__(self, workspace_dir, video_id, struct_type='', "
        "custom_config=None, requires_vector_db=False)`; must provide `addStructure`, "
        "`retrieveStructure`, and `_get_api_params` through inheritance or an ABI-compatible override.\n"
        "- localization: `__init__(self, env, struct_db=None, custom_config=None)`; "
        "tool methods must use the `tool_` prefix.\n"
        "- perception: `__init__(self, env, struct_db=None, custom_config=None)`; "
        "tool methods must use the `tool_` prefix.\n"
        "- memory: `__init__(self, workspace_dir, video_id, question, custom_config=None)`; "
        "must support the execution-layer memory methods including `get_context_packet`.\n"
        "- thinking: `__init__(self, env, custom_config=None)` and "
        "`__call__(self, question, video_context, work_context, force_finish=False)`.\n"
        "Do not omit `struct_db` for localization/perception; the Agent passes "
        "it by keyword during assembly.\n"
        "Use the neutral protocol helpers inherited from the bases: thinking must retain "
        "`set_runtime_context`, `get_runtime_context`, and `normalize_plan`; localization "
        "must return `self.make_localization_result(...)`; perception must return "
        "`self.make_perception_result(...)`. These are schemas, not default strategies.\n"
        "\n### Canonical structure-record ABI (mandatory)\n\n"
        "The generic execution-layer segment builder calls only:\n"
        "```python\n"
        "structuring.addStructure(start_sec=..., end_sec=..., "
        "multimodal_narration=<raw VLM text or JSON>)\n"
        "```\n"
        "It does not promise `raw_model_output`, `description`, `asr_text`, "
        "or pre-parsed entity fields. `VideoStructuringBase.addStructure` "
        "normalizes this payload into `canonical_evidence`, `retrieval_document`, "
        "and native fields. If you override `addStructure`, pass the original "
        "`multimodal_narration` through to `super().addStructure(**record)`; "
        "never replace it with empty placeholders. Localization must treat an "
        "empty canonical record as `unavailable`, never as a successful ranked "
        "candidate. The real smoke verifies that produced evidence remains retrievable.\n"
        "\n### Actual execution-layer thinking protocol\n\n"
        "The runtime call site is fixed and non-negotiable:\n"
        "```python\n"
        "thought, action, payload = self.thinker(\n"
        "    runtime_question, video_context, work_context, force_finish=is_last_step\n"
        ")\n"
        "if action == \"finish\":\n"
        "    answer = payload.get(\"answer\", \"\")\n"
        "else:\n"
        "    plan = self.thinker.normalize_plan(payload)\n"
        "```\n"
        "Therefore `thinking.__call__` MUST return exactly a 3-item tuple/list: "
        "`(thought: str, action: str, payload: list|dict)`. For normal tool use, "
        "return `(thought, \"act\", [subtask_dict, ...])`; for final answers, "
        "return `(thought, \"finish\", {\"answer\": \"...\", "
        "\"answer_type\": \"choice|multi_choice|free_text|count|unknown\", "
        "\"answer_confidence\": 0.0, \"evidence_summary\": \"...\"})`. "
        "Before every call the runtime supplies a structured per-turn context through "
        "`self.get_runtime_context()` (question, memory packet, and normalized prior module results); "
        "do not recover those fields by regex from work_context.\n"
        "`force_finish=True` is a mandatory termination boundary: on that call return "
        "`action == \"finish\"` with one concrete, parseable `payload[\"answer\"]`. "
        "Do not issue another plan, repeat a successful verification, or replace the "
        "answer with uncertainty/failure text when this flag is true.\n"
        "`payload['answer']` is the single canonical answer outlet used by the "
        "runtime evaluator; do not invent alternative final fields such as "
        "`prediction`, `result`, `final`, or `selected_option`. It must never "
        "return a dict such as `{\"decision\": ..., \"thought\": ..., "
        "\"action_input\": ...}`. It must never put a JSON string under "
        "`action_input.plan`; the runtime will treat that as an empty plan.\n"
        "Each `act` subtask dict must use execution-layer keys: "
        "`subtask_id`, `instruction`, `target_tool`, `tool_params`, and optional "
        "`export_vars`. `target_tool` is the runtime tool name without the "
        "`tool_` prefix, because the Agent dispatches it by checking "
        "`tool_{target_tool}` on localization/perception modules. Tool params "
        "must be a dict whose names match the selected tool method signature.\n"
        "\n## Formal Capability Boundary\n\n"
        "Unregistered external/local visual toolboxes are not part of this release. "
        "Use only the registered runtime_evidence adapters in the "
        "Runtime Capability Manual.\n"
        "\n## Runtime Capability Manual Contract\n\n"
        "The Runtime Input JSON includes `runtime_capability_manual`, the "
        "authoritative prompt-safe list of execution-layer model/tool calls "
        "available to generated code. Use it when deciding whether and how to "
        "call ASR, OCR, VLM, embedding, or LLM helpers. If your design relies "
        "on audio/speech, call the documented ASR path on real audio clips; do "
        "not label visual summaries or inferred context as transcripts. If "
        "your design emits confidence or conflict fields, compute them from "
        "evidence agreement instead of using constants. The manual is not a "
        "task hint and contains no benchmark answers.\n"
        f"Every VLM adapter call MUST use literal `profile_id='{INITIAL_VLM_PROFILE}'`. "
        "Use the VLM profile selected by the active capability configuration.\n"
        "\n## Mandatory Evidence Channel Contract\n\n"
        "Your module design must be internally executable, not just plausible "
        "on paper. First decide which evidence channels your design relies on "
        "for this raw-video distribution: visual frames, OCR/subtitles, audio/"
        "speech, temporal structure, entity/relationship state, retrieval "
        "metadata, or a different explicit channel. Then make sure the generated "
        "modules preserve those channels as non-empty, retrievable fields in "
        "the smoke-built artifacts. Do not let API parsing failures, fallback "
        "serialization, field-name mismatches, or tool-boundary conversions "
        "silently discard the channel your design needs. If an upstream model "
        "returns unstructured text instead of strict JSON, preserve the raw "
        "content and parsing diagnostics as retrievable evidence rather than "
        "collapsing it to placeholders. Smoke tests will summarize the created "
        "files, rows, fields, placeholder ratio, channel coverage, trajectory, "
        "tool calls, and final answer shape; Codex repair must use those facts "
        "to fix engineering/dataflow bugs before any full evaluation.\n"
        "\n## Final Decision Output Boundary\n\n"
        "The execution contract requires the thinking module to finish with a "
        "non-empty, evaluator-consumable decision derived from runtime evidence, "
        "rather than a process summary, an error message, or an uncertainty "
        "sentinel. This is an observable output boundary, not a prescribed "
        "reasoning algorithm: choose the internal control flow, tool sequence, "
        "evidence representation, and answer-generation method that best satisfy "
        "the five-module interfaces. The resulting `payload['answer']` must be "
        "a concise answer in the form requested by the runtime question; preserve "
        "limitations in structured confidence/evidence fields rather than making "
        "them the terminal answer. Use only documented `runtime_evidence` APIs "
        "for any capability call—never `utils.call_llm`, a provider SDK, HTTP, or "
        "a CLI.\n\n"
        "The generated answer must remain parseable by the unified evaluator. "
        "When choices are present, labels/text may be returned; multi-choice "
        "answers use semicolons, while open answers are concise. Do not assume a "
        "fixed number of choices or fixed question taxonomy. The answer-generation "
        "mechanism remains part of the generated agent design, not a prescribed "
        "set of task-specific helpers or examples.\n"
    )


MAX_CODEX_CONTEXT_CHARS = 900_000


def _context_artifact(artifact_id: str, title: str, purpose: str, content: str,
                      content_kind: str) -> Dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "title": title,
        "purpose": purpose,
        "content_kind": content_kind,
        "sha256": sha256_text(content),
        "chars": len(content),
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


def _context_manifest(artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {key: artifact[key] for key in ("artifact_id", "title", "purpose", "content_kind", "sha256", "chars")}
        for artifact in artifacts
    ]


def _assert_context_budget(artifacts: List[Dict[str, Any]], *, context_name: str) -> None:
    try:
        limit = int(os.environ.get("CODEX_MAX_CONTEXT_CHARS", str(MAX_CODEX_CONTEXT_CHARS)))
    except ValueError as exc:
        raise RuntimeError("CODEX_MAX_CONTEXT_CHARS must be an integer") from exc
    total = sum(int(artifact.get("chars", 0) or 0) for artifact in artifacts)
    if total > limit:
        raise RuntimeError(
            f"{context_name} contains {total} characters, exceeding CODEX_MAX_CONTEXT_CHARS={limit}; "
            "refusing to silently truncate an engineering artifact"
        )


def _render_embedded_artifacts(artifacts: List[Dict[str, Any]]) -> str:
    rendered = []
    for artifact in artifacts:
        artifact_id = str(artifact.get("artifact_id") or "unnamed_artifact")
        rendered.append(
            f"### {artifact_id}: {artifact.get('title', '')}\n"
            f"Purpose: {artifact.get('purpose', '')}\n"
            f"Content type: {artifact.get('content_kind', '')}; sha256={artifact.get('sha256', '')}; "
            f"characters={artifact.get('chars', 0)}\n"
            f"<<<BEGIN {artifact_id}>>>\n{artifact.get('content', '')}\n<<<END {artifact_id}>>>"
        )
    return "\n\n".join(rendered)


def build_initial_codegen_context(profile: Dict[str, Any], manifest_spec: Dict[str, Any],
                                  deep_research_summary: str, output_dir: str) -> Dict[str, Any]:
    """Create the immutable base context shared verbatim by codegen and repair."""
    docs = os.path.join(CURRENT_DIR, "agent_docs")
    artifacts: List[Dict[str, Any]] = [
        _context_artifact(
            "initial_codegen_rules", "Initial codegen rules",
            "Raw-video-only policy, design requirements, and canonical bundle output fields.",
            load_prompt(),
            "text",
        ),
        _context_json_artifact(
            "initial_execution_policy", "Initial execution policy",
            "Binding initial-stage policy that resolves conflicts among research, plans, and optional inventories.",
            {
                "priority": "This policy and generated_runtime_abi_witness override research suggestions, plans, and optional inventories on runtime ABI/capability conflicts.",
                "execution_llm_profile_id": DEFAULT_PROFILE_IDS["llm"],
                "asr_profile_id": INITIAL_ASR_PROFILE,
                "vlm_profile_id": INITIAL_VLM_PROFILE,
                "structure_build_lifecycle": (
                    "The execution-layer structure builder owns bounded media sampling and model calls. "
                    "The generated video_structuring module persists, indexes, and retrieves supplied records; "
                    "its addStructure must not make another model call."
                ),
            },
        ),
        _context_artifact(
            "five_module_interface_contract", "Five-module interface contract",
            "Authoritative responsibility and handoff ABI for all five generated modules.",
            _read_context_file(os.path.join(docs, "module_interface_contract.md")), "text",
        ),
        _context_artifact(
            "generated_runtime_abi_witness", "Generated runtime ABI witness",
            "The sole authority for exact base-method signatures, return shapes, and minimal calls.",
            _read_context_file(os.path.join(docs, "generated_runtime_abi_witness.json")), "json",
        ),
        _context_artifact(
            "runtime_module_protocol", "Execution-layer module protocol",
            "Actual neutral runtime protocol that generated modules must obey.",
            _read_context_file(os.path.join(PROJECT_ROOT, "execution", "action_runtime", "module_protocol.py")), "text",
        ),
        _context_artifact(
            "runtime_capability_manual", "Runtime capability manual",
            "Authoritative supported model/tool adapter APIs and failure semantics.",
            _read_context_file(os.path.join(docs, "runtime_capability_manual.md")), "text",
        ),
        _context_artifact(
            "runtime_capability_profiles", "Registered runtime capability profiles",
            "Authoritative profiles, transports, provenance, and selected VLM identity.",
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
        "Run-specific initial design/assembly plan.", os.path.join(output_dir, "initial_combo_plan.json"),
    )
    if plan:
        artifacts.append(plan)
    # The summary above is the single prompt-visible research channel.  The
    # full report often repeats it (and the initial contract) verbatim, which
    # dilutes exact ABI facts and makes repairs needlessly expensive.
    _assert_context_budget(artifacts, context_name="initial_codegen_context")
    return {
        "artifact_type": "initial_codegen_context", "schema_version": 1,
        "purpose": "immutable no-answer-supervision base context shared by initial codegen and all initial smoke repairs",
        "artifact_manifest": _context_manifest(artifacts),
        "artifacts": artifacts,
        "total_chars": sum(int(artifact["chars"]) for artifact in artifacts),
    }


def build_prompt(profile: Dict[str, Any], manifest_spec: Dict[str, Any], task_sample: List[dict],
                 deep_research_summary: str = "", output_path: str = "",
                 codegen_context: Dict[str, Any] | None = None) -> str:
    """Render the first-codegen prompt from the canonical shared context."""
    if not codegen_context:
        return _build_previous_codegen_contract(
            profile, manifest_spec, task_sample, deep_research_summary, output_path,
        )
    artifacts = codegen_context.get("artifacts") if isinstance(codegen_context.get("artifacts"), list) else []
    return (
        "You are the automatic code-authoring agent generating the initial executable five-module MetaVideoAgent bundle. "
        "Every authoritative policy input is embedded below. The isolated staging workspace also contains a read-only "
        "copy of the execution runtime, which you may inspect only to verify compatibility; it contains no data, prior outputs, or credentials. "
        "Follow this order: (1) generated_runtime_abi_witness, (2) five-module responsibilities "
        "and runtime capability profiles, (3) distribution profile, manifest, and research context, (4) write the "
        "bundle, (5) leave one canonical bundle JSON at requested_output_path, and (6) return only "
        "a short completion status.\n\n"
        "The host, not you, will run a real-runtime compile/inject/construct preflight after your response. "
        "When staged runtime source disagrees with an embedded witness/policy, follow the embedded witness/policy. "
        "Do not read datasets, previous outputs, task labels, or files outside the staging runtime.\n\n"
        f"requested_output_path: {output_path}\n"
        "Do not run synthetic fixtures or inspect datasets. Write the canonical bundle once; "
        "the host will perform the real-runtime preflight and smoke.\n\n"
        "## Initial Codegen Context Manifest\n\n"
        f"{json.dumps(codegen_context.get('artifact_manifest', []), ensure_ascii=False, indent=2)}\n\n"
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
    if int(bundle.get("schema_version", 0) or 0) != 1:
        issues.append("bundle must use the current schema")
    if str(bundle.get("mode") or "") != "initial_baseline":
        issues.append("mode must be 'initial_baseline'")
    for field in ("design_answers", "observed_channel_summary", "deep_research_summary"):
        if field not in bundle:
            issues.append(f"initial bundle is missing required field: {field}")
    execution_llm_profile = DEFAULT_PROFILE_IDS["llm"]
    evolution_llm_profile = LLM_ROLE_PROFILE_IDS["evolution"]
    for module in bundle.get("modules") or []:
        if not isinstance(module, dict):
            continue
        code = str(module.get("code") or "")
        if evolution_llm_profile != execution_llm_profile and evolution_llm_profile in code:
            issues.append(
                f"execution bundle references evolution-only profile {evolution_llm_profile}; "
                f"combo thinking must use {execution_llm_profile}"
            )
        if "call_text_llm(" in code and execution_llm_profile not in code:
            issues.append(
                "execution bundle calls text LLM without explicit "
                f"{execution_llm_profile} profile_id"
            )
    validation["issues"] = issues
    validation["passed"] = bool(validation.get("passed")) and not issues
    return validation


def _strip_json_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def _extract_json_object(text: str) -> Dict[str, Any]:
    value = _strip_json_fence(text)
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(value[start:end + 1])
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _create_codegen_staging_workspace(parent_dir: str, artifact_label: str) -> str:
    """Create a minimal write-isolated Codex workspace for one bundle.

    Generated bundle code is returned as JSON; Codex never needs the training
    workspace, datasets, previous outputs, credentials, or the formal project
    tree.  The copied execution runtime lets it inspect the actual bases and
    Agent assembly without granting it write access to the formal project.
    """
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact_label or "codegen")
    stage = tempfile.mkdtemp(prefix=f".codex_stage_{safe_label}_", dir=parent_dir)
    try:
        runtime_target = os.path.join(stage, "execution", "action_runtime")
        shutil.copytree(METAVIDEOAGENT_RUNTIME_DIR, runtime_target)
        _strip_stale_candidate_classes_from_staging_runtime(runtime_target)
        write_text(
            os.path.join(stage, "STAGING_SCOPE.md"),
            "This isolated workspace contains only the MetaVideoAgent execution runtime. "
            "Do not read data, formal outputs, or files outside this stage. "
            "Write exactly one canonical bundle to requested_bundle.json. "
            "The host performs ABI/injection preflight and real smoke after authoring exits.\n",
        )
        return stage
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _strip_stale_candidate_classes_from_staging_runtime(runtime_dir: str) -> None:
    """Remove prior Codex strategy classes from an isolated initial-codegen stage.

    The formal runtime checkout can contain source left by a previous candidate
    for debugging, but an initial bundle must start from neutral ABI helpers.
    Generated classes are uniformly marked with ``CODEX_BUNDLE_CHANGE_REASON``;
    removing only those marked classes (and their map registration assignments)
    keeps the real project untouched while preventing hidden candidate-pool
    contamination of a fresh Codex invocation.
    """
    filenames = (
        "video_structuring_modules.py", "localization_modules.py",
        "perception_modules.py", "memory_modules.py", "thinking_module.py",
    )
    for filename in filenames:
        path = os.path.join(runtime_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
            tree = ast.parse(content)
        except (OSError, SyntaxError):
            continue
        stale_names = {
            node.name for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "CODEX_BUNDLE_CHANGE_REASON"
                        for target in item.targets)
                for item in node.body
            )
        }
        if not stale_names:
            continue
        remove_ranges = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name in stale_names:
                remove_ranges.append((node.lineno - 1, node.end_lineno))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                            and isinstance(target.slice, ast.Constant)
                            and str(target.slice.value) in stale_names):
                        remove_ranges.append((node.lineno - 1, node.end_lineno))
                        break
        lines = content.splitlines(keepends=True)
        for start, end in sorted(remove_ranges, reverse=True):
            del lines[start:end]
        with open(path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)


def _run_codegen_command(prompt: str,
                         *,
                         output_path: str,
                         codex_cli: str = "",
                         timeout: int = 900,
                         artifact_label: str = "initial_baseline_codegen",
                         writable_artifacts: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Run the workflow's coding-agent adapter in an isolated workspace."""
    settings = resolve_codex_runtime_settings()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(artifact_label or "initial_baseline_codegen"))
    prompt_template_path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)), f"{safe_label}_prompt_template.txt"
    )
    write_text(prompt_template_path, prompt)
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    stage = _create_codegen_staging_workspace(parent_dir, safe_label)
    staged_output = os.path.join(stage, "requested_bundle.json")
    staged_last_message = os.path.join(stage, "last_message.json")
    staged_artifacts = {}
    for name, host_path in (writable_artifacts or {}).items():
        staged_artifacts[str(host_path)] = os.path.join(stage, f"requested_{name}.json")
    prompt_for_codex = prompt.replace(os.path.abspath(output_path), staged_output)
    for host_path, stage_path in staged_artifacts.items():
        prompt_for_codex = prompt_for_codex.replace(os.path.abspath(host_path), stage_path)
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"{safe_label}_prompt.txt")
    write_text(prompt_path, prompt_for_codex)
    started = time.time()
    executable = resolve_coding_agent_executable(codex_cli)
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
    env = os.environ.copy()
    env["INITIAL_BASELINE_PROMPT_PATH"] = prompt_path
    env["INITIAL_BASELINE_BUNDLE_OUTPUT"] = staged_output
    env["CODEX_MODEL"] = settings["model"]
    env["CODEX_REASONING_EFFORT"] = settings["reasoning_effort"]
    result = None
    bundle = {}
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
        if os.path.exists(staged_output):
            shutil.copy2(staged_output, output_path)
        for host_path, stage_path in staged_artifacts.items():
            if os.path.exists(stage_path):
                os.makedirs(os.path.dirname(os.path.abspath(host_path)), exist_ok=True)
                shutil.copy2(stage_path, host_path)
        if os.path.exists(output_path):
            try:
                bundle = read_json(output_path)
            except Exception:
                bundle = {}
        if os.path.exists(staged_last_message):
            shutil.copy2(staged_last_message, os.path.join(parent_dir, f"{safe_label}_last_message.json"))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {
        "ok": bool(result and result.returncode == 0 and bundle),
        "returncode": result.returncode if result is not None else None,
        "stdout": result.stdout if result is not None else "",
        "stderr": result.stderr if result is not None else "",
        "elapsed_sec": round(time.time() - started, 1),
        "prompt_path": prompt_path,
        "prompt_template_path": prompt_template_path,
        "output_path": output_path,
        "bundle": bundle,
        "cmd": cmd,
        "codex_runtime_settings": settings,
        "coding_agent_protocol": CODING_AGENT_PROTOCOL,
        "execution_scope": "isolated_codegen_staging_workspace",
    }


def _normalize_codegen_bundle(bundle: Dict[str, Any],
                              *,
                              workspace: str,
                              distribution_manifest: str,
                              generation_mode: str,
                              prompt: str,
                              profile: Dict[str, Any],
                              deep_research_summary: str,
                              task_sample: List[dict],
                              codegen_result: Dict[str, Any]) -> Dict[str, Any]:
    bundle = dict(bundle or {})
    # Codex may use ``initial_baseline_bundle`` as a transport envelope even
    # though the requested output path contains a single bundle.  Unwrap only
    # that exact object; this preserves its semantic contents and does not
    # synthesize any missing schema, combo, module, or design field.
    wrapped_bundle = bundle.get("initial_baseline_bundle")
    if isinstance(wrapped_bundle, dict):
        bundle = dict(wrapped_bundle)
    modules = bundle.get("modules") if isinstance(bundle.get("modules"), list) else []
    for module in modules:
        if not isinstance(module, dict):
            continue
        module_type = module.get("module_type")
        module.setdefault("source_file", SOURCE_FILES.get(module_type, ""))
        module.setdefault("evolution_mode", f"initial_baseline_bundle:{generation_mode}")
        module.setdefault("reuse_primitive", False)
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
        "prompt_sha256": sha256_text(prompt),
        "task_sample_size": 0,
        "task_sample": [],
        "task_sample_policy": (
            "No supervised task sample is supplied; unsupervised questions/options "
            "are embedded in the distribution profile records."
        ),
        "runtime_policy": {"required_vlm_profile_id": INITIAL_VLM_PROFILE},
        "codegen": {
            "ok": bool(codegen_result.get("ok")),
            "returncode": codegen_result.get("returncode"),
            "elapsed_sec": codegen_result.get("elapsed_sec"),
            "prompt_path": codegen_result.get("prompt_path", ""),
            "output_path": codegen_result.get("output_path", ""),
            "execution_scope": codegen_result.get("execution_scope", ""),
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
    task_sample = []
    output_dir = os.path.abspath(codegen_output_dir or os.getcwd())
    # This is the single canonical initial-bundle artifact.  Do not create a
    # second near-identical "codegen" bundle and later copy/guess fields into
    # the bundle that smoke actually executes.
    codegen_bundle_path = os.path.join(output_dir, "initial_baseline_bundle.json")
    codegen_context = build_initial_codegen_context(
        profile, manifest_spec, deep_research_summary, output_dir,
    )
    codegen_context_path = os.path.join(output_dir, "initial_codegen_context.json")
    write_json(codegen_context_path, codegen_context)
    prompt = build_prompt(
        profile,
        manifest_spec,
        task_sample,
        deep_research_summary=deep_research_summary,
        output_path=codegen_bundle_path if execute_codegen else "",
        codegen_context=codegen_context,
    )
    if not execute_codegen:
        return {}, prompt
    result = _run_codegen_command(
        prompt,
        output_path=codegen_bundle_path,
        codex_cli=codex_cli,
        timeout=codegen_timeout,
    )
    write_text(os.path.join(output_dir, "initial_baseline_codegen_stdout.txt"), result.get("stdout", ""))
    write_text(os.path.join(output_dir, "initial_baseline_codegen_stderr.txt"), result.get("stderr", ""))
    bundle = _normalize_codegen_bundle(
        result.get("bundle") or {},
        workspace=workspace,
        distribution_manifest=distribution_manifest,
        generation_mode="coding_agent",
        prompt=prompt,
        profile=profile,
        deep_research_summary=deep_research_summary,
        task_sample=task_sample,
        codegen_result=result,
    )
    bundle.setdefault("codegen", {})["context_path"] = codegen_context_path
    bundle["codegen"]["context_sha256"] = sha256_text(json.dumps(
        codegen_context.get("artifact_manifest", []), ensure_ascii=False, sort_keys=True,
    ))
    if not result.get("ok"):
        bundle.setdefault("validation", {})["passed"] = False
        bundle.setdefault("validation", {}).setdefault("issues", []).append(
            "codegen command failed or did not return bundle JSON"
        )
    write_json(codegen_bundle_path, bundle)
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
# prompt.  The bound applies only to a *single smoke artifact*; smoke uses a
# deliberately small suite, while full-eval JSONL is never a repair input.
# A one-question trajectory can exceed 512 KiB when a faulty candidate repeats
# bounded perception calls, so permit one MiB of complete no-label evidence.
# Failing closed above that remains safer than silently truncating it.
MAX_INLINE_SMOKE_ARTIFACT_BYTES = 1024 * 1024
_NO_LABEL_KEYS = {
    "answer", "answers", "gold", "gold_answer", "ground_truth", "groundtruth",
    "candidate_answer", "final_answer", "correct", "is_correct", "correctness",
    "score", "scores", "reward",
}


def _is_no_label_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return normalized in _NO_LABEL_KEYS or any(
        token in normalized
        for token in ("answer", "gold", "ground_truth", "groundtruth", "correct", "score", "reward")
    )


def _read_small_text(path: str, *, artifact_id: str) -> str:
    size = os.path.getsize(path)
    if size > MAX_INLINE_SMOKE_ARTIFACT_BYTES:
        raise RuntimeError(
            f"{artifact_id} is {size} bytes, exceeding the {MAX_INLINE_SMOKE_ARTIFACT_BYTES}-byte "
            "smoke-repair attachment limit; do not truncate or substitute it"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _redact_no_label(value: Any, *, key: str = "") -> Any:
    """Preserve runtime facts while excluding answer/correctness supervision."""
    if _is_no_label_key(key):
        return "<redacted_no_label_supervision>"
    if isinstance(value, dict):
        return {
            str(item_key): _redact_no_label(item_value, key=str(item_key))
            for item_key, item_value in value.items()
            if not _is_no_label_key(item_key)
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
        "sha256": sha256_text(content),
        "chars": len(content),
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


def _walk_ancestors(path: str) -> Iterable[str]:
    current = os.path.abspath(path)
    if os.path.isfile(current):
        current = os.path.dirname(current)
    while True:
        yield current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent


def _initial_context_root(bundle: Dict[str, Any], smoke_result: Dict[str, Any], output_dir: str) -> str:
    """Find this run's root, never a global/latest output directory."""
    candidates = [output_dir, str(smoke_result.get("engineering_execution_trace_path") or "")]
    codegen = bundle.get("codegen") if isinstance(bundle.get("codegen"), dict) else {}
    candidates.extend([str(codegen.get("prompt_path") or ""), str(bundle.get("prompt_path") or "")])
    for candidate in candidates:
        if not candidate:
            continue
        for directory in _walk_ancestors(candidate):
            if (
                os.path.isfile(os.path.join(directory, "initial_baseline_bundle.json"))
                and os.path.isfile(os.path.join(directory, "initial_combo_plan.json"))
            ):
                return directory
    return ""


def _existing_text_artifact(artifact_id: str, title: str, purpose: str,
                            path: str, *, required: bool = False) -> Dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        if required:
            raise RuntimeError(f"required repair-context artifact missing: {artifact_id}")
        return None
    return _artifact(artifact_id, title, purpose, _read_small_text(path, artifact_id=artifact_id), "text")


def _base_codegen_artifacts(bundle: Dict[str, Any], smoke_result: Dict[str, Any],
                            output_dir: str) -> Tuple[List[Dict[str, Any]], str]:
    """Reconstruct the initial codegen context from authoritative source files.

    We deliberately attach the source inputs, rather than the old composed
    prompt plus its constituent files.  This gives repair the same facts as
    codegen without wasting context on duplicate copies.
    """
    root = _initial_context_root(bundle, smoke_result, output_dir)
    codegen = bundle.get("codegen") if isinstance(bundle.get("codegen"), dict) else {}
    context_candidates = [str(codegen.get("context_path") or "")]
    if root:
        context_candidates.append(os.path.join(root, "initial_codegen_context.json"))
    for context_path in context_candidates:
        if not context_path or not os.path.isfile(context_path):
            continue
        try:
            context = read_json(context_path)
        except (OSError, json.JSONDecodeError):
            continue
        artifacts = context.get("artifacts") if isinstance(context.get("artifacts"), list) else []
        if context.get("artifact_type") == "initial_codegen_context" and artifacts:
            _assert_context_budget(artifacts, context_name="reused_initial_codegen_context")
            # A repair needs runtime facts, not a second copy of profile/research
            # prose already recorded inside the candidate bundle.  Keep only the
            # ABI/protocol/capability sources that govern the fix.
            required_ids = {
                "initial_codegen_rules",
                "five_module_interface_contract",
                "generated_runtime_abi_witness",
                "runtime_module_protocol",
                "runtime_capability_manual",
                "runtime_capability_profiles",
            }
            return [
                dict(artifact) for artifact in artifacts
                if isinstance(artifact, dict) and artifact.get("artifact_id") in required_ids
            ], root
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
    if root:
        contextual = [
            ("observed_distribution_profile", "Observed distribution profile", "Five-frame per-video records with associated questions/options; no answers, evidence annotations, or task labels.", os.path.join(root, "observed_distribution_profile.json")),
            ("initial_combo_plan", "Initial combo plan", "The code-generation design/assembly plan for this exact run.", os.path.join(root, "initial_combo_plan.json")),
        ]
        research_dir = os.path.join(os.path.dirname(root), "deep_research")
        contextual.extend([
            ("deep_research_report", "Codex deep-research report", "Research evidence available when the initial design was generated.", os.path.join(research_dir, "deep_research_report.json")),
            ("deep_research_design_brief", "Deep-research design brief", "Actionable design constraints produced by deep research.", os.path.join(research_dir, "deep_research_design_brief.json")),
            ("deep_research_summary", "Deep-research summary", "Human-readable research summary available to codegen.", os.path.join(research_dir, "deep_research_summary.md")),
        ])
        original_bundle_path = os.path.join(root, "initial_baseline_bundle.json")
        try:
            original_bundle = read_json(original_bundle_path)
        except (OSError, json.JSONDecodeError):
            original_bundle = {}
        manifest_path = str(original_bundle.get("distribution_manifest") or bundle.get("distribution_manifest") or "")
        if manifest_path:
            contextual.append(("distribution_manifest", "Dataset distribution manifest", "Prompt-safe dataset split/distribution specification used by initial codegen.", manifest_path))
        for artifact_id, title, purpose, path in contextual:
            item = _existing_text_artifact(artifact_id, title, purpose, path)
            if item:
                artifacts.append(item)
    return artifacts, root


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
        raw = _read_small_text(path, artifact_id=f"smoke_build_output_{index}")
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
    artifacts, context_root = _base_codegen_artifacts(bundle, smoke_result, output_dir)
    artifacts.append(_json_artifact(
        "current_candidate_bundle", "Current candidate bundle and five module sources",
        "Canonical current code to repair. It contains the complete source of every generated module exactly once; historical smoke feedback is intentionally excluded.",
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
            trace = json.loads(_read_small_text(trace_path, artifact_id="complete_smoke_execution_trace"))
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
    _assert_context_budget(artifacts, context_name="initial_smoke_repair_context")
    manifest = _context_manifest(artifacts)
    return {
        "artifact_type": "initial_smoke_repair_context", "schema_version": 1,
        "purpose": "minimal complete engineering repair context: current code, exact ABI, and this smoke failure",
        "repair_modules": repair_modules,
        "failure_contract": failure_contract,
        "context_root_found": bool(context_root),
        "artifact_manifest": manifest,
        "artifacts": artifacts,
        "total_chars": sum(int(artifact["chars"]) for artifact in artifacts),
    }


def build_smoke_repair_prompt(bundle: Dict[str, Any],
                              smoke_result: Dict[str, Any],
                              output_path: str,
                              repair_packet: Dict[str, Any] | None = None) -> str:
    # A new Codex invocation cannot inherit a previous codegen conversation or
    # assume it can read a path.  The packet therefore embeds all input files.
    repair_packet = repair_packet or build_smoke_repair_packet(bundle or {}, smoke_result)
    manifest = repair_packet.get("artifact_manifest") if isinstance(repair_packet.get("artifact_manifest"), list) else []
    embedded = []
    for artifact in repair_packet.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("artifact_id") or "unnamed_artifact")
        embedded.append(
            f"### {artifact_id}: {artifact.get('title', '')}\n"
            f"Purpose: {artifact.get('purpose', '')}\n"
            f"Content type: {artifact.get('content_kind', '')}; sha256={artifact.get('sha256', '')}; "
            f"characters={artifact.get('chars', 0)}\n"
            f"<<<BEGIN {artifact_id}>>>\n{artifact.get('content', '')}\n<<<END {artifact_id}>>>"
        )
    return (
        "You are the automatic code-authoring agent repairing a generated initial baseline bundle for "
        "MetaVideoAgent.\n\n"
        "This is a fresh repair invocation. Do not rely on prior conversation "
        "state or on reading input paths: every required input file is embedded "
        "below in full and annotated with its role. The original codegen context "
        "is reconstructed from authoritative source files once, rather than "
        "duplicated as an old composed prompt.\n\n"
        + "The smoke stage is for engineering correctness, not answer accuracy. "
        "Read the current candidate source, complete no-label trajectory, runtime "
        "output, and build/API output together. Locate the earliest real "
        "dataflow, control-flow, or interface break and repair all implicated "
        "modules; the listed target modules are starting points, not a restriction. "
        "The deterministic failure boundary is an acceptance contract: satisfy "
        "every falsifiable success condition, rather than hiding empty/error "
        "states or hard-coding a task.\n\n"
        "When repairing thinking, `force_finish=True` is non-negotiable: it must "
        "return `action == \"finish\"` with one concrete parseable `payload.answer`. "
        "After a successful perception result, do not repeat the same verification "
        "without state-changing evidence.\n\n"
        "Write exactly one canonical JSON object for `initial_baseline_bundle` to the requested "
        "path with the same schema as before: `combo`, five executable `modules`, "
        "`design_answers`, `observed_channel_summary`, and `deep_research_summary`. "
        "Do not include Markdown prose in that file.\n\n"
        "The embedded initial-codegen rules, five-module ABI, runtime protocol, "
        "capability manual, and registered profiles are binding. Use only their "
        "documented `runtime_evidence` adapters; every VLM call remains pinned "
        f"to `{INITIAL_VLM_PROFILE}`.\n\n"
        "If `requested_output_path` is provided, write the repaired bundle there "
        "as UTF-8 JSON and return only a short completion status. The host will run real-runtime "
        "preflight and smoke after completion; do not inspect unlisted workspace files.\n\n"
        f"requested_output_path: {output_path}\n\n"
        "## Repair Input Manifest\n\n"
        f"{json.dumps(manifest, ensure_ascii=False, indent=2)}\n\n"
        "## Embedded Repair Input Files\n\n"
        + "\n\n".join(embedded)
        + "\n"
    )


def repair_initial_baseline_bundle_with_smoke_feedback(
    bundle: Dict[str, Any],
    smoke_result: Dict[str, Any],
    *,
    output_dir: str,
    attempt: int = 1,
    codex_cli: str = "",
    timeout: int = 900,
    artifact_tag: str = "smoke",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(artifact_tag or "smoke"))
    repaired_path = os.path.join(
        output_dir, f"initial_baseline_bundle_repaired_{safe_tag}_attempt_{attempt}.json"
    )
    repair_packet_path = os.path.join(
        output_dir, f"initial_smoke_repair_context_{safe_tag}_attempt_{attempt}.json"
    )
    repair_packet = build_smoke_repair_packet(bundle, smoke_result, output_dir=output_dir)
    write_json(repair_packet_path, repair_packet)
    prompt = build_smoke_repair_prompt(
        bundle, smoke_result, repaired_path,
        repair_packet=repair_packet,
    )
    result = _run_codegen_command(
        prompt,
        output_path=repaired_path,
        codex_cli=codex_cli,
        timeout=timeout,
        artifact_label=f"bundle_smoke_repair_{safe_tag}_attempt_{attempt}",
    )
    prompt_path = str(result.get("prompt_path") or "")
    write_text(
        os.path.join(output_dir, f"bundle_smoke_repair_stdout_attempt_{attempt}.txt"),
        result.get("stdout", ""),
    )
    write_text(
        os.path.join(output_dir, f"bundle_smoke_repair_stderr_attempt_{attempt}.txt"),
        result.get("stderr", ""),
    )
    repaired = dict(result.get("bundle") or {})
    # A repair must emit a complete canonical bundle itself.  Do not silently
    # inherit combo/schema/module identity from the failed candidate: that
    # would make a malformed Codex response look repaired.
    repaired["generation_mode"] = "coding_agent_smoke_repair"
    repaired["smoke_repair"] = {
        "attempt": attempt,
        "prompt_path": prompt_path,
        "codegen_ok": bool(result.get("ok")),
        "repair_context_path": repair_packet_path,
        "repair_packet_path": repair_packet_path,
        "repair_context_total_chars": repair_packet.get("total_chars", 0),
        "failure_contract_sha256": sha256_text(json.dumps(
            repair_packet.get("failure_contract", {}), ensure_ascii=False, sort_keys=True,
        )),
        "engineering_execution_trace_path": str(
            smoke_result.get("engineering_execution_trace_path") or ""
        ),
    }
    repaired["runtime_policy"] = {"required_vlm_profile_id": INITIAL_VLM_PROFILE}
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
        "repair_context_total_chars": repair_packet.get("total_chars", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initial-bundle contract helpers used by the public execution entrypoint."
    )
    parser.parse_args()
    parser.error(
        "use execution/run_initial_agent.py; direct automatic bundle generation is not exposed"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
