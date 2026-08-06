"""Initial combo design from label-blind MetaVideoAgent distribution evidence.

This module creates the pre-evolution reference point:

1. Inspect the distribution/channel profile produced from the training set.
2. Answer the five module-design questions without seeing human type labels.
3. Emit a prompt-safe design brief for Codex initial bundle generation.

The implementation is deterministic and deliberately avoids registered module
keys.  The executable initial combo must be generated as a five-module bundle by
Codex after distribution profiling and deep research.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _channel_hypothesis(profile: Dict[str, Any]) -> Dict[str, Any]:
    hypothesis = (
        profile.get("information_channel_hypothesis")
        or profile.get("channel_hypothesis")
    )
    if isinstance(hypothesis, dict):
        return hypothesis
    return {}


def _ranked_channels(profile: Dict[str, Any]) -> list:
    hypothesis = _channel_hypothesis(profile)
    ranked = hypothesis.get("ranked_channels")
    if isinstance(ranked, list):
        return ranked
    scores = (
        hypothesis.get("channel_scores")
        if isinstance(hypothesis.get("channel_scores"), dict) else
        hypothesis.get("scores")
        if isinstance(hypothesis.get("scores"), dict) else
        {}
    )
    return [
        {"channel": channel, "score": score}
        for channel, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def _genericize_runtime_names(value: Any) -> Any:
    replacements = {
        "registered module": "generated module",
    }
    if isinstance(value, dict):
        return {key: _genericize_runtime_names(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_genericize_runtime_names(item) for item in value]
    if isinstance(value, str):
        text = value
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    return value


def _research_module_guidance(design_brief: Dict[str, Any]) -> Dict[str, str]:
    """Extract compact, source-backed module guidance for the initial plan."""
    design = design_brief.get("baseline_combo_design_spec") if isinstance(design_brief, dict) else {}
    if not isinstance(design, dict):
        return {}
    guidance: Dict[str, str] = {}
    for module in ("video_structuring", "localization", "perception", "memory", "thinking"):
        item = design.get(module)
        if not isinstance(item, dict):
            continue
        parts = [
            str(item.get("class_goal") or "").strip(),
            *[str(value).strip() for value in (item.get("core_mechanisms") or []) if str(value).strip()],
            *[str(value).strip() for value in (item.get("implementation_notes") or []) if str(value).strip()],
        ]
        text = " ".join(parts).strip()
        if text:
            guidance[module] = text[:1200]
    return guidance


def design_initial_combo(profile: Dict[str, Any],
                         manifest_path: str = "",
                         output_path: str = "",
                         deep_research_brief: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a prompt-safe initial combo plan.

    The plan records module choices plus design answers.  It deliberately avoids
    human category/type labels and only references observed channel statistics.
    """
    hypothesis = _channel_hypothesis(profile)
    primary = (
        hypothesis.get("primary_observed_channel")
        or hypothesis.get("primary_channel")
        or hypothesis.get("primary")
        or "mixed_or_unknown"
    )
    ranked = _ranked_channels(profile)
    raw_video_summary = (
        (profile.get("raw_video_signal_profile") or {}).get("summary")
        if isinstance(profile.get("raw_video_signal_profile"), dict) else {}
    ) or {}
    question_stats = (
        profile.get("observed_information_channels")
        or profile.get("question_stats")
        or {}
    )
    long_context_ratio = 0.0
    if isinstance(question_stats.get("channel_hit_ratio"), dict):
        long_context_ratio = float(
            question_stats["channel_hit_ratio"].get("long_context_reasoning", 0.0) or 0.0
        )

    if primary in ("speech_or_audio", "screen_text_or_subtitle"):
        struct_strategy = (
            "Generate a structuring module that preserves transcript/audio cues, "
            "screen text, visible actions, and timestamps in one evidence schema."
        )
        localization_strategy = (
            "Generate channel-aware localization that tries speech/text evidence "
            "early while keeping bounded visual verification fallback."
        )
    elif primary in ("visual_action", "visual_object_detail"):
        struct_strategy = (
            "Generate a visual-event/object evidence table with entities, object "
            "states, spatial relations, and visible state changes."
        )
        localization_strategy = (
            "Generate visual-aware localization that decomposes events/states and "
            "compares bounded windows instead of increasing search breadth blindly."
        )
    else:
        struct_strategy = (
            "Generate a conservative mixed-channel evidence store without assuming "
            "a human type label."
        )
        localization_strategy = (
            "Generate a bounded localization policy that first tests the most "
            "plausible observed channel and records uncertainty for later review."
        )

    if long_context_ratio >= 0.25:
        thinking_strategy = (
            "Generate thinking that tracks cross-window entities, relations, and "
            "state changes while keeping final answers normalized."
        )
    else:
        thinking_strategy = (
            "Generate bounded ReAct thinking that chooses an evidence channel, "
            "verifies it, and finalizes in the expected answer form."
        )

    design_answers = {
        "video_structuring": {
            "selected": "codex_generated_video_structuring_module",
            "answer": struct_strategy,
        },
        "localization": {
            "selected": "codex_generated_localization_module",
            "answer": localization_strategy,
        },
        "perception": {
            "selected": "codex_generated_perception_module",
            "answer": (
                "Generate perception tools that inspect only candidate windows and "
                "choose visual/text/audio verification according to the question."
            ),
        },
        "memory": {
            "selected": "codex_generated_memory_module",
            "answer": (
                "Generate memory that preserves searched windows, evidence channel, "
                "entities/states, contradictions, and answer-normalization decisions."
            ),
        },
        "thinking": {
            "selected": "codex_generated_thinking_module",
            "answer": thinking_strategy,
        },
    }
    research_guidance = _research_module_guidance(deep_research_brief or {})
    for module, guidance in research_guidance.items():
        if module in design_answers:
            design_answers[module]["answer"] += f" Research-backed implementation direction: {guidance}"

    capability_blueprints = []
    if primary == "speech_or_audio":
        capability_blueprints.append({
            "module_type": "localization",
            "capability": "speech_text_temporal_localization",
            "rationale": (
                "Observed distribution has high speech/audio evidence. The initial "
                "bundle should support speech/text localization and only expand it "
                "after trajectory evidence shows missed or poorly verified windows."
            ),
            "interface_contract": "Add a LocalizationBase-compatible class with generated tool_ methods; preserve bounded fallback behavior inside the generated code.",
        })
        capability_blueprints.append({
            "module_type": "video_structuring",
            "capability": "audio_text_aligned_structure",
            "rationale": (
                "Structure should preserve spoken claims, speaker turns, screen text, "
                "and their timestamps so long videos do not require visual-only scans."
            ),
            "interface_contract": "Add addStructure/retrieveStructure-compatible structuring class; no hard-coded task IDs.",
        })
    elif primary in ("visual_action", "visual_object_detail"):
        capability_blueprints.append({
            "module_type": "memory",
            "capability": "entity_state_or_role_graph_memory",
            "rationale": (
                "Observed visual/story evidence suggests failures may come from "
                "tracking entities, roles, object states, and cross-clip relations. "
                "The generated memory should preserve enough structured state to "
                "support later diagnosis."
            ),
            "interface_contract": "Keep WorkMemoryBase-compatible methods and expose compact entity/state summaries.",
        })
        capability_blueprints.append({
            "module_type": "localization",
            "capability": "visual_event_interval_decomposition",
            "rationale": (
                "Future localization should decompose visually rich long videos "
                "into event/state subqueries rather than increasing top_k blindly."
            ),
            "interface_contract": "Add generated LocalizationBase tool_ methods and use only registered runtime adapters.",
        })
    elif primary == "mixed_or_unknown":
        capability_blueprints.append({
            "module_type": "thinking",
            "capability": "channel_uncertainty_planner",
            "rationale": (
                "When the observed channel is uncertain, thinking should first choose "
                "which evidence channel to test from trajectory feedback instead of "
                "assuming a fixed video platform type."
            ),
            "interface_contract": "Keep ThinkingBase __init__/__call__ compatibility.",
        })
    else:
        capability_blueprints.append({
            "module_type": "video_structuring",
            "capability": f"{primary}_aware_evidence_structure",
            "rationale": (
                "Use the observed channel profile as auxiliary evidence for the first "
                "evolution target after Teacher reviews concrete failures."
            ),
            "interface_contract": "Keep video_structuring addStructure/retrieveStructure compatibility.",
        })

    plan = {
        "artifact_type": "metavideoagent_initial_combo_plan",
        "schema_version": 1,
        "created_at": int(time.time()),
        "source": "five_frame_query_aware_distribution_profile",
        "distribution_manifest": os.path.abspath(manifest_path) if manifest_path else "",
        "leakage_policy": (
            "The initial combo designer uses five-frame per-video observations and "
            "associated questions/options. Answers, evidence intervals, trajectories, "
            "and human type/category descriptions are not prompt-visible."
        ),
        "observed_profile_summary": {
            "primary_channel": primary,
            "ranked_channels": ranked[:5],
            "question_stats": question_stats,
            "raw_video_signal_summary": raw_video_summary,
        },
        "combo": {},
        "custom_configs": {},
        "design_answers": _genericize_runtime_names(design_answers),
        "initial_capability_blueprints": _genericize_runtime_names(capability_blueprints),
        "initial_execution_policy": {
            "asr_profile_id": os.environ.get("METAVIDEOAGENT_ASR_PROFILE", "paper.asr.default"),
            "vlm_profile_id": os.environ.get("METAVIDEOAGENT_VLM_PROFILE", "paper.vlm.default"),
            "structure_build_lifecycle": (
                "The execution-layer builder performs bounded media sampling and model calls. "
                "The generated video_structuring module persists, indexes, and retrieves supplied records; "
                "addStructure must not issue model calls."
            ),
        },
        "deep_research_input": {
            "consumed": bool(deep_research_brief),
            "design_brief_sha256": (
                hashlib.sha256(json.dumps(deep_research_brief, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
                if deep_research_brief else ""
            ),
            "module_guidance": _genericize_runtime_names(research_guidance),
        },
        "initial_codegen_policy": (
            "This deterministic plan is design context only. It must not be "
            "executed as a baseline combo. Initial execution must use the "
            "Codex-generated initial_baseline_bundle, then Diagnosis/Evolution "
            "must use concrete trajectory failures as the primary signal."
        ),
        "next_step": (
            "Generate a Codex initial_baseline_bundle from this profile/research "
            "context, run that bundle, then treat its trajectories as the initial "
            "reference."
        ),
    }
    if output_path:
        _write_json(output_path, plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Design an initial combo from a label-blind channel profile")
    parser.add_argument("--profile", required=True, help="observed_distribution_profile.json")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--deep-research-report", default="",
                        help="Validated initial Codex deep-research report consumed by this plan")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    profile = _read_json(args.profile)
    research_report = _read_json(args.deep_research_report) if args.deep_research_report else {}
    plan = design_initial_combo(
        profile,
        manifest_path=args.manifest,
        output_path=args.output,
        deep_research_brief=(research_report.get("design_brief") or {}),
    )
    print("INITIAL_COMBO_PLAN")
    print(f"output={args.output}")
    print(f"primary_channel={plan['observed_profile_summary'].get('primary_channel')}")
    print(f"combo={json.dumps(plan['combo'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
