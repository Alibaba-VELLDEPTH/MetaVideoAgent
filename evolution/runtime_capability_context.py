"""Prompt-safe runtime capability context backed by the profile manual.

The execution layer owns provider routing.  Diagnosis, research, and Codex see
only profile IDs and contracts, never keys, base URLs, SDK snippets, or an
unbounded provider catalogue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    from runtime_paths import runtime_dir
except ImportError:  # pragma: no cover - package import
    from .runtime_paths import runtime_dir

CURRENT_DIR = Path(__file__).resolve().parent
MANUAL_PATH = CURRENT_DIR / "agent_docs" / "runtime_capability_manual.md"
try:
    from capability_registry import DEFAULT_PROFILE_IDS, PROFILE_MANUAL_PATH
except ImportError:  # pragma: no cover - package import fallback
    from execution.action_runtime.capability_registry import (
        DEFAULT_PROFILE_IDS,
        PROFILE_MANUAL_PATH,
    )

DEFAULT_VLM_PROFILE_ID = DEFAULT_PROFILE_IDS["vlm"]
DEFAULT_EXECUTION_LLM_PROFILE_ID = DEFAULT_PROFILE_IDS["llm"]


def candidate_runtime_capabilities(diagnosis: dict) -> list[dict]:
    """Carry the latest research plan into execution, including an empty plan.

    Diagnosis suggests a direction; Research selects concrete registered
    adapters. Neither selection locks later code repair to one model profile.
    """
    brief = (diagnosis.get("codex_deep_research") or {}).get("design_brief") or {}
    needs = brief.get("runtime_capability_mapping")
    if needs is None:
        needs = (diagnosis.get("implementation_task") or {}).get("runtime_capabilities") or []
    if not isinstance(needs, list):
        raise ValueError("runtime capability plan must be a list")
    inventory = {item["profile_id"]: item for item in _profile_inventory()}
    plan = []
    for need in needs:
        if not isinstance(need, dict):
            raise ValueError("runtime capability plan entries must be objects")
        item = dict(need)
        capability = str(item.get("capability") or "")
        profile_id = str(item.get("profile_id") or DEFAULT_PROFILE_IDS.get(capability, ""))
        profile = inventory.get(profile_id)
        if not profile or profile.get("capability") != capability:
            raise ValueError(f"runtime capability plan uses an unregistered profile: {capability}/{profile_id}")
        item["profile_id"] = profile_id
        if profile.get("input_provenance"):
            item["input_provenance"] = profile["input_provenance"]
        plan.append(item)
    return plan


def runtime_profile_mismatches(expected: dict, observed: dict) -> dict:
    """Check the configured inventory, not equality with an earlier suggestion.

    A candidate may legitimately use several registered profiles for the same
    capability. Runtime events still must identify profiles in that capability's
    configured inventory; unknown or cross-capability bindings are errors.
    """
    allowed = {}
    for profile in _profile_inventory():
        allowed.setdefault(profile["capability"], set()).add(profile["profile_id"])
    return {
        capability: {"planned": expected.get(capability),
                     "allowed": sorted(allowed.get(capability, set())),
                     "observed": sorted(profiles)}
        for capability, profiles in observed.items()
        if set(profiles) - allowed.get(capability, set())
    }


def _read_manual() -> str:
    try:
        return MANUAL_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Runtime capability manual unavailable. Use only documented adapters. Error: {exc}"


def _profile_inventory() -> list[Dict[str, Any]]:
    try:
        payload = json.loads(PROFILE_MANUAL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    profiles = payload.get("profiles", []) if isinstance(payload, dict) else []
    safe_keys = (
        "profile_id", "capability", "provider", "model_id", "external_capability",
        "transport", "input_provenance", "output_contract", "cost_tier",
        "source_url", "enabled_by_default",
    )
    return [
        {key: profile.get(key) for key in safe_keys if profile.get(key) not in (None, "")}
        for profile in profiles if isinstance(profile, dict) and profile.get("profile_id")
    ]


def build_runtime_capability_context() -> Dict[str, Any]:
    """Return the model-safe execution inventory for planning/codegen prompts."""
    return {
        "context_type": "execution_layer_runtime_capability_manual",
        "manual_path": str(MANUAL_PATH),
        "profile_manual_path": str(PROFILE_MANUAL_PATH),
        "manual_text": _read_manual(),
        "runtime_module": str(Path(runtime_dir()) / "runtime_evidence.py"),
        "profile_selection_policy": [
            "Select a documented profile_id only when the diagnosed mechanism needs a different capability/model profile.",
            f"The configured default VLM profile is `{DEFAULT_VLM_PROFILE_ID}`.",
            f"The configured default execution LLM profile is `{DEFAULT_EXECUTION_LLM_PROFILE_ID}`.",
            "Generated code must call runtime_evidence adapters with profile_id; never construct provider clients, HTTP requests, SDK calls, or CLI commands.",
            "A profile is not proof of local availability. Runtime events and smoke/probe verify credentials, assets, and actual execution.",
            "Only profiles in the selected capability file are available to generated code.",
        ],
        "available_runtime_functions": {
            "llm": {
                "call": "runtime_evidence.call_text_llm(messages, profile_id=..., temperature=..., max_tokens=...)",
                "default_profile": DEFAULT_EXECUTION_LLM_PROFILE_ID,
            },
            "vlm": {
                "call": "runtime_evidence.inspect_active_frames(env, capability='vlm', prompt=..., max_windows=..., frames_per_window=..., profile_id=...)",
                "localized_call": "runtime_evidence.inspect_time_ranges(env, time_ranges, capability='vlm', prompt=..., max_windows=..., frames_per_window=..., profile_id=...)",
                "default_profile": DEFAULT_VLM_PROFILE_ID,
            },
            "asr": {
                "call": "runtime_evidence.transcribe_active_windows(env, max_windows=..., max_seconds_per_window=..., profile_id=...)",
                "localized_call": "runtime_evidence.transcribe_time_ranges(env, time_ranges, max_windows=..., max_seconds_per_window=..., profile_id=...)",
                "default_profile": DEFAULT_PROFILE_IDS["asr"],
            },
            "ocr": {
                "call": "runtime_evidence.inspect_active_frames(env, capability='ocr', prompt=..., max_windows=..., frames_per_window=..., profile_id=...)",
                "localized_call": "runtime_evidence.inspect_time_ranges(env, time_ranges, capability='ocr', prompt=..., max_windows=..., frames_per_window=..., profile_id=...)",
                "default_profile": DEFAULT_PROFILE_IDS["ocr"],
            },
            "embedding": {
                "call": "runtime_evidence.embed_text(input_text, profile_id=...) or runtime_evidence.embed_image(image_path, profile_id=...)",
                "default_profile": DEFAULT_PROFILE_IDS["embedding"],
            },
        },
        "profiles": _profile_inventory(),
        "module_usage_guidance": {
            "video_structuring": [
                "Use a profile only if its channel is necessary for the structure design.",
                "Persist native evidence fields and capability provenance, not a synthetic transcript or description.",
            ],
            "localization": [
                "Use existing structure records first; raw-media profiles remain bounded by active windows during smoke/probe.",
                "Return machine-readable candidate windows that preserved thinking can consume.",
            ],
            "perception": [
                "Use bounded frames/segments and compute confidence from evidence agreement rather than constants.",
                "When a tool accepts localization-produced time_ranges/candidate_windows, use the documented localized_call adapter. Those ranges are executable media selectors, not display metadata; the adapter records actual intersected windows for validation.",
            ],
            "thinking": [
                "Use only registered runtime tools and consume their structured evidence; do not construct provider calls.",
            ],
        },
        "failure_policy": [
            "Return explicit unavailable/error evidence on provider or asset failure.",
            "Do not use unbounded retries, full-video model calls, or provider discovery at runtime.",
        ],
    }
