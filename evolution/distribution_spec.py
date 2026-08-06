"""Prompt-safe distribution manifest loading for MetaVideoAgent evolution.

The manifest is a run input, not a semantic label source.
This module deliberately strips human descriptions such as "live", "surveillance",
or "narrative" before anything can be injected into Teacher/Diagnosis/Codex prompts.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

PROMPT_LEAK_FIELDS = {
    "description",
    "human_description",
    "human_note",
    "type",
    "type_id",
    "type_name",
    "category",
    "class_name",
    "expected_agent_strategy",
    "dominant_evidence_channels",
    "distribution_profile",
    "strategy",
    "notes",
}

PROMPT_FORBIDDEN_KEY_TERMS = (
    "question",
    "answer",
    "choice",
    "evidence",
    "clue",
    "trajectory",
    "trajector",
    "task",
    "domain",
    "category",
    "type",
)


def _resolve_path(base_dir: str, value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def resolve_distribution_manifest_path(path: str) -> str:
    """Resolve and validate one MetaVideoAgent distribution manifest path."""
    if not path:
        return ""
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"distribution manifest not found: {path}")
    with open(resolved, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("distribution manifest must be a JSON object")
    if not isinstance(raw.get("splits"), dict):
        raise ValueError("distribution manifest must define a splits object")
    return resolved


def load_distribution_spec(path: str) -> Dict[str, Any]:
    """Load a manifest and return only prompt-safe operational fields.

    Allowed fields are split paths and budgets. Human semantic
    descriptions are recorded only as redacted field names so downstream code can
    audit leakage prevention without seeing the content.
    """
    if not path:
        return {}
    path = resolve_distribution_manifest_path(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("distribution manifest must be a JSON object")

    base_dir = os.path.dirname(os.path.abspath(path))
    splits = raw.get("splits") or {}
    if not isinstance(splits, dict):
        raise ValueError("distribution manifest field 'splits' must be an object")
    video_splits = raw.get("video_splits") or raw.get("video_split_paths") or {}
    if video_splits and not isinstance(video_splits, dict):
        raise ValueError("distribution manifest field 'video_splits' must be an object")

    safe_splits = {
        key: _resolve_path(base_dir, value)
        for key, value in splits.items()
        if key in {"train", "val", "validation", "test"} and value
    }
    safe_video_splits = {
        key: _resolve_path(base_dir, value)
        for key, value in video_splits.items()
        if key in {"train", "val", "validation", "test"} and value
    }
    budgets = raw.get("budgets") if isinstance(raw.get("budgets"), dict) else {}
    artifacts_raw = raw.get("artifacts") if isinstance(raw.get("artifacts"), dict) else {}
    semantic_path = (
        raw.get("semantic_channel_observations")
        or raw.get("channel_observations")
        or artifacts_raw.get("semantic_channel_observations")
        or artifacts_raw.get("channel_observations")
        or ""
    )
    artifacts = {}
    if semantic_path:
        artifacts["semantic_channel_observations"] = _resolve_path(base_dir, semantic_path)

    redacted = sorted(k for k in raw.keys() if k in PROMPT_LEAK_FIELDS)
    spec = {
        "manifest_path": os.path.abspath(path),
        "distribution_id": str(raw.get("distribution_id") or raw.get("name") or "target_distribution"),
        "splits": safe_splits,
        "video_splits": safe_video_splits,
        "budgets": budgets,
        "artifacts": artifacts,
        "redacted_prompt_fields": redacted,
        "leakage_policy": (
            "Human type labels/descriptions are not prompt-visible. "
            "Only train/val/test paths, budgets, and the observed "
            "profile generated from data may be used by evolution agents. "
            "Optional semantic channel observation artifacts must be produced "
            "from sampled video/audio/text evidence, not human type labels."
        ),
    }
    return spec


def prompt_safe_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact version safe to include in reports/prompts."""
    if not spec:
        return {}
    budgets = spec.get("budgets", {}) if isinstance(spec.get("budgets"), dict) else {}
    safe_budgets = {
        key: value
        for key, value in budgets.items()
        if not any(term in str(key).lower() for term in PROMPT_FORBIDDEN_KEY_TERMS)
    }
    redacted_runtime_fields = sorted(
        list(set(budgets) - set(safe_budgets))
    )
    return {
        "distribution_id_prompt_visible": False,
        "split_names": sorted((spec.get("splits", {}) or {}).keys()),
        "video_split_names": sorted((spec.get("video_splits", {}) or {}).keys()),
        "split_paths_prompt_visible": False,
        "video_ids_prompt_visible": False,
        "budgets": safe_budgets,
        "artifacts": {
            key: True
            for key in (spec.get("artifacts", {}) or {}).keys()
        },
        "redacted_prompt_fields": spec.get("redacted_prompt_fields", []),
        "redacted_runtime_field_count": len(redacted_runtime_fields),
        "leakage_policy": spec.get("leakage_policy", ""),
    }
