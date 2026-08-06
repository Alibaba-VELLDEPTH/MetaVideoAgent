"""Combo contract helpers for MetaVideoAgent evolution.

The MetaVideoAgent workflow must not silently fall back to a fixed baseline combo.
These helpers enforce one public combo-key vocabulary and fail fast when a
formal MetaVideoAgent stage lacks one of the five required module slots.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

COMBO_KEYS: Tuple[str, ...] = (
    "video_structuring",
    "thinking",
    "memory",
    "localization",
    "perception",
)

_CANONICAL_KEYS = {
    "video_structuring": "video_structuring",
    "thinking": "thinking",
    "memory": "memory",
    "localization": "localization",
    "perception": "perception",
}


def normalize_agent_combo(combo: dict | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(combo, dict):
        return out
    for key, value in combo.items():
        mapped = _CANONICAL_KEYS.get(key)
        if mapped and value:
            out[mapped] = value
    return out


def normalize_evolution_combo(combo: dict | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(combo, dict):
        return out
    for key, value in combo.items():
        mapped = _CANONICAL_KEYS.get(key)
        if mapped and value:
            out[mapped] = value
    return out


def missing_combo_keys(combo: dict, required: Iterable[str]) -> list[str]:
    return [key for key in required if not combo.get(key)]


def require_complete_agent_combo(combo: dict | None, source: str = "combo") -> Dict[str, str]:
    normalized = normalize_agent_combo(combo)
    missing = missing_combo_keys(normalized, COMBO_KEYS)
    if missing:
        raise RuntimeError(
            f"{source} must explicitly define all five module slots; "
            f"missing {missing}. MetaVideoAgent requires an explicit complete combo."
        )
    return normalized


def require_complete_evolution_combo(combo: dict | None, source: str = "combo") -> Dict[str, str]:
    normalized = normalize_evolution_combo(combo)
    missing = missing_combo_keys(normalized, COMBO_KEYS)
    if missing:
        raise RuntimeError(
            f"{source} must explicitly define all five module slots; "
            f"missing {missing}. MetaVideoAgent requires an explicit complete combo."
        )
    return normalized



