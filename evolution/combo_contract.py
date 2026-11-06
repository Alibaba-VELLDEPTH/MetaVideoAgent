"""The five module slots in an executable MetaVideoAgent combo."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

COMBO_KEYS: Tuple[str, ...] = (
    "video_structuring",
    "thinking",
    "memory",
    "localization",
    "perception",
)


def normalize_agent_combo(combo: dict | None) -> Dict[str, str]:
    if not isinstance(combo, dict):
        return {}
    return {key: combo[key] for key in COMBO_KEYS if combo.get(key)}


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
