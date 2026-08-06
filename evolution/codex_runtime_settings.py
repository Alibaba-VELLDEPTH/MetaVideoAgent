"""Single-source defaults and validation for every Codex invocation."""

from __future__ import annotations

import os
from typing import Mapping

DEFAULT_CODEX_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING_EFFORT = "medium"


def resolve_codex_runtime_settings(*, model: str = "", reasoning_effort: str = "",
                                   environ: Mapping[str, str] | None = None) -> dict:
    """Return the selected Codex configuration with operational defaults.

    The paper does not prescribe a Codex model/effort pair; users may override
    these shipped operational defaults.
    """
    env = environ if environ is not None else os.environ
    effective_model = str(model or env.get("CODEX_MODEL") or DEFAULT_CODEX_MODEL).strip()
    effective_effort = str(
        reasoning_effort or env.get("CODEX_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT
    ).strip()
    errors = []
    if not effective_model:
        errors.append("model must not be empty")
    allowed_efforts = {"low", "medium", "high", "xhigh", "max", "ultra"}
    if effective_effort not in allowed_efforts:
        errors.append(
            f"reasoning_effort must be one of {sorted(allowed_efforts)}, got {effective_effort!r}"
        )
    if errors:
        raise ValueError("Invalid Codex runtime settings: " + "; ".join(errors))
    return {
        "model": effective_model,
        "reasoning_effort": effective_effort,
    }


def codex_exec_model_args(settings: Mapping[str, str]) -> list[str]:
    """Build only documented Codex CLI overrides from validated settings."""
    return [
        "--model", str(settings["model"]),
        "-c", f'model_reasoning_effort="{settings["reasoning_effort"]}"',
    ]
