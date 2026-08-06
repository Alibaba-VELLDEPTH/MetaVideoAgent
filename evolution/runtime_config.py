"""Early, non-persistent runtime configuration for formal CLI entrypoints."""

from __future__ import annotations

import os
from typing import Iterable

_CLI_ENV = {
    "--api-key": "METAVIDEOAGENT_API_KEY",
    "--base-url": "METAVIDEOAGENT_BASE_URL",
    "--exec-llm-model": "METAVIDEOAGENT_LLM_MODEL",
    "--llm-model": "METAVIDEOAGENT_LLM_MODEL",
    "--asr-model": "METAVIDEOAGENT_ASR_MODEL",
}


def bootstrap_cli_api_env(argv: Iterable[str]) -> None:
    """Apply explicit CLI configuration before importing runtime/provider modules.

    The full argparse validation still occurs in the owning entrypoint.  This
    tiny parser only fixes import order; it never records values and ignores
    absent/malformed flags for the proper parser to reject later.
    """
    values = list(argv or [])
    index = 0
    while index < len(values):
        item = str(values[index])
        flag, sep, value = item.partition("=")
        env_name = _CLI_ENV.get(flag)
        if env_name:
            if not sep and index + 1 < len(values):
                index += 1
                value = str(values[index])
            if value:
                os.environ[env_name] = value
        index += 1


def nonsecret_runtime_audit() -> dict:
    """Report selected profiles/configuration without persisting credentials."""
    return {
        "provider_resolution": "capability_profile_with_optional_cli_overrides",
        "configured": {
            "credential_present": bool(os.environ.get("METAVIDEOAGENT_API_KEY")),
            "base_url_present": bool(os.environ.get("METAVIDEOAGENT_BASE_URL")),
        },
    }
