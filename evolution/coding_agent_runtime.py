"""Executable resolution for the single automatic coding-agent workflow.

MetaVideoAgent publicly documents and defaults to Codex.  A compatible wrapper
may be supplied in place of the ``codex`` executable when it implements the
same command contract used by this workflow.  This is an execution adapter,
not a separate authoring or evolution route.
"""

from __future__ import annotations

import os
import shutil

CODING_AGENT_PROTOCOL = "codex_exec_compatible"
_RUNTIME_PROVIDER_ENV = {
    "METAVIDEOAGENT_API_KEY",
    "METAVIDEOAGENT_BASE_URL",
}


def coding_agent_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a subprocess environment without MetaVideoAgent provider secrets."""
    env = dict(os.environ)
    for name in _RUNTIME_PROVIDER_ENV:
        env.pop(name, None)
    env.update(extra or {})
    return env


def resolve_coding_agent_executable(explicit: str = "") -> str:
    """Return an executable implementing the workflow's Codex CLI contract."""
    candidate = str(explicit or "").strip()
    if candidate:
        resolved = shutil.which(candidate) or candidate
        if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved
        raise FileNotFoundError(f"Coding-agent executable is not executable: {candidate}")
    resolved = shutil.which("codex")
    if resolved:
        return resolved
    raise FileNotFoundError(
        "Codex CLI not found. Install Codex, add it to PATH, or pass --codex-cli "
        "with a Codex-compatible coding-agent adapter."
    )
