"""Executable resolution and environments for automatic coding tasks."""

from __future__ import annotations

import os
import shutil

from execution.action_runtime.capability_registry import iter_profiles

CODING_AGENT_PROTOCOL = "codex_exec_compatible"
_RUNTIME_PROVIDER_ENV = {
    "METAVIDEOAGENT_API_KEY",
    "METAVIDEOAGENT_BASE_URL",
}


def coding_agent_environment(
    extra: dict[str, str] | None = None, *, workspace: str = ""
) -> dict[str, str]:
    """Exclude the credentials and endpoints declared by runtime profiles."""
    env = dict(os.environ)
    env.update(extra or {})
    provider_names = _RUNTIME_PROVIDER_ENV | {
        name
        for profile in iter_profiles()
        for field in ("api_key_env", "base_url_env")
        for name in profile.get(field) or []
    }
    for name in provider_names:
        env.pop(name, None)
    if workspace:
        env["PYTHONPATH"] = os.pathsep.join(
            os.path.join(workspace, relative)
            for relative in ("", "evolution", "execution/action_runtime")
        )
        env["METAVIDEOAGENT_CAPABILITY_PROFILES"] = os.path.join(
            workspace,
            "evolution/agent_docs/capabilities/runtime_capability_profiles.json",
        )
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
