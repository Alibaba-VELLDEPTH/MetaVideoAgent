"""Build a runnable SDK workspace for initial authoring and candidate repair."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from execution.action_runtime.capability_registry import PROFILE_MANUAL_PATH

MODULE_BASES = {
    "video_structuring_modules.py": "VideoStructuringBase",
    "localization_modules.py": "LocalizationBase",
    "perception_modules.py": "PerceptionBase",
    "memory_modules.py": "WorkMemoryBase",
    "thinking_module.py": "ThinkingBase",
}


def create_authoring_workspace(parent_dir: str, prefix: str, *, initial: bool = False) -> str:
    project = Path(__file__).resolve().parents[1]
    stage = Path(tempfile.mkdtemp(prefix=prefix, dir=parent_dir))
    try:
        runtime = stage / "execution/action_runtime"
        shutil.copytree(
            project / "execution/action_runtime",
            runtime,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        for relative in (
            "execution/__init__.py",
            "evolution/__init__.py",
            "evolution/runtime_bases.py",
        ):
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project / relative, destination)
        profiles = stage / "evolution/agent_docs/capabilities/runtime_capability_profiles.json"
        profiles.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROFILE_MANUAL_PATH, profiles)
        if initial:
            for filename, base in MODULE_BASES.items():
                (runtime / filename).write_text(
                    f'"""Generated strategies use the shared {base} interface."""\n'
                    f"from evolution.runtime_bases import {base}\n",
                    encoding="utf-8",
                )
        (stage / "STAGING_SCOPE.md").write_text(
            "# MetaVideoAgent implementation workspace\n\n"
            "The execution runtime, shared ABI bases and capability profiles are available here. "
            "Use offline imports and focused interface checks while implementing the requested agent. "
            "The host runs runtime smoke and training evaluation after authoring.\n",
            encoding="utf-8",
        )
        return str(stage)
    except Exception:
        shutil.rmtree(stage)
        raise
