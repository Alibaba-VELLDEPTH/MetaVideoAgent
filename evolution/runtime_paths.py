"""Canonical paths for the bundled MetaVideoAgent runtime."""

from __future__ import annotations

import os
import sys

EVOLUTION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVOLUTION_DIR)
METAVIDEOAGENT_RUNTIME = os.path.join(PROJECT_ROOT, "execution", "action_runtime")
METAVIDEOAGENT_WORKSPACE = os.path.join(PROJECT_ROOT, "execution", "workspace")
DEFAULT_OUTPUT_ROOT_NAME = "metavideoagent_outputs"


def runtime_dir() -> str:
    return os.path.abspath(METAVIDEOAGENT_RUNTIME)


def use_runtime_path() -> str:
    path = runtime_dir()
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def runtime_id() -> str:
    return "metavideoagent"


def require_metavideoagent_runtime(context: str = "MetaVideoAgent workflow") -> str:
    path = runtime_dir()
    if not os.path.isdir(path):
        raise RuntimeError(f"{context} requires the bundled runtime at {path}.")
    return path


def default_workspace() -> str:
    return os.path.abspath(METAVIDEOAGENT_WORKSPACE)


def runtime_output_root(path: str = "") -> str:
    return metavideoagent_output_root(path)


def metavideoagent_output_root(path: str = "") -> str:
    return os.path.abspath(
        path
        or os.environ.get("METAVIDEOAGENT_OUTPUT_ROOT")
        or os.path.join(os.getcwd(), DEFAULT_OUTPUT_ROOT_NAME)
    )


def metavideoagent_run_dir(run_id: str = "", output_root: str = "") -> str:
    root = metavideoagent_output_root(output_root)
    if not run_id:
        return root
    if os.path.basename(os.path.normpath(root)) == run_id:
        return root
    return os.path.join(root, run_id)
