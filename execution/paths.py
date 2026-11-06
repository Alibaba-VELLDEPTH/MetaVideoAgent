"""Artifact-path helpers for MetaVideoAgent runs."""

from __future__ import annotations

import os

DEFAULT_OUTPUT_ROOT_NAME = "metavideoagent_outputs"


def output_root(path: str = "") -> str:
    configured = path or os.environ.get("METAVIDEOAGENT_OUTPUT_ROOT")
    return os.path.abspath(configured or os.path.join(os.getcwd(), DEFAULT_OUTPUT_ROOT_NAME))


def run_dir(run_id: str, root: str = "") -> str:
    if not run_id:
        raise ValueError("run_id is required")
    root_abs = output_root(root)
    # Accept either the output root or this run's directory.
    if os.path.basename(os.path.normpath(root_abs)) == run_id:
        return root_abs
    return os.path.join(root_abs, run_id)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
