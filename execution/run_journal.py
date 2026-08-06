"""Append-only run journal shared by the MetaVideoAgent entrypoints."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

try:
    from .paths import output_root as resolve_output_root
except ImportError:
    from paths import output_root as resolve_output_root


_SENSITIVE_ARG_TERMS = (
    "api_key",
    "base_url",
    "credential",
    "endpoint",
    "password",
    "secret",
    "token",
)
_URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


def _is_sensitive_arg(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return any(term in normalized for term in _SENSITIVE_ARG_TERMS)


def _safe_args(args: Any) -> dict:
    values = vars(args) if hasattr(args, "__dict__") else {}
    return {
        key: ("<redacted>" if _is_sensitive_arg(key) else value)
        for key, value in values.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def _safe_error(error: Any, args: Any) -> str:
    """Remove run-scoped provider values and URLs from persisted errors."""
    text = str(error or "")
    values = vars(args) if hasattr(args, "__dict__") else {}
    for key, value in values.items():
        if _is_sensitive_arg(key) and isinstance(value, str) and value:
            text = text.replace(value, "<redacted>")
    return _URL_PATTERN.sub("<redacted-url>", text)[:2000]


def append_operation(*, output_root: str, run_id: str, stage: str, args: Any,
                     started_at: float, status: str, report: dict | None = None,
                     error: str = "") -> str:
    root = resolve_output_root(output_root)
    target = os.path.join(root, run_id) if run_id else root
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, "operations.jsonl")
    payload = {
        "artifact_type": "run_operation",
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": time.time(),
        "elapsed_sec": round(time.time() - started_at, 3),
        "stage": stage,
        "status": status,
        "args": _safe_args(args),
        "report_path": (report or {}).get("report_path", ""),
        "output_paths": {key: value for key, value in (report or {}).items() if key.endswith("_path") and isinstance(value, str)},
        "error": _safe_error(error, args),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path
