"""Prompt-safe task views for evolution-layer analysis.

The evolution layer may use question text, answer keys, options, evidence
windows, and stable task identifiers. Benchmark category labels are deliberately
removed because they are not evidence needed to answer the video question.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

BLOCKED_TASK_FIELDS = {
    "question_type",
    "type",
    "sub_category",
    "category",
    "domain",
    "domain_name",
    "dataset_type",
    "video_type",
    "class",
    "class_name",
    "label_type",
}

ALLOWED_TASK_FIELDS = {
    "video_id",
    "video",
    "video_name",
    "task_id",
    "question",
    "query",
    "prompt",
    "options",
    "choices",
    "answer",
    "gt_answer",
    "correct_answer",
    "time_reference",
    "time_ref",
    "evidence_interval",
    "clue_intervals",
    "start",
    "end",
}


def stable_task_id(row: Dict[str, Any]) -> str:
    """Build a deterministic id from fields that are safe to expose."""
    if row.get("task_id"):
        return str(row["task_id"])
    seed = {
        "video_id": row.get("video_id") or row.get("video") or row.get("video_name") or "",
        "time_reference": row.get("time_reference") or row.get("time_ref") or row.get("evidence_interval") or "",
        "question": row.get("question") or row.get("query") or row.get("prompt") or "",
    }
    return hashlib.md5(
        json.dumps(seed, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def sanitize_task_for_evolution(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow, prompt-safe task row.

    This intentionally does not preserve benchmark-provided type/category
    fields. It keeps answer fields because Teacher diagnosis needs the official
    answer to build Gold Path evidence, but callers should still avoid exposing
    these rows to the execution Agent as task metadata.
    """
    row = dict(row or {})
    clean = {
        key: value
        for key, value in row.items()
        if key in ALLOWED_TASK_FIELDS and key not in BLOCKED_TASK_FIELDS
    }
    # ``choices`` is the canonical execution-layer field.  Public datasets
    # commonly use ``options`` instead, so normalize the alias once at the
    # boundary while retaining ``options`` for lossless evaluator metadata.
    if not clean.get("choices") and clean.get("options") not in (None, "", [], {}):
        clean["choices"] = clean["options"]
    # Some datasets export temporal supervision as ``clue_intervals``. Normalize
    # it to the provider-neutral ``time_reference`` field so strict
    # smoke can preserve the declared media scope without falling back to a
    # full video.  This field remains evaluator-only and is never part of the
    # distribution profile or initial-codegen context.
    if not clean.get("time_reference"):
        inherited_reference = (
            row.get("time_ref")
            or row.get("evidence_interval")
            or row.get("clue_intervals")
        )
        if inherited_reference not in (None, "", [], {}):
            clean["time_reference"] = inherited_reference
    clean.pop("clue_intervals", None)
    clean.setdefault("task_id", stable_task_id(clean))
    return clean
