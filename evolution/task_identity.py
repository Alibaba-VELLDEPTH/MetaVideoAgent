"""Shared task identity and strict answer parsing helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Dict

try:
    from .answer_normalizer import judge_answer
except ImportError:  # Support direct execution from the evolution directory.
    from answer_normalizer import judge_answer


def question_hash(question: str) -> str:
    text = " ".join(str(question or "").split())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def make_task_id(video_id: str, time_reference: str, question: str) -> str:
    return f"{video_id or 'unknown'}::{time_reference or 'unknown'}::{question_hash(question)}"




def identity_from_entry(entry: Dict[str, Any], fallback_video_id: str = "") -> Dict[str, str]:
    meta = entry.get("task_meta") or entry
    video_id = (
        meta.get("video_id")
        or entry.get("video_id")
        or fallback_video_id
        or ""
    )
    time_ref = (
        meta.get("time_reference")
        or entry.get("time_reference")
        or entry.get("question_time_reference")
        or ""
    )
    question = meta.get("question") or entry.get("question") or ""
    task_id = entry.get("task_id") or meta.get("task_id") or make_task_id(
        video_id, time_ref, question
    )
    return {
        "video_id": video_id,
        "time_reference": time_ref,
        "question": question,
        "task_id": task_id,
    }


def attach_task_identity(row: Dict[str, Any], fallback_video_id: str = "") -> Dict[str, Any]:
    ident = identity_from_entry(row, fallback_video_id=fallback_video_id)
    row["video_id"] = ident["video_id"]
    row["time_reference"] = ident["time_reference"]
    row["task_id"] = ident["task_id"]
    meta = row.setdefault("task_meta", {})
    if isinstance(meta, dict):
        meta.setdefault("video_id", ident["video_id"])
        meta.setdefault("time_reference", ident["time_reference"])
        meta.setdefault("question", ident["question"])
        meta.setdefault("task_id", ident["task_id"])
    return row




def judge_exact_answer(answer: str, gt_answer: str) -> bool:
    """Judge an answer when only prediction and gold strings are available."""
    return judge_answer(answer, gt_answer).get("is_correct", False)
