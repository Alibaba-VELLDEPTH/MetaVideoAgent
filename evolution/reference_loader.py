"""Load explicit MetaVideoAgent reference results and normalize trajectory rows."""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterable, List, Tuple

try:
    from answer_normalizer import extract_gold_answer
    from task_identity import attach_task_identity, identity_from_entry, make_task_id
except ImportError:  # pragma: no cover - script-mode fallback
    from .answer_normalizer import extract_gold_answer
    from .task_identity import attach_task_identity, identity_from_entry, make_task_id


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _resolve_report_source(report_path: str) -> str:
    if not report_path or not os.path.exists(report_path):
        return ""
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return ""
    paths = report.get("paths", {}) if isinstance(report.get("paths"), dict) else {}
    verification = (
        report.get("verification", {})
        if isinstance(report.get("verification"), dict) else {}
    )
    source = (
        report.get("results_path")
        or paths.get("results_path")
        or verification.get("results_path")
        or report.get("sandbox_dir")
        or paths.get("sandbox_dir")
        or verification.get("sandbox_dir")
        or ""
    )
    if source and not os.path.isabs(source):
        candidate = os.path.normpath(os.path.join(os.path.dirname(report_path), source))
        if os.path.exists(candidate):
            return candidate
    return source


def load_reference_rows(source: str) -> List[dict]:
    """Load rows from a JSONL file, report JSON, or sandbox directory."""
    if not source:
        return []
    source = os.path.abspath(source)
    if not os.path.exists(source):
        return []
    if os.path.isdir(source):
        rows: List[dict] = []
        patterns = [
            os.path.join(source, "*_sandbox.jsonl"),
            os.path.join(source, "*_results.jsonl"),
            os.path.join(source, "*.jsonl"),
        ]
        seen = set()
        for pattern in patterns:
            for path in sorted(glob.glob(pattern)):
                if path in seen:
                    continue
                seen.add(path)
                rows.extend(_iter_jsonl(path))
        return rows
    if source.endswith(".jsonl"):
        return list(_iter_jsonl(source))
    if source.endswith(".json"):
        resolved = _resolve_report_source(source)
        if resolved and resolved != source:
            return load_reference_rows(resolved)
        try:
            with open(source, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return []
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        rows = payload.get("results") if isinstance(payload, dict) else None
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _answer_from_row(row: dict) -> str:
    return (
        row.get("final_agent_answer")
        or row.get("answer")
        or row.get("final_answer")
        or ""
    )


def normalize_reference_entry(row: dict, reference_label: str = "reference") -> dict:
    """Return a normalized trajectory entry for one result row."""
    if not isinstance(row, dict):
        return {}
    entry = attach_task_identity(dict(row), fallback_video_id=row.get("video_id", ""))
    ident = identity_from_entry(entry)
    meta = dict(entry.get("task_meta") or {})

    question = ident.get("question") or entry.get("question") or meta.get("question", "")
    video_id = ident.get("video_id") or entry.get("video_id") or meta.get("video_id", "")
    time_ref = ident.get("time_reference") or entry.get("time_reference") or meta.get("time_reference", "")
    task_id = ident.get("task_id") or entry.get("task_id") or meta.get("task_id", "")
    if not task_id and video_id and time_ref and question:
        task_id = make_task_id(video_id, time_ref, question)
    if not task_id and not question:
        return {}

    gt = extract_gold_answer(entry)
    meta.update({
        "question": question,
        "gt_answer": gt,
        "answer": meta.get("answer", gt),
        "time_reference": time_ref,
        "video_id": video_id,
        "task_id": task_id,
    })
    entry["gt_answer"] = gt
    normalized = dict(entry)
    normalized["task_meta"] = meta
    normalized["final_agent_answer"] = _answer_from_row(entry)
    normalized.setdefault("trajectory", entry.get("trajectory", []) or [])
    normalized.setdefault(
        "architecture_combo",
        entry.get("architecture_combo") or entry.get("combo") or {},
    )
    normalized["reference_label"] = reference_label
    if task_id:
        normalized["task_id"] = task_id
    if video_id:
        normalized["video_id"] = video_id
    if time_ref:
        normalized["time_reference"] = time_ref
    if question:
        normalized["question"] = question
    return attach_task_identity(normalized, fallback_video_id=video_id)


def load_reference_trajectories(source: str,
                                reference_label: str = "reference") -> List[dict]:
    entries = []
    for row in load_reference_rows(source):
        entry = normalize_reference_entry(row, reference_label=reference_label)
        if entry:
            entries.append(entry)
    return entries


def combo_from_reference(source: str) -> Dict[str, str]:
    """Return the first executable combo found in a reference result source.

    The returned mapping uses the five canonical public module keys.
    """
    default_keys = {
        "video_structuring": "video_structuring",
        "thinking": "thinking",
        "memory": "memory",
        "localization": "localization",
        "perception": "perception",
    }
    for entry in load_reference_trajectories(source, reference_label="reference"):
        raw_combo = (
            entry.get("architecture_combo")
            or entry.get("combo")
            or entry.get("agent_combo")
            or {}
        )
        if not isinstance(raw_combo, dict) or not raw_combo:
            continue
        combo: Dict[str, str] = {}
        for key, value in raw_combo.items():
            mapped = default_keys.get(key)
            if mapped and value:
                combo[mapped] = value
        if combo:
            return combo
    return {}


def group_trajectories_by_ref(entries: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for entry in entries:
        ident = identity_from_entry(entry)
        task_id = ident.get("task_id", "")
        if not task_id:
            continue
        grouped.setdefault(task_id, []).append(entry)
    for rows in grouped.values():
        rows.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
    return grouped


def load_reference_grouped(source: str,
                           reference_label: str = "reference") -> Dict[str, List[dict]]:
    return group_trajectories_by_ref(
        load_reference_trajectories(source, reference_label=reference_label)
    )


def load_reference_by_ref_and_question(source: str,
                                       reference_label: str = "reference") -> Tuple[Dict[str, dict], Dict[str, dict]]:
    by_ref: Dict[str, dict] = {}
    by_question: Dict[str, dict] = {}
    for entry in load_reference_trajectories(source, reference_label=reference_label):
        ident = identity_from_entry(entry)
        task_id = ident.get("task_id", "")
        question = ident.get("question", "")
        if task_id and task_id not in by_ref:
            by_ref[task_id] = entry
        if question and question not in by_question:
            by_question[question] = entry
    return by_ref, by_question
