"""Shared accuracy/cost/trajectory audit for evolution decisions.

This module is intentionally model-free.  It normalizes full-eval and probe
evidence into one compact schema so Teacher review, DiagnosisAgent, Codex
repair, and post-full-eval decisions judge candidates with the same standards.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

try:
    from cost_calculator import compute_cost_from_entry, cost_score
except ImportError:  # pragma: no cover - package import
    from .cost_calculator import compute_cost_from_entry, cost_score


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    if not path or not os.path.exists(path):
        return rows
    paths = [path]
    if os.path.isdir(path):
        paths = [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if name.endswith(".jsonl")
        ]
    for item in paths:
        try:
            with open(item, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return rows


def _task_id(row: dict) -> str:
    meta = row.get("task_meta") if isinstance(row.get("task_meta"), dict) else {}
    return (
        str(row.get("task_id") or "")
        or str(meta.get("task_id") or "")
        or _task_key(row)
    )


def _task_key(row: dict) -> str:
    meta = row.get("task_meta") if isinstance(row.get("task_meta"), dict) else {}
    video_id = row.get("video_id") or meta.get("video_id") or ""
    question = row.get("question") or meta.get("question") or ""
    time_ref = row.get("time_reference") or meta.get("time_reference") or ""
    return f"{video_id}::{time_ref}::{question[:80]}"


def _is_judged(row: dict) -> bool:
    return "is_correct" in row and bool(_task_id(row))


def _raw_row_index(rows: Iterable[dict]) -> Dict[str, dict]:
    """Index raw trajectory rows that may carry exact CostTracker fields."""
    out: Dict[str, dict] = {}
    for row in rows:
        if _is_judged(row):
            continue
        tid = _task_id(row)
        if tid:
            out[tid] = row
    return out


def _judged_row_index(rows: Iterable[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in rows:
        if _is_judged(row):
            out[_task_id(row)] = row
    return out


def _tool_counts(row: dict) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for step in row.get("trajectory", []) or []:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if action:
            counts[action] = counts.get(action, 0) + 1
    return counts


def _max_repeated_tool_calls(tool_counts: Dict[str, int]) -> int:
    if not tool_counts:
        return 0
    return max(int(v or 0) for v in tool_counts.values())


def _safe_ratio(num: float, den: float) -> float | None:
    if den is None or den <= 0:
        return None
    return num / den


def _round_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            out[key] = round(float(value), 4)
        else:
            out[key] = value
    return out


def _cost_for_row(row: dict, raw_row: dict | None = None) -> Dict[str, float]:
    source = raw_row if raw_row and raw_row.get("cost") else row
    metrics = compute_cost_from_entry(source or row or {})
    metrics["cost_score"] = cost_score(metrics)
    return _round_metrics(metrics)


def _aggregate_cost(items: List[dict], prefix: str) -> Dict[str, float]:
    valid = [item for item in items if item.get(prefix)]
    if not valid:
        return {"sample_size": 0}
    keys = [
        "frames", "vlm_calls", "llm_calls", "weighted_calls",
        "total_tokens", "prompt_tokens", "completion_tokens", "ocr_calls",
        "asr_calls", "embedding_calls", "audio_seconds_submitted",
        "unique_frame_assets", "latency_sec", "cost_score",
    ]
    totals = {
        key: sum(float((item[prefix] or {}).get(key, 0)) for item in valid)
        for key in keys
    }
    avg = {key: value / len(valid) for key, value in totals.items()}
    # ``latency_sec`` is the sum of per-task measured latency.  It must not be
    # labelled as wall-clock runtime because questions can execute concurrently.
    return {
        "sample_size": len(valid),
        "totals": _round_metrics(totals),
        "per_question_avg": _round_metrics(avg),
        "latency_semantics": "sum_of_per_task_latency_sec_not_wall_clock",
    }


def _change_sets(report: dict) -> Tuple[set, set]:
    corrections = {
        str(item.get("task_id") or "")
        for item in report.get("corrections", []) or []
        if isinstance(item, dict) and item.get("task_id")
    }
    regressions = {
        str(item.get("task_id") or "")
        for item in report.get("regressions", []) or []
        if isinstance(item, dict) and item.get("task_id")
    }
    return corrections, regressions


def _issue(code: str, severity: str, task_id: str, rationale: str,
           repair_goal: str) -> dict:
    return {
        "code": code,
        "severity": severity,
        "task_id": task_id,
        "rationale": rationale,
        "repair_goal": repair_goal,
    }


def build_full_eval_audit(report: dict,
                          candidate_rows: List[dict],
                          reference_rows: List[dict] | None = None,
                          sandbox_dir: str = "") -> dict:
    """Build a full-eval audit with accuracy, cost, and behavior classes."""
    report = report or {}
    reference_rows = reference_rows or []
    candidate_by_id = _judged_row_index(candidate_rows)
    reference_by_id = _judged_row_index(reference_rows)
    raw_candidate = _raw_row_index(read_jsonl(sandbox_dir))
    corrections, regressions = _change_sets(report)

    per_task = []
    issues = []
    for task_id, row in sorted(candidate_by_id.items()):
        ref = reference_by_id.get(task_id, {})
        candidate_cost = _cost_for_row(row, raw_candidate.get(task_id))
        reference_cost = _cost_for_row(ref, None) if ref else {}
        ratio = _safe_ratio(candidate_cost.get("cost_score", 0), reference_cost.get("cost_score", 0))
        tool_counts = _tool_counts(row)
        is_correct = bool(row.get("is_correct"))
        ref_correct = bool(ref.get("is_correct")) if ref else None
        if task_id in corrections:
            change_type = "correction"
        elif task_id in regressions:
            change_type = "regression"
        elif is_correct and ref_correct:
            change_type = "unchanged_correct"
        elif (not is_correct) and ref_correct is False:
            change_type = "unchanged_wrong"
        else:
            change_type = "unknown"

        if (
            candidate_cost.get("frames", 0) >= 120
            or candidate_cost.get("vlm_calls", 0) >= 3
            or candidate_cost.get("llm_calls", 0) >= 20
            or (ratio is not None and ratio >= 1.8)
        ):
            issues.append(_issue(
                "cost_explosion",
                "high",
                task_id,
                "Candidate consumed many frames, VLM calls, LLM calls, or a high cost ratio on this task.",
                "Reduce redundant retrieval and only escalate to expensive inspection after compact evidence is insufficient.",
            ))
        if _max_repeated_tool_calls(tool_counts) >= 4 or sum(tool_counts.values()) >= 10:
            issues.append(_issue(
                "non_convergent_tool_loop",
                "high",
                task_id,
                "Trajectory repeatedly called one or more runtime tools without quickly converging to a final answer.",
                "Improve evidence handoff and stopping criteria so the thinking loop can answer after sufficient evidence.",
            ))
        if change_type == "regression":
            issues.append(_issue(
                "accuracy_regression",
                "critical",
                task_id,
                "Candidate regressed against the explicit reference/current-best result.",
                "Preserve current-best behavior on this task family while keeping beneficial corrections.",
            ))
        if change_type in ("regression", "unchanged_wrong") and candidate_cost.get("cost_score", 0) > 1.0:
            issues.append(_issue(
                "expensive_wrong_answer",
                "high",
                task_id,
                "Candidate spent substantial compute but still answered incorrectly.",
                "Avoid spending more compute on poorly ranked evidence; improve retrieval precision and verification gating.",
            ))
        if change_type == "correction" and ratio is not None and ratio >= 2.0:
            issues.append(_issue(
                "expensive_correction",
                "medium",
                task_id,
                "A correction was achieved with a much higher cost than the reference.",
                "Keep the corrected behavior but compress the evidence path and reduce repeated inspection.",
            ))

        per_task.append({
            "task_id": task_id,
            "video_id": row.get("video_id", ""),
            "change_type": change_type,
            "candidate_correct": is_correct,
            "reference_correct": ref_correct,
            "candidate_answer": row.get("answer", ""),
            "reference_answer": ref.get("answer", "") if ref else "",
            "gt_answer": row.get("gt_answer", ""),
            "steps": row.get("steps", len(row.get("trajectory", []) or [])),
            "reference_steps": ref.get("steps", len(ref.get("trajectory", []) or [])) if ref else None,
            "tool_counts": tool_counts,
            "candidate_cost": candidate_cost,
            "reference_cost": reference_cost,
            "cost_ratio_vs_reference": round(ratio, 4) if ratio is not None else None,
        })

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    issues.sort(key=lambda item: (severity_order.get(item["severity"], 9), item["code"], item["task_id"]))
    issue_summary: Dict[str, dict] = {}
    for item in issues:
        rec = issue_summary.setdefault(item["code"], {
            "code": item["code"],
            "severity": item["severity"],
            "count": 0,
            "sample_task_ids": [],
            "repair_goal": item["repair_goal"],
        })
        rec["count"] += 1
        if len(rec["sample_task_ids"]) < 8:
            rec["sample_task_ids"].append(item["task_id"])

    candidate_cost = _aggregate_cost(per_task, "candidate_cost")
    reference_cost = _aggregate_cost(per_task, "reference_cost")
    cand_avg = (candidate_cost.get("per_question_avg") or {}).get("cost_score")
    ref_avg = (reference_cost.get("per_question_avg") or {}).get("cost_score")
    cost_ratio = _safe_ratio(cand_avg or 0, ref_avg or 0)

    return {
        "schema_version": 1,
        "audit_type": "full_eval_accuracy_cost_trajectory",
        "report_path": report.get("report_path", ""),
        "results_path": report.get("results_path", ""),
        "sandbox_dir": sandbox_dir or report.get("sandbox_dir", ""),
        "reference_label": report.get("reference_label", ""),
        "total_questions": report.get("total_questions", len(per_task)),
        "candidate_correct": report.get("candidate_correct"),
        "reference_correct": report.get("reference_correct"),
        "accuracy_delta": report.get("accuracy_delta", report.get("accuracy_delta_vs_reference")),
        "corrections": len(report.get("corrections", []) or []),
        "regressions": len(report.get("regressions", []) or []),
        "candidate_cost": candidate_cost,
        "reference_cost": reference_cost,
        "cost_ratio_vs_reference": round(cost_ratio, 4) if cost_ratio is not None else None,
        "issue_summary": sorted(issue_summary.values(), key=lambda item: (severity_order.get(item["severity"], 9), item["code"])),
        "issues": issues,
        "per_task": per_task,
        "decision_guidance": {
            "accuracy_is_primary": True,
            "cost_is_required_for_next_round": True,
            "net_positive_with_cost_explosion": (
                (report.get("accuracy_delta", report.get("accuracy_delta_vs_reference")) or 0) > 0
                and bool(issue_summary.get("cost_explosion"))
            ),
            "recommended_next_focus": _recommended_focus(report, issue_summary),
        },
    }


def _recommended_focus(report: dict, issue_summary: Dict[str, dict]) -> str:
    delta = report.get("accuracy_delta", report.get("accuracy_delta_vs_reference")) or 0
    if report.get("engineering_invalid"):
        return "engineering_repair"
    if delta <= 0:
        return "accuracy_recovery_or_module_shift"
    if issue_summary.get("accuracy_regression") and issue_summary.get("cost_explosion"):
        return "preserve_corrections_repair_regressions_reduce_cost"
    if issue_summary.get("accuracy_regression"):
        return "preserve_corrections_repair_regressions"
    if issue_summary.get("cost_explosion"):
        return "preserve_accuracy_reduce_cost"
    return "consider_finalize_or_target_remaining_failures"


def audit_markdown(audit: dict) -> str:
    lines = [
        "# Evaluation Audit",
        "",
        f"- total_questions: `{audit.get('total_questions')}`",
        f"- candidate_correct: `{audit.get('candidate_correct')}`",
        f"- reference_correct: `{audit.get('reference_correct')}`",
        f"- accuracy_delta: `{audit.get('accuracy_delta')}`",
        f"- corrections: `{audit.get('corrections')}`",
        f"- regressions: `{audit.get('regressions')}`",
        f"- cost_ratio_vs_reference: `{audit.get('cost_ratio_vs_reference')}`",
        f"- recommended_next_focus: `{(audit.get('decision_guidance') or {}).get('recommended_next_focus')}`",
        "",
        "## Issue Summary",
    ]
    for item in audit.get("issue_summary", []) or []:
        lines.append(
            f"- `{item.get('code')}` ({item.get('severity')}, count={item.get('count')}): "
            f"{item.get('repair_goal')}"
        )
    lines.extend(["", "## Cost Averages"])
    for label in ("candidate_cost", "reference_cost"):
        avg = (audit.get(label) or {}).get("per_question_avg") or {}
        totals = (audit.get(label) or {}).get("totals") or {}
        lines.append(
            f"- {label}: frames={avg.get('frames')}, vlm_calls={avg.get('vlm_calls')}, "
            f"llm_calls={avg.get('llm_calls')}, tokens={avg.get('total_tokens')}, "
            f"task_latency_sec={avg.get('latency_sec')}, score={avg.get('cost_score')}"
        )
        lines.append(
            f"  totals: frames={totals.get('frames')}, vlm_calls={totals.get('vlm_calls')}, "
            f"llm_calls={totals.get('llm_calls')}, tokens={totals.get('total_tokens')}, "
            f"task_latency_sec={totals.get('latency_sec')} (sum of per-task latency; not wall-clock)"
        )
    return "\n".join(lines) + "\n"


def write_audit_files(audit: dict, json_path: str, markdown_path: str = "") -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    if markdown_path:
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(audit_markdown(audit))
