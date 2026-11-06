"""Select the current best from valid training evaluations and summarize evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

if not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution.config import (
    POST_DECISION_LLM_API_KEY,
    POST_DECISION_LLM_BASE_URL,
    POST_DECISION_LLM_MODEL,
    POST_DECISION_LLM_PROFILE_ID,
    POST_DECISION_LLM_REQUEST_OPTIONS,
)

CURRENT_BEST_SELECTION_SPLIT = "train"


def read_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def full_evaluation_is_valid(report: dict) -> bool:
    """A selection result needs a complete execution and a valid comparison."""
    return (
        report.get("evaluation_complete") is True
        and report.get("comparison_valid") is True
        and not report.get("engineering_invalid")
    )


def apply_current_best_acceptance_policy(decision: dict, report: dict,
                                         audit: dict) -> dict:
    """Prefer higher train accuracy, then non-increasing cost at equal accuracy."""
    if not full_evaluation_is_valid(report):
        raise ValueError("current-best selection requires a complete, engineering-valid evaluation and comparison")
    decision = dict(decision)
    best_correct = int(report["reference_correct"])
    candidate_correct = int(report["candidate_correct"])
    cost_ratio = audit.get("cost_ratio_vs_reference")
    cost_non_increasing = isinstance(cost_ratio, (int, float)) and cost_ratio <= 1.0

    if candidate_correct > best_correct:
        accepted, policy = True, "positive_accuracy_accept"
    elif candidate_correct == best_correct and cost_non_increasing:
        accepted, policy = True, "equal_accuracy_non_increasing_cost_accept"
    elif candidate_correct == best_correct:
        accepted, policy = False, "equal_accuracy_higher_cost_reject"
    else:
        accepted, policy = False, "negative_accuracy_never_accept"

    decision.update({
        "update_current_best": accepted,
        "state_transition": "accepted_new_current_best" if accepted else "rejected_retain_previous_current_best",
        "acceptance_policy": policy,
        "next_action": "continue_evolution",
        "current_best_comparison": {
            "candidate_correct": candidate_correct,
            "current_best_correct": best_correct,
            "total_questions": report["total_questions"],
            "cost_ratio_vs_reference": cost_ratio,
        },
    })
    if report.get("evolution_phase_policy"):
        decision["evolution_phase_policy"] = report["evolution_phase_policy"]
    return decision


def current_best_update_allowed(report: dict, decision: dict) -> bool:
    """Only a valid, accepted training evaluation may promote a candidate."""
    selection = report.get("current_best_selection") or {}
    return bool(
        full_evaluation_is_valid(report)
        and decision.get("update_current_best") is True
        and selection.get("selection_split") == CURRENT_BEST_SELECTION_SPLIT
        and selection.get("this_eval_split") == CURRENT_BEST_SELECTION_SPLIT
    )


def build_prompt(report: dict, audit: dict, decision: dict) -> str:
    """Give the optional explanation model the measured decision and its evidence."""
    evidence = {
        "selection": decision,
        "accuracy_delta": report.get("accuracy_delta_vs_reference", report.get("accuracy_delta")),
        "corrections": len(report.get("corrections") or []),
        "regressions": len(report.get("regressions") or []),
        "candidate_cost": audit.get("candidate_cost") or {},
        "reference_cost": audit.get("reference_cost") or {},
        "issues": audit.get("issue_summary") or [],
        "evidence_files": {
            "evaluation": report.get("report_path", ""),
            "trajectories": report.get("results_path", ""),
            "audit": report.get("evaluation_audit_path", ""),
        },
    }
    return (
        "Explain the recorded training-policy decision for MetaVideoAgent. "
        "Describe the measured improvement or rejection, regression risks, cost tradeoffs, "
        "and useful evidence for the next review. Return JSON with one rationale string.\n\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
    )


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    content = fenced.group(1) if fenced else text[text.find("{"):text.rfind("}") + 1]
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _chat_completion_content(api_key: str, base_url: str, model: str, prompt: str) -> str:
    """Request an evidence summary from the configured compatible endpoint."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1600,
        **POST_DECISION_LLM_REQUEST_OPTIONS,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"] or ""


def decide_post_full_eval(report: dict, audit: dict, *,
                          model: str = "", call_llm: bool = True) -> dict:
    decision = apply_current_best_acceptance_policy({
        "schema_version": 1,
        "decision_source": "train_policy",
        "rationale": "Select by training accuracy, with non-increasing cost required at equal accuracy.",
    }, report, audit)
    if not call_llm or not POST_DECISION_LLM_API_KEY or not POST_DECISION_LLM_BASE_URL:
        return decision
    try:
        content = _chat_completion_content(
            POST_DECISION_LLM_API_KEY, POST_DECISION_LLM_BASE_URL,
            model or POST_DECISION_LLM_MODEL, build_prompt(report, audit, decision),
        )
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        explanation = _extract_json(content)
        if not isinstance(explanation.get("rationale"), str) or not explanation["rationale"].strip():
            raise ValueError("explanation has no rationale")
        decision["rationale"] = explanation["rationale"]
        decision["explanation_provider"] = POST_DECISION_LLM_PROFILE_ID
        decision["explanation_model"] = model or POST_DECISION_LLM_MODEL
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        decision["explanation_error"] = f"{type(exc).__name__}: {exc}"
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Select current best after a valid full evaluation")
    parser.add_argument("--report", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    decision = decide_post_full_eval(
        read_json(args.report), read_json(args.audit),
        model=args.model, call_llm=not args.no_llm,
    )
    write_json(args.out, decision)
    print(f"post_full_eval_decision={args.out}")
    print(f"next_action={decision['next_action']}")
    print(f"update_current_best={decision['update_current_best']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
