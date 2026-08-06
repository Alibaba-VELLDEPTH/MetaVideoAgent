"""Deterministic state transition after a full evaluation.

Each completed, engineering-valid full evaluation has exactly two executable
outcomes: accept the post-evolution combo as the new current best, or retain
the pre-evolution current best. An optional LLM may describe the evidence, but
it never creates a third control-flow branch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request

try:
    from config import (
        POST_DECISION_LLM_API_KEY,
        POST_DECISION_LLM_BASE_URL,
        POST_DECISION_LLM_MODEL,
        POST_DECISION_LLM_PROFILE_ID,
        POST_DECISION_LLM_REQUEST_OPTIONS,
    )
except ImportError:  # pragma: no cover - package import
    from .config import (
        POST_DECISION_LLM_API_KEY,
        POST_DECISION_LLM_BASE_URL,
        POST_DECISION_LLM_MODEL,
        POST_DECISION_LLM_PROFILE_ID,
        POST_DECISION_LLM_REQUEST_OPTIONS,
    )

POST_DECISION_MODEL = POST_DECISION_LLM_MODEL
POST_DECISION_PROVIDER = POST_DECISION_LLM_PROFILE_ID
POST_DECISION_REQUEST_OPTIONS = POST_DECISION_LLM_REQUEST_OPTIONS
# Candidate selection uses only the evolution split. Held-out evaluation is
# read-only reporting and never participates in this state transition.
CURRENT_BEST_SELECTION_SPLIT = "train"


def read_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _compact(obj, limit: int = 12000) -> str:
    text = json.dumps(obj or {}, ensure_ascii=False, indent=2)
    if len(text) > limit:
        return text[:limit] + "\n...<truncated>..."
    return text


DECISION_PROMPT_REDACT_KEYS = {
    "task_id",
    "video_id",
    "question",
    "time_reference",
    "question_time_reference",
    "answer",
    "final_agent_answer",
    "gt_answer",
    "ground_truth",
    "correct_answer",
    "trajectory",
    "task_meta",
    "evidence_interval",
    "evidence_intervals",
    "evidence_window",
    "evidence_windows",
}


def _decision_prompt_safe_payload(value):
    """Remove per-task content that the round-level decider does not need."""
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str in DECISION_PROMPT_REDACT_KEYS:
                if key_str == "trajectory":
                    safe["trajectory_present"] = bool(item)
                    safe["trajectory_steps"] = len(item) if isinstance(item, list) else 0
                elif key_str in {"answer", "final_agent_answer", "gt_answer", "ground_truth", "correct_answer"}:
                    safe[f"{key_str}_present"] = bool(str(item or "").strip())
                elif key_str == "question":
                    safe["question_chars"] = len(str(item or ""))
                else:
                    safe[f"{key_str}_redacted"] = bool(item)
                continue
            safe[key_str] = _decision_prompt_safe_payload(item)
        return safe
    if isinstance(value, list):
        return [_decision_prompt_safe_payload(item) for item in value]
    return value


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return {}


def _metric_score(report: dict) -> tuple:
    correct = int(report.get("candidate_correct") or 0)
    total = int(report.get("total_questions") or 0)
    delta = int(report.get("accuracy_delta") or report.get("accuracy_delta_vs_reference") or 0)
    regressions = len(report.get("regressions", []) or [])
    return correct, total, delta, -regressions


def _ledger_current_best(ledger: dict) -> dict:
    return ledger.get("current_best") or {}


def _ledger_best_score(ledger: dict) -> tuple:
    best = _ledger_current_best(ledger)
    metrics = best.get("metrics") or {}
    return (
        int(metrics.get("correct") or metrics.get("candidate_correct") or 0),
        int(metrics.get("total_questions") or 0),
        int(metrics.get("accuracy_delta") or 0),
        -int(metrics.get("regressions") or 0),
    )


def apply_current_best_acceptance_policy(decision: dict, report: dict,
                                         audit: dict, ledger: dict) -> dict:
    """Make neutral-cost improvements deterministic rather than LLM preference.

    Equal accuracy with non-increasing cost is an accepted current-best update.
    Negative accuracy never becomes current best.  The decision still records
    regressions/corrections for the next causal review instead of discarding
    the candidate's learning evidence.
    """
    decision = dict(decision or {})
    ledger = ledger or {}
    if report.get("evaluation_complete", True) is not True:
        raise ValueError("post-decision requires a completed full eval")
    candidate_correct = int(report.get("candidate_correct") or 0)
    best_correct = _ledger_best_score(ledger)[0]
    has_best = bool(_ledger_current_best(ledger))
    cost_ratio = audit.get("cost_ratio_vs_reference")
    cost_non_increasing = isinstance(cost_ratio, (int, float)) and cost_ratio <= 1.0

    phase = dict(report.get("evolution_phase_policy") or decision.get("evolution_phase_policy") or {})
    compression = phase.get("phase") == "efficiency_compression"
    if compression and has_best:
        # Compression is not a new accuracy-search round.  A candidate must
        # retain the accuracy-best score and strictly improve cost.
        accepted = candidate_correct == best_correct and cost_non_increasing
        policy = "compression_accuracy_guard_cost_accept" if accepted else "compression_guard_or_cost_reject"
    elif not has_best:
        accepted = True
        policy = "first_executed_reference_accept"
    elif candidate_correct > best_correct:
        accepted = True
        policy = "positive_accuracy_accept"
    elif candidate_correct == best_correct and cost_non_increasing:
        accepted = True
        policy = "equal_accuracy_non_increasing_cost_accept"
    elif candidate_correct == best_correct:
        accepted = False
        policy = "equal_accuracy_higher_cost_reject"
    else:
        accepted = False
        policy = "negative_accuracy_never_accept"

    # The branch is a state transition, not an LLM-controlled workflow. Every
    # valid full-eval outcome continues into the next review/diagnosis round.
    decision["update_current_best"] = accepted
    decision["state_transition"] = (
        "accepted_new_current_best"
        if accepted else "rejected_retain_previous_current_best"
    )
    decision["acceptance_policy"] = policy
    decision["evolution_phase_policy"] = phase
    decision["next_action"] = "continue_evolution"
    decision["target_for_next_round"] = (
        "regression_repair" if accepted else "candidate_failure_analysis"
    )
    decision["next_round_evidence_policy"] = (
        "Use the accepted post-evolution combo as current best; retain this "
        "round's corrections and regressions as preservation/guard evidence."
        if accepted else
        "Retain the previous current best; carry this rejected candidate's "
        "corrections, regressions, costs, and engineering evidence into the next diagnosis."
    )
    if accepted and candidate_correct == best_correct:
        decision["candidate_quality"] = "accepted_neutral_cost"
    return decision


def current_best_update_allowed(report: dict, decision: dict) -> bool:
    """Return whether an accepted full evaluation may update the current best."""
    report = report or {}
    decision = decision or {}
    selection = report.get("current_best_selection") or {}
    if not isinstance(selection, dict) or not selection:
        return False
    selection_split = str(selection.get("selection_split") or "")
    evaluated_split = str(selection.get("this_eval_split") or "")
    if selection_split != CURRENT_BEST_SELECTION_SPLIT:
        return False
    if evaluated_split != CURRENT_BEST_SELECTION_SPLIT:
        return False
    return bool(
        decision.get("update_current_best") is True
        and report.get("evaluation_complete", True)
    )


def heuristic_decision(report: dict, audit: dict, ledger: dict = None,
                       reason: str = "") -> dict:
    ledger = ledger or {}
    if report.get("evaluation_complete", True) is not True:
        raise ValueError("post-decision requires a completed full eval")
    candidate_score = _metric_score(report)
    best_score = _ledger_best_score(ledger)
    has_best = bool(_ledger_current_best(ledger))
    delta = int(report.get("accuracy_delta") or report.get("accuracy_delta_vs_reference") or 0)
    corrections = len(report.get("corrections", []) or [])
    regressions = len(report.get("regressions", []) or [])
    issue_codes = {item.get("code") for item in audit.get("issue_summary", []) or []}
    # `apply_current_best_acceptance_policy()` owns the executable decision.
    # Keep this value only as an audit hint for the heuristic payload; do not
    # reference an undefined completion flag when the LLM fallback is used.
    update_best = not has_best or candidate_score > best_score

    if delta > 0 and corrections >= 1:
        quality = "good" if regressions or "cost_explosion" in issue_codes else "excellent"
        next_action = "continue_evolution"
        if regressions and "cost_explosion" in issue_codes:
            target = "cost_and_regression_repair"
        elif regressions:
            target = "regression_repair"
        elif "cost_explosion" in issue_codes:
            target = "cost"
        else:
            target = "accuracy_or_finalize"
    else:
        quality = "mixed" if corrections else "bad"
        next_action = "continue_evolution"
        target = "accuracy"

    return apply_current_best_acceptance_policy({
        "schema_version": 1,
        "decision_source": "heuristic_fallback",
        "candidate_quality": quality,
        "update_current_best": bool(update_best),
        "next_action": next_action,
        "target_for_next_round": target,
        "rationale": reason or "Heuristic decision from full-eval metrics and evaluation audit.",
        "required_next_inputs": [
            "full_eval_report.json",
            "full_eval_results.jsonl",
            "evaluation_audit.json",
            "current_best_ledger.json",
        ],
        "current_best_comparison": {
            "candidate_score": candidate_score,
            "ledger_best_score": best_score,
            "has_ledger_best": has_best,
        },
    }, report, audit, ledger)


def build_prompt(report: dict, audit: dict, ledger: dict) -> str:
    return f"""You are the post-full-eval decision judge for an autonomous video-agent evolution loop.

Describe the evidence for one of two deterministic state transitions: accept
the just-tested post-evolution combo as current best, or retain the
pre-evolution current best. The run always continues to the next round;
all corrections, regressions, and costs are retained.

Hard constraints:
- Do not update current_best unless this candidate is better than the ledger current_best, or it has equal accuracy with non-increasing cost. Equal-accuracy, lower-cost candidates are accepted and remain subject to later regression repair.
- A net-positive candidate may update current_best even with regressions, but regressions and cost explosion should usually push next_action=continue_evolution.
- Return JSON only.

Required JSON schema:
{{
  "schema_version": 1,
  "decision_source": "llm",
  "candidate_quality": "excellent|good|mixed|bad|invalid",
  "update_current_best": true,
  "next_action": "continue_evolution",
  "target_for_next_round": "accuracy|cost|stability|regression_repair|cost_and_regression_repair|module_shift|candidate_failure_analysis",
  "rationale": "short evidence-grounded explanation",
  "required_next_inputs": ["..."],
  "current_best_comparison": {{
    "candidate_is_better_than_ledger_best": true,
    "comparison_basis": "correct/delta/cost/regression evidence"
  }}
}}

FULL_EVAL_REPORT:
{_compact(_decision_prompt_safe_payload(report), 14000)}

EVALUATION_AUDIT:
{_compact(_decision_prompt_safe_payload(audit), 16000)}

CURRENT_BEST_LEDGER:
{_compact(_decision_prompt_safe_payload(ledger), 8000)}
"""


def _post_decision_provider_config() -> tuple[str, str]:
    """Resolve the configured OpenAI-compatible route at call time.

    Full-eval runners may inject the explicit MetaVideoAgent key immediately before
    spawning this stage.  Looking it up here prevents an imported config
    snapshot from silently falling back to a different provider or heuristic.
    """
    api_key = POST_DECISION_LLM_API_KEY
    base_url = POST_DECISION_LLM_BASE_URL
    return str(api_key or ""), str(base_url or "")


def _chat_completion_content(api_key: str, base_url: str, model: str, prompt: str) -> str:
    """Call the selected judge through an OpenAI-compatible endpoint.

    The project environment may combine an older openai SDK with a newer httpx
    release, which breaks SDK construction around proxy handling. This direct
    request keeps the post-eval judge independent from that dependency pair.
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1600,
    }
    payload.update(POST_DECISION_REQUEST_OPTIONS)
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")


def decide_post_full_eval(report: dict, audit: dict, ledger: dict = None,
                          model: str = "", call_llm: bool = True) -> dict:
    ledger = ledger or {}
    if report.get("evaluation_complete", True) is not True:
        raise ValueError("post-decision requires a completed full eval")
    if not call_llm:
        return heuristic_decision(report, audit, ledger, reason="LLM decision disabled by caller.")
    decision_model = str(model or POST_DECISION_MODEL)
    api_key, base_url = _post_decision_provider_config()
    if not api_key or not base_url:
        return heuristic_decision(
            report,
            audit,
            ledger,
            reason="The configured post-evaluation judge is unavailable.",
        )
    try:
        content = _chat_completion_content(
            api_key,
            base_url,
            decision_model,
            build_prompt(report, audit, ledger),
        )
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        parsed = _extract_json(content or "")
        if not parsed:
            return heuristic_decision(report, audit, ledger, reason="LLM returned unparsable decision JSON.")
        parsed.setdefault("schema_version", 1)
        parsed.setdefault("decision_source", "llm")
        parsed.setdefault("decision_provider", POST_DECISION_PROVIDER)
        parsed.setdefault("decision_model", decision_model)
        parsed.setdefault("decision_request_options", dict(POST_DECISION_REQUEST_OPTIONS))
        parsed.setdefault("required_next_inputs", [
            "full_eval_report.json",
            "full_eval_results.jsonl",
            "evaluation_audit.json",
            "current_best_ledger.json",
        ])
        return apply_current_best_acceptance_policy(parsed, report, audit, ledger)
    except Exception as exc:
        return heuristic_decision(report, audit, ledger, reason=f"LLM decision failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide what to do after full eval")
    parser.add_argument("--report", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--ledger", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    decision = decide_post_full_eval(
        read_json(args.report),
        read_json(args.audit),
        read_json(args.ledger),
        model=args.model,
        call_llm=not args.no_llm,
    )
    write_json(args.out, decision)
    print(f"post_full_eval_decision={args.out}")
    print(f"next_action={decision.get('next_action')}")
    print(f"update_current_best={decision.get('update_current_best')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
