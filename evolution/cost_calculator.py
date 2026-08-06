"""Compute normalized provider and execution costs from trajectory records."""

from typing import Dict, List

# ============================================================
# Relative capability weights used by the paper-reproduction configuration.
# ============================================================
MODEL_WEIGHTS = {
    # Multimodal calls dominate frame-processing cost.
    "vlm": 3.0,
    # Plain text LLM
    "llm": 1.0,
    # Embedding calls have a small relative weight.
    "embedding": 0.05,
    # ASR
    "asr": 0.1,
}

# Approximate tokens per frame for multi-image VLM encoding.
TOKENS_PER_FRAME = 258

# Normalization basis (used to map absolute values to the 0-1 range)
NORM_FRAMES = 50       # 50 frames is considered 1.0
NORM_WEIGHTED_CALLS = 10  # 10 weighted calls count as 1.0
NORM_TOKENS = 50000    # 50k tokens are considered 1.0
NORM_LATENCY = 60      # 60s is considered 1.0

# Aggregate metric weights.
W_FRAMES = 0.4
W_CALLS = 0.3
W_TOKENS = 0.2
W_LATENCY = 0.1

# Hard constraint: the evolution cost must not exceed 1.2 times the baseline
MAX_COST_RATIO = 1.2


def _looks_like_visual_tool(action: str, observation: str = "") -> bool:
    text = f"{action} {observation[:500]}".lower()
    visual_terms = (
        "visual", "vision", "vlm", "ocr", "frame", "image", "video",
        "perception", "observe", "verify", "evidence_window",
        "Vision", "picture", "frame extraction", "pictures", "video", "Review",
    )
    return any(term in text for term in visual_terms)


def _looks_like_retrieval_tool(action: str, observation: str = "") -> bool:
    text = f"{action} {observation[:500]}".lower()
    retrieval_terms = (
        "retrieve", "retrieval", "search", "localize", "locate", "window",
        "structure", "semantic", "embedding", "Search", "Positioning", "structure",
    )
    return any(term in text for term in retrieval_terms)


def _looks_like_audio_tool(action: str, observation: str = "") -> bool:
    text = f"{action} {observation[:500]}".lower()
    audio_terms = ("audio", "asr", "speech", "transcript", "voice", "Audio", "Voice", "Transcribe")
    return any(term in text for term in audio_terms)


def compute_cost_from_entry(entry: dict) -> Dict[str, float]:
    """Extract cost metrics from one trajectory record.

    Provider-reported values take precedence; otherwise the function estimates
    cost from the recorded reasoning and tool steps.
    """
    cost_data = entry.get("cost")

    if cost_data and isinstance(cost_data, dict):
        # Exact mode: data recorded directly with CostTracker
        frames = cost_data.get("vision_frame_inputs", cost_data.get("frames_viewed", 0))
        llm_calls = cost_data.get("llm_calls", 0)
        vlm_calls = cost_data.get("vlm_calls", 0)
        total_tokens = cost_data.get("total_tokens", 0)
        latency = cost_data.get("latency_sec", 0)

        # Weighted number of calls
        weighted_calls = (
            vlm_calls * MODEL_WEIGHTS["vlm"]
            + llm_calls * MODEL_WEIGHTS["llm"]
        )

        return {
            "frames": frames,
            "vlm_calls": vlm_calls,
            "llm_calls": llm_calls,
            "weighted_calls": weighted_calls,
            "total_tokens": total_tokens,
            "latency_sec": latency,
            "prompt_tokens": cost_data.get("prompt_tokens", 0),
            "completion_tokens": cost_data.get("completion_tokens", 0),
            "ocr_calls": cost_data.get("ocr_calls", 0),
            "asr_calls": cost_data.get("asr_calls", 0),
            "embedding_calls": cost_data.get("embedding_calls", 0),
            "audio_seconds_submitted": cost_data.get("audio_seconds_submitted", 0),
            "unique_frame_assets": cost_data.get("unique_frame_assets", frames),
            "usage_source": "provider_reported",
        }

    # Fallback: estimated from trajectory
    trajectory = entry.get("trajectory", [])
    frames = 0
    vlm_calls = 0
    llm_calls = 0
    token_estimate = 0
    latency = 0

    # Calculate delay from timestamp
    timestamps = [s.get("timestamp", 0) for s in trajectory if s.get("timestamp")]
    if len(timestamps) >= 2:
        latency = timestamps[-1] - timestamps[0]

    for step in trajectory:
        step_type = step.get("step_type", "")

        if step_type == "reasoning":
            llm_calls += 1
            thought = step.get("thought", "")
            token_estimate += len(thought) // 4  # Rough estimate of output tokens

        elif step_type == "tool_execution":
            action = step.get("action", "")
            obs = step.get("observation", "")
            token_estimate += len(obs) // 4

            if _looks_like_visual_tool(action, obs):
                vlm_calls += 1
                frames += 25

            elif "browse" in action.lower() or "summar" in action.lower():
                llm_calls += 1  # Internal LLM summary

            elif _looks_like_retrieval_tool(action, obs):
                pass  # embedding call, cost ignored

            elif _looks_like_audio_tool(action, obs):
                pass  # ASR costs very little

    weighted_calls = (
        vlm_calls * MODEL_WEIGHTS["vlm"]
        + llm_calls * MODEL_WEIGHTS["llm"]
    )
    total_tokens = token_estimate + frames * TOKENS_PER_FRAME

    return {
        "frames": frames,
        "vlm_calls": vlm_calls,
        "llm_calls": llm_calls,
        "weighted_calls": weighted_calls,
        "total_tokens": total_tokens,
        "latency_sec": latency,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "ocr_calls": 0,
        "asr_calls": 0,
        "embedding_calls": 0,
        "audio_seconds_submitted": 0,
        "unique_frame_assets": frames,
        "usage_source": "trajectory_estimated",
    }


def cost_score(metrics: Dict[str, float]) -> float:
    """Return a normalized aggregate score; larger values indicate higher cost."""
    return (
        W_FRAMES * (metrics["frames"] / NORM_FRAMES)
        + W_CALLS * (metrics["weighted_calls"] / NORM_WEIGHTED_CALLS)
        + W_TOKENS * (metrics["total_tokens"] / NORM_TOKENS)
        + W_LATENCY * (metrics["latency_sec"] / NORM_LATENCY)
    )



def compute_reference_cost(reference_entries: List[dict]) -> Dict:
    """Compute the same budget from explicit reference trajectories."""
    entries = [entry for entry in reference_entries or [] if isinstance(entry, dict)]
    if not entries:
        return {"error": "reference trajectories are empty"}
    all_metrics = [compute_cost_from_entry(e) for e in entries]
    n = len(all_metrics)
    avg_metrics = {
        "frames": sum(m["frames"] for m in all_metrics) / n,
        "vlm_calls": sum(m["vlm_calls"] for m in all_metrics) / n,
        "llm_calls": sum(m["llm_calls"] for m in all_metrics) / n,
        "weighted_calls": sum(m["weighted_calls"] for m in all_metrics) / n,
        "total_tokens": sum(m["total_tokens"] for m in all_metrics) / n,
        "latency_sec": sum(m["latency_sec"] for m in all_metrics) / n,
    }
    avg_score = cost_score(avg_metrics)
    return {
        "per_question_avg": avg_metrics,
        "cost_score_avg": round(avg_score, 4),
        "cost_budget": round(avg_score * MAX_COST_RATIO, 4),
        "max_cost_ratio": MAX_COST_RATIO,
        "sample_size": n,
        "source": "explicit_reference",
    }


def format_cost_budget(baseline_result: Dict) -> str:
    """Format the reference cost budget for a diagnosis prompt."""
    if "error" in baseline_result:
        return f"Cost data unavailable: {baseline_result['error']}"

    avg = baseline_result["per_question_avg"]
    score = baseline_result["cost_score_avg"]
    budget = baseline_result["cost_budget"]
    ratio = baseline_result["max_cost_ratio"]

    return f"""## Cost reference

Current reference average cost per question:
- VLM frames: {avg['frames']:.0f} across {avg['vlm_calls']:.1f} calls
- LLM calls: {avg['llm_calls']:.1f}
- Estimated tokens: {avg['total_tokens']:.0f}
- Latency: {avg['latency_sec']:.1f} seconds
- Aggregate cost score: {score:.4f}

Prioritize measurable effectiveness, then cost. A candidate may increase cost by
at most {(ratio - 1) * 100:.0f}% (`cost_score <= {budget:.4f}`). Prefer the lower-cost
candidate when effectiveness is otherwise equivalent."""
