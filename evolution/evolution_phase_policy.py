"""Iteration-independent objective policy for target-distribution evolution."""

from __future__ import annotations


def for_iteration(iteration: int | None) -> dict:
    """Return the open-ended policy used by every evolution iteration."""
    index = max(0, int(iteration or 0))
    return {
        "phase": "initial_design" if index == 0 else "iterative_evolution",
        "iteration_index": index,
        "evolution_round": index + 1,
        "primary_objective": "training_evidence_improvement",
        "cost_policy": "record_only",
        "allow_expensive_capabilities": True,
        "current_best_selection": "evolution_split_performance_and_trajectory_review",
        "selection_split": "train",
        "efficiency_constraints": {"enforced": False},
        "instructions": [
            "Use evolution-split trajectories for review, diagnosis, and candidate design.",
            "Treat held-out evaluation as read-only reporting evidence.",
            "Record model and perception cost within the outer workflow's configured update budget.",
        ],
    }
