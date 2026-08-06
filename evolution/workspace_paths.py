"""Iteration-scoped paths used by MetaVideoAgent review and diagnosis."""

from __future__ import annotations

import os
import re
from typing import Optional


class IterationPaths:
    """Resolve artifacts for one iteration in a MetaVideoAgent workspace."""

    def __init__(self, workspace_dir: str, iteration: Optional[int] = None):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.iterations_root = os.path.join(self.workspace_dir, "iterations")
        os.makedirs(self.iterations_root, exist_ok=True)
        self.iteration = self._detect_latest_iteration() if iteration is None else int(iteration)
        self._iter_dir = os.path.join(self.iterations_root, f"iter_{self.iteration}")
        os.makedirs(self._iter_dir, exist_ok=True)

    def _directory(self, name: str) -> str:
        path = os.path.join(self._iter_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def trajectories(self) -> str:
        return self._directory("trajectories")

    @property
    def question_reviews(self) -> str:
        return self._directory("question_reviews")

    @property
    def diagnosis(self) -> str:
        return self._directory("diagnosis")

    @property
    def evolution(self) -> str:
        return self._directory("evolution")

    @property
    def agent_status(self) -> str:
        return os.path.join(self._iter_dir, "agent_status.md")

    @property
    def raw_videos(self) -> str:
        return os.path.join(self.workspace_dir, "raw_videos")

    @property
    def datasets(self) -> str:
        return os.path.join(self.workspace_dir, "datasets")

    def previous(self) -> Optional["IterationPaths"]:
        if self.iteration <= 0:
            return None
        return IterationPaths(self.workspace_dir, self.iteration - 1)

    def get_all_evolution_conclusions(self) -> str:
        chunks = []
        for index in range(self.iteration + 1):
            path = os.path.join(
                self.iterations_root, f"iter_{index}", "evolution", "conclusion.md"
            )
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    chunks.append(handle.read().strip())
        return "\n\n---\n\n".join(chunk for chunk in chunks if chunk)

    def _detect_latest_iteration(self) -> int:
        latest = 0
        if not os.path.isdir(self.iterations_root):
            return latest
        for name in os.listdir(self.iterations_root):
            match = re.fullmatch(r"iter_(\d+)", name)
            if match:
                latest = max(latest, int(match.group(1)))
        return latest

    def detect_video_ids(self) -> list[str]:
        ids = [
            name.removesuffix("_trajectory.jsonl")
            for name in os.listdir(self.trajectories)
            if name.endswith("_trajectory.jsonl")
        ]
        if ids:
            return sorted(ids)
        if not os.path.isdir(self.datasets):
            return []
        return sorted(
            name.removesuffix(".jsonl")
            for name in os.listdir(self.datasets)
            if name.endswith(".jsonl")
        )

    def __repr__(self) -> str:
        return (
            f"IterationPaths(workspace={self.workspace_dir!r}, "
            f"iteration={self.iteration}, iter_dir={self._iter_dir!r})"
        )
