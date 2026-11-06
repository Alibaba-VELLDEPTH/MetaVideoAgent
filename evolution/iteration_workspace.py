"""One current Agent and explicit stage results for each evolution iteration."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

STAGES = (
    "initialization",
    "teacher",
    "diagnosis",
    "research",
    "codegen",
    "smoke",
    "probe",
    "full_eval",
    "review",
)


def write_json(path: str | Path, value: dict) -> None:
    """Publish a complete local result before advancing its stage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".result_", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def candidate_stage_paths(candidate_run: str | Path, stage: str) -> dict:
    """Follow the execution receipt written by this candidate attempt."""
    return (read_json(Path(candidate_run) / "run_record.json").get("stage_results") or {}).get(
        stage
    ) or {}


def _reopen_stage(record: dict) -> None:
    """Keep the latest available findings when reopening work."""
    previous = record.pop("result", None)
    if previous:
        record["previous_result"] = previous
    record["status"] = "pending"


class IterationWorkspace:
    """Own the editable source, stage attempts and pending task for one iter_N.

    Execution snapshots and exported bundles describe a particular execution;
    only root/agent and root/agent.json are the current editable implementation.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    @property
    def progress_path(self) -> Path:
        return self.root / "progress.json"

    def progress(self) -> dict:
        return read_json(self.progress_path)

    def reserve(self, stage: str) -> Path:
        progress = self.progress()
        for name in STAGES[STAGES.index(stage) + 1 :]:
            downstream = progress.get("stages", {}).get(name)
            if downstream:
                _reopen_stage(downstream)
        record = progress.setdefault("stages", {}).setdefault(stage, {"attempts": []})
        index = len(record["attempts"])
        directory = self.root / stage / f"attempt_{index}"
        directory.mkdir(parents=True, exist_ok=False)
        record["attempts"].append(str(directory.relative_to(self.root)))
        _reopen_stage(record)
        record["attempt"] = index
        progress.update(current_stage=stage, status="pending")
        write_json(self.progress_path, progress)
        return directory

    def record(
        self,
        stage: str,
        result_path: str | Path,
        *,
        status: str,
        next_task: dict | None = None,
        notes_path: str | Path = "",
    ) -> None:
        progress = self.progress()
        record = progress.setdefault("stages", {}).setdefault(stage, {"attempts": []})
        record.update(status=status, result=str(Path(result_path).resolve()))
        if notes_path:
            record["notes"] = str(Path(notes_path).resolve())
        progress.update(current_stage=stage, status=status, next_task=next_task or {})
        write_json(self.progress_path, progress)

    def bind_base(self, *, bundle: str, report: str = "", results: str = "") -> None:
        progress = self.progress()
        if not progress.get("base_reference"):
            progress["base_reference"] = {"bundle": bundle, "report": report, "results": results}
            write_json(self.progress_path, progress)

    def context(self, stage: str) -> dict:
        """Conclusions navigate to complete notes/results; they do not replace them."""
        progress = self.progress()
        allowed = STAGES[: STAGES.index(stage) + 1]
        conclusions, materials = {}, {}
        for name in allowed:
            record = (progress.get("stages") or {}).get(name) or {}
            result_path = record.get("result") or record.get("previous_result")
            if result_path:
                result = read_json(result_path)
                conclusions[name] = {
                    key: result[key]
                    for key in ("summary", "open_questions", "next_task")
                    if key in result
                }
                materials[name] = {
                    "result": result_path,
                    "notes": record.get("notes", ""),
                    "status": record["status"],
                }
        return {
            "iteration_dir": str(self.root),
            "base_reference": progress.get("base_reference") or {},
            "stage_conclusions": conclusions,
            "stage_materials": materials,
            "pending_task": progress.get("next_task") or {},
        }

    def restart(self, stage: str) -> None:
        """Reopen affected work while retaining source, notes and execution files."""
        progress = self.progress()
        for name in STAGES[STAGES.index(stage) :]:
            record = progress.get("stages", {}).get(name)
            if record:
                _reopen_stage(record)
        progress.update(current_stage=stage, status="pending", next_task={"stage": stage})
        write_json(self.progress_path, progress)

    def finish(self, status: str) -> None:
        """The next physical iteration need not use this candidate as its base."""
        progress = self.progress()
        progress["status"] = status
        write_json(self.progress_path, progress)
        suffix = self.root.name.removeprefix("iter_")
        if self.root.name.startswith("iter_") and suffix.isdigit():
            index = int(suffix)
            run_path = self.root.parent / "run.json"
            run_state = read_json(run_path)
            run_state.update(
                iteration=index,
                status=status,
                next_iteration=index + 1 if status == "complete" else index,
            )
            write_json(run_path, run_state)
