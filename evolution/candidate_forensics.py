"""Training trajectories, source snapshots and investigation notes for Codex."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

FORENSIC_ARTIFACT_TYPE = "supervised_forensic_workspace"


def _jsonl_metadata(path: Path) -> dict[str, Any]:
    """Stream metadata only; never load a trajectory JSONL as one object."""
    line_count = 0
    task_ids: list[str] = []
    malformed_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            line_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            if isinstance(row, dict):
                task_id = str(row.get("task_id") or "")
                if task_id:
                    task_ids.append(task_id)
    return {
        "line_count": line_count,
        "task_ids": task_ids,
        "malformed_rows": malformed_rows,
    }


def _iter_artifact_files(paths: Iterable[str], *, jsonl_only: bool) -> Iterable[Path]:
    seen: set[str] = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            pattern = "*.jsonl" if jsonl_only else "*"
            candidates = [item for item in sorted(path.rglob(pattern)) if item.is_file()]
        else:
            continue
        for candidate in candidates:
            if jsonl_only and candidate.suffix != ".jsonl":
                continue
            canonical = str(candidate)
            if canonical in seen:
                continue
            seen.add(canonical)
            yield candidate


def _copy_artifacts(paths: Iterable[str], destination: Path, *, role: str,
                    jsonl_only: bool) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, source in enumerate(_iter_artifact_files(paths, jsonl_only=jsonl_only), start=1):
        target = destination / f"{index:03d}_{source.name}"
        shutil.copy2(source, target)
        record: dict[str, Any] = {
            "role": role,
            "source_path": str(source),
            "path": str(target.resolve()),
            "bytes": target.stat().st_size,
        }
        if target.suffix == ".jsonl":
            record.update(_jsonl_metadata(target))
        records.append(record)
    return records


def _copy_source_tree(source_dir: str, destination: Path, *, role: str) -> list[dict[str, Any]]:
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for source in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({
            "role": role,
            "source_path": str(source),
            "path": str(target.resolve()),
            "bytes": target.stat().st_size,
        })
    return records


def materialize_supervised_forensic_workspace(
    workspace_dir: str,
    *,
    candidate_source_dir: str,
    reference_source_dir: str = "",
    candidate_trajectory_paths: Iterable[str] = (),
    reference_trajectory_paths: Iterable[str] = (),
    reference_artifact_paths: Iterable[str] = (),
    task_manifest_paths: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a training feedback workspace with copied source and trajectories.

    ``candidate_trajectory_paths`` and ``reference_trajectory_paths`` are copied
    as raw JSONL only.  Supporting report/bundle/manifest files are copied as
    ordinary artifacts. The index records roles, paths and streamed row/task
    counts, while Codex reads the copied originals themselves.
    """
    root = Path(workspace_dir).resolve()
    if root.exists():
        raise FileExistsError(f"forensic workspace already exists: {root}")
    root.mkdir(parents=True)
    candidate_source = _copy_source_tree(
        candidate_source_dir, root / "candidate_source", role="candidate_source"
    )
    if not candidate_source:
        raise FileNotFoundError(f"candidate source snapshot unavailable: {candidate_source_dir}")
    reference_source = _copy_source_tree(
        reference_source_dir, root / "reference_source", role="reference_source"
    ) if reference_source_dir else []
    candidate_traces = _copy_artifacts(
        candidate_trajectory_paths, root / "candidate_trajectories",
        role="candidate_trajectory", jsonl_only=True,
    )
    if not candidate_traces:
        raise FileNotFoundError("supervised forensic workspace requires complete candidate JSONL trajectories")
    reference_traces = _copy_artifacts(
        reference_trajectory_paths, root / "reference_trajectories",
        role="reference_trajectory", jsonl_only=True,
    )
    reference_artifacts = _copy_artifacts(
        reference_artifact_paths, root / "reference_artifacts",
        role="reference_artifact", jsonl_only=False,
    )
    task_manifests = _copy_artifacts(
        task_manifest_paths, root / "task_manifests",
        role="task_manifest", jsonl_only=False,
    )
    task_ids = []
    for record in candidate_traces:
        if int(record.get("malformed_rows") or 0):
            raise ValueError(
                "supervised forensic workspace refuses malformed candidate trajectory JSONL"
            )
        if int(record.get("line_count") or 0) != len(record.get("task_ids") or []):
            raise ValueError(
                "supervised forensic workspace requires a task_id for every candidate trajectory row"
            )
        for task_id in record.get("task_ids") or []:
            if task_id not in task_ids:
                task_ids.append(task_id)
    if not task_ids:
        raise ValueError(
            "supervised forensic workspace requires task_id on every candidate trace set; "
            "cannot verify per-row Codex accounting from anonymous JSONL"
        )
    report_path = root / "codex_forensic_report.json"
    manifest = {
        "artifact_type": FORENSIC_ARTIFACT_TYPE,
        "schema_version": 1,
        "workspace": str(root),
        "copy_policy": "raw trajectory/source artifacts copied byte-for-byte; index is navigational only",
        "candidate_source": candidate_source,
        "reference_source": reference_source,
        "candidate_trajectories": candidate_traces,
        "reference_trajectories": reference_traces,
        "reference_artifacts": reference_artifacts,
        "task_manifests": task_manifests,
        "candidate_task_ids": task_ids,
        "candidate_task_count": len(task_ids),
        "forensic_report_path": str(report_path.resolve()),
        "metadata": dict(metadata or {}),
    }
    manifest_path = root / "forensics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def forensic_prompt_block(manifest: dict[str, Any]) -> str:
    """Guide causal debugging and carry the investigation into the next attempt."""
    return f"""
## Training investigation and implementation handoff

The artifact index is at `{manifest.get('manifest_path', '')}`. It links complete
candidate/reference trajectories, current source, evaluation outcomes and prior
findings. Inspect the raw records needed to explain the failure and regression
patterns; an aggregate score is not an explanation. Training questions, options,
gold answers, windows and provider events are available for debugging, never as
runtime lookup rules. No held-out evaluation evidence is supplied.

Before editing, distinguish an implementation failure from an unsupported causal
hypothesis and a provider/asset failure. Follow the earliest producer/consumer
break through the actual source and downstream use. For a successful producer,
check whether its result was consumed and changed the reasoning state; do not
merely add fields to an envelope. Compare affected and preserved behavior.

Continue the previous analysis instead of restarting from the score. Explain the
general behavioral change, why it should address the observed mechanism, which
modules actually need changes, and what observable result would falsify it.
Account for corrections to preserve and regressions to guard. If the available
evidence contradicts the diagnosis, state the disagreement and remaining work.

Write the detailed investigation to `{manifest.get('forensic_report_path', '')}`
as JSON with analysis, evidence_references, proposed_changes, preserved_behavior,
open_questions and next_task. next_task should specify the next edit or check
and its expected observable result. Smoke and probe execute the implementation
and provide evidence for the next attempt.

Do not embed task/video IDs, questions, answers, options, annotated timestamps
or transcript literals in candidate source. Keep changes reusable across tasks.
"""


def load_codex_forensic_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """Load the coding stage's analysis for the next attempt."""
    report_path = Path(str(manifest.get("forensic_report_path") or ""))
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "report_path": str(report_path), "warning": str(exc)}
    if not isinstance(report, dict):
        return {"available": False, "report_path": str(report_path), "warning": "analysis notes must be an object"}
    return {"available": True, "report_path": str(report_path), "report": report}
