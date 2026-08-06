"""Training-only forensic workspaces for Codex bundle self-debug.

This module deliberately copies raw artifacts rather than deriving a compact
feedback summary.  The caller may add a small index so Codex can navigate the
workspace, but the index never replaces the candidate/reference JSONL traces
or source snapshots.  Candidate source remains subject to the existing
anti-specialization checks after Codex edits it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

FORENSIC_ARTIFACT_TYPE = "supervised_forensic_workspace"
FORENSIC_REPORT_ARTIFACT_TYPE = "codex_forensic_report"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "sha256": _sha256(target),
            "copied_byte_for_byte": True,
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
            "sha256": _sha256(target),
            "copied_byte_for_byte": True,
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
    """Create an immutable, complete training feedback workspace.

    ``candidate_trajectory_paths`` and ``reference_trajectory_paths`` are copied
    as raw JSONL only.  Supporting report/bundle/manifest files are copied as
    ordinary artifacts.  The manifest records checksums and streamed row/task
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
    report_schema = {
        "artifact_type": FORENSIC_REPORT_ARTIFACT_TYPE,
        "schema_version": 1,
        "required": {
            "all_rows_read": True,
            "row_analyses": [
                {
                    "task_id": "must match one candidate trajectory row",
                    "candidate_answer": "observed candidate answer",
                    "gold_answer": "training supervision value",
                    "earliest_divergence": {
                        "trajectory_step": "integer or null when no trajectory step exists",
                        "module_or_boundary": "source module/function or runtime boundary",
                        "evidence_event_ids": ["optional real event IDs"],
                        "observed_evidence_excerpt": "non-empty exact excerpt from this candidate row/trajectory",
                    },
                    "mechanism_hypothesis": "generalizable implementation hypothesis",
                    "preserve_or_change": "what a repaired candidate should preserve/change",
                }
            ],
            "cross_row_mechanisms": ["general mechanisms, never task lookup rules"],
            "preservation_and_regression_claims": {
                "preserve_symbols": ["real candidate source symbol or runtime:/handoff: boundary"],
                "change_symbols": ["real candidate source symbol or runtime:/handoff: boundary"],
                "regression_risk_symbols": ["real candidate source symbol or runtime:/handoff: boundary"],
            },
        },
        "source_policy": {
            "training_feedback_is_allowed": True,
            "candidate_source_must_not_embed": [
                "task ids", "video ids", "question text", "answers", "options",
                "timestamps", "evidence phrases", "dataset-specific branches",
            ],
        },
    }
    (root / "codex_forensic_report_schema.json").write_text(
        json.dumps(report_schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
        "forensic_report_schema_path": str((root / "codex_forensic_report_schema.json").resolve()),
        "metadata": dict(metadata or {}),
    }
    manifest_path = root / "forensics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def forensic_prompt_block(manifest: dict[str, Any]) -> str:
    """Return instructions that point Codex to complete local artifacts."""
    return f"""
## Supervised training forensic workspace

This is a training-set development repair. Before changing source, read every
line of every JSONL listed in `{manifest.get('manifest_path', '')}` and inspect
the copied candidate/reference source and artifacts. The manifest is an index,
not a substitute for raw files. Questions, options, gold answers, windows,
provider outputs and trajectory events are intentionally available for
supervised debugging.

Write `{manifest.get('forensic_report_path', '')}` as valid JSON matching
`{manifest.get('forensic_report_schema_path', '')}` BEFORE emitting code. It
must account for every candidate task ID, identify the earliest observed
divergence for each row, cite a real candidate source symbol (for example
`Class.method`) or a `runtime:`/`handoff:` boundary, and include an exact non-empty
`observed_evidence_excerpt` from that row/trajectory. Candidate and gold answers in
the report must exactly match the copied row. State only reusable mechanisms. Fill
the preservation/regression symbol lists as well. Do not encode any
training literal, task/video ID, answer, option, timestamp, or evidence phrase
in candidate source; source is checked separately for specialization.
"""


def _candidate_trace_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Stream copied candidate JSONL into only the fields needed for checks."""
    index: dict[str, dict[str, Any]] = {}
    for record in manifest.get("candidate_trajectories") or []:
        path = Path(str(record.get("path") or ""))
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                task_id = str(row.get("task_id") or "")
                if not task_id:
                    continue
                trajectory = row.get("trajectory") if isinstance(row.get("trajectory"), list) else []
                step_numbers = set()
                event_ids = set()
                for ordinal, step in enumerate(trajectory, start=1):
                    if not isinstance(step, dict):
                        continue
                    raw_step = step.get("step")
                    if isinstance(raw_step, int):
                        step_numbers.add(raw_step)
                    step_numbers.add(ordinal)
                    for event_id in step.get("evidence_event_ids") or []:
                        if str(event_id):
                            event_ids.add(str(event_id))
                for event in row.get("capability_events") or []:
                    if not isinstance(event, dict):
                        continue
                    for key in ("event_id", "id", "source_event_id"):
                        if str(event.get(key) or ""):
                            event_ids.add(str(event[key]))
                def first_observed_value(*keys: str) -> Any:
                    for key in keys:
                        if key in row and row.get(key) not in (None, ""):
                            return row.get(key)
                    return ""

                # A forensic evidence excerpt has to come from runtime
                # observation, not the task/question/answer envelope.  This
                # makes the claimed cross-field diagnosis auditable while
                # still retaining only a small derived index in memory.
                runtime_trace = {
                    key: row.get(key)
                    for key in (
                        "trajectory", "capability_events", "observations",
                        "module_outputs", "runtime_context", "handoffs",
                    )
                    if key in row
                }
                index[task_id] = {
                    "trajectory_steps": step_numbers,
                    "evidence_event_ids": event_ids,
                    "candidate_answer": first_observed_value(
                        "answer", "final_agent_answer", "final_answer"
                    ),
                    "gold_answer": first_observed_value(
                        "gold_answer", "gt_answer", "ground_truth"
                    ),
                    # The report must quote one real runtime fact instead of
                    # merely asserting that it read the trajectory.  JSON
                    # serialization makes nested observations/event payloads
                    # available without retaining the whole JSONL in memory.
                    "row_text": json.dumps(row, ensure_ascii=False, sort_keys=True),
                    "runtime_trace_text": json.dumps(runtime_trace, ensure_ascii=False, sort_keys=True),
                }
    return index


def _source_symbols(source_root: Path) -> set[str]:
    symbols: set[str] = set()
    for path in source_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.add(node.name)
            elif isinstance(node, ast.ClassDef):
                symbols.add(node.name)
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.add(f"{node.name}.{child.name}")
                        symbols.add(child.name)
    return symbols


def _report_symbol_is_valid(value: str, symbols: set[str]) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith(("runtime:", "handoff:", "adapter:", "orchestrator:")):
        return True
    return text in symbols


def _training_literals(manifest: dict[str, Any]) -> dict[str, set[str]]:
    """Extract only high-signal supervision literals without loading JSONL wholesale."""
    values = {"identity": set(), "text": set(), "answer": set(), "interval": set()}
    for record in manifest.get("candidate_trajectories") or []:
        path = Path(str(record.get("path") or ""))
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                for key in ("task_id", "video_id"):
                    value = str(row.get(key) or "").strip()
                    if value:
                        values["identity"].add(value)
                value = str(row.get("time_reference") or "").strip()
                if value:
                    values["interval"].add(value)
                for key in ("question",):
                    value = str(row.get(key) or "").strip()
                    if len(value) >= 16:
                        values["text"].add(value)
                for choice in row.get("choices") or []:
                    value = str(choice or "").strip()
                    if len(value) >= 16:
                        values["text"].add(value)
                for key in ("answer", "gold_answer", "gt_answer", "final_answer"):
                    value = str(row.get(key) or "").strip()
                    if value:
                        values["answer"].add(value)
                for step in row.get("trajectory") or []:
                    if not isinstance(step, dict):
                        continue
                    for key in ("observation", "output", "content"):
                        value = str(step.get(key) or "").strip()
                        if len(value) >= 48:
                            values["text"].add(value)
    return values


def scan_candidate_source_for_specialization(bundle: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when supervised task literals cross into candidate source.

    Generic AST checks remain useful, but they cannot know this attempt's task
    IDs or gold answers. This scan compares the generated source with the raw,
    copied training records and treats a literal answer as unsafe only when it
    occupies a final-answer outlet, avoiding false positives for normal tool
    names or generic strings.
    """
    literals = _training_literals(manifest)
    hits: list[dict[str, str]] = []
    modules = (bundle or {}).get("modules") or {}
    records = modules.values() if isinstance(modules, dict) else modules
    for record in records or []:
        if not isinstance(record, dict):
            continue
        code = str(record.get("code") or "")
        module = str(record.get("module_type") or record.get("name") or "unknown")
        for category in ("identity", "interval", "text"):
            for literal in literals[category]:
                if literal and literal in code:
                    hits.append({"module": module, "category": category, "literal_sha256": hashlib.sha256(literal.encode("utf-8")).hexdigest()})
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or str(key.value) != "answer":
                    continue
                if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value in literals["answer"]:
                    hits.append({"module": module, "category": "gold_or_candidate_answer_outlet", "literal_sha256": hashlib.sha256(value.value.encode("utf-8")).hexdigest()})
    return {"passed": not hits, "hit_count": len(hits), "hits": hits}


def validate_codex_forensic_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate report completeness against streamed candidate trajectory IDs."""
    report_path = Path(str(manifest.get("forensic_report_path") or ""))
    if not report_path.is_file():
        return {"passed": False, "error": "codex_forensic_report_missing"}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "error": f"codex_forensic_report_invalid_json: {exc}"}
    if not isinstance(report, dict):
        return {"passed": False, "error": "codex_forensic_report_not_object"}
    if report.get("artifact_type") != FORENSIC_REPORT_ARTIFACT_TYPE or report.get("schema_version") != 1:
        return {"passed": False, "error": "codex_forensic_report_schema_version"}
    if report.get("all_rows_read") is not True:
        return {"passed": False, "error": "codex_forensic_report_all_rows_read_missing"}
    expected = {str(value) for value in (manifest.get("candidate_task_ids") or []) if str(value)}
    trace_index = _candidate_trace_index(manifest)
    source_root = Path(str(manifest.get("workspace") or "")) / "candidate_source"
    source_symbols = _source_symbols(source_root)
    analyses = report.get("row_analyses") if isinstance(report.get("row_analyses"), list) else []
    observed = set()
    invalid_rows = []
    for item in analyses:
        if not isinstance(item, dict):
            invalid_rows.append("non_object")
            continue
        task_id = str(item.get("task_id") or "")
        divergence = item.get("earliest_divergence")
        mechanism = str(item.get("mechanism_hypothesis") or "").strip()
        preserve_or_change = str(item.get("preserve_or_change") or "").strip()
        if (
            not task_id or not isinstance(divergence, dict) or not mechanism
            or not preserve_or_change
            or "candidate_answer" not in item or "gold_answer" not in item
            or not str(divergence.get("module_or_boundary") or "").strip()
            or not str(divergence.get("observed_evidence_excerpt") or "").strip()
        ):
            invalid_rows.append(task_id or "missing_task_id")
            continue
        if not _report_symbol_is_valid(str(divergence.get("module_or_boundary") or ""), source_symbols):
            invalid_rows.append(task_id + ":unknown_source_symbol")
            continue
        observed_trace = trace_index.get(task_id, {})
        if item.get("candidate_answer") != observed_trace.get("candidate_answer", ""):
            invalid_rows.append(task_id + ":candidate_answer_mismatch")
            continue
        if item.get("gold_answer") != observed_trace.get("gold_answer", ""):
            invalid_rows.append(task_id + ":gold_answer_mismatch")
            continue
        excerpt = str(divergence.get("observed_evidence_excerpt") or "").strip()
        if excerpt not in str(observed_trace.get("runtime_trace_text") or ""):
            invalid_rows.append(task_id + ":observed_evidence_not_in_trace")
            continue
        trajectory_step = divergence.get("trajectory_step")
        if trajectory_step is not None:
            if not isinstance(trajectory_step, int) or trajectory_step not in observed_trace.get("trajectory_steps", set()):
                invalid_rows.append(task_id + ":unknown_trajectory_step")
                continue
        cited_event_ids = divergence.get("evidence_event_ids", [])
        if not isinstance(cited_event_ids, list):
            invalid_rows.append(task_id + ":invalid_evidence_event_ids")
            continue
        unknown_event_ids = [
            str(value) for value in cited_event_ids
            if str(value) and str(value) not in observed_trace.get("evidence_event_ids", set())
        ]
        if unknown_event_ids:
            invalid_rows.append(task_id + ":unknown_evidence_event_id")
            continue
        observed.add(task_id)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if invalid_rows or missing or unknown:
        return {
            "passed": False,
            "error": "codex_forensic_report_row_accounting_failed",
            "invalid_rows": invalid_rows,
            "missing_task_ids": missing,
            "unknown_task_ids": unknown,
        }
    mechanisms = report.get("cross_row_mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        return {"passed": False, "error": "codex_forensic_report_cross_row_mechanisms_missing"}
    preservation = report.get("preservation_and_regression_claims")
    if not isinstance(preservation, dict):
        return {"passed": False, "error": "codex_forensic_report_preservation_claims_missing"}
    for field in ("preserve_symbols", "change_symbols", "regression_risk_symbols"):
        values = preservation.get(field)
        if not isinstance(values, list):
            return {"passed": False, "error": f"codex_forensic_report_{field}_missing"}
        invalid = [str(value) for value in values if not _report_symbol_is_valid(str(value), source_symbols)]
        if invalid:
            return {"passed": False, "error": f"codex_forensic_report_{field}_unknown_symbol", "symbols": invalid}
    return {
        "passed": True,
        "report_path": str(report_path),
        "task_count": len(expected),
        "row_analysis_count": len(analyses),
        "report": report,
    }
