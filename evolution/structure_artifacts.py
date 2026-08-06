"""Report-scoped resolution of immutable MetaVideoAgent structure artifacts."""

from __future__ import annotations

import json
import os


def read_json_optional(path: str) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_report_artifact_dir(value: str, report_path: str) -> str:
    """Resolve an absolute or report-relative artifact directory."""
    if not isinstance(value, str) or not value:
        return ""
    candidates = [value]
    if report_path and not os.path.isabs(value):
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(report_path)), value))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return ""


def reference_structure_artifact_dir(reference_report: dict, report_path: str) -> str:
    """Return the immutable structure directory attested by a reference report."""
    report = reference_report if isinstance(reference_report, dict) else {}
    artifact = report.get("structure_artifact") or {}
    if isinstance(artifact, dict):
        for key in ("materialized_dir", "source_dir"):
            resolved = resolve_report_artifact_dir(str(artifact.get(key) or ""), report_path)
            if resolved:
                return resolved

    expected_report = os.path.abspath(report_path) if report_path else ""
    current_dir = os.path.dirname(expected_report) if expected_report else ""
    while current_dir:
        ledger = read_json_optional(os.path.join(current_dir, "current_best_ledger.json"))
        current_best = ledger.get("current_best") if isinstance(ledger, dict) else {}
        ledger_report = str((current_best or {}).get("report_path") or "")
        if ledger_report and os.path.abspath(ledger_report) == expected_report:
            structure = ((current_best or {}).get("artifacts") or {}).get("structure") or {}
            if isinstance(structure, dict):
                for key in ("materialized_dir", "source_dir"):
                    resolved = resolve_report_artifact_dir(
                        str(structure.get(key) or ""), report_path
                    )
                    if resolved:
                        return resolved
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break
        current_dir = parent
    return ""
