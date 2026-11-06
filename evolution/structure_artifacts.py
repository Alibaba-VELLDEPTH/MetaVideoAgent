"""Resolve the video structures recorded by an execution report."""

from __future__ import annotations

import os


def resolve_report_artifact_dir(value: str, report_path: str) -> str:
    """Resolve an absolute or report-relative artifact directory."""
    if not isinstance(value, str) or not value:
        return ""
    if report_path and not os.path.isabs(value):
        value = os.path.join(os.path.dirname(os.path.abspath(report_path)), value)
    return os.path.abspath(value) if os.path.isdir(value) else ""


def reference_structure_artifact_dir(reference_report: dict, report_path: str) -> str:
    """Return the structure directory supplied by the reference report."""
    report = reference_report if isinstance(reference_report, dict) else {}
    artifact = report.get("structure_artifact") or {}
    if isinstance(artifact, dict):
        for key in ("materialized_dir", "source_dir"):
            resolved = resolve_report_artifact_dir(str(artifact.get(key) or ""), report_path)
            if resolved:
                return resolved

    return ""
