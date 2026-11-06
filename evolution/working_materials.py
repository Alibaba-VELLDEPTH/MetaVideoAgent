"""Named working materials and paginated reads for evolution stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def material_paths(context: dict[str, Any]) -> dict[str, str]:
    """Name the current training materials without searching output trees."""
    reference = context.get("reference") or {}
    previous = context.get("previous_round") or {}
    history = context.get("history") or {}

    def path(value: Any) -> str:
        return str(value.get("path") or "") if isinstance(value, dict) else str(value or "")

    candidates = {
        "profile": path(context.get("observed_distribution_profile")),
        "current_agent": path(reference.get("initial_baseline_bundle")),
        "current_results": path(reference.get("results")),
        "current_report": path(reference.get("report")),
        "review": path(previous.get("evolution_review")),
        "previous_diagnosis": path(previous.get("diagnosis")),
        "candidate_feedback": path(previous.get("candidate_rejection_context")),
        "candidate_agent": path(previous.get("candidate_bundle")),
        "candidate_evaluation": path(previous.get("full_eval_report")),
        "history": path(history.get("evolution_memory_index_path")),
    }
    candidates.update(context.get("working_materials") or {})
    for stage, record in ((context.get("stage_context") or {}).get("stage_materials") or {}).items():
        if record.get("result"):
            candidates[f"{stage}_result"] = record["result"]
        if record.get("notes"):
            candidates[f"{stage}_notes"] = record["notes"]
    return {name: value for name, value in candidates.items() if isinstance(value, str) and value}


def read_material(materials: dict[str, str], name: str, *, offset: int = 0,
                  limit: int = 160, module: str = "") -> dict[str, Any]:
    """Read a named artifact and return the continuation offset."""
    if name not in materials:
        raise ValueError(f"Unknown material {name!r}; available: {', '.join(materials)}")
    offset = max(0, int(offset))
    limit = max(1, min(400, int(limit)))
    source = Path(materials[name])
    if not source.is_file():
        raise ValueError(f"Material {name!r} is not a readable file")
    content = source.read_text(encoding="utf-8")
    if source.suffix == ".json":
        payload = json.loads(content)
        if module:
            if not isinstance(payload, dict):
                raise ValueError(f"Material {name!r} is not an agent bundle")
            records = payload.get("modules") or {}
            if isinstance(records, list):
                records = {item.get("module_type"): item for item in records if isinstance(item, dict)}
            record = records.get(module) if isinstance(records, dict) else None
            if not isinstance(record, dict) or not isinstance(record.get("code"), str):
                raise ValueError(f"Material {name!r} has no source for module {module!r}")
            content = record["code"]
        else:
            content = json.dumps(payload, ensure_ascii=False, indent=2)
    lines = content.splitlines()
    end = min(len(lines), offset + limit)
    return {
        "material": name,
        "module": module,
        "path": str(source),
        "offset": offset,
        "content": "\n".join(lines[offset:end]),
        "next_offset": end if end < len(lines) else None,
        "total_lines": len(lines),
    }
