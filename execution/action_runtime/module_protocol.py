"""Neutral handoff protocol for the five MetaVideoAgent modules.

The protocol defines data shapes and provenance only.  It deliberately does
not decide *which* window to retrieve, *which* model to call, how to summarize
memory, or how to answer a question; those remain generated bundle behavior.
"""

from __future__ import annotations

import json
import re
from typing import Any

PROTOCOL_VERSION = "metavideoagent_module_protocol"
RESULT_STATUSES = {"ok", "unavailable", "error"}


def parse_json_object(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_time_range(value: Any) -> dict | None:
    """Normalize one interval without inventing missing temporal evidence."""
    start = end = None
    if isinstance(value, dict):
        for key in ("start_sec", "start", "start_time"):
            if key in value:
                start = value.get(key)
                break
        for key in ("end_sec", "end", "end_time"):
            if key in value:
                end = value.get(key)
                break
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        start, end = value[0], value[1]
    try:
        start_f, end_f = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if end_f <= start_f:
        return None
    result = {"start_sec": start_f, "end_sec": end_f}
    if isinstance(value, dict):
        for key in ("source", "record_id", "evidence_id", "modality", "purpose"):
            if value.get(key) not in (None, ""):
                result[key] = value[key]
    return result


def normalize_time_ranges(value: Any) -> list[dict]:
    if isinstance(value, dict):
        for key in ("time_ranges", "candidate_windows", "windows", "verified_windows", "ranges"):
            if key in value:
                return normalize_time_ranges(value.get(key))
        item = normalize_time_range(value)
        return [item] if item else []
    if not isinstance(value, (list, tuple)):
        return []
    # A two-number list is one range; a list of lists/dicts is many.
    item = normalize_time_range(value)
    if item:
        return [item]
    result, seen = [], set()
    for raw in value:
        item = normalize_time_range(raw)
        if not item:
            continue
        key = (round(item["start_sec"], 6), round(item["end_sec"], 6), str(item.get("source", "")))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def time_range_pairs(ranges: Any) -> list[list[float]]:
    return [[item["start_sec"], item["end_sec"]] for item in normalize_time_ranges(ranges)]


def _ranges_from_text(text: str) -> list[dict]:
    matches = re.findall(
        r"\[\s*(\d+(?:\.\d+)?)\s*s?\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*s?\s*\]",
        str(text or ""), re.I,
    )
    return normalize_time_ranges([[start, end] for start, end in matches])


def _status_from_value(value: Any) -> str:
    if isinstance(value, dict) and str(value.get("status") or "").lower() in RESULT_STATUSES:
        return str(value["status"]).lower()
    text = str(value or "").strip().lower()
    if not text:
        return "unavailable"
    if "error" in text or "exception" in text:
        return "error"
    if any(token in text for token in ("unavailable", "not found", "no structure records", "no candidate")):
        return "unavailable"
    return "ok"


def make_tool_request(producer_module: str, tool_name: str, params: dict | None,
                      instruction: str = "") -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "tool_request",
        "requested_module": str(producer_module or ""),
        "tool_name": str(tool_name or ""),
        "params": dict(params or {}),
        "instruction": str(instruction or ""),
    }


def make_module_result(*, producer_module: str, tool_name: str, status: str,
                       time_ranges: Any = None, structure_records: list | None = None,
                       perception_requests: list | None = None, evidence: dict | None = None,
                       error: str = "", request: dict | None = None) -> dict:
    """Create a native localization/perception result without choosing policy."""
    module = str(producer_module or "")
    if module not in {"localization", "perception"}:
        raise ValueError("producer_module must be localization or perception")
    normalized_status = str(status or "unavailable").lower()
    if normalized_status not in RESULT_STATUSES:
        raise ValueError(f"unsupported result status: {status!r}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": f"{module}_result",
        "status": normalized_status,
        "producer_module": module,
        "tool_name": str(tool_name or ""),
        "request": dict(request or {}),
        "time_ranges": normalize_time_ranges(time_ranges),
        "structure_records": list(structure_records or []),
        "perception_requests": list(perception_requests or []),
        "evidence": dict(evidence or {}),
        "error": str(error or ""),
    }


def normalize_tool_result(value: Any, *, producer_module: str, tool_name: str,
                          request: dict | None = None) -> dict:
    """Normalize generated tool output into one auditable envelope."""
    payload = parse_json_object(value)
    if payload.get("protocol_version") == PROTOCOL_VERSION and payload.get("kind") in {
        "localization_result", "perception_result",
    }:
        result = dict(payload)
        result.setdefault("producer_module", producer_module)
        result.setdefault("tool_name", tool_name)
        result["time_ranges"] = normalize_time_ranges(result.get("time_ranges"))
        result["status"] = _status_from_value(result)
        if not isinstance(result.get("request"), dict) or (request and not result.get("request")):
            result["request"] = dict(request or {})
        if not isinstance(result.get("evidence"), dict):
            result["evidence"] = {"raw_output": result.get("evidence")} if result.get("evidence") else {}
        if not isinstance(result.get("structure_records"), list):
            result["structure_records"] = [result["structure_records"]] if result.get("structure_records") else []
        if not isinstance(result.get("perception_requests"), list):
            result["perception_requests"] = [result["perception_requests"]] if result.get("perception_requests") else []
        result.setdefault("normalization", "native_protocol")
        return result

    source = payload if payload else value
    ranges = normalize_time_ranges(payload) if payload else _ranges_from_text(str(value or ""))
    if producer_module == "localization":
        kind = "localization_result"
        structure_records = (
            (payload.get("structure_records") or payload.get("evidence_cards") or [])
            if payload else []
        )
        perception_requests = (payload.get("perception_requests") or []) if payload else []
        # Candidate windows are localization requests for later
        # verification, not already-perceived evidence.
        if ranges and not perception_requests and not structure_records:
            perception_requests = [
                {"time_range": item, "modality": "visual", "purpose": "verify localized evidence"}
                for item in ranges
            ]
    else:
        kind = "perception_result"
        structure_records, perception_requests = [], []
    status = _status_from_value(source)
    if producer_module == "localization" and status == "ok" and not ranges:
        status = "unavailable"
    evidence = payload.get("evidence") if payload else None
    if not isinstance(evidence, dict):
        evidence = {}
    if value not in (None, "", {}, []):
        evidence.setdefault("raw_output", value if isinstance(value, str) else payload)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": kind,
        "status": status,
        "producer_module": producer_module,
        "tool_name": str(tool_name or ""),
        "request": dict(request or {}),
        "time_ranges": ranges,
        "structure_records": structure_records if isinstance(structure_records, list) else [structure_records],
        "perception_requests": perception_requests if isinstance(perception_requests, list) else [perception_requests],
        "evidence": evidence,
        "error": str(payload.get("error") or "") if payload else "",
        "normalization": "generated_output_adapter",
    }


def result_is_usable(result: dict) -> bool:
    if not isinstance(result, dict) or result.get("status") != "ok":
        return False
    if result.get("kind") == "localization_result":
        return bool(normalize_time_ranges(result.get("time_ranges")))
    if result.get("kind") == "perception_result":
        evidence = result.get("evidence") or {}
        if not isinstance(evidence, dict):
            return False
        return bool(evidence) and bool(evidence.get("raw_output") or evidence.get("claims") or evidence.get("channels"))
    return False


def extract_export_values(result: dict, export_vars: dict | None) -> dict:
    """Deterministically expose protocol fields; never ask a second LLM to parse."""
    if not isinstance(result, dict) or not isinstance(export_vars, dict):
        return {}
    values: dict[str, Any] = {}
    aliases = {
        "candidate_windows": time_range_pairs(result.get("time_ranges")),
        "time_ranges": time_range_pairs(result.get("time_ranges")),
        "windows": time_range_pairs(result.get("time_ranges")),
        "verified_windows": time_range_pairs(result.get("time_ranges")),
        "perception_requests": result.get("perception_requests") or [],
        "structure_records": result.get("structure_records") or [],
        "evidence_cards": result.get("structure_records") or [],
        "perception_evidence": result.get("evidence") or {},
        "evidence": result.get("evidence") or {},
        "tool_result": result,
    }
    for name in export_vars:
        key = str(name)
        if key in aliases:
            values[key] = aliases[key]
        elif key in result:
            values[key] = result[key]
        elif isinstance(result.get("evidence"), dict) and key in result["evidence"]:
            values[key] = result["evidence"][key]
    return {key: value for key, value in values.items() if value is not None}


def render_result_for_context(result: dict) -> str:
    """Serialize a module result losslessly for runtime memory and traces.

    Tool outputs are execution evidence. Truncating the serialized JSON can
    cut through a string literal, producing an unparsable trajectory entry and
    silently dropping evidence from later reasoning.  Keep the complete JSON;
    callers that need a compact model prompt must create a separate, explicitly
    valid summary rather than corrupting the persisted/runtime result.
    """
    return json.dumps(result if isinstance(result, dict) else {}, ensure_ascii=False, sort_keys=True)


def make_thinking_context(*, runtime_question: str, video_context: dict,
                          memory_context: dict, recent_results: list[dict]) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "thinking_context",
        "runtime_question": str(runtime_question or ""),
        "video_context": dict(video_context or {}),
        "memory": dict(memory_context or {}),
        "recent_module_results": [item for item in (recent_results or []) if isinstance(item, dict)],
    }


def normalize_thinking_return(value: Any) -> tuple[str, str, Any, str]:
    """Validate the fixed thinking tuple without supplying a fallback policy."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return "", "", {}, "thinking must return exactly (thought, action, payload)"
    thought, action, payload = value
    action = str(action or "")
    if action not in {"act", "finish"}:
        return str(thought or ""), action, payload, "thinking action must be 'act' or 'finish'"
    if action == "act" and not isinstance(payload, list):
        # A generated thinking strategy may attach durable decision state to
        # its executable plan.  The agent's existing ``normalize_plan``
        # consumes ``payload['plan']`` while working memory retains sibling
        # fields such as ``decision_state`` for the next reasoning turn.
        # Support both the concise list form and the structured form that
        # carries cross-turn decision state beside the executable plan.
        if not (isinstance(payload, dict) and isinstance(payload.get("plan"), list)):
            return str(thought or ""), action, payload, "thinking act payload must be a list or an object with a plan list"
    if action == "finish" and not isinstance(payload, dict):
        return str(thought or ""), action, payload, "thinking finish payload must be an object"
    if action == "finish" and not str(payload.get("answer") or "").strip():
        return str(thought or ""), action, payload, "thinking finish payload must contain a non-empty answer"
    return str(thought or ""), action, payload, ""
