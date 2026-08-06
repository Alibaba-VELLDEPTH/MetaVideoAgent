"""Canonical, provider-independent structure-record contract.

The execution layer supplies observations and bounded media references to a
generated ``video_structuring`` module.  This module normalizes that payload
without adding perception or answer policy, so every generated implementation
inherits the same retrievable record shape.
"""

from __future__ import annotations

import json
from typing import Any

CANONICAL_STRUCTURE_SCHEMA_VERSION = 1
STRUCTURE_BUILD_REQUEST_TYPE = "metavideoagent_structure_build_request"


def _as_mapping(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start:end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _as_list(value: Any) -> list:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item not in (None, "")]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return [value.strip()] if value.strip() else []
        if isinstance(parsed, list):
            return [item for item in parsed if item not in (None, "")]
        return [value.strip()] if value.strip() else []
    return [value]


def _first_text(*values: Any) -> str:
    for value in values:
        text = _as_text(value)
        if text:
            return text
    return ""


def _first_list(*values: Any) -> list:
    for value in values:
        items = _as_list(value)
        if items:
            return items
    return []


def _normalize_media_assets(value: Any, *, start_sec: float, end_sec: float) -> list:
    assets, seen = [], set()
    for raw in _as_list(value):
        item = dict(raw) if isinstance(raw, dict) else {"kind": "opaque", "ref": str(raw)}
        if item.get("path_or_ref") not in (None, "") and not (item.get("path") or item.get("ref")):
            item["ref"] = item["path_or_ref"]
        try:
            asset_start = float(item.get("start_sec", item.get("timestamp_sec", start_sec)))
        except (TypeError, ValueError):
            asset_start = start_sec
        try:
            asset_end = float(item.get("end_sec", item.get("timestamp_sec", end_sec)))
        except (TypeError, ValueError):
            asset_end = end_sec
        if asset_end < asset_start:
            asset_start, asset_end = asset_end, asset_start
        item.update({"start_sec": asset_start, "end_sec": asset_end})
        item.setdefault("kind", "opaque")
        item.setdefault("status", "available")
        key = (
            str(item.get("kind")), round(asset_start, 6), round(asset_end, 6),
            str(item.get("path") or item.get("ref") or ""),
        )
        if key not in seen:
            seen.add(key)
            assets.append(item)
    return assets


def canonicalize_structure_input(record: dict | None) -> dict:
    """Return a lossless, retrievable record for one bounded observation."""
    source = dict(record or {})
    existing = _as_mapping(source.get("canonical_evidence"))
    raw_narration = _first_text(
        source.get("multimodal_narration"), source.get("raw_model_output"),
        existing.get("raw_multimodal_narration"), source.get("visual_summary"),
        source.get("description"),
    )
    parsed = _as_mapping(source.get("multimodal_narration"))
    if not parsed:
        parsed = _as_mapping(source.get("raw_model_output"))

    description = _first_text(
        source.get("description"), source.get("visual_summary"),
        existing.get("description"), parsed.get("description"), parsed.get("summary"),
        parsed.get("caption"), raw_narration,
    )
    transcript = _first_text(
        source.get("asr_text"), source.get("transcript"),
        source.get("transcript_summary"), existing.get("transcript"),
    )
    narration_transcript_summary = _first_text(
        existing.get("narration_transcript_summary"), parsed.get("asr_text"),
        parsed.get("transcript"), parsed.get("transcript_summary"),
    )
    entities = _first_list(
        source.get("visual_entities"), existing.get("visual_entities"),
        parsed.get("visual_entities"), parsed.get("visible_entities"),
        parsed.get("entities"), parsed.get("objects"),
    )
    actions = _first_list(
        source.get("visual_actions"), existing.get("visual_actions"),
        parsed.get("visual_actions"), parsed.get("actions"),
        parsed.get("action_inventory"),
    )
    screen_text = _first_list(
        source.get("screen_text"), source.get("ocr_text"), existing.get("screen_text"),
        parsed.get("screen_text"), parsed.get("ocr_text"), parsed.get("text"),
    )
    relations = _first_list(
        source.get("spatial_relations"), existing.get("spatial_relations"),
        parsed.get("spatial_relations"), parsed.get("relations"),
    )
    uncertainties = _first_list(
        source.get("uncertainties"), existing.get("uncertainties"),
        parsed.get("uncertainties"),
    )
    keywords = _first_list(
        source.get("discriminative_keywords"), existing.get("retrieval_keywords"),
        parsed.get("discriminative_keywords"), parsed.get("retrieval_keywords"),
    )

    try:
        start_sec = float(source.get("start_sec", (existing.get("window") or {}).get("start_sec", 0.0)))
    except (TypeError, ValueError):
        start_sec = 0.0
    try:
        end_sec = float(source.get("end_sec", (existing.get("window") or {}).get("end_sec", start_sec)))
    except (TypeError, ValueError):
        end_sec = start_sec
    media_assets = _normalize_media_assets(
        source.get("media_assets") or existing.get("media_assets") or source.get("artifact_refs"),
        start_sec=start_sec,
        end_sec=end_sec,
    )

    builder_request = _as_mapping(source.get("builder_request"))
    if not builder_request:
        builder_request = _as_mapping(existing.get("builder_request"))
    builder_request.setdefault("artifact_type", STRUCTURE_BUILD_REQUEST_TYPE)
    builder_request.setdefault("schema_version", 1)
    builder_request["window"] = {"start_sec": start_sec, "end_sec": end_sec}
    requested_modalities = _as_list(builder_request.get("requested_modalities"))
    if not requested_modalities:
        if raw_narration:
            requested_modalities = ["multimodal_narration"]
        elif transcript:
            requested_modalities = ["audio_asr"]
        elif media_assets:
            requested_modalities = ["media_reference"]
    builder_request["requested_modalities"] = requested_modalities

    canonical_evidence = {
        "schema_version": CANONICAL_STRUCTURE_SCHEMA_VERSION,
        "window": {"start_sec": start_sec, "end_sec": end_sec},
        "raw_multimodal_narration": raw_narration,
        "description": description,
        "transcript": transcript,
        "narration_transcript_summary": narration_transcript_summary,
        "visual_entities": entities,
        "visual_actions": actions,
        "screen_text": screen_text,
        "spatial_relations": relations,
        "uncertainties": uncertainties,
        "retrieval_keywords": keywords,
        "media_assets": media_assets,
        "builder_request": builder_request,
        "parse_status": (
            "structured" if parsed else "raw_text" if raw_narration
            else "non_vlm_evidence" if (transcript or screen_text or entities or actions or media_assets)
            else "empty"
        ),
    }
    retrieval_document = "\n".join([
        f"window: {start_sec:.3f}-{end_sec:.3f}",
        f"description: {description}",
        f"transcript: {transcript}",
        "entities: " + json.dumps(entities, ensure_ascii=False),
        "actions: " + json.dumps(actions, ensure_ascii=False),
        "screen_text: " + json.dumps(screen_text, ensure_ascii=False),
        "raw_multimodal_narration: " + raw_narration[:6000],
    ])

    normalized = dict(source)
    normalized.update({
        "start_sec": start_sec,
        "end_sec": end_sec,
        "description": description,
        "asr_text": transcript,
        "transcript_summary": transcript[:3000],
        "asr_status": str(source.get("asr_status") or ("ok" if transcript else "not_supplied")),
        "visual_entities": entities,
        "visual_actions": actions,
        "screen_text": screen_text,
        "spatial_relations": relations,
        "uncertainties": uncertainties,
        "discriminative_keywords": keywords,
        "raw_model_output": _first_text(source.get("raw_model_output"), raw_narration),
        "media_assets": media_assets,
        "builder_request": builder_request,
        "canonical_evidence": canonical_evidence,
        "retrieval_document": retrieval_document,
    })
    return normalized


def validate_canonical_structure_record(record: dict | None) -> list[str]:
    """Validate a persisted record without using task or provider data."""
    value = record if isinstance(record, dict) else {}
    evidence = value.get("canonical_evidence")
    if not isinstance(evidence, dict):
        return ["canonical_evidence is missing"]
    issues: list[str] = []
    if int(evidence.get("schema_version", 0) or 0) != CANONICAL_STRUCTURE_SCHEMA_VERSION:
        issues.append("canonical_evidence.schema_version is invalid")
    window = evidence.get("window") or {}
    try:
        if float(window["end_sec"]) <= float(window["start_sec"]):
            issues.append("canonical_evidence.window must have end_sec > start_sec")
    except (KeyError, TypeError, ValueError):
        issues.append("canonical_evidence.window is missing or invalid")
    has_evidence = bool(
        _as_text(evidence.get("raw_multimodal_narration"))
        or _as_text(evidence.get("transcript"))
        or evidence.get("screen_text")
        or evidence.get("visual_entities")
        or evidence.get("visual_actions")
        or evidence.get("media_assets")
    )
    if not has_evidence:
        issues.append("canonical_evidence has no observed or time-bounded source evidence")
    if not _as_text(value.get("retrieval_document")):
        issues.append("retrieval_document is empty")
    request = value.get("builder_request")
    if not isinstance(request, dict):
        issues.append("builder_request provenance is missing")
    elif request.get("artifact_type") != STRUCTURE_BUILD_REQUEST_TYPE:
        issues.append("builder_request.artifact_type is invalid")
    return issues

