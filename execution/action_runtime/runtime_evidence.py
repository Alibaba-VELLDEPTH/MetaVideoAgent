"""Bounded, observable raw-media capability adapters for generated modules."""

from __future__ import annotations

import os
import tempfile
from contextvars import ContextVar
from typing import Any, Dict, List

try:
    import utils
    from capability_registry import CapabilityProfileError, resolve_profile
except ImportError:  # pragma: no cover - package import
    from . import utils
    from .capability_registry import CapabilityProfileError, resolve_profile

_EVENTS: ContextVar[tuple] = ContextVar("runtime_evidence_events", default=())
_TRACE: ContextVar[tuple] = ContextVar("runtime_evidence_trace", default=())


def begin_evidence_run():
    """Start an event buffer for one agent run and return its reset token."""
    return _EVENTS.set(())


def reset_evidence_run(token) -> None:
    if token is not None:
        _EVENTS.reset(token)


def get_evidence_events() -> List[Dict[str, Any]]:
    return [dict(item) for item in _EVENTS.get()]


def begin_runtime_trace():
    """Start a full, in-process execution trace for one evaluation run.

    This trace is intentionally separate from the compact capability-event
    audit: it retains model/tool inputs and returns (never credentials) so a
    repair can diagnose an actual failed decision rather than infer from
    status-only events.
    """
    return _TRACE.set(())


def reset_runtime_trace(token) -> None:
    if token is not None:
        _TRACE.reset(token)


def get_runtime_trace() -> List[Dict[str, Any]]:
    return [dict(item) for item in _TRACE.get()]


def record_runtime_trace(stage: str, *, payload: Any = None, result: Any = None,
                         status: str = "", **details: Any) -> None:
    """Append one lossless, credential-free runtime record.

    Callers provide already-materialized public inputs/results; API keys and
    environment are never read here.  ``default=str`` keeps an individual
    provider oddity from dropping the rest of the trace.
    """
    _TRACE.set((*_TRACE.get(), {
        "trace_index": len(_TRACE.get()) + 1,
        "stage": str(stage or "runtime"),
        "status": str(status or ""),
        "payload": payload,
        "result": result,
        **{key: value for key, value in details.items() if value is not None},
    }))


def _emit(capability: str, event: str, status: str, **details: Any) -> str:
    # This identifier is local to one begin_evidence_run() buffer.  It gives
    # review/diagnosis a stable, bounded anchor without leaking media paths or
    # raw model payloads into the trajectory.
    event_id = f"ev_{len(_EVENTS.get()) + 1:04d}"
    record = {
        "evidence_event_id": event_id,
        "capability": capability,
        "event": event,
        "status": status,
    }
    record.update({key: value for key, value in details.items() if value is not None})
    _EVENTS.set((*_EVENTS.get(), record))
    return event_id


def mark_output_consumed(*, consumer: str, tool_name: str = "", producer_module: str = "",
                         output_nonempty: bool = True, output_protocol: str = "",
                         contract_valid: bool = True,
                         source_event_ids: List[str] | None = None,
                         output_fields: List[str] | None = None) -> None:
    """Record that the agent passed a tool result to the next reasoning turn.

    This is framework instrumentation, not a declaration made by generated
    code.  It lets smoke/probe distinguish "tool returned text" from "the
    preserved consumer actually received the target-module output".
    """
    _emit(
        "module_output",
        "consumer_consumed",
        "ok" if output_nonempty and contract_valid else "no_content",
        consumer=consumer,
        tool_name=tool_name,
        producer_module=producer_module,
        output_protocol=output_protocol,
        contract_valid=bool(contract_valid),
        source_event_ids=[str(value) for value in (source_event_ids or []) if str(value)],
        output_fields=[str(value) for value in (output_fields or []) if str(value)],
    )




def _active_windows(env, max_windows: int | None) -> List[tuple[float, float]]:
    getter = getattr(env, "get_active_time_windows", None)
    raw = getter() if callable(getter) else []
    windows: List[tuple[float, float]] = []
    for item in raw or []:
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end > start:
            windows.append((start, end))
    return windows[:max_windows] if max_windows else windows


def _status(text: str, error_prefix: str, *, no_speech: bool = False) -> str:
    value = str(text or "").strip()
    normalized = value.upper()
    if normalized in {"[NO SPEECH DETECTED]", "NO SPEECH DETECTED", "[NO SPEECH]"}:
        return "no_speech" if no_speech else "no_content"
    if not value:
        return "no_content"
    # ``utils.call_llm`` preserves an empty provider completion as this
    # sentinel so callers can retain the raw response in their trace.  It is
    # not a successful inference: treating it as ``ok`` masks a failed
    # decision attempt and prevents a candidate controller from replanning
    # against the real cause.
    if normalized == "LLM RETURNED EMPTY RESPONSE":
        return "no_content"
    # Adapters preserve the provider's retry context, e.g. ``OCR Error after
    # 2 attempts: ...``.  That is still a provider failure even when the
    # caller supplied the shorter ``Error calling OCR API`` prefix.
    provider_error = (
        value.lower().startswith(error_prefix.lower())
        or normalized.startswith(("OCR ERROR", "VLM ERROR", "ERROR CALLING"))
    )
    return "api_error" if provider_error else "ok"


def _overall_status(items: List[Dict[str, Any]]) -> str:
    """Preserve the most useful terminal status without hiding provider faults."""
    statuses = [str(item.get("status") or "") for item in items]
    if "ok" in statuses:
        return "ok"
    for status in ("api_error", "asset_error", "no_speech", "no_content"):
        if status in statuses:
            return status
    return "unavailable"


def _emit_exception(capability: str, event: str, exc: Exception, **details: Any) -> str:
    """Never lose a provider/asset failure before the evaluator can route it."""
    status = "asset_error" if isinstance(exc, (OSError, ValueError)) else "api_error"
    event_id = _emit(
        capability, event, status,
        error_type=type(exc).__name__, error_message=str(exc)[:300], **details,
    )
    record_runtime_trace(
        "%s.%s" % (capability, event), status=status,
        payload=details,
        result={"error_type": type(exc).__name__, "error_message": str(exc)},
    )
    return event_id


def _profile_details(capability: str, profile_id: str | None) -> Dict[str, Any]:
    """Resolve one registered profile and produce event-safe audit metadata."""
    profile = resolve_profile(capability, profile_id)
    return {
        "profile_id": str(profile.get("profile_id") or ""),
        "provider": str(profile.get("provider") or ""),
        "resolved_model_id": str(profile.get("model_id") or profile.get("external_capability") or ""),
        "transport": str(profile.get("transport") or ""),
        "input_provenance_contract": str(profile.get("input_provenance") or ""),
        "external_capability": str(profile.get("external_capability") or ""),
    }


def transcribe_active_windows(
    env,
    *,
    max_windows: int = 3,
    max_seconds_per_window: float = 45.0,
    model_name: str | None = None,
    profile_id: str | None = None,
) -> Dict[str, Any]:
    """Use segmented ASR only on exact runtime-visible media windows.

    The available ASR provider emits transcript text, not word timestamps.
    Returned evidence consequently retains each extracted segment interval as
    the only valid temporal provenance.
    """
    try:
        profile_details = _profile_details("asr", profile_id)
    except CapabilityProfileError as exc:
        _emit("asr", "request_rejected", "unavailable", reason="unknown_profile", error_message=str(exc))
        return {"status": "unavailable", "reason": "unknown_profile", "windows": []}
    windows = _active_windows(env, max_windows)
    if not windows:
        _emit("asr", "request_rejected", "unavailable", reason="no_active_time_windows", **profile_details)
        return {"status": "unavailable", "reason": "no_active_time_windows", "windows": []}
    video_path = str(getattr(env, "raw_video_path", "") or "")
    if not os.path.isfile(video_path):
        _emit("asr", "request_rejected", "unavailable", reason="raw_video_missing", **profile_details)
        return {"status": "unavailable", "reason": "raw_video_missing", "windows": []}

    evidence = []
    max_seconds = max(1.0, float(max_seconds_per_window or 1.0))
    with tempfile.TemporaryDirectory(prefix="metavideoagent_asr_") as temp_dir:
        for index, (start, end) in enumerate(windows):
            end = min(end, start + max_seconds)
            audio_path = os.path.join(temp_dir, f"window_{index:02d}.wav")
            extraction_exception = False
            try:
                extracted = utils.extract_audio_segment(video_path, start, end, audio_path)
            except Exception as exc:
                _emit_exception(
                    "asr", "audio_extraction", exc, start_sec=start, end_sec=end,
                    input_provenance="active_time_window_segment", **profile_details,
                )
                extraction_exception = True
                extracted = False
            if not extracted:
                if not extraction_exception:
                    _emit("asr", "audio_extraction", "asset_error", start_sec=start, end_sec=end,
                          input_provenance="active_time_window_segment", **profile_details)
                evidence.append({
                    "start_sec": start, "end_sec": end, "status": "asset_error",
                    "text": "", "raw_result": "", "parsed_text": "",
                })
                continue
            _emit(
                "asr", "audio_extraction", "ok", start_sec=start, end_sec=end,
                input_provenance="active_time_window_segment", **profile_details,
            )
            try:
                asr_result = utils.call_asr(
                    audio_path, model_name=model_name or utils.ASR_MODEL,
                    profile_id=profile_details["profile_id"], return_attempts=True,
                )
            except Exception as exc:
                _emit_exception(
                    "asr", "transcription", exc, start_sec=start, end_sec=end,
                    input_provenance="active_time_window_segment", call_index=index, **profile_details,
                )
                evidence.append({
                    "start_sec": start, "end_sec": end, "status": "api_error",
                    "text": "", "raw_result": "", "parsed_text": "",
                })
                continue
            text, attempts = (
                asr_result if isinstance(asr_result, tuple) and len(asr_result) == 2
                else (asr_result, [])
            )
            for attempt in attempts[:-1]:
                if str((attempt or {}).get("status") or "") != "api_error":
                    continue
                _emit(
                    "asr", "transcription_retry", "retrying", start_sec=start, end_sec=end,
                    input_provenance="active_time_window_segment", call_index=index,
                    attempt=int((attempt or {}).get("attempt") or 1),
                    error_type=str((attempt or {}).get("error_type") or ""),
                    error_message=str((attempt or {}).get("error_message") or "")[:300],
                    **profile_details,
                )
            status = _status(text, "Error calling ASR API", no_speech=True)
            _emit(
                "asr", "transcription", status, start_sec=start, end_sec=end,
                input_provenance="active_time_window_segment",
                call_index=index, attempt_count=max(1, len(attempts)),
                retried=bool(len(attempts) > 1), **profile_details,
            )
            raw_text = str(text or "")
            record_runtime_trace(
                "asr.transcription", status=status,
                payload={"start_sec": start, "end_sec": end, "profile": profile_details,
                         "attempts": attempts},
                result=raw_text,
            )
            evidence.append({
                "start_sec": start, "end_sec": end, "status": status,
                "text": raw_text, "raw_result": raw_text, "parsed_text": raw_text,
            })
    overall = _overall_status(evidence)
    return {
        "status": overall, "source": "active_window_segment_asr",
        "call_count": len(evidence), "windows": evidence, "profile": profile_details,
    }




class _ScopedActiveWindowEnv:
    """Delegate an environment while narrowing it to already-active windows."""

    def __init__(self, env: Any, windows: List[tuple[float, float]]):
        self._env = env
        self._windows = windows

    def get_active_time_windows(self):
        return list(self._windows)

    def __getattr__(self, name: str):
        return getattr(self._env, name)


def scoped_media_env(env: Any, time_ranges: Any, *, max_windows: int | None = None) -> tuple[Any, List[tuple[float, float]]]:
    """Return a public, bounded view of ``env`` for localized media work.

    Generated modules must not treat localization ranges as descriptive
    metadata.  This helper normalizes the requested ranges, intersects them
    with the runtime's currently permitted media window, and exposes only that
    intersection to the registered adapters.  It deliberately returns both the
    wrapped environment and actual windows so callers can report an explicit
    ``unavailable`` result when localization produced no usable media.

    No provider is called here; provider access remains in the capability
    adapters below.  The helper is public specifically so generated code never
    needs to discover or depend on the implementation-only wrapper class.
    """
    requested: List[tuple[float, float]] = []
    items = time_ranges if isinstance(time_ranges, (list, tuple)) else []
    for item in items:
        try:
            if isinstance(item, dict):
                start = float(item.get("start_sec", item.get("start")))
                end = float(item.get("end_sec", item.get("end")))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start, end = float(item[0]), float(item[1])
            else:
                continue
        except (TypeError, ValueError):
            continue
        if end > start:
            requested.append((start, end))

    active = _active_windows(env, None)
    actual: List[tuple[float, float]] = []
    for start, end in requested:
        for active_start, active_end in active:
            lo, hi = max(start, active_start), min(end, active_end)
            if hi > lo and (lo, hi) not in actual:
                actual.append((lo, hi))
    actual.sort()
    if max_windows:
        actual = actual[:max(1, int(max_windows))]
    return _ScopedActiveWindowEnv(env, actual), actual


def inspect_time_ranges(
    env: Any,
    time_ranges: Any,
    *,
    capability: str = "vlm",
    prompt: str = "Describe only visible evidence in this frame.",
    max_windows: int = 3,
    frames_per_window: int = 2,
    flat_total_samples: int | None = None,
    batch_vlm: bool = False,
    stitch_frames: bool = False,
    stitch_columns: int = 4,
    crop_region: Dict[str, float] | None = None,
    model_name: str | None = None,
    profile_id: str | None = None,
) -> Dict[str, Any]:
    """Inspect exactly localized ranges through the registered VLM/OCR adapter.

    This is the public range-scoped counterpart to ``inspect_active_frames``.
    Capability events retain their normal timestamp provenance, while the
    return payload records the normalized actual windows for smoke/probe
    auditing.  ``crop_region`` is optional normalized image geometry
    (``x0, y0, x1, y1`` in [0, 1]); it enlarges the same selected frames
    locally before the single VLM request and never expands the media range.
    """
    scoped_env, actual = scoped_media_env(env, time_ranges, max_windows=max_windows)
    if not actual:
        _emit(
            capability, "localized_request_rejected", "unavailable",
            reason="no_intersection_with_active_media",
            input_provenance="localized_time_range_frame",
            requested_window_count=len(time_ranges or []) if isinstance(time_ranges, (list, tuple)) else 0,
            profile_id=str(profile_id or ""),
        )
        return {
            "status": "unavailable",
            "reason": "no_intersection_with_active_media",
            "requested_time_ranges": list(time_ranges or []) if isinstance(time_ranges, (list, tuple)) else [],
            "actual_time_ranges": [],
            "frames": [],
        }
    # A flattened request samples several localized windows as one ordered
    # timeline while retaining scoped-media provenance for every frame.
    if batch_vlm and capability == "vlm" and flat_total_samples is not None:
        if stitch_frames:
            result = _inspect_flattened_vlm_collage(
                scoped_env, actual, prompt=prompt,
                total_samples=flat_total_samples, columns=stitch_columns,
                crop_region=crop_region, model_name=model_name, profile_id=profile_id,
            )
        else:
            result = _inspect_flattened_vlm_batch(
                scoped_env, actual, prompt=prompt,
                total_samples=flat_total_samples, model_name=model_name,
                profile_id=profile_id,
            )
    else:
        result = inspect_active_frames(
            scoped_env, capability=capability, prompt=prompt, max_windows=len(actual),
            frames_per_window=frames_per_window, model_name=model_name, profile_id=profile_id,
        )
    result = dict(result or {})
    result["requested_time_ranges"] = list(time_ranges or []) if isinstance(time_ranges, (list, tuple)) else []
    result["actual_time_ranges"] = [[start, end] for start, end in actual]
    result["source"] = "localized_time_range_frame_inspection"
    return result


def _inspect_flattened_vlm_collage(
    scoped_env: Any,
    windows: list[tuple[float, float]],
    *,
    prompt: Any,
    total_samples: int,
    columns: int,
    crop_region: Dict[str, float] | None,
    model_name: str | None,
    profile_id: str | None,
) -> Dict[str, Any]:
    """Inspect a chronological scoped contact sheet in one VLM image request.

    The source frames are acquired only through ``scoped_env``.  The derived
    contact sheet is transient local media: it is removed immediately after
    the synchronous adapter call, while the event stream preserves every
    source timestamp and the single VLM inference lineage.
    """
    try:
        from PIL import Image, ImageDraw
        profile_details = _profile_details("vlm", profile_id)
    except Exception as exc:
        event_id = _emit_exception(
            "vlm", "request_rejected", exc,
            input_provenance="localized_contact_sheet_frame",
            profile_id=str(profile_id or ""),
        )
        return {"status": "unavailable", "source": "localized_contact_sheet_vlm", "frames": [], "capability_event_ids": [event_id]}

    intervals = sorted(
        [(float(start), float(end)) for start, end in windows if float(end) > float(start)],
        key=lambda item: item[0],
    )
    duration = sum(end - start for start, end in intervals)
    sample_count = max(1, int(total_samples or 1))
    if duration <= 0:
        return {"status": "unavailable", "source": "localized_contact_sheet_vlm", "frames": []}

    timestamps: list[float] = []
    for index in range(sample_count):
        # Sample the center of each equal-duration bin.  Besides avoiding an
        # endpoint bias, this matches the five uniformly spaced overview
        # frames materialized by the distribution observer.
        offset = duration * (index + 0.5) / sample_count
        elapsed = 0.0
        for start, end in intervals:
            width = end - start
            if elapsed <= offset < elapsed + width:
                timestamps.append(start + offset - elapsed)
                break
            elapsed += width

    selected: list[tuple[float, str, str]] = []
    frame_rows: list[dict] = []
    for timestamp in timestamps:
        try:
            image_path = scoped_env.get_frame_at_timestamp(float(timestamp))
        except Exception as exc:
            event_id = _emit_exception(
                "vlm", "frame_selection", exc, timestamp_sec=timestamp,
                input_provenance="localized_contact_sheet_frame", **profile_details,
            )
            frame_rows.append({"timestamp_sec": timestamp, "status": "asset_error", "text": "", "raw_result": "", "parsed_text": "", "capability_event_ids": [event_id]})
            continue
        if not image_path:
            event_id = _emit(
                "vlm", "frame_selection", "asset_error", timestamp_sec=timestamp,
                input_provenance="localized_contact_sheet_frame", **profile_details,
            )
            frame_rows.append({"timestamp_sec": timestamp, "status": "asset_error", "text": "", "raw_result": "", "parsed_text": "", "capability_event_ids": [event_id]})
            continue
        event_id = _emit(
            "vlm", "frame_selection", "ok", timestamp_sec=timestamp,
            input_provenance="localized_contact_sheet_frame", **profile_details,
        )
        selected.append((timestamp, image_path, event_id))

    if not selected:
        return {"status": "asset_error", "source": "localized_contact_sheet_vlm", "call_count": 0, "frames": frame_rows, "profile": profile_details}

    normalized_crop = None
    if isinstance(crop_region, dict):
        try:
            x0, y0 = float(crop_region.get("x0", 0.0)), float(crop_region.get("y0", 0.0))
            x1, y1 = float(crop_region.get("x1", 1.0)), float(crop_region.get("y1", 1.0))
            if 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0:
                normalized_crop = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        except (TypeError, ValueError):
            normalized_crop = None

    columns = max(1, min(int(columns or 4), len(selected)))
    # Keep the overview compact (4-column/12-frame sheet), while a detail
    # caller can request a 2-column sheet whose individual frames retain twice
    # the linear resolution for text and small UI controls.
    # A crop is an explicit request to spend the same VLM call on enlarged
    # local evidence.  Use two columns so the cropped content is not shrunk
    # back to overview resolution.
    cell_width = 640 if (columns <= 2 or normalized_crop) else 320
    cell_height = 360 if (columns <= 2 or normalized_crop) else 180
    label_height = 30 if columns <= 2 else 25
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        for index, (timestamp, image_path, _event_id) in enumerate(selected):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                if normalized_crop:
                    width, height = image.size
                    left = max(0, min(width - 1, int(width * normalized_crop["x0"])))
                    top = max(0, min(height - 1, int(height * normalized_crop["y0"])))
                    right = max(left + 1, min(width, int(width * normalized_crop["x1"])))
                    bottom = max(top + 1, min(height, int(height * normalized_crop["y1"])))
                    image = image.crop((left, top, right, bottom))
                image.thumbnail((cell_width, cell_height))
                x = (index % columns) * cell_width
                y = (index // columns) * (cell_height + label_height)
                sheet.paste(image, (x + (cell_width - image.width) // 2, y + (cell_height - image.height) // 2))
                draw.rectangle((x, y + cell_height, x + cell_width, y + cell_height + label_height), fill="white")
                label = f"#{index + 1}  {timestamp:.2f}s"
                if normalized_crop:
                    label += "  zoom"
                draw.text((x + 4, y + cell_height + 4), label, fill="black")
        with tempfile.NamedTemporaryFile(prefix="metavideoagent_scoped_contact_sheet_", suffix=".jpg", delete=False) as handle:
            collage_path = handle.name
        sheet.save(collage_path, format="JPEG", quality=90)
    except Exception as exc:
        event_id = _emit_exception("vlm", "contact_sheet_build", exc, input_provenance="localized_contact_sheet_frame", **profile_details)
        return {"status": "asset_error", "source": "localized_contact_sheet_vlm", "call_count": 0, "frames": frame_rows, "profile": profile_details, "capability_event_ids": [event_id]}

    try:
        text = utils.call_vlm(prompt, [collage_path], model_name=model_name or utils.VLM_MODEL, profile_id=profile_details["profile_id"])
    except Exception as exc:
        inference_event_id = _emit_exception(
            "vlm", "inference", exc,
            input_provenance="localized_contact_sheet_vlm", sampled_time_ranges=[[start, end] for start, end in intervals],
            frame_count=len(selected), contact_sheet=True, **profile_details,
        )
        status = "api_error"
        text = ""
    else:
        status = _status(text, "Error calling VLM API")
        inference_event_id = _emit(
            "vlm", "inference", status, input_provenance="localized_contact_sheet_vlm",
            start_sec=min(timestamp for timestamp, _path, _event in selected),
            end_sec=max(timestamp for timestamp, _path, _event in selected),
            sampled_time_ranges=[[start, end] for start, end in intervals], frame_count=len(selected),
            contact_sheet=True, **profile_details,
        )
    finally:
        try:
            os.unlink(collage_path)
        except OSError:
            pass

    raw_text = str(text or "")
    record_runtime_trace(
        "vlm.contact_sheet_inference", status=status,
        payload={"timestamps_sec": [timestamp for timestamp, _path, _event in selected], "columns": columns, "rows": rows, "crop_region": normalized_crop, "prompt": prompt, "profile": profile_details},
        result=raw_text,
    )
    for timestamp, _path, selection_event_id in selected:
        frame_rows.append({"timestamp_sec": timestamp, "status": status, "text": raw_text, "raw_result": raw_text, "parsed_text": raw_text, "capability_event_ids": [selection_event_id, inference_event_id]})
    return {"status": status, "source": "localized_contact_sheet_vlm", "call_count": 1, "frame_count": len(selected), "frames": frame_rows, "crop_region": normalized_crop, "profile": profile_details}






def _inspect_flattened_vlm_batch(
    scoped_env: Any,
    windows: list[tuple[float, float]],
    *,
    prompt: Any,
    total_samples: int,
    model_name: str | None,
    profile_id: str | None,
) -> Dict[str, Any]:
    """Sample flattened intervals and issue one scoped multi-image VLM call."""
    try:
        profile_details = _profile_details("vlm", profile_id)
    except CapabilityProfileError as exc:
        _emit("vlm", "request_rejected", "unavailable", reason="unknown_profile", error_message=str(exc))
        return {"status": "unavailable", "reason": "unknown_profile", "frames": []}

    intervals = sorted(
        [(float(start), float(end)) for start, end in windows if float(end) > float(start)],
        key=lambda item: item[0],
    )
    total_duration = sum(end - start for start, end in intervals)
    sample_count = max(1, int(total_samples or 1))
    if total_duration <= 0:
        return {"status": "unavailable", "reason": "no_positive_time_ranges", "frames": [], "profile": profile_details}

    timestamps: list[float] = []
    for index in range(sample_count):
        # Use bin centers so a flattened request covers the full interval
        # uniformly without over-representing its first instant.
        offset = total_duration * (index + 0.5) / sample_count
        elapsed = 0.0
        for start, end in intervals:
            width = end - start
            if elapsed <= offset < elapsed + width:
                timestamps.append(start + (offset - elapsed))
                break
            elapsed += width

    selected: list[tuple[float, str, str]] = []
    frame_rows: list[dict] = []
    for timestamp in timestamps:
        try:
            image_path = scoped_env.get_frame_at_timestamp(float(timestamp))
        except Exception as exc:
            selection_event_id = _emit_exception(
                "vlm", "frame_selection", exc, timestamp_sec=timestamp,
                input_provenance="localized_flattened_time_range_frame", **profile_details,
            )
            frame_rows.append({"timestamp_sec": timestamp, "status": "asset_error", "text": "", "raw_result": "", "parsed_text": "", "capability_event_ids": [selection_event_id]})
            continue
        if not image_path:
            selection_event_id = _emit(
                "vlm", "frame_selection", "asset_error", timestamp_sec=timestamp,
                input_provenance="localized_flattened_time_range_frame", **profile_details,
            )
            frame_rows.append({"timestamp_sec": timestamp, "status": "asset_error", "text": "", "raw_result": "", "parsed_text": "", "capability_event_ids": [selection_event_id]})
            continue
        selection_event_id = _emit(
            "vlm", "frame_selection", "ok", timestamp_sec=timestamp,
            input_provenance="localized_flattened_time_range_frame", **profile_details,
        )
        selected.append((timestamp, image_path, selection_event_id))

    if not selected:
        return {"status": "asset_error", "source": "localized_flattened_vlm_batch", "call_count": 0, "frames": frame_rows, "profile": profile_details}
    try:
        text = utils.call_vlm(
            prompt, [path for _, path, _ in selected], model_name=model_name or utils.VLM_MODEL,
            profile_id=profile_details["profile_id"],
        )
    except Exception as exc:
        inference_event_id = _emit_exception(
            "vlm", "inference", exc, input_provenance="localized_flattened_time_range_frame_batch",
            sampled_time_ranges=[[start, end] for start, end in intervals], **profile_details,
        )
        for timestamp, _path, selection_event_id in selected:
            frame_rows.append({"timestamp_sec": timestamp, "status": "api_error", "text": "", "raw_result": "", "parsed_text": "", "capability_event_ids": [selection_event_id, inference_event_id]})
        return {"status": "api_error", "source": "localized_flattened_vlm_batch", "call_count": 1, "frames": frame_rows, "profile": profile_details}

    status = _status(text, "Error calling VLM API")
    inference_event_id = _emit(
        "vlm", "inference", status, input_provenance="localized_flattened_time_range_frame_batch",
        # A batch has no single frame timestamp.  Persist both its enclosing
        # temporal bounds and exact disjoint inputs so smoke can verify that
        # this successful VLM event intersects the tool's requested ranges.
        start_sec=min(timestamp for timestamp, _path, _event in selected),
        end_sec=max(timestamp for timestamp, _path, _event in selected),
        sampled_time_ranges=[[start, end] for start, end in intervals], frame_count=len(selected), **profile_details,
    )
    raw_text = str(text or "")
    record_runtime_trace(
        "vlm.flattened_batch_inference", status=status,
        payload={"timestamps_sec": [timestamp for timestamp, _path, _event in selected], "prompt": prompt, "profile": profile_details},
        result=raw_text,
    )
    for timestamp, _path, selection_event_id in selected:
        frame_rows.append({
            "timestamp_sec": timestamp, "status": status, "text": raw_text,
            "raw_result": raw_text, "parsed_text": raw_text,
            "capability_event_ids": [selection_event_id, inference_event_id],
        })
    return {
        "status": status, "source": "localized_flattened_vlm_batch", "call_count": 1,
        "frame_count": len(selected), "frames": frame_rows, "profile": profile_details,
    }


def transcribe_time_ranges(
    env: Any,
    time_ranges: Any,
    *,
    max_windows: int = 3,
    max_seconds_per_window: float = 45.0,
    model_name: str | None = None,
    profile_id: str | None = None,
) -> Dict[str, Any]:
    """Transcribe exactly localized ranges through the registered ASR adapter."""
    scoped_env, actual = scoped_media_env(env, time_ranges, max_windows=max_windows)
    if not actual:
        _emit(
            "asr", "localized_request_rejected", "unavailable",
            reason="no_intersection_with_active_media",
            input_provenance="localized_time_range_segment",
            requested_window_count=len(time_ranges or []) if isinstance(time_ranges, (list, tuple)) else 0,
            profile_id=str(profile_id or ""),
        )
        return {
            "status": "unavailable",
            "reason": "no_intersection_with_active_media",
            "requested_time_ranges": list(time_ranges or []) if isinstance(time_ranges, (list, tuple)) else [],
            "actual_time_ranges": [],
            "windows": [],
        }
    result = transcribe_active_windows(
        scoped_env, max_windows=len(actual), max_seconds_per_window=max_seconds_per_window,
        model_name=model_name, profile_id=profile_id,
    )
    result = dict(result or {})
    result["requested_time_ranges"] = list(time_ranges or []) if isinstance(time_ranges, (list, tuple)) else []
    result["actual_time_ranges"] = [[start, end] for start, end in actual]
    result["source"] = "localized_time_range_segment_asr"
    return result




def inspect_active_frames(
    env,
    *,
    capability: str = "vlm",
    prompt: str = "Describe only visible evidence in this frame.",
    max_windows: int = 3,
    frames_per_window: int = 2,
    model_name: str | None = None,
    profile_id: str | None = None,
) -> Dict[str, Any]:
    """Run bounded VLM/OCR inspection over frames sampled from active windows."""
    if capability not in {"vlm", "ocr"}:
        raise ValueError("capability must be 'vlm' or 'ocr'")
    try:
        profile_details = _profile_details(capability, profile_id)
    except CapabilityProfileError as exc:
        _emit(capability, "request_rejected", "unavailable", reason="unknown_profile", error_message=str(exc))
        return {"status": "unavailable", "reason": "unknown_profile", "frames": []}
    windows = _active_windows(env, max_windows)
    if not windows:
        _emit(capability, "request_rejected", "unavailable", reason="no_active_time_windows", **profile_details)
        return {"status": "unavailable", "reason": "no_active_time_windows", "frames": []}
    frames = []
    count = max(1, int(frames_per_window or 1))
    for start, end in windows:
        for index in range(count):
            timestamp = start + (end - start) * ((index + 0.5) / count)
            frame_selection_exception = False
            try:
                image_path = env.get_frame_at_timestamp(timestamp)
            except Exception as exc:
                _emit_exception(
                    capability, "frame_selection", exc, timestamp_sec=timestamp,
                    input_provenance="active_time_window_frame", **profile_details,
                )
                frame_selection_exception = True
                image_path = ""
            if not image_path:
                if not frame_selection_exception:
                    _emit(capability, "frame_selection", "asset_error", timestamp_sec=timestamp,
                          input_provenance="active_time_window_frame", **profile_details)
                frames.append({
                    "timestamp_sec": timestamp, "status": "asset_error",
                    "text": "", "raw_result": "", "parsed_text": "",
                })
                continue
            frame_selection_event_id = _emit(
                capability, "frame_selection", "ok", timestamp_sec=timestamp,
                input_provenance="active_time_window_frame", **profile_details,
            )
            try:
                text = (
                    utils.call_vlm(
                        prompt, [image_path], model_name=model_name or utils.VLM_MODEL,
                        profile_id=profile_details["profile_id"],
                    )
                    if capability == "vlm"
                    else utils.call_ocr(
                        image_path, model_name=model_name or utils.OCR_MODEL,
                        profile_id=profile_details["profile_id"],
                    )
                )
            except Exception as exc:
                _emit_exception(
                    capability, "inference", exc, timestamp_sec=timestamp,
                    input_provenance="active_time_window_frame", **profile_details,
                )
                frames.append({
                    "timestamp_sec": timestamp, "status": "api_error",
                    "text": "", "raw_result": "", "parsed_text": "",
                })
                continue
            status = _status(text, "Error calling VLM API" if capability == "vlm" else "Error calling OCR API")
            inference_event_id = _emit(
                capability, "inference", status, timestamp_sec=timestamp,
                input_provenance="active_time_window_frame",
                call_index=len(frames), **profile_details,
            )
            raw_text = str(text or "")
            record_runtime_trace(
                "%s.inference" % capability, status=status,
                payload={"timestamp_sec": timestamp, "prompt": prompt, "profile": profile_details},
                result=raw_text,
            )
            frames.append({
                "timestamp_sec": timestamp, "status": status,
                "text": raw_text, "raw_result": raw_text, "parsed_text": raw_text,
                # Per-frame event lineage lets generated evidence records cite
                # their exact adapter calls instead of an entire tool batch.
                "capability_event_ids": [frame_selection_event_id, inference_event_id],
            })
    overall = _overall_status(frames)
    return {
        "status": overall, "source": "active_window_frame_inspection",
        "call_count": len(frames), "frames": frames, "profile": profile_details,
    }


def call_text_llm(messages, *, profile_id: str | None = None, temperature: float = 0.1,
                  max_tokens: int | None = 4096, response_format: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Bounded text reasoning through a registered LLM profile."""
    try:
        profile_details = _profile_details("llm", profile_id)
    except CapabilityProfileError as exc:
        _emit("llm", "request_rejected", "unavailable", reason="unknown_profile", error_message=str(exc))
        return {"status": "unavailable", "reason": "unknown_profile", "text": ""}
    try:
        text = utils.call_llm(
            list(messages or []), temperature=temperature, max_tokens=max_tokens,
            profile_id=profile_details["profile_id"], response_format=response_format,
        )
    except Exception as exc:
        _emit_exception("llm", "inference", exc,
                        input_provenance="bounded_text_context", **profile_details)
        result = {"status": "api_error", "text": "", "profile": profile_details}
        record_runtime_trace("llm.inference", status="api_error",
                             payload={"messages": list(messages or []), "temperature": temperature,
                                      "max_tokens": max_tokens, "response_format": response_format,
                                      "profile": profile_details}, result=result)
        return result
    status = _status(text, "LLM Error")
    _emit("llm", "inference", status, input_provenance="bounded_text_context", **profile_details)
    result = {"status": status, "text": str(text or ""), "profile": profile_details}
    record_runtime_trace("llm.inference", status=status,
                         payload={"messages": list(messages or []), "temperature": temperature,
                                  "max_tokens": max_tokens, "response_format": response_format,
                                  "profile": profile_details}, result=result)
    return result


def embed_text(input_text: str, *, profile_id: str | None = None) -> Dict[str, Any]:
    """Create one registered text embedding and expose its provenance."""
    try:
        profile_details = _profile_details("embedding", profile_id)
    except CapabilityProfileError as exc:
        _emit("embedding", "request_rejected", "unavailable", reason="unknown_profile", error_message=str(exc))
        return {"status": "unavailable", "reason": "unknown_profile", "embedding": []}
    try:
        vector = utils.call_embedding(
            str(input_text or ""), profile_id=profile_details["profile_id"],
        )
    except Exception as exc:
        _emit_exception("embedding", "inference", exc,
                        input_provenance="bounded_text_context", **profile_details)
        return {"status": "api_error", "embedding": [], "profile": profile_details}
    status = "ok" if vector and any(float(value) != 0.0 for value in vector) else "no_content"
    _emit("embedding", "inference", status, input_provenance="bounded_text_context", **profile_details)
    result = {"status": status, "embedding": vector, "profile": profile_details}
    record_runtime_trace("embedding.inference", status=status,
                         payload={"input_text": str(input_text or ""), "profile": profile_details}, result=result)
    return result


def embed_image(image_path: str, *, profile_id: str | None = None) -> Dict[str, Any]:
    """Embed a local frame in the same space as ``embed_text``.

    This is intentionally a public runtime adapter so generated localization
    code can perform text-to-image retrieval without provider SDK/HTTP calls.
    """
    try:
        profile_details = _profile_details("embedding", profile_id)
        vector = utils.call_image_embedding(str(image_path or ""), profile_id=profile_details["profile_id"])
    except Exception as exc:
        _emit_exception("embedding", "image_inference", exc,
                        input_provenance="active_time_window_frame")
        return {"status": "api_error", "embedding": []}
    status = "ok" if vector and any(float(value) != 0.0 for value in vector) else "no_content"
    _emit("embedding", "image_inference", status,
          input_provenance="active_time_window_frame", **profile_details)
    result = {"status": status, "embedding": vector, "profile": profile_details}
    record_runtime_trace("embedding.image_inference", status=status,
                         payload={"image_path": str(image_path or ""), "profile": profile_details}, result=result)
    return result
