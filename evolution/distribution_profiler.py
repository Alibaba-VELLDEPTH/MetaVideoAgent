"""Distribution-aware profiling for MetaVideoAgent initialization.

The initial profile combines five uniformly sampled overview frames per
evolution video with its associated questions and answer options.  Answer
labels, evidence annotations, human task categories, trajectories, and
execution artifacts remain excluded.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List

try:
    from .distribution_spec import prompt_safe_spec
    from .question_set_loader import dataset_files_from_path
    from .runtime_paths import metavideoagent_output_root
except ImportError:
    from distribution_spec import prompt_safe_spec
    from question_set_loader import dataset_files_from_path
    from runtime_paths import metavideoagent_output_root


def _conda_sibling_binary(name: str) -> str:
    candidate = os.path.join(os.path.dirname(sys.executable), name)
    return candidate if os.path.exists(candidate) and os.access(candidate, os.X_OK) else ""


def _ffmpeg_bin() -> str:
    env_value = os.environ.get("FFMPEG_BIN") or os.environ.get("METAVIDEOAGENT_FFMPEG_BIN")
    if env_value:
        return env_value
    sibling = _conda_sibling_binary("ffmpeg")
    if sibling:
        return sibling
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _ffprobe_bin() -> str:
    env_value = os.environ.get("FFPROBE_BIN") or os.environ.get("METAVIDEOAGENT_FFPROBE_BIN")
    if env_value:
        return env_value
    sibling = _conda_sibling_binary("ffprobe")
    if sibling:
        return sibling
    return "ffprobe"


def _iter_jsonl(path: str, limit: int = 0) -> Iterable[dict]:
    if not path or not os.path.exists(path):
        return
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            if limit and count >= limit:
                return


def _iter_split_rows(split_path: str, limit: int = 0) -> Iterable[dict]:
    if not split_path:
        return
    yielded = 0
    for item in dataset_files_from_path(split_path):
        remaining = max(0, limit - yielded) if limit else 0
        if limit and remaining <= 0:
            return
        for row in _iter_jsonl(item, limit=remaining):
            yield row
            yielded += 1
            if limit and yielded >= limit:
                return


def _video_ids_from_split(split_path: str, limit: int = 0) -> List[str]:
    """Read stable video identifiers from an evolution split."""
    ids = []
    seen = set()
    for row in _iter_split_rows(split_path, limit=0):
        video_id = (
            row.get("video_id")
            or row.get("video_uid")
            or row.get("video")
            or row.get("video_name")
            or ""
        )
        video_id = str(video_id).strip()
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        ids.append(video_id)
        if limit and len(ids) >= limit:
            break
    return ids


def _queries_by_video_from_split(split_path: str) -> Dict[str, List[dict]]:
    """Return query text/options without answer or evidence supervision."""
    grouped: Dict[str, List[dict]] = {}
    for row in _iter_split_rows(split_path, limit=0):
        video_id = str(
            row.get("video_id")
            or row.get("video_uid")
            or row.get("video")
            or row.get("video_name")
            or ""
        ).strip()
        question = str(row.get("question") or row.get("query") or row.get("prompt") or "").strip()
        options = row.get("choices")
        if options in (None, "", [], {}):
            options = row.get("options")
        if not video_id or not question:
            continue
        grouped.setdefault(video_id, []).append({
            "question": question,
            "choices": options if options not in (None, "") else [],
        })
    return grouped


def _distribution_records(video_ids: List[str], queries_by_video: Dict[str, List[dict]],
                          video_stats: Dict[str, Any], semantic_rows: List[dict]) -> List[dict]:
    """Build prompt-safe per-video records matching the paper profile contract."""
    metadata = {
        str(item.get("video_id") or ""): {
            key: item.get(key)
            for key in (
                "duration_seconds", "fps", "frame_count", "avg_frame_delta",
                "scene_change_samples", "low_visual_motion", "has_audio_stream",
            )
        }
        for item in ((video_stats or {}).get("per_video") or [])
        if isinstance(item, dict) and item.get("video_id")
    }
    observations: Dict[str, List[dict]] = {}
    for row in semantic_rows:
        if not isinstance(row, dict):
            continue
        video_id = str(row.get("video_id") or "")
        model = row.get("model_observation") if isinstance(row.get("model_observation"), dict) else {}
        parsed = model.get("vlm_json") if isinstance(model.get("vlm_json"), dict) else {}
        asset = row.get("asset") if isinstance(row.get("asset"), dict) else {}
        timestamps = [
            frame.get("time_sec")
            for frame in (asset.get("frames") or [])
            if isinstance(frame, dict) and frame.get("time_sec") is not None
        ]
        observations.setdefault(video_id, []).append({
            "overview_frame_timestamps_sec": timestamps[:5],
            "overview_summary": parsed.get("visual_summary") or model.get("vlm_raw") or "",
            "query_relevant_requirements": parsed.get("query_relevant_requirements") or [],
            "dominant_information_channel": parsed.get("dominant_information_channel") or "unknown",
        })
    return [
        {
            "video_id": video_id,
            "non_answer_metadata": metadata.get(video_id, {}),
            "overview_frame_count": 5,
            "overview_observations": observations.get(video_id, []),
            "associated_queries": queries_by_video.get(video_id, []),
        }
        for video_id in video_ids
    ]


def _video_ids_from_workspace(workspace_dir: str, limit: int = 0) -> List[str]:
    raw_dir = os.path.join(workspace_dir, "raw_videos")
    ids = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.mp4"))):
        ids.append(os.path.splitext(os.path.basename(path))[0])
        if limit and len(ids) >= limit:
            break
    return ids


def _level(value: float, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def _channel_hypothesis(channel_counter: Counter, tool_counts: Counter,
                        structure_stats: Dict[str, Any], total_q: int,
                        video_stats: Dict[str, Any] | None = None,
                        semantic_stats: Dict[str, Any] | None = None) -> Dict[str, Any]:
    total_q = max(1, total_q)
    density = structure_stats.get("density", {}) or {}
    video_summary = (video_stats or {}).get("summary", {}) or {}
    semantic_density = (semantic_stats or {}).get("density", {}) or {}
    audio_stream_ratio = video_summary.get("audio_stream_ratio")
    audio_activity_ratio = video_summary.get("audio_activity_ratio")
    visual_dynamic_level = video_summary.get("visual_dynamics", "unknown")
    low_motion_ratio = float(video_summary.get("low_visual_motion_ratio") or 0.0)
    audio_bonus = float(audio_stream_ratio or 0.0) * 0.2 if audio_stream_ratio is not None else 0.0
    audio_bonus += float(audio_activity_ratio or 0.0) * 0.35 if audio_activity_ratio is not None else 0.0
    static_visual_penalty = 0.15 if low_motion_ratio >= 0.7 else 0.0
    visual_motion_bonus = 0.15 if visual_dynamic_level == "high" else 0.05 if visual_dynamic_level == "medium" else 0.0
    scores = {
        "speech_or_audio": (
            channel_counter["speech_or_dialogue_semantics"] / total_q
            + density.get("speech_or_audio_mentions", 0)
            + min(1.0, tool_counts.get("mentions_audio_or_speech", 0) / max(1, total_q))
            + audio_bonus
            + float(semantic_density.get("speech_or_audio", 0.0) or 0.0)
        ),
        "screen_text_or_subtitle": (
            channel_counter["screen_text_or_subtitle"] / total_q
            + density.get("subtitle_or_transcript_mentions", 0)
            + density.get("ocr_or_screen_text_mentions", 0)
            + min(1.0, tool_counts.get("mentions_text_or_ocr", 0) / max(1, total_q))
            + float(semantic_density.get("screen_text_or_subtitle", 0.0) or 0.0)
        ),
        "visual_action": (
            channel_counter["visual_event_or_action"] / total_q
            + density.get("visual_action_mentions", 0)
            + float(semantic_density.get("visual_action", 0.0) or 0.0)
            + visual_motion_bonus
            - static_visual_penalty
        ),
        "visual_object_detail": (
            channel_counter["visual_detail_or_object"] / total_q
            + density.get("visual_object_detail_mentions", 0)
            + float(semantic_density.get("visual_object_detail", 0.0) or 0.0)
            + max(0.0, visual_motion_bonus / 2)
        ),
        "long_context_reasoning": channel_counter["long_context_reasoning"] / total_q,
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary = ranked[0][0] if ranked and ranked[0][1] > 0 else "mixed_or_unknown"
    secondary = [name for name, score in ranked[1:4] if score > 0]
    return {
        "primary_observed_channel": primary,
        "secondary_observed_channels": secondary,
        "channel_scores": {k: round(v, 3) for k, v in ranked},
        "raw_video_signal_evidence": video_summary,
        "use_in_diagnosis": (
            "Use this as auxiliary evidence only. Concrete wrong trajectories, "
            "Teacher micro-reviews, full-eval deltas, and probe evidence remain primary."
        ),
    }


def _stable_channel_schema(channel_counter: Counter,
                           total_q: int,
                           interval_lengths: List[int],
                           video_stats: Dict[str, Any],
                           structure_stats: Dict[str, Any],
                           semantic_stats: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return the information-channel schema used by MetaVideoAgent prompts.

    The estimate is intentionally lightweight: sample a few frames across each
    video for visual dynamics and a short local audio segment for speech/audio
    activity. No VLM/LLM/API call is made here.
    """
    total_q = max(1, total_q)
    video_summary = (video_stats or {}).get("summary", {}) or {}
    raw_video_available = (video_stats or {}).get("sampled_videos", 0) > 0 and video_summary.get("status") == "sampled_raw_videos"
    density = (structure_stats or {}).get("density", {}) or {}
    semantic_density = (semantic_stats or {}).get("density", {}) or {}
    avg_motion = float(video_summary.get("avg_frame_delta") or 0.0) if raw_video_available else 0.0
    low_motion_ratio = float(video_summary.get("low_visual_motion_ratio") or 0.0) if raw_video_available else 0.0
    audio_stream_ratio = video_summary.get("audio_stream_ratio")
    audio_activity_ratio = float(video_summary.get("audio_activity_ratio") or 0.0)
    text_density = (
        channel_counter["screen_text_or_subtitle"] / total_q
        + float(density.get("subtitle_or_transcript_mentions", 0.0) or 0.0)
        + float(density.get("ocr_or_screen_text_mentions", 0.0) or 0.0)
        + float(semantic_density.get("screen_text_or_subtitle", 0.0) or 0.0)
    )
    visual_action_density = (
        channel_counter["visual_event_or_action"] / total_q
        + float(density.get("visual_action_mentions", 0.0) or 0.0)
        + float(semantic_density.get("visual_action", 0.0) or 0.0)
        + min(1.0, avg_motion * 8.0)
    )
    visual_detail_density = (
        channel_counter["visual_detail_or_object"] / total_q
        + float(density.get("visual_object_detail_mentions", 0.0) or 0.0)
        + float(semantic_density.get("visual_object_detail", 0.0) or 0.0)
        + min(1.0, avg_motion * 4.0)
    )
    speech_density = (
        channel_counter["speech_or_dialogue_semantics"] / total_q
        + float(density.get("speech_or_audio_mentions", 0.0) or 0.0)
        + float(semantic_density.get("speech_or_audio", 0.0) or 0.0)
        + audio_activity_ratio
    )
    scene_change_density = max(0.0, 1.0 - low_motion_ratio) if raw_video_available else 0.0
    long_interval_ratio = (
        sum(1 for x in interval_lengths if x >= 120) / max(1, len(interval_lengths))
        if interval_lengths else 0.0
    )
    channel_scores = {
        "audio_speech": round(speech_density, 3),
        "visual_motion": round(visual_action_density, 3),
        "visual_detail": round(visual_detail_density, 3),
        "text_subtitle_ocr": round(text_density, 3),
        "long_context": round(
            channel_counter["long_context_reasoning"] / total_q + long_interval_ratio,
            3,
        ),
    }
    dominant = max(channel_scores.items(), key=lambda item: item[1])[0] if channel_scores else "mixed"
    sorted_scores = sorted(channel_scores.values(), reverse=True)
    confidence = 0.0
    if sorted_scores:
        confidence = min(1.0, max(0.0, sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)))
    return {
        "schema_version": 1,
        "sampling_method": (
            "five uniformly sampled overview frames per video plus associated "
            "questions/options and local media metadata. Answers, evidence windows, "
            "task categories, trajectories, and structure text are excluded"
        ),
        "semantic_observation_available": bool((semantic_stats or {}).get("sampled_observations")),
        "raw_video_sample_available": raw_video_available,
        "visual_motion_density": round(avg_motion, 4) if raw_video_available else None,
        "visual_scene_change_density": round(scene_change_density, 3) if raw_video_available else None,
        "speech_activity_density": round(audio_activity_ratio, 3),
        "audio_stream_ratio": round(float(audio_stream_ratio), 3) if audio_stream_ratio is not None else None,
        "subtitle_or_ocr_density": round(min(1.0, text_density), 3),
        "visual_action_density": round(min(1.0, visual_action_density), 3),
        "visual_detail_density": round(min(1.0, visual_detail_density), 3),
        "long_context_density": round(min(1.0, long_interval_ratio), 3),
        "audio_visual_alignment_hint": (
            "audio_dominant_static_visual"
            if speech_density >= 0.5 and low_motion_ratio >= 0.5 else
            "visual_dominant_dynamic"
            if visual_action_density >= speech_density and scene_change_density >= 0.5 else
            "mixed_or_uncertain"
        ),
        "dominant_information_channel": dominant,
        "channel_scores": channel_scores,
        "channel_confidence": round(confidence, 3),
        "prompt_safety": {
            "uses_human_type_labels": False,
            "excluded_fields": ["type", "category", "sub_category", "question_type"],
        },
    }




def _stable_sample_values(values: List[str], budget: int) -> List[str]:
    values = sorted({str(v) for v in values if v})
    if not budget or len(values) <= budget:
        return values
    return sorted(values, key=lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())[:budget]


def _has_audio_stream(video_path: str) -> bool | None:
    try:
        proc = subprocess.run(
            [
                _ffprobe_bin(), "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "default=nw=1:nk=1", video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return "audio" in (proc.stdout or "").lower()


def _audio_activity_stats(video_path: str, duration: float = 0.0) -> Dict[str, Any]:
    """Estimate audio density from a short local sample without calling APIs."""
    if not video_path or not os.path.exists(video_path):
        return {"available": False, "reason": "missing_video"}
    sample_seconds = 90.0
    if duration and duration > 0:
        sample_seconds = min(sample_seconds, max(15.0, duration * 0.08))
    try:
        proc = subprocess.run(
            [
                _ffmpeg_bin(), "-nostdin", "-hide_banner", "-v", "info",
                "-t", f"{sample_seconds:.2f}",
                "-i", video_path,
                "-map", "0:a:0", "-af", "volumedetect",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "reason": f"ffmpeg_failed: {exc}"}
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    mean_match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", text)
    max_match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", text)
    if proc.returncode != 0 and not mean_match:
        return {"available": False, "reason": "no_audio_or_ffmpeg_error"}
    mean_db = float(mean_match.group(1)) if mean_match else None
    max_db = float(max_match.group(1)) if max_match else None
    if mean_db is None:
        activity = "unknown"
    elif mean_db >= -28:
        activity = "high"
    elif mean_db >= -42:
        activity = "medium"
    else:
        activity = "low"
    return {
        "available": True,
        "sample_seconds": round(sample_seconds, 2),
        "mean_volume_db": mean_db,
        "max_volume_db": max_db,
        "audio_activity": activity,
    }


def _video_signal_stats(workspace_dir: str, video_ids: List[str],
                        budget: int = 24) -> Dict[str, Any]:
    """Sample raw videos locally to estimate observable information channels.

    This is deliberately API-free.  It provides coarse evidence for whether a
    distribution is visually dynamic, low-motion/static, long-form, or likely
    audio-bearing.  It does not use human type labels.
    """
    try:
        import cv2
    except Exception as exc:
        return {"sampled_videos": 0, "error": f"cv2 unavailable: {exc}"}

    raw_dir = os.path.join(workspace_dir, "raw_videos")
    ids = _stable_sample_values(video_ids, budget)
    per_video = []
    for video_id in ids:
        path = os.path.join(raw_dir, f"{video_id}.mp4")
        if not os.path.exists(path):
            per_video.append({"video_id": video_id, "path": path, "exists": False})
            continue
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            per_video.append({"video_id": video_id, "path": path, "exists": True, "readable": False})
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        sample_count = 5
        motion_scores = []
        scene_changes = 0
        prev_gray = None
        for idx in range(sample_count):
            if frame_count > 1:
                frame_idx = int((idx + 0.5) * frame_count / sample_count)
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, frame_count - 1))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(diff.mean()) / 255.0
                motion_scores.append(score)
                if score >= 0.12:
                    scene_changes += 1
            prev_gray = gray
        cap.release()
        avg_motion = sum(motion_scores) / max(1, len(motion_scores))
        audio_stream = _has_audio_stream(path)
        audio_activity = (
            _audio_activity_stats(path, duration=duration)
            if audio_stream else
            {"available": False, "reason": "no_audio_stream"}
        )
        per_video.append({
            "video_id": video_id,
            "path": path,
            "exists": True,
            "readable": True,
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 3),
            "frame_count": frame_count,
            "avg_frame_delta": round(avg_motion, 4),
            "scene_change_samples": scene_changes,
            "low_visual_motion": avg_motion < 0.025,
            "has_audio_stream": audio_stream,
            "audio_activity": audio_activity,
        })

    readable = [v for v in per_video if v.get("readable")]
    if not readable:
        return {
            "sampled_videos": 0,
            "requested_videos": len(ids),
            "per_video": per_video,
            "summary": {"status": "no_readable_raw_videos"},
        }
    durations = [v.get("duration_seconds", 0.0) for v in readable]
    motions = [v.get("avg_frame_delta", 0.0) for v in readable]
    audio_known = [v for v in readable if v.get("has_audio_stream") is not None]
    audio_ratio = (
        sum(1 for v in audio_known if v.get("has_audio_stream")) / max(1, len(audio_known))
        if audio_known else None
    )
    low_motion_ratio = sum(1 for v in readable if v.get("low_visual_motion")) / max(1, len(readable))
    active_audio = [
        v for v in readable
        if ((v.get("audio_activity") or {}).get("audio_activity") in ("high", "medium"))
    ]
    audio_activity_ratio = len(active_audio) / max(1, len(readable))
    summary = {
        "status": "sampled_raw_videos",
        "avg_duration_seconds": round(sum(durations) / max(1, len(durations)), 2),
        "avg_frame_delta": round(sum(motions) / max(1, len(motions)), 4),
        "low_visual_motion_ratio": round(low_motion_ratio, 3),
        "audio_stream_ratio": round(audio_ratio, 3) if audio_ratio is not None else None,
        "audio_activity_ratio": round(audio_activity_ratio, 3),
        "visual_dynamics": _level(sum(motions) / max(1, len(motions)), 0.025, 0.08),
        "audio_availability": (
            "unknown" if audio_ratio is None else
            ("high" if audio_ratio >= 0.8 else "medium" if audio_ratio >= 0.3 else "low")
        ),
        "audio_activity": _level(audio_activity_ratio, 0.3, 0.7),
    }
    return {
        "sampled_videos": len(readable),
        "requested_videos": len(ids),
        "summary": summary,
        "per_video": per_video[:budget],
    }


def _semantic_sampling_plan(video_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Return the deterministic five-frame overview plan used at initialization."""
    per_video = (video_stats or {}).get("per_video", []) or []
    plan = []
    for item in per_video:
        if not item.get("readable"):
            continue
        duration = float(item.get("duration_seconds") or 0.0)
        if duration <= 0:
            continue
        plan.append({
            "video_id": item.get("video_id", ""),
            "video_duration_seconds": duration,
            "frame_timestamps_sec": [
                round(duration * (index + 0.5) / 5, 2)
                for index in range(5)
            ],
        })
    return {
        "schema_version": 1,
        "purpose": (
            "Five uniformly sampled full-video frames per evolution video, "
            "summarized in one query-aware VLM request."
        ),
        "frames_per_video": 5,
        "coverage": "complete_video_duration",
        "sampled_videos": len(plan),
        "plan": plan,
    }


def _load_semantic_observation_rows(path: str, budget: int) -> List[dict]:
    if not path or not os.path.exists(path):
        return []
    if os.path.isdir(path):
        rows = []
        for item in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
            rows.extend(_iter_jsonl(item, limit=max(1, budget - len(rows))))
            if len(rows) >= budget:
                break
        return rows[:budget]
    if path.endswith(".jsonl"):
        return list(_iter_jsonl(path, limit=budget))
    if path.endswith(".json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return []
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)][:budget]
        if isinstance(payload, dict):
            rows = payload.get("observations") or payload.get("rows") or payload.get("samples")
            if isinstance(rows, list):
                return [x for x in rows if isinstance(x, dict)][:budget]
    return []


def _semantic_observation_stats(path: str, budget: int) -> Dict[str, Any]:
    rows = _load_semantic_observation_rows(path, budget)
    if not rows:
        return {
            "sampled_observations": 0,
            "source_available": bool(path and os.path.exists(path)),
            "source_basename": os.path.basename(path) if path else "",
            "planned_video_records": 0,
            "density": {},
            "keyword_counts": {},
        }
    counts = Counter()
    usable = 0
    for row in rows:
        model_obs = row.get("model_observation", {}) if isinstance(row.get("model_observation"), dict) else {}
        vlm_json = model_obs.get("vlm_json", {}) if isinstance(model_obs.get("vlm_json"), dict) else {}
        has_model_signal = bool(vlm_json or model_obs.get("vlm_raw"))
        if not has_model_signal:
            continue
        usable += 1
        fields = [
            model_obs.get("vlm_raw"), vlm_json.get("visual_summary"),
            vlm_json.get("reason"), vlm_json.get("dominant_information_channel"),
            " ".join(str(item) for item in (vlm_json.get("query_relevant_requirements") or [])),
        ]
        text = " ".join(str(x or "") for x in fields).lower()
        channel = str(vlm_json.get("dominant_information_channel") or "").lower()
        if channel == "speech":
            counts["speech_or_audio"] += 1
        elif channel in {"subtitle_text", "screen_text"}:
            counts["screen_text_or_subtitle"] += 1
        elif channel == "visual_action":
            counts["visual_action"] += 1
        elif channel == "visual_detail":
            counts["visual_object_detail"] += 1
        if any(k in text for k in ("say", "speech", "voice", "talk", "audio", "transcript", "speak", "sound")):
            counts["speech_or_audio"] += 1
        if any(k in text for k in ("subtitle", "caption", "ocr", "screen text")):
            counts["screen_text_or_subtitle"] += 1
        if any(k in text for k in ("walk", "run", "move", "pick", "put", "enter", "leave", "action")):
            counts["visual_action"] += 1
        if any(k in text for k in ("object", "color", "wear", "product", "number", "merchandise")):
            counts["visual_object_detail"] += 1
    denom = max(1, usable)
    return {
        "schema_version": 1,
        "sampled_observations": usable,
        "source_available": True,
        "source_basename": os.path.basename(path),
        "planned_video_records": len(rows),
        "keyword_counts": dict(counts),
        "density": {key: round(value / denom, 3) for key, value in counts.items()},
        "prompt_safety": {
            "uses_human_type_labels": False,
            "accepted_fields": [
                "vlm_raw", "visual_summary", "reason",
                "dominant_information_channel", "query_relevant_requirements",
            ],
        },
    }








def build_observed_distribution_profile(
    workspace_dir: str,
    distribution_spec: Dict[str, Any] | None = None,
    output_path: str | None = None,
    include_execution_artifacts: bool = True,
) -> Dict[str, Any]:
    spec = distribution_spec or {}
    budgets = spec.get("budgets", {}) if isinstance(spec.get("budgets"), dict) else {}
    video_id_budget = int(budgets.get("profile_video_ids", 0) or 0)
    video_budget = int(budgets.get("profile_videos", 24) or 24)
    semantic_budget = int(budgets.get("profile_semantic_observations", 200) or 200)

    video_split_path = (spec.get("video_splits", {}) or {}).get("train")
    question_split_path = (spec.get("splits", {}) or {}).get("train")
    split_path = video_split_path or question_split_path
    if split_path:
        video_ids = _video_ids_from_split(split_path, limit=video_id_budget)
    else:
        video_ids = _video_ids_from_workspace(workspace_dir, limit=video_id_budget)
    queries_by_video = (
        _queries_by_video_from_split(question_split_path)
        if question_split_path else {}
    )
    channel_counter = Counter()
    tool_counts = Counter()
    interval_lengths: List[int] = []
    structure_stats = {
        "sampled_segments": 0,
        "keyword_counts": {},
        "density": {},
        "skipped_reason": "initial_profile_excludes_execution_artifacts",
    }
    video_signal_stats = _video_signal_stats(
        workspace_dir,
        video_ids,
        budget=video_budget,
    )
    semantic_artifacts = spec.get("artifacts", {}) if isinstance(spec.get("artifacts"), dict) else {}
    semantic_observation_path = semantic_artifacts.get("semantic_channel_observations", "")
    semantic_observation_stats = _semantic_observation_stats(
        semantic_observation_path,
        semantic_budget,
    )
    semantic_rows = _load_semantic_observation_rows(
        semantic_observation_path, semantic_budget,
    )
    distribution_records = _distribution_records(
        video_ids, queries_by_video, video_signal_stats, semantic_rows,
    )
    question_count = sum(len(items) for items in queries_by_video.values())
    semantic_sampling_plan = _semantic_sampling_plan(video_signal_stats)

    total_videos = max(1, len(video_ids))
    profile = {
        "timestamp": int(time.time()),
        "profile_source": "five_frame_query_aware_train_distribution",
        "leakage_policy": (
            "No human type label, description, or expected strategy is used. "
            "For each evolution video, the profile uses five uniformly sampled "
            "overview frames, non-answer video metadata, and associated questions "
            "with answer options. It never reads answer labels, evidence intervals, "
            "task categories, trajectories, or structure databases."
        ),
        "distribution_spec": prompt_safe_spec(spec),
        "sampling": {
            "video_ids_seen": len(video_ids),
            "video_split_source": (
                "video_splits.train"
                if video_split_path else
                ("splits.train_fallback" if split_path else "workspace.raw_videos")
            ),
            "video_ids_profiled": min(len(video_ids), video_budget or len(video_ids)),
            "question_rows_profiled": question_count,
            "baseline_trajectories_profiled": 0,
            "structure_segments_profiled": 0,
            "raw_videos_profiled": video_signal_stats.get("sampled_videos", 0),
            "unique_videos_profiled": len(video_ids),
            "include_execution_artifacts": False,
            "ignored_include_execution_artifacts_arg": bool(include_execution_artifacts),
            "supervision_used": False,
            "query_context_used": True,
            "forbidden_inputs": [
                "answer", "gt_answer", "correct_answer", "time_reference",
                "clue_intervals", "question_type", "domain", "sub_category",
                "trajectories", "structure_text",
            ],
        },
        "distribution_records": distribution_records,
        "raw_video_signal_profile": video_signal_stats,
        "information_channel_schema": _stable_channel_schema(
            channel_counter,
            total_videos,
            interval_lengths,
            video_signal_stats,
            structure_stats,
            semantic_observation_stats,
        ),
        "observed_information_channels": {
            "source": "five_frame_query_aware_records",
            "dominant_channels": [],
            "note": (
                "Questions and answer options are retained as unsupervised design "
                "context; answers and evidence annotations are excluded."
            ),
        },
        "observed_artifact_channels": structure_stats,
        "semantic_channel_observation_profile": semantic_observation_stats,
        "semantic_channel_sampling_plan": semantic_sampling_plan,
        "information_channel_hypothesis": _channel_hypothesis(
            channel_counter, tool_counts, structure_stats, total_videos,
            video_stats=video_signal_stats,
            semantic_stats=semantic_observation_stats,
        ),
        "baseline_behavior_observed": {
            "skipped": True,
            "reason": "initial profile does not inspect execution trajectories",
            "sampled_tool_counts": {},
            "retrieval_heavy": False,
            "mentions_audio_or_speech": 0,
            "mentions_text_or_ocr": 0,
        },
        "diagnosis_guidance": {
            "must_not_use_human_type_label": True,
            "prompt_instruction": (
                "Initial baseline design may use the five-frame per-video records, "
                "their associated questions/options, and deep research. It must not "
                "use answers, evidence windows, trajectories, or human type labels."
            ),
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    return profile


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Build a five-frame, query-aware distribution profile"
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    spec = {}
    if args.manifest:
        try:
            from .distribution_spec import load_distribution_spec
        except ImportError:
            from distribution_spec import load_distribution_spec
        spec = load_distribution_spec(args.manifest)

    output = args.output
    if not output:
        output_dir = os.path.join(metavideoagent_output_root(), "profiles")
        output = os.path.join(
            output_dir, f"observed_distribution_profile_{int(time.time())}.json"
        )
    profile = build_observed_distribution_profile(
        args.workspace,
        distribution_spec=spec,
        output_path=output,
        include_execution_artifacts=False,
    )
    print(output)
    print(json.dumps({
        "dominant_channels": (
            profile.get("observed_information_channels", {}) or {}
        ).get("dominant_channels", []),
        "profiled_questions": (
            profile.get("sampling", {}) or {}
        ).get("question_rows_profiled", 0),
        "redacted_prompt_fields": (
            profile.get("distribution_spec", {}) or {}
        ).get("redacted_prompt_fields", []),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
