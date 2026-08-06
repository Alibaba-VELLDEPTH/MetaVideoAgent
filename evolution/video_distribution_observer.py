"""Five-frame, query-aware observer for MetaVideoAgent initial profiling.

For every evolution video, this module uniformly samples five overview frames
and summarizes them together with the video's associated questions and answer
options.  It never reads answer labels, evidence annotations, task categories,
trajectories, or structure database text.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List

try:
    import cv2
except Exception:  # pragma: no cover - reported at runtime
    cv2 = None

try:
    from distribution_spec import load_distribution_spec
    from question_set_loader import dataset_files_from_path
    from runtime_paths import metavideoagent_run_dir
except ImportError:  # pragma: no cover
    from .distribution_spec import load_distribution_spec
    from .question_set_loader import dataset_files_from_path
    from .runtime_paths import metavideoagent_run_dir


CHANNELS = [
    "speech",
    "subtitle_text",
    "screen_text",
    "visual_action",
    "visual_detail",
    "mixed",
    "unknown",
]


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


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def train_video_queries_from_manifest(manifest_path: str) -> tuple[List[str], Dict[str, List[dict]]]:
    spec = load_distribution_spec(manifest_path)
    train_path = (
        (spec.get("video_splits", {}) or {}).get("train")
        or (spec.get("splits", {}) or {}).get("train")
    )
    if not train_path:
        return []
    ids = []
    seen = set()
    queries: Dict[str, List[dict]] = defaultdict(list)
    for split_file in dataset_files_from_path(train_path):
        for row in _iter_jsonl(split_file):
            video_id = (
                row.get("video_id")
                or row.get("video_uid")
                or row.get("video")
                or row.get("video_name")
                or ""
            )
            video_id = str(video_id).strip()
            if not video_id or video_id in seen:
                if not video_id:
                    continue
            else:
                seen.add(video_id)
                ids.append(video_id)
            question = str(row.get("question") or row.get("query") or row.get("prompt") or "").strip()
            choices = row.get("choices")
            if choices in (None, "", [], {}):
                choices = row.get("options")
            if question:
                queries[video_id].append({
                    "question": question,
                    "choices": choices if choices not in (None, "") else [],
                })
    return ids, dict(queries)


def _video_meta(path: str) -> Dict[str, Any]:
    if cv2 is None:
        return {"readable": False, "error": "cv2 unavailable"}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"readable": False, "error": "open_failed"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "readable": duration > 0 and frame_count > 0,
        "duration_seconds": round(duration, 3),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


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


def _frame_quality(frame) -> Dict[str, Any]:
    if cv2 is None or frame is None:
        return {"valid": False, "brightness": None, "contrast": None, "low_information": True}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    return {
        "valid": True,
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "low_information": brightness < 8.0 or contrast < 4.0,
    }


def extract_overview_frames(video_path: str, video_id: str,
                            window: Dict[str, float], assets_dir: str) -> Dict[str, Any]:
    os.makedirs(assets_dir, exist_ok=True)
    if cv2 is None:
        return {"ok": False, "error": "cv2 unavailable", "frames": []}
    meta = _video_meta(video_path)
    duration = float(meta.get("duration_seconds") or 0.0)
    if not meta.get("readable"):
        return {"ok": False, "error": meta.get("error", "unreadable_video"), "frames": []}

    start = max(0.0, float(window["start_sec"]))
    end = min(duration, float(window["end_sec"]))
    if end <= start:
        return {"ok": False, "error": "invalid_video_duration", "frames": []}
    cap = cv2.VideoCapture(video_path)
    frames = []
    qualities = []
    for frame_idx in range(5):
        ts = start + ((frame_idx + 0.5) / 5) * (end - start)
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        quality = _frame_quality(frame)
        qualities.append(quality)
        out_path = os.path.join(
            assets_dir, f"{video_id}_overview_f{frame_idx:02d}_{int(ts):06d}s.jpg",
        )
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        frames.append({"time_sec": round(ts, 3), "path": out_path, "quality": quality})
    cap.release()
    if len(frames) != 5:
        return {"ok": False, "error": "incomplete_five_frame_overview", "frames": frames}
    low_ratio = sum(1 for q in qualities if q.get("low_information")) / 5
    return {
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "frames": frames,
        "low_information_frame_ratio": round(low_ratio, 3),
        "ok": True,
    }




def _json_from_text(text: str) -> Dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(value[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _observer_prompt(queries: List[dict]) -> str:
    return """You are profiling a training-video distribution for a video QA agent.
Inspect exactly five uniformly sampled overview frames together with the associated
questions and answer options below. Do not infer or reveal correct answers. Do not
infer benchmark categories, evidence intervals, or question labels.

Return exactly one JSON object with:
{
  "visual_summary": "...",
  "subtitle_presence": "none|low|medium|high",
  "screen_text_presence": "none|low|medium|high",
  "visual_motion": "low|medium|high",
  "scene_change": "low|medium|high",
  "dominant_information_channel": "speech|subtitle_text|screen_text|visual_action|visual_detail|mixed|unknown",
  "query_relevant_requirements": ["distribution-level modality, localization, or evidence-granularity requirement"],
  "reason": "brief evidence from the five overview frames"
}

Associated questions and answer options (no labels):
""" + json.dumps(queries or [], ensure_ascii=False)[:12000]


def observe_overview_with_model(row: Dict[str, Any], runtime_dir: str,
                                vlm_model: str = "") -> Dict[str, Any]:
    """Observe one five-frame overview through the auditable VLM adapter."""
    del vlm_model  # formal profiles, not caller model strings, select transports
    runtime_dir = os.path.abspath(runtime_dir)
    if runtime_dir and runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)
    import runtime_evidence  # type: ignore

    asset = row.get("asset", {}) or {}
    planned = row.get("planned_window", {}) or {}
    start = float(planned.get("start_sec", 0.0) or 0.0)
    end = float(planned.get("end_sec", start) or start)

    class _ObserverEnv:
        raw_video_path = str(row.get("video_path") or "")
        video_length_secs = float((row.get("video_meta") or {}).get("duration_seconds") or end)

        def get_active_time_windows(self):
            return [(start, end)] if end > start else []

        def get_frame_at_timestamp(self, timestamp):
            frames = asset.get("frames", []) if isinstance(asset, dict) else []
            candidates = [item for item in frames if isinstance(item, dict) and item.get("path")]
            if not candidates:
                return ""
            nearest = min(candidates, key=lambda item: abs(float(item.get("time_sec", start)) - float(timestamp)))
            return str(nearest.get("path") or "")

    env = _ObserverEnv()
    token = runtime_evidence.begin_evidence_run()
    try:
        visual = runtime_evidence.inspect_time_ranges(
            env, [[start, end]], capability="vlm", max_windows=1,
            frames_per_window=5,
            flat_total_samples=5,
            batch_vlm=True,
            prompt=_observer_prompt(row.get("associated_queries") or []),
        )
        frame_texts = [
            str(item.get("text") or "")
            for item in (visual.get("frames") or []) if isinstance(item, dict)
        ]
        # The five overview frames are sent in one ordered VLM batch. Parse the
        # response as one per-video distribution record.
        parsed_frames = [_json_from_text(text) for text in frame_texts]
        parsed = next((item for item in parsed_frames if item), {})
        vlm_text = next((text for text in frame_texts if text.strip()), "")
        events = runtime_evidence.get_evidence_events()
    finally:
        runtime_evidence.reset_evidence_run(token)
    return {
        "vlm_raw": vlm_text,
        "vlm_json": parsed,
        "vlm_frame_json": parsed_frames,
        "dominant_information_channel": parsed.get("dominant_information_channel", "unknown"),
        "capability_events": events,
    }


def _level_to_score(value: str) -> float:
    value = str(value or "").lower()
    return {"high": 1.0, "medium": 0.6, "low": 0.25, "none": 0.0}.get(value, 0.0)


def aggregate_observations(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    channel_counts = Counter()
    scalar = defaultdict(float)
    usable = 0
    for row in rows:
        obs = row.get("model_observation", {}) or {}
        parsed = obs.get("vlm_json", {}) if isinstance(obs.get("vlm_json"), dict) else {}
        channel = (
            parsed.get("dominant_information_channel")
            or obs.get("dominant_information_channel")
            or "unknown"
        )
        if channel not in CHANNELS:
            channel = "mixed"
        channel_counts[channel] += 1
        if parsed:
            usable += 1
            scalar["subtitle_text"] += _level_to_score(parsed.get("subtitle_presence"))
            scalar["screen_text"] += _level_to_score(parsed.get("screen_text_presence"))
            scalar["visual_action"] += _level_to_score(parsed.get("visual_motion"))
            scalar["visual_detail"] += max(
                _level_to_score(parsed.get("screen_text_presence")),
                _level_to_score(parsed.get("visual_motion")) * 0.5,
            )
    denom = max(1, len(rows))
    channel_mix = {key: round(channel_counts.get(key, 0) / denom, 3) for key in CHANNELS}
    semantic_density = {
        key: round(value / max(1, usable), 3)
        for key, value in scalar.items()
    }
    primary = max(channel_mix.items(), key=lambda item: item[1])[0] if rows else "unknown"
    top = sorted(channel_mix.values(), reverse=True)
    confidence = round(max(0.0, top[0] - (top[1] if len(top) > 1 else 0.0)), 3) if top else 0.0
    if primary == "unknown":
        confidence = 0.0
    return {
        "sampled_video_records": len(rows),
        "usable_model_observations": usable,
        "channel_mix": channel_mix,
        "semantic_density": semantic_density,
        "dominant_information_channel": primary,
        "channel_confidence": confidence,
        "distribution_homogeneity": (
            "unknown" if primary == "unknown" else
            "high" if confidence >= 0.35 else
            "medium" if confidence >= 0.15 else "low"
        ),
    }


def run_observer(*,
                 workspace: str,
                 distribution_manifest: str,
                 output_dir: str,
                 mode: str = "plan",
                 video_budget: int = 0,
                 runtime_dir: str = "",
                 vlm_model: str = "",
                 observe_workers: int = 1) -> Dict[str, Any]:
    if mode not in {"plan", "observe"}:
        raise ValueError("mode must be plan or observe")
    workspace = os.path.abspath(workspace)
    output_dir = os.path.abspath(output_dir)
    assets_dir = os.path.join(output_dir, "overview_frames")
    os.makedirs(assets_dir, exist_ok=True)
    video_ids, queries_by_video = train_video_queries_from_manifest(distribution_manifest)
    if video_budget and len(video_ids) > video_budget:
        video_ids = video_ids[:video_budget]
    rows = []
    failures = []
    for video_id in video_ids:
        video_path = os.path.join(workspace, "raw_videos", f"{video_id}.mp4")
        if not os.path.exists(video_path):
            failures.append({"video_id": video_id, "error": "missing_video", "path": video_path})
            continue
        meta = _video_meta(video_path)
        if not meta.get("readable"):
            failures.append({"video_id": video_id, "error": meta.get("error", "unreadable"), "path": video_path})
            continue
        has_audio = _has_audio_stream(video_path)
        duration = float(meta.get("duration_seconds") or 0.0)
        # The paper profile is one record per video with five frames sampled
        # uniformly across the complete duration.
        windows = [{"window_index": 0, "start_sec": 0.0, "end_sec": duration}]
        for window in windows:
            overview_dir = os.path.join(assets_dir, video_id)
            asset = extract_overview_frames(video_path, video_id, window, overview_dir)
            row = {
                "schema_version": 1,
                "source": "five_frame_query_aware_overview",
                "video_id": video_id,
                "video_path": video_path,
                "video_meta": meta,
                "has_audio_stream": has_audio,
                "associated_queries": queries_by_video.get(video_id, []),
                "planned_window": window,
                "asset": asset,
                "model_observation": {},
            }
            rows.append(row)
    if mode == "observe":
        workers = max(1, int(observe_workers or 1))
        indexed_rows = [
            (idx, row) for idx, row in enumerate(rows)
            if (row.get("asset", {}) or {}).get("ok")
        ]
        def _observe(item):
            idx, row = item
            try:
                observation = observe_overview_with_model(
                    row,
                    runtime_dir=runtime_dir,
                    vlm_model=vlm_model,
                )
            except Exception as exc:
                observation = {
                    "vlm_raw": f"VLM_ERROR: {exc}",
                    "vlm_json": {},
                    "dominant_information_channel": "unknown",
                }
            return idx, observation
        if workers == 1:
            for item in indexed_rows:
                idx, observation = _observe(item)
                rows[idx]["model_observation"] = observation
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_observe, item) for item in indexed_rows]
                for future in as_completed(futures):
                    idx, observation = future.result()
                    rows[idx]["model_observation"] = observation
    summary = aggregate_observations(rows)
    low_info = [
        r for r in rows
        if (r.get("asset", {}) or {}).get("low_information_frame_ratio", 1.0) >= 0.75
    ]
    profile = {
        "schema_version": 1,
        "generated_at": int(time.time()),
        "mode": mode,
        "profile_source": "five_frame_query_aware_observer",
        "leakage_policy": (
            "Five uniformly sampled frames, non-answer video metadata, and associated "
            "questions/options are used. Answers, evidence intervals, trajectories, "
            "and human type labels are not read."
        ),
        "workspace": workspace,
        "distribution_manifest": os.path.abspath(distribution_manifest),
        "video_count": len(video_ids),
        "record_count": len(rows),
        "audio_stream_video_ratio": round(
            sum(1 for row in rows if row.get("has_audio_stream") is True) / max(1, len(rows)), 3
        ),
        "low_information_video_ratio": round(len(low_info) / max(1, len(rows)), 3),
        "observe_workers": max(1, int(observe_workers or 1)) if mode == "observe" else 0,
        "summary": summary,
        "failures": failures,
    }
    observations_path = os.path.join(output_dir, "video_overview_observations.jsonl")
    with open(observations_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    profile_path = os.path.join(output_dir, "video_distribution_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    summary_path = os.path.join(output_dir, "video_distribution_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(render_summary(profile))
    profile.update({
        "observations_path": observations_path,
        "profile_path": profile_path,
        "summary_path": summary_path,
        "assets_dir": assets_dir,
    })
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return profile


def render_summary(profile: Dict[str, Any]) -> str:
    summary = profile.get("summary", {}) or {}
    lines = [
        "# Raw Video Distribution Profile",
        "",
        profile.get("leakage_policy", ""),
        "",
        f"- mode: `{profile.get('mode')}`",
        f"- videos: {profile.get('video_count')}",
        f"- records: {profile.get('record_count')}",
        f"- dominant_information_channel: `{summary.get('dominant_information_channel', 'unknown')}`",
        f"- distribution_homogeneity: `{summary.get('distribution_homogeneity', 'unknown')}`",
        f"- channel_mix: `{json.dumps(summary.get('channel_mix', {}), ensure_ascii=False)}`",
        f"- semantic_density: `{json.dumps(summary.get('semantic_density', {}), ensure_ascii=False)}`",
        "",
        "Use this profile as the first-step video-distribution evidence for deep research and initial baseline design.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build five-frame query-aware distribution observations")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--distribution-manifest", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--mode", choices=("plan", "observe"), default="plan")
    parser.add_argument("--video-budget", type=int, default=0)
    parser.add_argument("--runtime-dir", default="")
    parser.add_argument("--vlm-model", default="")
    parser.add_argument("--observe-workers", type=int, default=1)
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(
            metavideoagent_run_dir(args.run_id or f"video_distribution_{int(time.time())}", args.output_root),
            "video_distribution_observer",
        )
    profile = run_observer(
        workspace=args.workspace,
        distribution_manifest=args.distribution_manifest,
        output_dir=output_dir,
        mode=args.mode,
        video_budget=args.video_budget,
        runtime_dir=args.runtime_dir,
        vlm_model=args.vlm_model,
        observe_workers=args.observe_workers,
    )
    print("VIDEO_DISTRIBUTION_OBSERVER_DONE")
    print(f"profile_path={profile.get('profile_path')}")
    print(f"observations_path={profile.get('observations_path')}")
    print(json.dumps(profile.get("summary", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
