import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Lock

try:
    import utils
except ImportError:  # pragma: no cover - package import
    from . import utils

# Global lock to protect JSONL + ChromaDB concurrent writes
_jsonl_write_lock = Lock()


def _default_segment_api_config() -> dict:
    return {
        "input_modality": "clip_frames_with_transcript",
        "active_model": utils.VLM_MODEL,
        "model": utils.VLM_MODEL,
        "temperature": 0.0,
        "prompt_template": (
            "You are building a neutral searchable video evidence store.\n"
            "Analyze the clip from {start_sec}s to {end_sec}s using the frames "
            "and transcript. Do not infer beyond visible/audible evidence.\n\n"
            "Transcript:\n{transcript}\n\n"
            "Return concise JSON if possible with keys such as description, "
            "visible_entities, actions, screen_text, transcript_summary, "
            "uncertainties, and retrieval_keywords. If JSON is not possible, "
            "return dense natural-language evidence."
        ),
    }


def _ensure_structure_api_config(api_cfg: dict, struct_type: str) -> dict:
    """Ensure generated structure builders always have the fields needed to run."""
    merged = _default_segment_api_config()
    if isinstance(api_cfg, dict):
        merged.update(api_cfg)
    model = utils.VLM_MODEL
    merged["model"] = model
    merged.setdefault("active_model", model)
    if not merged.get("prompt_template"):
        merged["prompt_template"] = _default_segment_api_config()["prompt_template"]
        print(
            f"   ⚠️ [Structure] {struct_type} api_cfg missing prompt_template; "
            "using default segment_text template."
        )
    merged.setdefault("input_modality", "clip_frames_with_transcript")
    return merged


def _structure_build_window(video_length: float) -> tuple:
    """Return the runtime-requested structure build window.

    Normal full evaluations leave this unset and build the whole video. Smoke
    and probe runs can set STRUCTURE_BUILD_START_SEC/END_SEC to build only the
    evidence window in an isolated sandbox, avoiding accidental full-video VLM
    rebuilds.
    """
    try:
        total = max(0.0, float(video_length or 0.0))
    except Exception:
        total = 0.0
    start_raw = os.environ.get("STRUCTURE_BUILD_START_SEC", "").strip()
    end_raw = os.environ.get("STRUCTURE_BUILD_END_SEC", "").strip()
    if not start_raw and not end_raw:
        return 0.0, total
    try:
        start = float(start_raw) if start_raw else 0.0
    except Exception:
        start = 0.0
    try:
        end = float(end_raw) if end_raw else total
    except Exception:
        end = total
    start = max(0.0, min(start, total))
    end = max(start, min(end, total))
    if end <= start:
        end = min(total, start + 1.0)
    return start, end


def _merge_windows(windows: list, max_gap: float = 0.0) -> list:
    valid = []
    for item in windows or []:
        try:
            start, end = float(item[0]), float(item[1])
        except Exception:
            continue
        if end <= start:
            continue
        valid.append((start, end))
    valid.sort()
    merged = []
    for start, end in valid:
        if not merged or start > merged[-1][1] + max_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(float(s), float(e)) for s, e in merged]


def _structure_build_windows(video_length: float) -> list:
    """Return one or more runtime-requested structure build windows."""
    try:
        total = max(0.0, float(video_length or 0.0))
    except Exception:
        total = 0.0
    windows = []
    # Sandboxed probe workers carry their exact media scope in a ContextVar.
    # Unlike environment variables this is task-local, so independent probe
    # questions may build isolated structures concurrently without borrowing
    # another question's time window. Normal evaluation has no active scope
    # and may use the explicitly configured environment window.
    try:
        windows.extend(utils.get_active_media_windows() or [])
    except Exception:
        pass
    raw = os.environ.get("STRUCTURE_BUILD_WINDOWS_JSON", "").strip()
    if not windows and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        windows.append((item.get("start"), item.get("end")))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        windows.append((item[0], item[1]))
        except Exception:
            windows = []
    if not windows:
        windows = [_structure_build_window(video_length)]
    clipped = []
    for start, end in windows:
        try:
            s = max(0.0, min(float(start), total))
            e = max(s, min(float(end), total))
        except Exception:
            continue
        if e <= s:
            e = min(total, s + 1.0)
        if e > s:
            clipped.append((s, e))
    return _merge_windows(clipped) or [(0.0, total)]


def _iter_clip_ranges(start: float, end: float, interval: float):
    try:
        step = max(0.1, float(interval or 10.0))
    except Exception:
        step = 10.0
    cur = float(start)
    idx_guard = 0
    while cur < float(end) and idx_guard < 100000:
        nxt = min(cur + step, float(end))
        if nxt <= cur:
            break
        yield cur, nxt
        cur = nxt
        idx_guard += 1


def _iter_window_clip_ranges(windows: list, interval: float):
    for window_start, window_end in windows:
        yield from _iter_clip_ranges(window_start, window_end, interval)

class VideoQAEnv:
    def __init__(self, workspace_dir, video_id, target_fps=1.0):
        self.workspace_dir = workspace_dir
        self.video_id = video_id
        self.target_fps = target_fps

        self.raw_video_path = os.path.join(workspace_dir, "raw_videos", f"{video_id}.mp4")
        self.frames_dir = os.path.join(workspace_dir, "extracted_frames", video_id)
        # ``allowed_time_windows`` is an execution boundary used by targeted
        # probes.  It is intentionally empty for normal smoke/full evaluation.
        # Generated modules may query it to keep ASR/OCR/VLM work inside the
        # media segment supplied by the evaluator.
        self.allowed_time_windows = None
        self.active_window_scope = "full_video"

        if not os.path.exists(self.raw_video_path):
            raise FileNotFoundError(f"Video file not found: {self.raw_video_path}")

        self.video_length_secs = 0.0
        self._initialize_video_metadata()

    def _initialize_video_metadata(self):
        import cv2
        cap = cv2.VideoCapture(self.raw_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_length_secs = frame_count / fps if fps > 0 else 0.0
        cap.release()

    def set_active_time_windows(self, windows, scope: str = "probe_time_reference") -> list:
        """Install normalized, exact media windows for this environment.

        The sandbox uses this for oracle-style probes.  The method is generic:
        it carries no question or answer information and can be used by any
        runtime capability that needs bounded raw-media access.
        """
        normalized = []
        total = max(0.0, float(self.video_length_secs or 0.0))
        for item in windows or []:
            try:
                start, end = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            start = max(0.0, min(start, total))
            end = max(start, min(end, total))
            if end > start:
                normalized.append((round(start, 3), round(end, 3)))
        normalized.sort()
        merged = []
        for start, end in normalized:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        self.allowed_time_windows = [(float(start), float(end)) for start, end in merged] or None
        self.active_window_scope = scope if self.allowed_time_windows else "full_video"
        return list(self.allowed_time_windows or [])

    def get_active_time_windows(self) -> list:
        """Return runtime-visible media windows, or the full video by default."""
        if self.allowed_time_windows:
            return [tuple(item) for item in self.allowed_time_windows]
        total = max(0.0, float(self.video_length_secs or 0.0))
        return [(0.0, total)] if total > 0 else []


    def get_frame_at_timestamp(self, timestamp: float) -> str:
        """Extract and cache the frame nearest to a timestamp."""
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            return ""
        allowed_windows = getattr(self, "allowed_time_windows", None)
        if allowed_windows:
            in_window = False
            for start, end in allowed_windows:
                try:
                    if float(start) <= timestamp <= float(end):
                        in_window = True
                        break
                except Exception:
                    continue
            if not in_window:
                return ""

        idx = int(round(timestamp * self.target_fps))
        filename = f"frame_{idx:06d}.jpg"
        frame_path = os.path.join(self.frames_dir, filename)

        if not os.path.exists(frame_path):
            os.makedirs(self.frames_dir, exist_ok=True)
            import cv2
            cap = cv2.VideoCapture(self.raw_video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_path, frame)
            cap.release()

        return frame_path if os.path.exists(frame_path) else ""


def _transcribe_clip_audio(clip_index, start_sec, end_sec, env,
                           cost_observer=None, cost_observer_metadata=None):
    """Run ASR as an independent, bounded perception task."""
    transcript, asr_status, asr_error = "", "asset_unavailable", ""
    safe_video_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in env.video_id)
    temp_wav = os.path.join(env.workspace_dir, f"temp_asr_{safe_video_id}_{clip_index}_{threading.get_ident()}.wav")
    try:
        if utils.extract_audio_segment(env.raw_video_path, start_sec, end_sec, temp_wav):
            result = utils.call_asr(
                temp_wav, profile_id=utils.DEFAULT_PROFILE_IDS["asr"],
                cost_observer=cost_observer,
                cost_observer_metadata=cost_observer_metadata,
            )
            if str(result or "").strip().lower().startswith("error calling asr api"):
                asr_status, asr_error = "api_error", str(result)
            else:
                transcript = str(result or "").strip()
                asr_status = "ok" if transcript else "no_speech"
    except Exception as exc:
        asr_status, asr_error = "api_error", str(exc)
        print(f"      ⚠️ [Clip {clip_index}] ASR call failed: {exc}")
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
    return transcript, asr_status, asr_error


def _process_single_clip_full(clip_index, start_sec, end_sec, physics, api_cfg, env, struct_db,
                              asr_executor=None):
    """Extract frames, submit ASR and VLM independently, then merge evidence."""
    try:
        frames_per_clip = physics.get("frames_per_clip_parallel", 10)
        clip_frames = []
        step = (end_sec - start_sec) / frames_per_clip
        for i in range(frames_per_clip):
            f_path = env.get_frame_at_timestamp(start_sec + i * step)
            if f_path:
                clip_frames.append(f_path)
        if not clip_frames:
            print(f"      ⚠️ [Clip {clip_index}] No valid frame, skip")
            return (clip_index, start_sec, end_sec, False)

        needs_asr = "transcript" in api_cfg.get("input_modality", "")
        structure_cost_observer = getattr(struct_db, "record_structure_api_call", None)
        structure_cost_metadata = {
            "phase": "structure_build",
            "structure_window": [float(start_sec), float(end_sec)],
        }
        transcript, asr_status, asr_error = "", "not_requested", ""
        asr_future = None
        if needs_asr:
            if asr_executor is None:
                transcript, asr_status, asr_error = _transcribe_clip_audio(
                    clip_index, start_sec, end_sec, env,
                    cost_observer=structure_cost_observer,
                    cost_observer_metadata=structure_cost_metadata,
                )
            else:
                asr_future = asr_executor.submit(
                    _transcribe_clip_audio, clip_index, start_sec, end_sec, env,
                    structure_cost_observer, structure_cost_metadata,
                )

        # The visual sensor must start as soon as frames exist.  Transcript is
        # merged after both sensor calls; it is never a VLM start dependency.
        transcript_placeholder = (
            "[ASR is being collected independently for this exact range. "
            "Analyze only the supplied frames; do not infer unheard speech.]"
            if asr_future else transcript
        )
        prompt = api_cfg["prompt_template"].format(
            start_sec=start_sec, end_sec=end_sec, transcript=transcript_placeholder,
        )
        res = utils.call_vlm(
            prompt, clip_frames, model_name=api_cfg["model"],
            profile_id=utils.DEFAULT_PROFILE_IDS["vlm"],
            cost_observer=structure_cost_observer,
            cost_observer_metadata=structure_cost_metadata,
        )
        response_text = str(res or "").strip()
        vlm_status = (
            "api_error" if response_text.startswith("Error calling VLM API")
            else "no_content" if not response_text or response_text == "VLM returned empty response"
            else "ok"
        )
        if asr_future is not None:
            try:
                transcript, asr_status, asr_error = asr_future.result()
            except Exception as exc:
                asr_status, asr_error = "api_error", str(exc)
                print(f"      ⚠️ [Clip {clip_index}] ASR worker failed: {exc}")
        frame_assets = [
            {"kind": "frame", "timestamp_sec": round(start_sec + i * step, 4), "path": path, "status": "available"}
            for i, path in enumerate(clip_frames)
        ]
        if vlm_status != "ok":
            # A provider safety filter is not evidence about the video.  In
            # an explicitly enabled formal-recovery pass, retain the bounded
            # media provenance and any independently collected ASR result,
            # but never manufacture a visual description or treat this as a
            # successful VLM observation.  The thinking module can then use
            # the explicit limitation as context rather than silently losing
            # the time range altogether.
            safety_blocked = "datainspectionfailed" in response_text.lower()
            persist_blocked = os.environ.get(
                "METAVIDEOAGENT_PERSIST_VLM_SAFETY_BLOCKED", ""
            ).lower() in ("1", "true", "yes")
            if not (safety_blocked and persist_blocked):
                print(f"      ⚠️ [Clip {clip_index}] VLM returned {vlm_status}; evidence record not persisted")
                return (clip_index, start_sec, end_sec, False)
            with _jsonl_write_lock:
                struct_db.addStructure(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    multimodal_narration=(
                        "Visual perception unavailable for this bounded clip: "
                        "the VLM provider safety filter blocked the response. "
                        "No visual claim is recorded; do not infer visual details "
                        "from this placeholder."
                    ),
                    asr_text=transcript,
                    asr_status=asr_status,
                    asr_error=asr_error,
                    media_assets=frame_assets,
                    builder_request={
                        "artifact_type": "metavideoagent_structure_build_request", "schema_version": 1,
                        "window": {"start_sec": start_sec, "end_sec": end_sec},
                        "requested_modalities": ["frames"] + (["audio_asr"] if needs_asr else []),
                        "vlm_profile_id": utils.DEFAULT_PROFILE_IDS["vlm"],
                        "asr_profile_id": utils.DEFAULT_PROFILE_IDS["asr"] if needs_asr else "",
                        "vlm_status": "provider_safety_blocked",
                        "vlm_error": response_text,
                        "sensor_execution": "asr_vlm_independent_then_merged",
                    },
                    source="generated_segment_builder_provider_safety_fallback",
                )
            print(f"      ⚠️ [Clip {clip_index}] VLM provider safety block persisted without visual inference")
            return (clip_index, start_sec, end_sec, True)
        with _jsonl_write_lock:
            struct_db.addStructure(
                start_sec=start_sec, end_sec=end_sec, multimodal_narration=res,
                asr_text=transcript, asr_status=asr_status, asr_error=asr_error,
                media_assets=frame_assets,
                builder_request={
                    "artifact_type": "metavideoagent_structure_build_request", "schema_version": 1,
                    "window": {"start_sec": start_sec, "end_sec": end_sec},
                    "requested_modalities": ["frames"] + (["audio_asr"] if needs_asr else []),
                    "vlm_profile_id": utils.DEFAULT_PROFILE_IDS["vlm"],
                    "asr_profile_id": utils.DEFAULT_PROFILE_IDS["asr"] if needs_asr else "",
                    "vlm_status": vlm_status,
                    "sensor_execution": "asr_vlm_independent_then_merged",
                },
                source="generated_segment_builder",
            )
        return (clip_index, start_sec, end_sec, True)
    except Exception as exc:
        print(f"      ⚠️ [Clip {clip_index}] Processing failed: {exc}")
        return (clip_index, start_sec, end_sec, False)


def _build_generated_segment_structure(env, struct_db):
    """Build generated segment records with bounded parallel media calls.

    The generated structuring module defines the evidence schema, prompt, and
    structure type. This runtime function handles slicing, frame extraction,
    ASR and VLM calls, and scheduling.
    """
    struct_type = getattr(struct_db, "struct_type", struct_db.__class__.__name__)
    physics = struct_db.config["physics"]
    api_cfg = _ensure_structure_api_config(
        struct_db._get_api_params(struct_type), struct_type
    )

    interval = (
        physics.get(f"sample_interval_{struct_type}")
        or physics.get("sample_interval_segment")
        or physics.get("sample_interval")
        or 10.0
    )
    # A clip worker dispatches VLM and ASR independently, but waits to merge
    # both bounded observations before persisting the one coherent record.
    # Therefore the ordinary structure-worker value alone cannot be used as
    # the VLM in-flight pool when ASR is intentionally rate-limited: after
    # the initial clips, all ordinary workers would otherwise wait on queued
    # ASR calls and starve the VLM start-rate gate.  Formal full-eval runs can
    # supply a larger, explicitly audited pool; it changes scheduling only,
    # not evidence content or the ASR concurrency/rate limits.
    max_workers = physics.get("parallel_workers", 3)
    try:
        configured_inflight_workers = int(
            os.environ.get("METAVIDEOAGENT_STRUCTURE_INFLIGHT_WORKERS", "0") or 0
        )
    except ValueError:
        configured_inflight_workers = 0
    if configured_inflight_workers > 0:
        max_workers = max(int(max_workers or 1), configured_inflight_workers)
    video_length = env.video_length_secs
    build_windows = _structure_build_windows(video_length)
    resume_build = os.environ.get("VIDEO_STRUCT_RESUME_BUILD", "").lower() in ("1", "true", "yes")
    existing_ranges = set()
    if resume_build and os.path.exists(struct_db.db_path):
        try:
            with open(struct_db.db_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    existing_ranges.add((round(float(r.get("start_sec", -1)), 3),
                                         round(float(r.get("end_sec", -1)), 3)))
            if existing_ranges:
                print(f"   [Generated Build] Resuming after {len(existing_ranges)} completed clips.")
        except Exception as e:
            print(f"   [Generated Build] Could not read the existing JSONL; rebuilding normally: {e}")
            existing_ranges = set()

    # Step 1: Generate clip interval list
    print("[Generated Build] Preparing clip intervals...")
    clip_ranges = []
    clip_idx = 0
    for start_sec, end_sec in _iter_window_clip_ranges(build_windows, interval):
        if end_sec <= start_sec:
            break
        key = (round(float(start_sec), 3), round(float(end_sec), 3))
        if key in existing_ranges:
            continue
        clip_idx += 1
        clip_ranges.append((clip_idx, start_sec, end_sec))

    total_clips = len(clip_ranges)
    if len(build_windows) != 1 or build_windows[0] != (0.0, video_length):
        preview = ", ".join(f"{s:.1f}-{e:.1f}s" for s, e in build_windows[:6])
        suffix = " ..." if len(build_windows) > 6 else ""
        print(f"   [Generated Build] windows: {preview}{suffix} of {video_length:.1f}s")
    print(f"   {total_clips} clips to process at a {interval}s interval.")
    if not clip_ranges:
        print("[Generated Build] No valid clips; structure construction stopped.")
        return

    # Step 2: Continuous in-transit parallel processing + real-time writes.
    #
    # Do not submit one fixed batch and wait for its slowest provider call
    # before admitting the next batch.  VLM latency is variable, so that
    # pattern leaves completed workers idle and lowers real QPS.  The sliding
    # window below maintains up to ``max_workers`` clip tasks in flight: as
    # soon as *any* task completes, exactly one next clip is admitted.  Formal
    # start-rate runs use a sufficiently large VLM in-flight pool; the VLM
    # start gate is then the QPS authority rather than a low worker count.
    print(
        "[Generated Build] Start continuous in-flight parallel processing"
        f"(inflight_workers={max_workers}, no batch barrier; "
        f"asr_workers={os.environ.get('METAVIDEOAGENT_ASR_BUILD_WORKERS', '16')})..."
    )
    success_count = 0
    fail_count = 0
    max_workers = max(1, int(max_workers or 1))
    pending = iter(clip_ranges)

    def _submit_next(executor, futures, asr_executor) -> bool:
        try:
            idx, start_sec, end_sec = next(pending)
        except StopIteration:
            return False
        future = executor.submit(
            _process_single_clip_full,
            idx, start_sec, end_sec, physics, api_cfg, env, struct_db, asr_executor,
        )
        futures[future] = (idx, start_sec, end_sec)
        return True

    # ASR has a separate bounded queue, so a slow/rate-limited audio request
    # cannot consume a VLM worker or prevent the next VLM start window.
    asr_workers = max(1, min(16, int(os.environ.get("METAVIDEOAGENT_ASR_BUILD_WORKERS", "16") or 16)))
    with ThreadPoolExecutor(max_workers=asr_workers) as asr_executor, \
            ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for _ in range(min(max_workers, total_clips)):
            _submit_next(executor, futures, asr_executor)

        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                fallback_idx, fallback_start, fallback_end = futures.pop(future)
                try:
                    clip_idx_batch, start_sec_batch, end_sec_batch, success = future.result()
                except Exception as exc:
                    clip_idx_batch, start_sec_batch, end_sec_batch, success = (
                        fallback_idx, fallback_start, fallback_end, False
                    )
                    print(f"   ⚠️ [Clip {clip_idx_batch}/{total_clips}] worker exception: {exc}")
                if success:
                    success_count += 1
                    print(f"   ✅ [Clip {clip_idx_batch}/{total_clips}] [{start_sec_batch:.1f}s - {end_sec_batch:.1f}s] completed and persisted")
                else:
                    fail_count += 1
                    print(f"   ⚠️ [Clip {clip_idx_batch}/{total_clips}] [{start_sec_batch:.1f}s - {end_sec_batch:.1f}s] failed, skipped")
                # Refill immediately after every completed task.  There is no
                # batch delay and no wait for unrelated slow requests.
                _submit_next(executor, futures, asr_executor)

    print(f"[Generated Build] Completed: {success_count}/{total_clips} succeeded, {fail_count} failed.")
    if success_count <= 0:
        raise RuntimeError(
            "structure_build_no_usable_records: all segment builds failed; "
            "do not continue with an empty/error-only structure artifact"
        )


def build_video_structure(env, struct_db):
    """Build the active video structure.

    A generated module may implement ``buildStructure(env)`` to own the build;
    otherwise the runtime uses the generic segment builder above.
    """
    struct_type = struct_db.struct_type
    begin_cost_capture = getattr(struct_db, "begin_structure_cost_capture", None)
    finish_cost_capture = getattr(struct_db, "finish_structure_cost_capture", None)
    if callable(begin_cost_capture):
        begin_cost_capture()

    def _finish_cost_capture():
        if callable(finish_cost_capture):
            finish_cost_capture()
    api_cfg = _ensure_structure_api_config(
        struct_db._get_api_params(struct_type), struct_type
    )
    modality = api_cfg.get("input_modality", "clip_frames")

    # Check whether records already exist to avoid repeated database creation
    resume_build = os.environ.get("VIDEO_STRUCT_RESUME_BUILD", "").lower() in ("1", "true", "yes")
    if os.path.exists(struct_db.db_path) and os.path.getsize(struct_db.db_path) > 0 and not resume_build:
        # Rebuild an artifact containing only failed provider responses.
        try:
            with open(struct_db.db_path, "r", encoding="utf-8") as _f:
                _lines = [line.strip() for line in _f if line.strip()]
            _all_error = (
                all("Error calling VLM API" in line or '"Error' in line for line in _lines)
                if _lines else False
            )
            if not _all_error:
                print(f"[Structure] Reusing existing {struct_type} records.")
                _finish_cost_capture()
                return
            print(f"[Structure] Existing {struct_type} records contain only errors; rebuilding.")
            backup_path = f"{struct_db.db_path}.error_backup.{int(time.time())}"
            os.replace(struct_db.db_path, backup_path)
            print(f"   Preserved the failed JSONL at: {backup_path}")
        except Exception:
            print(f"[Structure] Reusing existing {struct_type} records.")
            _finish_cost_capture()
            return

    print(f"[Structure] Building records | type={struct_type} | modality={modality}")

    if hasattr(struct_db, "buildStructure") and callable(getattr(struct_db, "buildStructure")):
        struct_db.buildStructure(env)
        if not os.path.isfile(struct_db.db_path) or os.path.getsize(struct_db.db_path) <= 0:
            raise RuntimeError(
                "custom_structure_build_no_persisted_records: custom buildStructure "
                "must persist time-bounded canonical records or media references"
            )
        print("[Structure] Custom build completed.")
        _finish_cost_capture()
        return

    _build_generated_segment_structure(env, struct_db)
    print("[Structure] Generated segment build completed.")
    _finish_cost_capture()
    return

if __name__ == "__main__":
    raise SystemExit(
        "Direct execution of execution/action_runtime/run_video_qa.py is "
        "disabled. Import VideoQAEnv/build_video_structure through the MetaVideoAgent "
        "runners instead."
    )
