import base64
import copy
import fcntl
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.request
from contextvars import ContextVar

import cv2
from openai import OpenAI

try:
    from capability_registry import (
        DEFAULT_PROFILE_IDS,
        CapabilityProfileError,
        provider_config,
        resolve_profile,
    )
except ImportError:  # pragma: no cover - package import
    from .capability_registry import (
        DEFAULT_PROFILE_IDS,
        CapabilityProfileError,
        provider_config,
        resolve_profile,
    )


# Per-thread/task probe media contract.  The evaluator sets this around one
# agent run, so concurrent probe questions cannot leak each other's windows.
_ACTIVE_MEDIA_WINDOWS = ContextVar("metavideoagent_active_media_windows", default=None)
_PROVIDER_AUDIT_LOCK = threading.Lock()


def _provider_audit(channel, event, started, error=""):
    path = str(os.environ.get("RUNTIME_PROVIDER_AUDIT_PATH") or "").strip()
    if not path:
        return
    row = {"ts": time.time(), "channel": channel, "event": event,
           "elapsed_sec": round(time.monotonic() - started, 3), "error": str(error)[:300]}
    try:
        with _PROVIDER_AUDIT_LOCK, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def set_active_media_windows(windows):
    """Set exact raw-media windows for the current runtime context.

    Returns a ContextVar token that the evaluator must reset after the agent
    finishes.  Normal evaluation leaves the value unset and remains full-video.
    """
    normalized = []
    for item in windows or []:
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end > start:
            normalized.append((start, end))
    return _ACTIVE_MEDIA_WINDOWS.set(tuple(normalized) or None)


def reset_active_media_windows(token) -> None:
    if token is not None:
        _ACTIVE_MEDIA_WINDOWS.reset(token)


def get_active_media_windows() -> list:
    return [tuple(item) for item in (_ACTIVE_MEDIA_WINDOWS.get() or ())]


def clamp_to_active_media_windows(start_sec: float, end_sec: float):
    """Return the largest overlap with the active probe media, if any.

    ``extract_audio_segment`` produces one contiguous file, so a request that
    spans multiple disjoint probe windows deterministically uses its largest
    visible overlap.  Callers needing all windows should iterate
    ``get_active_media_windows()`` and extract one segment per interval.
    """
    windows = get_active_media_windows()
    if not windows:
        return float(start_sec), float(end_sec)
    try:
        start, end = float(start_sec), float(end_sec)
    except (TypeError, ValueError):
        return None
    if end < start:
        start, end = end, start
    overlaps = [
        (max(start, active_start), min(end, active_end))
        for active_start, active_end in windows
        if min(end, active_end) > max(start, active_start)
    ]
    if not overlaps:
        return None
    return max(overlaps, key=lambda item: item[1] - item[0])


def _explicit_proxy_url(env: dict | None = None) -> str:
    """Return one validated local proxy URL for provider clients.

    The runtime used to let every httpx client independently inherit the
    process proxy environment.  That makes provider transport depend on
    mixed-case/overlapping proxy variables and hides the actual route from
    execution logs.  Formal runs instead select one conventional proxy URL
    explicitly and construct a UTF-8 client with ``trust_env=False``.
    """
    env = os.environ if env is None else env
    for key in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"):
        value = str(env.get(key) or "").strip()
        if not value:
            continue
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{key} must be an ASCII proxy URL") from exc
        return value
    return ""


def _httpx_provider_client(timeout: float):
    """Build one deterministic UTF-8 httpx client for all provider calls."""
    import httpx
    force_ipv4 = os.environ.get("METAVIDEOAGENT_FORCE_IPV4", "").lower() in ("1", "true", "yes")
    proxy_url = "" if force_ipv4 else _explicit_proxy_url()
    kwargs = {
        "timeout": httpx.Timeout(timeout, connect=min(10.0, timeout)),
        # Do not additionally inherit a different proxy/no_proxy setting from
        # httpx after selecting the explicit route above.
        "trust_env": False,
        "default_encoding": "utf-8",
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url
    if force_ipv4:
        # Bind an IPv4 local address so DNS connection attempts cannot stall
        # on a transient IPv6 route.  Proxy routing is intentionally bypassed.
        kwargs["transport"] = httpx.HTTPTransport(local_address="0.0.0.0", retries=0)
    return httpx.Client(**kwargs)


def make_openai_client(api_key: str, base_url: str, timeout: float = 60.0) -> OpenAI:
    """Create an OpenAI-compatible client with deterministic transport settings."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
        http_client=_httpx_provider_client(timeout),
    )


# =====================================================================
# Cost Counter
# =====================================================================
_COST_STATE = ContextVar("metavideoagent_cost_state", default=None)


class CostTracker:
    """Per-question API usage ledger safe under question-level threading.

    Only a successful provider response is recorded.  Failed/retried requests
    therefore do not inflate billed usage, while each returned usage record is
    retained so later audits can distinguish measured provider tokens from a
    provider that omitted usage metadata.
    """

    @staticmethod
    def _fresh_state() -> dict:
        return {
            "vision_frame_inputs": 0,
            "frame_assets": set(),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens_reported": 0,
            "llm_calls": 0,
            "vlm_calls": 0,
            "ocr_calls": 0,
            "asr_calls": 0,
            "embedding_calls": 0,
            "audio_seconds_submitted": 0.0,
            "api_calls_with_usage": 0,
            "api_calls_without_usage": 0,
            "api_call_ledger": [],
            "start_time": None,
        }

    def _state(self) -> dict:
        state = _COST_STATE.get()
        if state is None:
            state = self._fresh_state()
            _COST_STATE.set(state)
        return state

    def reset(self):
        """Begin a fresh ledger in the current task/thread context."""
        _COST_STATE.set(self._fresh_state())

    def start_timer(self):
        self._state()["start_time"] = time.time()

    def stop_timer(self) -> float:
        state = self._state()
        if state["start_time"] is None:
            return 0.0
        elapsed = time.time() - state["start_time"]
        state["start_time"] = None
        return elapsed

    @property
    def total_tokens(self):
        state = self._state()
        # Some providers expose only total_tokens.  Preserve that measured
        # total instead of silently replacing it with a zero prompt/completion
        # decomposition.
        return max(
            int(state["total_tokens_reported"] or 0),
            int(state["prompt_tokens"] or 0) + int(state["completion_tokens"] or 0),
        )

    @property
    def elapsed(self) -> float:
        started = self._state()["start_time"]
        return 0.0 if started is None else time.time() - started

    @staticmethod
    def _usage_fields(response) -> tuple[int, int, int, bool]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if not usage:
            return 0, 0, 0, False
        getter = usage.get if isinstance(usage, dict) else lambda key, default=0: getattr(usage, key, default)
        prompt = int(getter("prompt_tokens", 0) or 0)
        completion = int(getter("completion_tokens", 0) or 0)
        total = int(getter("total_tokens", 0) or 0)
        return prompt, completion, total, bool(prompt or completion or total)

    def record_api_call(self, response, *, call_type: str, profile: dict | None = None,
                        model_name: str = "", frame_paths: list | None = None,
                        audio_seconds: float = 0.0):
        """Record one successful API response and its provider usage metadata."""
        state = self._state()
        call_type = str(call_type or "unknown")
        counter = f"{call_type}_calls"
        if counter in state:
            state[counter] += 1
        else:
            # Unknown future API types still remain visible in the ledger.
            state[counter] = int(state.get(counter, 0)) + 1
        paths = [str(path) for path in (frame_paths or []) if str(path)]
        if call_type in {"vlm", "ocr", "embedding"}:
            state["vision_frame_inputs"] += len(paths)
            state["frame_assets"].update(paths)
        if call_type == "asr":
            state["audio_seconds_submitted"] += max(0.0, float(audio_seconds or 0.0))
        prompt, completion, total, has_usage = self._usage_fields(response)
        state["prompt_tokens"] += prompt
        state["completion_tokens"] += completion
        state["total_tokens_reported"] += total
        state["api_calls_with_usage" if has_usage else "api_calls_without_usage"] += 1
        details = profile or {}
        state["api_call_ledger"].append({
            "call_index": len(state["api_call_ledger"]) + 1,
            "capability": call_type,
            "profile_id": str(details.get("profile_id") or ""),
            "provider": str(details.get("provider") or ""),
            "model_id": str(model_name or details.get("model_id") or details.get("resolved_model_id") or ""),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "usage_available": has_usage,
            "image_inputs_submitted": len(paths),
            "audio_seconds_submitted": round(max(0.0, float(audio_seconds or 0.0)), 4),
        })

    def summary(self) -> dict:
        state = self._state()
        return {
            "artifact_type": "metavideoagent_api_cost_ledger", "schema_version": 1,
            # This public metric counts image inputs that reached a successful
            # VLM or OCR request.
            "frames_viewed": state["vision_frame_inputs"],
            "vision_frame_inputs": state["vision_frame_inputs"],
            "unique_frame_assets": len(state["frame_assets"]),
            "llm_calls": state["llm_calls"],
            "vlm_calls": state["vlm_calls"],
            "ocr_calls": state["ocr_calls"],
            "asr_calls": state["asr_calls"],
            "embedding_calls": state["embedding_calls"],
            "api_successful_calls": len(state["api_call_ledger"]),
            "api_calls_with_usage": state["api_calls_with_usage"],
            "api_calls_without_usage": state["api_calls_without_usage"],
            "audio_seconds_submitted": round(state["audio_seconds_submitted"], 4),
            "prompt_tokens": state["prompt_tokens"],
            "completion_tokens": state["completion_tokens"],
            "total_tokens": self.total_tokens,
            "latency_sec": round(self.elapsed if state["start_time"] else 0.0, 3),
            "api_call_ledger": [dict(item) for item in state["api_call_ledger"]],
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n{'─'*50}")
        print("Cost statistics:")
        print(f"   Visual inputs: {s['vision_frame_inputs']} frame(s) ({s['vlm_calls']} VLM call(s), {s['ocr_calls']} OCR call(s))")
        print(f"   Tokens: {s['total_tokens']:,} (prompt {s['prompt_tokens']:,} + completion {s['completion_tokens']:,})")
        print(f"   LLM calls: {s['llm_calls']}")
        print(f"   Inference latency: {s['latency_sec']}s")
        print(f"{'─'*50}")


# Global singleton
cost_tracker = CostTracker()

# =====================================================================
# Provider configuration
# =====================================================================
# Provider routing is defined by the selected capability profile file.
LLM_MODEL = str(resolve_profile("llm").get("model_id") or "")
VLM_MODEL = str(resolve_profile("vlm").get("model_id") or "")
OCR_MODEL = str(resolve_profile("ocr").get("model_id") or "")
ASR_MODEL = str(resolve_profile("asr").get("model_id") or "")
EMBEDDING_MODEL = str(resolve_profile("embedding").get("model_id") or "")


def _parallel_limit_from_env(name: str, default: int, ceiling: int | None = None):
    """Return an optional process-local API limiter configured by the run."""
    try:
        limit = int(os.environ.get(name, str(default)) or default)
    except ValueError:
        limit = default
    # A caller may lower a limit for a constrained provider, but may never
    # silently raise it above the channel's explicit run-wide ceiling.
    # ``default`` and ``ceiling`` intentionally differ for GLM: ordinary
    # formal runs retain the conservative default of 8, while an explicitly
    # requested, separately audited high-throughput run may use at most 16.
    limit = max(1, min(limit, int(ceiling if ceiling is not None else default)))
    return threading.BoundedSemaphore(limit)


_API_LIMITERS = {}
_API_LIMITER_SIGNATURE = None
_API_LIMITER_LOCK = threading.Lock()


class _FixedWindowStartRateGate:
    """Evenly pace provider starts without waiting for previous responses.

    This is intentionally different from a semaphore.  A semaphore bounds
    in-flight HTTP calls; this gate reserves the next start slot at a fixed
    cadence.  Fixed *calendar* windows can place two batches on opposite sides
    of a boundary and create a visible 30--40 request burst in one observed
    second.  Pacing prevents that boundary burst while retaining the intended
    high-throughput behavior: a slow response never holds the next start.
    """

    def __init__(self, starts_per_second: float):
        self.starts_per_second = max(0.001, float(starts_per_second))
        self._lock = threading.Lock()
        self._next_start = 0.0

    def acquire(self) -> None:
        spacing = 1.0 / self.starts_per_second
        with self._lock:
            # Sleep while retaining the gate lock.  Releasing it before the
            # sleep lets OS scheduling reorder queued callers: a late earlier
            # caller and an on-time later caller can then start together.
            # Serializing only this tiny admission section does not serialize
            # the HTTP requests themselves.
            now = time.monotonic()
            delay = self._next_start - now
            if delay > 0:
                time.sleep(delay)
            admitted = time.monotonic()
            self._next_start = admitted + spacing


_API_START_RATE_GATES = {}
_API_START_RATE_SIGNATURE = None
_API_START_RATE_LOCK = threading.Lock()


class _SharedFixedWindowStartRateGate:
    """Opt-in fixed-window start limiter shared by separate runtime processes.

    Formal train and test evaluation may execute in independent processes
    while sharing a provider credential.  A process-local semaphore/rate gate
    cannot enforce that provider's combined limit.  When explicitly enabled,
    this gate coordinates one capability through a small flock-protected state
    file; normal single-process execution keeps the existing local gate.
    """

    def __init__(self, channel: str, starts_per_second: int, state_dir: str):
        self.channel = str(channel)
        self.starts_per_second = max(1, int(starts_per_second))
        self.state_dir = os.path.abspath(str(state_dir))
        self.state_path = os.path.join(
            self.state_dir, f"{self.channel}_shared_start_rate.json"
        )
        self._lock = threading.Lock()

    def acquire(self) -> None:
        os.makedirs(self.state_dir, exist_ok=True)
        while True:
            now = time.time()
            window = int(now)
            with self._lock, open(self.state_path, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.seek(0)
                    try:
                        state = json.load(handle)
                    except (json.JSONDecodeError, OSError, ValueError):
                        state = {}
                    if int(state.get("window", -1)) != window:
                        state = {"window": window, "started": 0}
                    started = int(state.get("started", 0) or 0)
                    if started < self.starts_per_second:
                        state["started"] = started + 1
                        handle.seek(0)
                        handle.truncate()
                        json.dump(state, handle, ensure_ascii=True)
                        handle.flush()
                        return
                    sleep_seconds = max(0.001, (window + 1.0) - now)
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            time.sleep(sleep_seconds)


def _configured_starts_per_second(channel: str) -> float:
    """Return an opt-in provider start-rate limit, or zero when disabled.

    File-style ASR models publish an RPM quota rather than a concurrency
    quota.  ``METAVIDEOAGENT_<CHANNEL>_STARTS_PER_MINUTE`` therefore takes
    precedence and is converted to a smooth per-second pacing rate.
    """
    minute_key = f"METAVIDEOAGENT_{str(channel).upper()}_STARTS_PER_MINUTE"
    try:
        per_minute = float(os.environ.get(minute_key, "0") or 0)
    except ValueError:
        per_minute = 0.0
    if per_minute > 0:
        return max(0.0, min(per_minute / 60.0, 100.0))
    key = f"METAVIDEOAGENT_{str(channel).upper()}_STARTS_PER_SECOND"
    try:
        value = float(os.environ.get(key, "0") or 0)
    except ValueError:
        value = 0.0
    return max(0.0, min(value, 100.0))


def _configured_shared_starts_per_second(channel: str) -> int:
    """Return an opt-in cross-process provider start limit."""
    key = f"METAVIDEOAGENT_{str(channel).upper()}_GLOBAL_STARTS_PER_SECOND"
    try:
        value = int(os.environ.get(key, "0") or 0)
    except ValueError:
        value = 0
    return max(0, min(value, 100))


def _current_start_rate_gates() -> dict:
    global _API_START_RATE_GATES, _API_START_RATE_SIGNATURE
    channels = ("llm", "vlm", "ocr", "asr", "embedding")
    shared_dir = str(os.environ.get("METAVIDEOAGENT_SHARED_START_RATE_GATE_DIR", "") or "").strip()
    signature = tuple(
        (channel, _configured_starts_per_second(channel),
         _configured_shared_starts_per_second(channel), shared_dir)
        for channel in channels
    )
    with _API_START_RATE_LOCK:
        if signature != _API_START_RATE_SIGNATURE:
            gates = {}
            for channel, local_rate, shared_rate, configured_dir in signature:
                if shared_rate > 0:
                    if not configured_dir:
                        raise RuntimeError(
                            f"{channel} shared start-rate limiting requires "
                            "METAVIDEOAGENT_SHARED_START_RATE_GATE_DIR"
                        )
                    gates[channel] = _SharedFixedWindowStartRateGate(
                        channel, shared_rate, configured_dir
                    )
                elif local_rate > 0:
                    gates[channel] = _FixedWindowStartRateGate(local_rate)
            _API_START_RATE_GATES = gates
            _API_START_RATE_SIGNATURE = signature
        return _API_START_RATE_GATES


def _current_api_limiters() -> dict:
    """Build limiters from the current run environment, not import-time state."""
    global _API_LIMITERS, _API_LIMITER_SIGNATURE
    # A default run retains the formal 32-request VLM in-flight ceiling.  An
    # explicitly enabled fixed-window VLM start-rate run may use a larger
    # in-flight pool so slow responses do not prevent the next second's starts.
    vlm_ceiling = 128 if _configured_starts_per_second("vlm") > 0 else 32
    names = {
        # Execution defaults are hard ceilings, rather than opt-in knobs: a
        # nested video-structure worker pool must not bypass the run's
        # VLM=32 / LLM=8 provider limits.
        "llm": ("METAVIDEOAGENT_LLM_MAX_CONCURRENCY", 8, 16),
        "vlm": ("METAVIDEOAGENT_VLM_MAX_CONCURRENCY", 32, vlm_ceiling),
        "ocr": ("METAVIDEOAGENT_OCR_MAX_CONCURRENCY", 16, 16),
        "asr": ("METAVIDEOAGENT_ASR_MAX_CONCURRENCY", 16, 16),
        "embedding": ("METAVIDEOAGENT_EMBEDDING_MAX_CONCURRENCY", 16, 16),
    }
    signature = tuple(
        (channel, os.environ.get(env_name, str(default)))
        for channel, (env_name, default, _ceiling) in names.items()
    )
    with _API_LIMITER_LOCK:
        if signature != _API_LIMITER_SIGNATURE:
            _API_LIMITERS = {
                channel: _parallel_limit_from_env(env_name, default, ceiling)
                for channel, (env_name, default, ceiling) in names.items()
            }
            _API_LIMITER_SIGNATURE = signature
        return _API_LIMITERS


def _limited_api_call(channel: str, request):
    """Apply optional start-rate and in-flight limiters around one HTTP call."""
    start_gate = _current_start_rate_gates().get(channel)
    if start_gate is not None:
        # Acquire before the semaphore: a call waiting for its next start
        # window must not consume an in-flight slot.
        start_gate.acquire()
    started = time.monotonic()
    _provider_audit(channel, "start", started)
    limiter = _current_api_limiters().get(channel)
    try:
        if limiter is None:
            result = request()
        else:
            with limiter:
                result = request()
    except Exception as exc:
        _provider_audit(channel, "error", started, exc)
        raise
    _provider_audit(channel, "end", started)
    return result


def _is_transient_provider_error(exc: Exception) -> bool:
    """Return whether a failed provider request is safe to retry.

    Retries are useful for a dropped connection, a read timeout, or a
    provider-side overload.  Repeating an authentication, schema, or invalid
    request error only spends quota and obscures the real configuration fault.
    The caller still records every attempt through ``RUNTIME_PROVIDER_AUDIT_PATH``.
    """
    text = str(exc or "").lower()
    transient_markers = (
        "timeout", "timed out", "read error", "connection", "reset by peer",
        "temporarily unavailable", "service unavailable", "internal server error",
        "rate limit", "too many requests", " 429", " 500", " 502", " 503", " 504",
    )
    return any(marker in text for marker in transient_markers)


def _configured_model_name(requested: str, default: str) -> str:
    """Resolve an optional caller value without introducing another provider."""
    return requested or default


def _vlm_model_name(requested: str) -> str:
    """Resolve the configured VLM model."""
    return _configured_model_name(requested, VLM_MODEL)


def _profile_client(profile_id: str, capability: str) -> tuple[dict, OpenAI]:
    """Resolve one registered provider profile into an OpenAI-compatible client.

    This is intentionally internal. Generated modules select a ``profile_id``
    through ``runtime_evidence``; they never receive provider credentials or
    construct their own client.
    """
    profile = resolve_profile(capability, profile_id)
    config = provider_config(profile)
    if not config["api_key"]:
        raise CapabilityProfileError(
            f"Profile {profile_id!r} is not configured: no provider API key is present in its runtime environment."
        )
    if not config["base_url"]:
        raise CapabilityProfileError(
            f"Profile {profile_id!r} is not configured: no provider base URL is available."
        )
    # These values become HTTP Authorization and URL values; validate them
    # here so transport errors identify the configuration field directly.
    for field in ("api_key", "base_url"):
        try:
            str(config[field]).encode("ascii")
        except UnicodeEncodeError as exc:
            raise CapabilityProfileError(
                f"Profile {profile_id!r} has a non-ASCII {field}; check copied credentials/endpoints for trailing punctuation."
            ) from exc
    timeout = max(1.0, float(os.environ.get("METAVIDEOAGENT_PROVIDER_TIMEOUT_SEC", "60") or 60))
    return profile, make_openai_client(config["api_key"], config["base_url"], timeout=timeout)


def _runtime_profile_client(capability: str, profile_id: str | None = None) -> tuple[dict, OpenAI]:
    """Resolve every provider call through the live profile registry.

    Profiles are supplied by the active capability configuration.
    """
    return _profile_client(profile_id or DEFAULT_PROFILE_IDS[capability], capability)


def _asr_profile_client(profile_id: str | None = None) -> tuple[dict, OpenAI]:
    """Resolve ASR through the same provider-profile path as other capabilities."""
    return _runtime_profile_client("asr", profile_id)


def _response_content_text(response) -> str:
    """Normalize OpenAI-compatible text/content-part responses."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    content = getattr(getattr(choices[0], "message", None), "content", "")
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(getattr(part, "text", part))
            for part in content
        ).strip()
    return str(content or "").strip()


# =====================================================================
# Physics tools (local data preprocessing)
# =====================================================================
def local_image_to_data_url(image_path, max_size=512):
    """Image compression and Base64 encoding (API data preparation).
    Automatically resize images to prevent exceeding VLM Token limits and speed up network transfers.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        raise ValueError("Image compression encoding failed")

    base64_encoded_data = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_encoded_data}"


def extract_audio_segment(video_path: str, start_sec: float, end_sec: float, output_path: str) -> bool:
    """Audio physical interception tool"""
    try:
        bounded = clamp_to_active_media_windows(start_sec, end_sec)
        if bounded is None:
            print("⚠️ The audio interception request is outside the visible range of the current probe and has been rejected.")
            return False
        start_sec, end_sec = bounded
        ffmpeg_bin = "ffmpeg"
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        command = [
            ffmpeg_bin, "-y",
            "-i", video_path,
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-q:a", "0",
            "-map", "a",
            output_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"⚠️ Audio extraction failed: {e}")
        return False


# =====================================================================
# Large model API call center (decoupled into dedicated channels for different modes)
# =====================================================================

def call_llm(messages: list, model_name: str = LLM_MODEL, temperature: float = 0.1,
             max_tokens: int | None = 4096, profile_id: str | None = None,
             response_format: dict | None = None) -> str:
    """Call the text-LLM capability selected by the active runtime profile."""
    try:
        profile, client = _runtime_profile_client("llm", profile_id)
    except CapabilityProfileError as exc:
        return f"LLM Error: {exc}"
    if profile.get("transport") != "openai_chat_text":
        return f"LLM Error: profile {profile_id!r} does not support text chat transport"
    model_name = str(profile.get("model_id") or _configured_model_name(model_name, LLM_MODEL))
    print(f"  [API] Call LLM ({model_name}) (Temp={temperature})...")

    # Value is total attempts; smoke may set it to one to disable retries.
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2") or "2")
    max_retries = max(1, max_retries)
    last_error = ""
    for attempt in range(max_retries):
        try:
            request = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
            # Optional strict structured-output request.  Existing callers
            # keep their prior behavior; runtime controllers may opt in when
            # a provider-supported JSON object is their explicit ABI.
            if isinstance(response_format, dict) and response_format:
                request["response_format"] = dict(response_format)
            # The agent controller requires a bounded final text/JSON decision,
            # so honor the
            # capability-profile request contract instead of silently
            # consuming the entire completion budget in ``reasoning_content``.
            request_options = profile.get("request_options") or {}
            if "enable_thinking" in request_options:
                request["extra_body"] = {
                    "enable_thinking": bool(request_options["enable_thinking"])
                }
            response = _limited_api_call(
                "llm",
                lambda: client.chat.completions.create(**request),
            )
            cost_tracker.record_api_call(
                response, call_type="llm", profile=profile, model_name=model_name,
            )
            content = _response_content_text(response)
            return content if content else "LLM returned empty response"
        except Exception as e:
            last_error = str(e)
            print(f"⚠️ [LLM API Error] Attempt {attempt + 1}/{max_retries}: {last_error}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return f"LLM Error: {last_error}"


def call_vlm(prompt_or_messages, image_paths: list, model_name: str = VLM_MODEL,
             temperature: float = 0.2, max_tokens: int = 4096,
             profile_id: str | None = None, *, cost_observer=None,
             cost_observer_metadata: dict | None = None) -> str:
    """Call the VLM selected by the active capability profile."""
    try:
        profile, client = _runtime_profile_client("vlm", profile_id)
    except CapabilityProfileError as exc:
        return f"Error calling VLM API: {exc}"
    if profile.get("transport") != "openai_chat_vision":
        return f"Error calling VLM API: profile {profile_id!r} does not support image chat transport"
    model_name = str(profile.get("model_id") or _vlm_model_name(model_name))

    # 1. Parse the input format (convert the string into a standard Message structure)
    if isinstance(prompt_or_messages, str):
        messages = [
            {"role": "system", "content": "You are an expert video understanding agent with precise visual perception."},
            {"role": "user", "content": [{"type": "text", "text": prompt_or_messages}]}
        ]
    else:
        # Deep copy to prevent contamination of the external original messages list
        messages = copy.deepcopy(prompt_or_messages)
        # Make sure the content of the last user message is a list structure so that images can be appended
        last_msg = messages[-1]
        if isinstance(last_msg["content"], str):
            last_msg["content"] = [{"type": "text", "text": last_msg["content"]}]

    # 2. Convert the image to Base64 and append it to the last User Message
    valid_images = 0
    valid_image_paths = []
    for fpath in image_paths:
        if os.path.exists(fpath):
            messages[-1]["content"].append({
                "type": "image_url",
                "image_url": {"url": local_image_to_data_url(fpath)}
            })
            valid_images += 1
            valid_image_paths.append(fpath)

    print(f"  [API] Calling VLM {model_name} with {valid_images} frame(s) (temperature={temperature})...")

    max_retries = max(1, int(os.environ.get("VLM_MAX_RETRIES", "2") or "2"))
    for attempt in range(max_retries):
        try:
            # The selected VLM profile is an evidence extractor, not the
            # agent's reasoning module. Request options therefore
            # live in the capability profile and explicitly disable model
            # thinking for every visual call (structuring and scoped
            # perception alike).  This prevents a provider default from
            # silently adding a hidden reasoning pass and changing latency or
            # evidence semantics between runs.
            request_options = profile.get("request_options") or {}
            extra_body = {}
            if "enable_thinking" in request_options:
                extra_body["enable_thinking"] = bool(request_options["enable_thinking"])
            request = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if extra_body:
                request["extra_body"] = extra_body
            response = _limited_api_call(
                "vlm",
                lambda: client.chat.completions.create(**request),
            )
            cost_tracker.record_api_call(
                response, call_type="vlm", profile=profile, model_name=model_name,
                frame_paths=valid_image_paths,
            )
            if callable(cost_observer):
                try:
                    cost_observer(
                        response, call_type="vlm", profile=profile, model_name=model_name,
                        frame_paths=valid_image_paths, metadata=cost_observer_metadata,
                    )
                except Exception:
                    # Output accounting must never change a successful media call.
                    pass
            content = response.choices[0].message.content
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", str(part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            return content.strip() if content else "VLM returned empty response"
        except Exception as e:
            print(f"⚠️ [VLM API Error] Attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1 and _is_transient_provider_error(e):
                time.sleep(2 ** attempt)
            elif attempt < max_retries - 1:
                return f"Error calling VLM API after {attempt + 1} attempts: non-retryable provider error: {str(e)}"
            else:
                return f"Error calling VLM API after {max_retries} attempts: {str(e)}"


def _audio_duration_seconds(path: str) -> float:
    """Best-effort duration for successful ASR billing provenance only."""
    try:
        import wave
        with wave.open(path, "rb") as handle:
            rate = float(handle.getframerate() or 0)
            return (float(handle.getnframes() or 0) / rate) if rate else 0.0
    except Exception:
        return 0.0


def call_asr(audio_path: str, model_name: str = ASR_MODEL, profile_id: str | None = None,
             *, return_attempts: bool = False, cost_observer=None,
             cost_observer_metadata: dict | None = None):
    """Call the formal ASR profile, retrying one transient provider failure.

    A single failed HTTP/provider request is not evidence that a localized
    audio segment is unavailable.  Run the identical request one more time
    and let ``runtime_evidence`` record both attempts.  ``return_attempts``
    returns a string by default and can additionally expose attempt metadata.
    """
    attempts = []

    def _return(text: str):
        return (text, attempts) if return_attempts else text

    try:
        profile, client = _asr_profile_client(profile_id)
    except CapabilityProfileError as exc:
        attempts.append({"attempt": 1, "status": "api_error", "error_type": type(exc).__name__,
                         "error_message": str(exc)})
        return _return(f"Error calling ASR API: {exc}")
    model_name = str(profile.get("model_id") or _configured_model_name(model_name, ASR_MODEL))
    print(f"[API] Call ASR ({model_name}) Extract speech and audio features...")

    try:
        max_attempts = int(os.environ.get("ASR_MAX_RETRIES", "2") or 2)
    except ValueError:
        max_attempts = 2
    max_attempts = max(1, min(4, max_attempts))
    for attempt in range(1, max_attempts + 1):
        try:
            if profile and profile.get("transport") == "openai_chat_audio":
                with open(audio_path, "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode("ascii")
                # The registered chat-audio transport receives a data URI in
                # ``input_audio.data``. Runtime audio clips are local and
                # ephemeral, so this preserves the active-window contract.
                audio_data_uri = f"data:audio/wav;base64,{audio_base64}"
                response = _limited_api_call(
                    "asr",
                    lambda: client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "user",
                            "content": [{
                                "type": "input_audio",
                                "input_audio": {"data": audio_data_uri},
                            }],
                        }],
                    ),
                )
                text = _response_content_text(response)
            elif not profile or profile.get("transport") == "openai_audio_transcription":
                with open(audio_path, "rb") as f:
                    response = _limited_api_call(
                        "asr",
                        lambda: client.audio.transcriptions.create(
                            model=model_name,
                            file=f,
                        ),
                    )
                text = getattr(response, "text", "")
            else:
                message = (
                    f"profile {profile_id!r} has unsupported transport "
                    f"{profile.get('transport')!r}"
                )
                attempts.append({"attempt": attempt, "status": "api_error", "error_message": message})
                return _return(f"Error calling ASR API: {message}")
            cost_tracker.record_api_call(
                response, call_type="asr", profile=profile, model_name=model_name,
                audio_seconds=_audio_duration_seconds(audio_path),
            )
            if callable(cost_observer):
                try:
                    cost_observer(
                        response, call_type="asr", profile=profile, model_name=model_name,
                        audio_seconds=_audio_duration_seconds(audio_path),
                        metadata=cost_observer_metadata,
                    )
                except Exception:
                    # Output accounting must never change a successful media call.
                    pass
            value = text.strip() if text else "[NO SPEECH DETECTED]"
            attempts.append({"attempt": attempt, "status": "ok"})
            return _return(value)
        except Exception as exc:
            attempts.append({"attempt": attempt, "status": "api_error",
                             "error_type": type(exc).__name__, "error_message": str(exc)})
            if attempt < max_attempts:
                # Do not release a synchronized retry burst after a 429.
                # The next call also passes through the ASR start-rate gate.
                delay = min(8.0, 2.0 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                print(
                    f"  ⚠️ [ASR API Error] attempt {attempt}/{max_attempts} failed; "
                    f"retrying after {delay:.2f}s: {exc}"
                )
                time.sleep(delay)
                continue
            return _return(f"Error calling ASR API after {max_attempts} attempts: {exc}")

    # The loop returns on every branch; keep a fail-closed fallback for type
    # checkers and future transport additions.
    return _return(f"Error calling ASR API after {max_attempts} attempts: unknown failure")


def call_embedding(input_text: str, model_name: str = EMBEDDING_MODEL,
                   profile_id: str | None = None) -> list:
    """Vectorized model (Embedding) calling interface.
    Used for: Converting search query or description text into high-dimensional vectors in multi-modal RAG (such as CLIP/CLAP database building).

    The paper profile uses a native multimodal transport. Provider-neutral
    profiles may instead select the standard OpenAI-compatible text embedding
    transport.
    """
    try:
        profile, client = _runtime_profile_client("embedding", profile_id)
    except CapabilityProfileError:
        return [0.0] * 1024
    transport = str(profile.get("transport") or "")
    if transport not in {"dashscope_multimodal_embedding", "openai_embedding"}:
        return [0.0] * 1024
    model_name = str(profile.get("model_id") or EMBEDDING_MODEL)

    max_retries = max(1, int(os.environ.get("EMBEDDING_MAX_RETRIES", "2") or "2"))
    for attempt in range(max_retries):
        try:
            if transport == "openai_embedding":
                response = _limited_api_call(
                    "embedding",
                    lambda: client.embeddings.create(
                        model=model_name,
                        input=str(input_text or ""),
                    ),
                )
                cost_tracker.record_api_call(
                    response, call_type="embedding", profile=profile,
                    model_name=model_name,
                )
                rows = getattr(response, "data", None) or []
                vector = getattr(rows[0], "embedding", None) if rows else None
                if isinstance(vector, list) and vector:
                    return vector
                raise RuntimeError(
                    "OpenAI-compatible embedding response did not include data[0].embedding"
                )
            config = provider_config(profile)
            # Text/vision/LLM use the OpenAI-compatible endpoint, but the paper
            # multimodal embedding profile uses its native provider route. A run
            # may set METAVIDEOAGENT_BASE_URL to a compatibility endpoint for chat models;
            # do not append the native embedding path below that compat prefix.
            base_url = config["base_url"].rstrip("/")
            if "/compatible-mode/" in base_url:
                base_url = base_url.split("/compatible-mode/", 1)[0]
            endpoint = base_url + "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
            payload = json.dumps({
                "model": model_name,
                "input": {"contents": [{"text": str(input_text or "")} ]},
                "parameters": {"dimension": 1024},
            }).encode("utf-8")
            def _request_embedding():
                request = urllib.request.Request(
                    endpoint, data=payload,
                    headers={"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8"))
            response = _limited_api_call("embedding", _request_embedding)
            cost_tracker.record_api_call(
                response, call_type="embedding", profile=profile, model_name=model_name,
            )
            embeddings = ((response.get("output") or {}).get("embeddings") or []) if isinstance(response, dict) else []
            vector = embeddings[0].get("embedding") if embeddings and isinstance(embeddings[0], dict) else None
            if isinstance(vector, list) and vector:
                return vector
            raise RuntimeError(
                "Native multimodal embedding response did not include "
                "output.embeddings[0].embedding"
            )
        except Exception as e:
            print(f"⚠️ Embedding API Error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)

    # The fail-soft vector preserves the module contract without fabricating
    # a successful capability event (runtime_evidence records no_content).
    return [0.0] * 1024


def call_image_embedding(image_path: str, model_name: str = EMBEDDING_MODEL,
                         profile_id: str | None = None) -> list:
    """Embed one local image when the selected profile is multimodal."""
    if not image_path or not os.path.isfile(image_path):
        return [0.0] * 1024
    try:
        profile, _ = _runtime_profile_client("embedding", profile_id)
        if profile.get("transport") != "dashscope_multimodal_embedding":
            return [0.0] * 1024
        config = provider_config(profile)
        # See call_embedding: the paper multimodal transport uses its native
        # endpoint rather than the OpenAI-compatible chat endpoint.
        base_url = config["base_url"].rstrip("/")
        if "/compatible-mode/" in base_url:
            base_url = base_url.split("/compatible-mode/", 1)[0]
        endpoint = base_url + "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
        payload = json.dumps({
            "model": str(profile.get("model_id") or model_name),
            "input": {"contents": [{"image": local_image_to_data_url(image_path, max_size=1024)}]},
            "parameters": {"dimension": 1024},
        }).encode("utf-8")
        request = urllib.request.Request(
            endpoint, data=payload,
            headers={"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}, method="POST",
        )
        with _current_api_limiters()["embedding"]:
            with urllib.request.urlopen(request, timeout=60) as response_handle:
                response = json.loads(response_handle.read().decode("utf-8"))
        cost_tracker.record_api_call(response, call_type="embedding", profile=profile,
                                     model_name=str(profile.get("model_id") or model_name), frame_paths=[image_path])
        rows = ((response.get("output") or {}).get("embeddings") or [])
        vector = rows[0].get("embedding") if rows and isinstance(rows[0], dict) else None
        return vector if isinstance(vector, list) and vector else [0.0] * 1024
    except Exception as exc:
        print(f"⚠️ Image embedding API Error: {exc}")
        return [0.0] * 1024


def call_ocr(image_path: str, model_name: str = OCR_MODEL, temperature: float = 0.0,
             max_tokens: int = 256, profile_id: str | None = None) -> str:
    """Call the OCR capability selected by the registered OCR profile.

    Args:
        image_path: image file path
        model_name: model name selected by the active OCR profile
        temperature: temperature parameter (OCR must be 0.0)
        max_tokens: maximum number of output tokens
    Returns:
        Recognized text, or ``No text detected`` when no text is present.
    """
    try:
        profile, client = _runtime_profile_client("ocr", profile_id)
    except CapabilityProfileError as exc:
        return f"OCR Error: {exc}"
    if profile.get("transport") != "openai_chat_vision":
        return f"OCR Error: profile {profile_id!r} does not support image chat transport"
    model_name = str(profile.get("model_id") or _configured_model_name(model_name, OCR_MODEL))

    image_data_url = local_image_to_data_url(image_path, max_size=768)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract ALL visible text from this video frame accurately. Include subtitles, captions, labels, signs, and any on-screen text. If no text is visible, reply 'No text detected'."},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]
    }]

    max_retries = max(1, int(os.environ.get("OCR_MAX_RETRIES", "2") or "2"))
    last_error = ""
    for attempt in range(max_retries):
        try:
            response = _limited_api_call(
                "ocr",
                lambda: client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
            cost_tracker.record_api_call(
                response, call_type="ocr", profile=profile, model_name=model_name,
                frame_paths=[image_path],
            )
            content = response.choices[0].message.content
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", str(part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            return content.strip() if content else "No text detected"
        except Exception as exc:
            last_error = str(exc)
            print(f"⚠️ [OCR API Error] Attempt {attempt + 1}/{max_retries}: {last_error}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return f"OCR Error after {max_retries} attempts: {last_error}"


# =====================================================================
# Global Tools: JSON Formatting (common to all Agents)
# =====================================================================
def format_json(raw_text: str, max_tokens: int = 4096) -> dict | None:
    """Use the selected execution LLM to recover one JSON object from text."""
    profile, client = _runtime_profile_client("llm")

    last_error = ""
    for attempt in range(2):
        try:
            response = _limited_api_call(
                "llm",
                lambda: client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": (
                            "Extract the intended JSON object from the user's text and repair only "
                            "syntactic defects such as trailing commas, unmatched delimiters, or "
                            "surrounding prose. Preserve the original values. Return exactly one valid "
                            "JSON object with no Markdown fence or explanation."
                        )},
                        {"role": "user", "content": raw_text},
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                ),
            )
            cost_tracker.record_api_call(
                response, call_type="llm", profile=profile, model_name=LLM_MODEL,
            )
            content = response.choices[0].message.content or ""
            # OpenAI-compatible content list compatibility
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", str(part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            content = content.strip()
            # Remove possible ```json packages
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            return json.loads(content)
        except Exception as exc:
            last_error = str(exc)
            print(f"⚠️ [format_json] Attempt {attempt + 1}/2 failed: {last_error}")
            if attempt == 0:
                time.sleep(1.0)
    return None
