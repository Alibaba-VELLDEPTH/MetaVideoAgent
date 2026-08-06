"""Trajectory-first attribution and comparison for MetaVideoAgent.

The reviewer first establishes the evidence-supported solution inside the
annotated interval, then reads the complete execution trajectory and locates
the earliest causal divergence. Bounded video, audio, transcript, and
structure review is available only when the trajectory does not resolve an
evidentiary question. Every disjoint interval remains independent and no
review tool expands it into unrestricted media access.
"""

import ast
import glob
import importlib
import json
import os
import re
import subprocess
import sys

import cv2
from openai import OpenAI

try:
    from answer_normalizer import judge_answer, parse_answer_labels
    from capability_registry import DEFAULT_PROFILE_IDS
    from config import (
        TEACHER_ASR_MODEL,
        TEACHER_LLM_API_KEY,
        TEACHER_LLM_BASE_URL,
        TEACHER_LLM_MODEL,
        TEACHER_LLM_REQUEST_OPTIONS,
        TEACHER_MAX_FRAMES,
    )
    from config import (
        TEACHER_ENHANCED_VLM_MODEL as CONFIG_TEACHER_ENHANCED_VLM_MODEL,
    )
    from config import (
        TEACHER_MULTIMODAL_MODEL as CONFIG_TEACHER_MULTIMODAL_MODEL,
    )
except ImportError:  # pragma: no cover - package import
    from execution.action_runtime.capability_registry import DEFAULT_PROFILE_IDS

    from .answer_normalizer import judge_answer, parse_answer_labels
    from .config import (
        TEACHER_ASR_MODEL,
        TEACHER_LLM_API_KEY,
        TEACHER_LLM_BASE_URL,
        TEACHER_LLM_MODEL,
        TEACHER_LLM_REQUEST_OPTIONS,
        TEACHER_MAX_FRAMES,
    )
    from .config import (
        TEACHER_ENHANCED_VLM_MODEL as CONFIG_TEACHER_ENHANCED_VLM_MODEL,
    )
    from .config import (
        TEACHER_MULTIMODAL_MODEL as CONFIG_TEACHER_MULTIMODAL_MODEL,
    )

TEACHER_ACTIVE_VLM_MODEL = CONFIG_TEACHER_MULTIMODAL_MODEL
TEACHER_ACTIVE_ENHANCED_VLM_MODEL = CONFIG_TEACHER_ENHANCED_VLM_MODEL

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DOCS_DIR = os.path.join(_CURRENT_DIR, "agent_docs")


def _load_prompt(filename: str, fallback: str = "") -> str:
    """Load prompt word files from agent_docs/."""
    fpath = os.path.join(_DOCS_DIR, filename)
    if os.path.exists(fpath):
        return open(fpath, "r", encoding="utf-8").read().strip()
    return fallback


MAX_ROUNDS = 15         # ReAct maximum number of rounds; use the number of rounds to prevent infinite loops without setting a low tool budget
MAX_TOOL_CALLS = MAX_ROUNDS
API_TIMEOUT_SEC = 120   # Prevent a single provider call from hanging indefinitely
WATCH_MAX_FRAMES = min(TEACHER_MAX_FRAMES, 96)
LONG_WATCH_MAX_FRAMES = min(TEACHER_MAX_FRAMES, 48)
LONG_WATCH_THRESHOLD_SEC = 120
# The selected VLM receives bounded image batches. One review window gets enough
# chronological samples to preserve short actions while keeping the request
# shape aligned with the execution VLM adapter. Each labelled window is
# reviewed independently, so a
# multi-window question can never lose an interval to a shared global cap.
REVIEW_FRAMES_PER_WINDOW = 10
REVIEW_DETAIL_FRAMES_PER_WINDOW = 8
SEEK_WINDOW = 120       # seek fast forward/rewind window (seconds)
DETAIL_MAX_SEC = 5      # detailed_watch review with narrow video clips, up to 5 seconds
EVIDENCE_EPS = 1e-3     # Evidence interval clipping tolerance

TEACHER_OUTPUT_TOOL_REPLACEMENTS = {
    "DIAG_VIDEO_AUDIT": "Video evidence review",
    "DIAG_DETAIL_VIDEO_AUDIT": "Detailed video review",
    "AUDIT_VIDEO": "Video evidence review",
    "AUDIT_DETAIL_VIDEO": "Detailed video review",
    "AUDIT_ENHANCED_VIDEO": "Enhanced video review",
    "AUDIT_STRUCTURE": "Structure library review",
    "AUDIT_TRANSCRIPT": "Subtitle/transcription review",
    "watch_video": "Video evidence review",
    "video_review": "Video evidence review",
    "teacher_video_verification": "Video evidence review",
    "detailed_watch": "Detailed video review",
    "detail_video_review": "Detailed video review",
    "teacher_detailed_video_verification": "Detailed video review",
    "enhanced_watch": "Enhanced video review",
    "enhanced_video_review": "Enhanced video review",
    "teacher_enhanced_video_verification": "Enhanced video review",
    "browse_struct_db": "Structure library review",
    "structure_review": "Structure library review",
    "teacher_structure_db_audit": "Structure library review",
    "read_transcript": "Subtitle/transcription review",
    "transcript_review": "Subtitle/transcription review",
    "teacher_transcript_audit": "Subtitle/transcription review",
    "listen_audio": "Audio review",
    "frame_inspect": "visual perception ability",
    "single_frame_inspect": "visual perception ability",
    "inspect_frames": "visual perception ability",
    "runtime_visual_observation_tool": "visual perception ability",
    "runtime_visual_observation_capability": "visual perception ability",
    "clip_search": "Structure retrieval/location capabilities",
    "frame_clip_search": "Structure retrieval/location capabilities",
    "search_evidence": "Evidence retrieval/location capabilities",
    "runtime_localization_or_structure_retrieval_tool": "Structure retrieval/location capabilities",
    "runtime_localization_or_structure_retrieval_capability": "Structure retrieval/location capabilities",
    "runtime_evidence_retrieval_capability": "Structure retrieval/location capabilities",
    "global_browse": "Global structure browsing capabilities",
    "runtime_broad_structure_retrieval_tool": "Global structure browsing capabilities",
    "runtime_broad_structure_retrieval_capability": "Global structure browsing capabilities",
    "audio_analysis": "Audio/subtitle understanding ability",
    "runtime_audio_or_transcript_evidence_tool": "Audio/subtitle understanding ability",
    "runtime_audio_or_transcript_capability": "Audio/subtitle understanding ability",
}


# =====================================================================
# Helpers
# =====================================================================

def _parse_time_reference_windows(time_ref) -> list[tuple[float, float]]:
    """Parse every approved interval in a ``time_reference`` value.

    A time reference is a *set* of disjoint evidence windows, not an envelope.
    In particular, ``[[30, 35], [40, 50]]`` must never silently become
    ``(30, 35)`` or an unrestricted ``(30, 50)`` request.  The latter would
    either omit labelled evidence or inspect the unlabelled gap.

    Supported examples:
    - ``00:15-00:19`` / ``15-19``
    - ``[15, 19]`` / ``[[15, 19], [30, 35]]``
    - Python/JSON list values
    - task refs containing a window, e.g. ``video::[[15, 19]]::hash``
    """
    if not time_ref:
        return []

    def _to_sec(value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if ":" in text:
            fields = [float(x) for x in re.split(r"[::]", text) if x != ""]
            if len(fields) == 3:
                return fields[0] * 3600 + fields[1] * 60 + fields[2]
            if len(fields) == 2:
                return fields[0] * 60 + fields[1]
            if len(fields) == 1:
                return fields[0]
            raise ValueError(text)
        return float(text)

    def _collect(obj) -> list[tuple[float, float]]:
        if isinstance(obj, dict):
            for start_key, end_key in (
                ("start_sec", "end_sec"),
                ("start", "end"),
                ("begin", "end"),
            ):
                if start_key in obj and end_key in obj:
                    return [(_to_sec(obj[start_key]), _to_sec(obj[end_key]))]
            return []
        if isinstance(obj, (list, tuple)):
            # A two-scalar list is one interval.  Any nested list is a list
            # of intervals and must retain every member.
            if len(obj) == 2 and not isinstance(obj[0], (list, tuple, dict)):
                return [(_to_sec(obj[0]), _to_sec(obj[1]))]
            windows = []
            for item in obj:
                windows.extend(_collect(item))
            return windows
        return []

    parsed = []
    if isinstance(time_ref, (dict, list, tuple)):
        try:
            parsed = _collect(time_ref)
        except (TypeError, ValueError, IndexError):
            parsed = []
    else:
        text = str(time_ref).strip()
        # A task id has the canonical JSON interval list between its first two
        # ``::`` separators.  Parse that value before using a conservative
        # numeric-pair fallback.
        candidates = [text]
        parts = text.split("::")
        if len(parts) >= 3:
            candidates.insert(0, parts[1])
        for candidate in candidates:
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = _collect(loader(candidate))
                    if parsed:
                        break
                except Exception:
                    continue
            if parsed:
                break
        if not parsed:
            pairs = re.findall(
                r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
                text,
            )
            parsed = [(float(start), float(end)) for start, end in pairs]
        if not parsed:
            parts = re.split(r"\\s*(?:-|~|to|—|–)\\s*", text, maxsplit=1)
            if len(parts) == 2:
                try:
                    parsed = [(_to_sec(parts[0]), _to_sec(parts[1]))]
                except (TypeError, ValueError, IndexError):
                    parsed = []

    # Reject malformed windows and deduplicate without changing the labelled
    # ordering.  Reordering can change the temporal narrative presented to
    # the review model.
    seen = set()
    windows = []
    for start, end in parsed:
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        key = (round(start, 6), round(end, 6))
        if key not in seen:
            seen.add(key)
            windows.append((start, end))
    return windows



def _sanitize_teacher_output_terms(value, preserve_trace: bool = False):
    """Remove internal Teacher/runtime tool names from saved review fields.

    Teacher ReAct still uses concrete internal tool names for ACTION dispatch,
    but micro-review JSON is later consumed by DiagnosisAgent and Codex.  Those
    downstream stages should receive capability-level descriptions, not tool
    names that are unavailable in the execution-layer candidate runtime.
    By default this also sanitizes ``react_trace`` because generated reports are
    directly reused as diagnosis/Codex context.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not preserve_trace and key == "react_trace":
                # ReAct traces contain Teacher-only dispatch tokens and raw
                # observations.  They are useful while debugging locally, but
                # harmful as downstream Diagnosis/Codex context because
                # generated runtime modules may mistake them for executable
                # tools.  Drop the field entirely from saved reports.
                continue
            if not preserve_trace and key in {
                "tool",
                "tool_name",
                "tool_call",
                "target_tool",
                "action",
            }:
                out[key] = _sanitize_teacher_output_terms(item, preserve_trace=preserve_trace)
                continue
            if preserve_trace and key == "react_trace":
                out[key] = item
            else:
                out[key] = _sanitize_teacher_output_terms(item, preserve_trace=preserve_trace)
        return out
    if isinstance(value, list):
        return [_sanitize_teacher_output_terms(item, preserve_trace=preserve_trace) for item in value]
    if isinstance(value, str):
        text = value
        for term, replacement in TEACHER_OUTPUT_TOOL_REPLACEMENTS.items():
            text = re.sub(re.escape(term), replacement, text, flags=re.IGNORECASE)
        return text
    return value


def _extract_content(response) -> str:
    """Extract text from either scalar or content-part provider responses."""
    content = response.choices[0].message.content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content).strip()
    return (content or "").strip()


def _extract_answer_letter(answer_raw: str) -> str:
    """Extract an explicit A-Z answer label without scanning option prose."""
    labels, _ = parse_answer_labels(answer_raw)
    return labels[0] if labels else ""


def _answer_is_correct(answer_raw: str, gt_answer: str) -> bool:
    """Use the shared answer contract for teacher-side review metadata.

    Execution answers may contain both a label and its text while a dataset
    stores only the option text. The shared normalizer handles both forms.
    """
    return bool(judge_answer(answer_raw, gt_answer).get("is_correct"))


def _review_prompt_value(value, *, depth: int = 0, key: str = ""):
    """Bound one persisted trace for a review prompt without editing the trace.

    Full-eval trajectories retain raw provider outputs for reproducibility and
    can be several megabytes per question.  Sending two such traces plus
    capability events to the teacher exceeds the provider request-body and
    context limits.  Keep the original JSONL untouched, but pass the teacher
    a deterministic, indexed excerpt: all execution steps and their anchors
    remain visible while verbose raw observations/prompts are clipped.
    """
    if depth >= 6:
        return "<nested value omitted for review prompt>"
    if isinstance(value, str):
        limit = 2600 if key in {"observation", "raw_output", "evidence", "content"} else 1100
        if len(value) <= limit:
            return value
        head = max(200, int(limit * 0.8))
        return value[:head] + f"\n...[{len(value) - limit} chars omitted; full value remains in execution JSONL]...\n" + value[-(limit - head):]
    if isinstance(value, list):
        limit = 16 if depth < 2 else 8
        items = [_review_prompt_value(item, depth=depth + 1, key=key) for item in value[:limit]]
        if len(value) > limit:
            items.append({"_omitted_items": len(value) - limit})
        return items
    if isinstance(value, dict):
        priority = (
            "step_type", "step", "action", "decision", "thought", "tool_name", "tool_params",
            "action_input", "plan_output", "module_result", "observation", "evidence", "error",
            "status", "time_ranges", "active_time_windows", "actual_time_ranges", "answer",
            "final_answer", "capability", "profile_id", "event_id", "evidence_event_ids",
        )
        keys = [name for name in priority if name in value]
        keys.extend(name for name in value if name not in keys)
        limit = 28 if depth < 2 else 16
        result = {
            str(name): _review_prompt_value(value[name], depth=depth + 1, key=str(name))
            for name in keys[:limit]
        }
        if len(keys) > limit:
            result["_omitted_keys"] = len(keys) - limit
        return result
    return value


def _review_prompt_json(value, *, max_chars: int) -> str:
    """Serialize a deterministic excerpt and enforce a hard prompt budget."""
    text = json.dumps(_review_prompt_value(value), ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[review prompt capped at {max_chars} chars; full trace remains on disk]"


def _format_trajectory(trajectory: list) -> str:
    """Render an anchored, bounded trajectory excerpt for Teacher prompts."""
    if not isinstance(trajectory, list) or not trajectory:
        return "(No trajectory data)"
    rendered = []
    tool_step_index = 0
    for record_index, step in enumerate(trajectory, start=1):
        if not isinstance(step, dict):
            rendered.append({
                "trace_record_index": record_index,
                "raw_record": step,
            })
            continue
        item = _review_prompt_value(dict(step))
        item["trace_record_index"] = record_index
        if item.get("step_type") == "tool_execution":
            tool_step_index += 1
            item["trajectory_step_index"] = tool_step_index
        rendered.append(item)
    return _review_prompt_json(rendered, max_chars=90000)


def _extract_student_time_windows(trajectory: list) -> list:
    """Extract time windows the student agent explicitly requested in tool params.

    These windows are not treated as ground-truth evidence. They are only used
    to let Teacher inspect the structured DB text that likely influenced the
    student's wrong search path.
    """
    windows = []

    def _walk(obj):
        if isinstance(obj, dict):
            yield obj
            for value in obj.values():
                yield from _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                yield from _walk(item)

    start_keys = ("start_sec", "start", "start_time", "begin_sec", "from_sec")
    end_keys = ("end_sec", "end", "end_time", "stop_sec", "to_sec")

    for step in trajectory or []:
        candidates = []
        if step.get("step_type") == "planning":
            for sub in step.get("plan_output", []) or []:
                if isinstance(sub, dict):
                    candidates.append(sub.get("tool_params", {}))
        elif step.get("step_type") == "tool_execution":
            candidates.extend([
                step.get("tool_params", {}),
                step.get("params", {}),
                step.get("arguments", {}),
            ])

        for cand in candidates:
            for obj in _walk(cand):
                start = next((obj.get(k) for k in start_keys if k in obj), None)
                end = next((obj.get(k) for k in end_keys if k in obj), None)
                try:
                    if start is None or end is None:
                        continue
                    s = float(start)
                    e = float(end)
                except (TypeError, ValueError):
                    continue
                if e < s:
                    s, e = e, s
                if e - s > EVIDENCE_EPS:
                    windows.append((s, e))

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for s, e in windows:
        key = (round(s, 3), round(e, 3))
        if key not in seen:
            seen.add(key)
            unique.append((s, e))
    return unique


def _format_time_windows(windows: list, limit: int = 12) -> str:
    if not windows:
        return "(No explicit time window was resolved from the trajectory parameters)"
    parts = [f"{s:.1f}s-{e:.1f}s" for s, e in windows[:limit]]
    if len(windows) > limit:
        parts.append(f"...and {len(windows) - limit} more")
    return ", ".join(parts)


def _robust_json_extract(text: str) -> dict:
    """Extract JSON from LLM output."""
    if not text:
        return {}
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        pass
    return {}


def _phase_max_tokens(phase_name: str) -> int:
    """Token budget per Teacher phase.

    Divergence needs enough room for fault steps and iteration-change analysis.
    """
    if phase_name == "divergence_diagnosis":
        return 5000
    if phase_name == "student_path":
        return 3600
    return 3000




# =====================================================================
# System prompt
# =====================================================================

SYSTEM_PROMPT = _load_prompt("teacher_prompt.md")


# =====================================================================
# Evolutionary comparative analysis Prompt
# =====================================================================

COMPARE_EVOLUTION_PROMPT = _load_prompt("teacher_compare_prompt.md")


# =====================================================================
# TeacherAgent
# =====================================================================

class TeacherAgent:
    """Attribute trajectory failures using bounded, evidence-first review."""

    def __init__(self, workspace_dir: str, video_id: str, structure_dir: str = ""):
        self.workspace_dir = workspace_dir
        self.video_id = video_id
        self.structure_dirs = []
        for directory in (
            structure_dir,
            os.path.join(workspace_dir, "video_structure"),
        ):
            if directory and directory not in self.structure_dirs:
                self.structure_dirs.append(directory)
        self.video_path = os.path.join(workspace_dir, "raw_videos", f"{video_id}.mp4")
        self.cache_dir = os.path.join(
            workspace_dir, "evolution_history", "teacher_cache", video_id
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        # ReAct main-loop client through the configured compatible endpoint.
        self.llm_client = OpenAI(
            api_key=TEACHER_LLM_API_KEY,
            base_url=TEACHER_LLM_BASE_URL,
            timeout=API_TIMEOUT_SEC,
        )
        self._model_registry = self._load_model_registry()

    @staticmethod
    def _review_runtime_utils():
        """Load the bundled runtime's selected VLM/ASR adapters.

        Teacher review is orchestration code, while provider transport belongs
        to the bundled execution runtime. Reusing these adapters guarantees
        the same capability profiles and request options,
        timeout handling and direct-server routing as formal full evaluation.
        """
        project_root = os.path.dirname(_CURRENT_DIR)
        runtime_dir = os.path.abspath(
            os.path.join(project_root, "execution", "action_runtime")
        )
        if not os.path.isfile(os.path.join(runtime_dir, "utils.py")):
            raise RuntimeError(f"review runtime utils unavailable: {runtime_dir}")
        if runtime_dir not in sys.path:
            sys.path.insert(0, runtime_dir)
        module = importlib.import_module("utils")
        module_path = os.path.abspath(str(getattr(module, "__file__", "")))
        expected = os.path.abspath(os.path.join(runtime_dir, "utils.py"))
        if module_path != expected:
            raise RuntimeError(
                f"review resolved an incompatible utils module: {module_path}; expected {expected}"
            )
        return module

    # ==================================================================
    # Model registration information loading
    # ==================================================================

    def _load_model_registry(self) -> str:
        """Return the prompt-safe capability inventory for this run."""
        from runtime_capability_context import build_runtime_capability_context
        return json.dumps(build_runtime_capability_context(), ensure_ascii=False, indent=2)

    # ==================================================================
    # Public entry
    # ==================================================================

    def solve(self, question: str, gt_answer: str, time_reference: str,
              agent_trajectory: list = None, agent_answer: str = "",
              cost: dict = None, force_review: bool = False,
              mode: str = "baseline_micro",
              evolution_info: dict = None) -> dict:
        """Review one trajectory and attribute its earliest causal failure.

        Args:
            agent_trajectory: Agent execution trajectory
            agent_answer: Agent final answer original text
            cost: Agent cost statistics for this question (frames_viewed, total_tokens, latency_sec)
            force_review: Force review even if the current answer is correct, used for cross-round change attribution.
            mode: review mode label.
            evolution_info: Cross-round candidate context.

        Returns:
            dict: fault_step, fault_evidence, fault_module, fault_type,
                  module_design_issue, key_evidence_summary, success
        """
        evidence_windows = _parse_time_reference_windows(time_reference)
        if not evidence_windows or not os.path.exists(self.video_path):
            return self._fail("No time_reference or video not found")

        # Correct answers need review only when cross-round attribution is requested.
        agent_letter = _extract_answer_letter(agent_answer)
        is_wrong = not _answer_is_correct(agent_answer, gt_answer)
        if not is_wrong and not force_review:
            return {
                "key_evidence_summary": "The Agent's answer is correct and the review is skipped.",
                "fault_module": None,
                "fault_type": "",
                "fault_steps": [],
                "fault_evidence": "",
                "module_design_issue": "",
                "success": True,
                "skipped": True,
            }

            print(f"     Reviewing trajectory [{time_reference}] agent={agent_letter or 'No answer'} gt={gt_answer}")
        return self.review_trajectory_three_phase(
            question=question,
            gt_answer=gt_answer,
            agent_answer=agent_letter or "No answer",
            time_reference=time_reference,
            trajectory=agent_trajectory or [],
            mode=mode,
            cost=cost,
            evolution_info=evolution_info,
        )

    def review_trajectory_three_phase(
        self,
        question: str,
        gt_answer: str,
        time_reference: str,
        trajectory: list,
        agent_answer: str = "",
        mode: str = "baseline_micro",
        baseline_review: dict = None,
        evolution_info: dict = None,
        cost: dict = None,
        reference_trajectory: list = None,
        reference_answer: str = "",
        capability_events: list = None,
    ) -> dict:
        """Run Teacher review as explicit Gold/Student/Divergence phases."""
        evidence_windows = _parse_time_reference_windows(time_reference)
        if not evidence_windows or not os.path.exists(self.video_path):
            return self._fail("No time_reference or video not found")
        start_sec, end_sec = evidence_windows[0]
        evidence_window_text = _format_time_windows(evidence_windows)

        # Gold Path must start from a complete observation of *all* annotated
        # evidence, rather than leaving coverage up to a later ReAct tool
        # choice.  Every disjoint interval is sampled for the selected VLM and
        # transcribed by the selected ASR separately.  If any interval lacks a
        # successful visual observation, this question has no valid Gold
        # evidence basis and must not yield a plausible-looking micro review.
        gold_video_evidence = self._observe_complete_time_reference(
            evidence_windows=evidence_windows,
            question=question,
        )
        if not gold_video_evidence.get("success"):
            failed = self._fail(
                "Gold Path could not obtain complete VLM and ASR evidence for "
                f"time_reference {evidence_window_text}: "
                f"{gold_video_evidence.get('error', 'unknown error')}"
            )
            failed.update({
                "time_reference": time_reference,
                "question": question,
                "gt_answer": gt_answer,
                "agent_answer": agent_answer,
                "gold_evidence_coverage": gold_video_evidence,
            })
            return failed

        traj_text = _format_trajectory(trajectory or [])
        student_windows = _extract_student_time_windows(trajectory or [])
        answer_letter = _extract_answer_letter(agent_answer) or agent_answer or "No answer"
        reference_answer_letter = _extract_answer_letter(reference_answer) or reference_answer or "Not provided"
        reference_traj_text = _format_trajectory(reference_trajectory or [])

        cost_line = ""
        if cost:
            cost_line = f"""## Cost baseline for this question
- Frames viewed: {cost.get('frames_viewed', '?')}
- Tokens consumed: {cost.get('total_tokens', '?')}
- Inference latency: {cost.get('latency_sec', '?')} seconds
- LLM/VLM calls: {cost.get('llm_calls', '?')} / {cost.get('vlm_calls', '?')}"""

        baseline_review_text = "No earlier micro-review is available."
        if baseline_review:
            baseline_review_text = _review_prompt_json(baseline_review, max_chars=12000)

        evolution_info_text = "No candidate-change context was supplied."
        pre_evolution_role = "Pre-evolution current best"
        post_evolution_role = "Candidate result"
        initial_reference_mode = False
        if evolution_info:
            evolution_info_text = _review_prompt_json(evolution_info, max_chars=12000)
            pre_evolution_role = str(
                evolution_info.get("pre_evolution_role") or pre_evolution_role
            )
            post_evolution_role = str(
                evolution_info.get("post_evolution_role") or post_evolution_role
            )
            initial_reference_mode = (
                evolution_info.get("review_mode") == "initial_reference_single_baseline"
                or evolution_info.get("comparison_available") is False
            )

        # Retain event anchors and compact observations.  Full provider
        # payloads remain in the execution JSONL; duplicating them here would
        # exceed the teacher's provider request limits on long trajectories.
        event_index = [item for item in (capability_events or []) if isinstance(item, dict)]
        event_index_text = _review_prompt_json(event_index, max_chars=48000)
        comparison_context = (
            "No predecessor combo exists for this question. Review only the executed current best. "
            "Do not invent pre/post-evolution differences, candidate changes, or regressions. Explain "
            "the current best's observed fault chain and provide evidence that later diagnosis can use."
            if initial_reference_mode else
            "A current-best reference is available. Judge candidate effects only from observable "
            "differences between the two complete trajectories."
        )
        common_context = f"""## Question
{question}

## Answers and evidence scope
- Correct answer: ({gt_answer})
- Agent answer: ({answer_letter})
- Annotated evidence reference: {time_reference}
- Allowed windows: {evidence_window_text}; gaps between windows are out of scope
- Diagnostic mode: {mode}
- Comparison semantics: {comparison_context}
{cost_line}

## {post_evolution_role} execution trace
{traj_text}

## {pre_evolution_role} control trace (if available)
- Pre-evolution answer: ({reference_answer_letter})
{reference_traj_text if reference_trajectory else "No pre-evolution control trace is available for this question."}

## Windows actually visited by the candidate
{_format_time_windows(student_windows)}

## Runtime evidence events
Only the evidence_event_id values listed below may be cited.
{event_index_text}

## Earlier micro-review (read-only, if available)
{baseline_review_text}

## Evolution metadata (if available)
{evolution_info_text}
"""

        system_content = SYSTEM_PROMPT.replace("{model_registry}", self._model_registry)
        react_trace = []

        gold_prompt = common_context + f"""You are now only performing Phase 1: Gold Path.

Goal:
- Explain why the authoritative answer ({gt_answer}) is correct using only the annotated evidence windows ({time_reference}).
- Use the complete media review below as the audiovisual basis. It contains selected-VLM observations and selected-ASR transcripts for every annotated interval; do not ignore any interval.
- You may use internal structure, subtitle, transcript, audio, or visual-review capabilities for additional detail, but every request must remain inside the annotated evidence windows. Supplemental calls must not replace the complete review.
- Internal ACTION names are protocol details. In final JSON, use capability descriptions such as "video review," "structure-store review," "subtitle review," "audio transcription," or "visual perception."
- Do not analyze the candidate's errors in this phase, and do not inspect media outside the annotated evidence windows.

## Complete authoritative media review (all annotated intervals covered)
{_review_prompt_json(gold_video_evidence, max_chars=36000)}

Output `ACTION: FINISH`, followed by JSON:
```json
{{
  "gold_path": {{
    "teacher_answer": "{gt_answer}",
    "solution_path": ["Step 1", "Step 2"],
    "key_evidence": "Key evidence supporting the correct answer within the authoritative evidence interval",
    "evidence_quality": "strong / medium / weak",
    "remaining_uncertainty": "Explain any unresolved ambiguity within the evidence windows; otherwise use an empty string",
    "gold_channel": {{      "primary": "visual / audio / subtitle / ocr / structured_retrieval / reasoning",
      "secondary": ["visual / audio / subtitle / ocr / structured_retrieval / reasoning"],
      "evidence": "Why the correct solution mainly relies on these information channels",
      "confidence": "high / medium / low"
    }}
  }}
}}
```"""
        gold = self._run_teacher_phase(
            phase_name="gold_path",
            system_content=system_content,
            user_prompt=gold_prompt,
            start_sec=start_sec,
            end_sec=end_sec,
            time_reference=time_reference,
            evidence_windows=evidence_windows,
            student_windows=[],
            max_rounds=5,
            allow_tools=True,
        )
        react_trace.extend(gold.get("react_trace", []))

        student_prompt = common_context + f"""You are now only performing Phase 2: Student Path.

Gold Path results:
{_review_prompt_json(gold, max_chars=18000)}

Goal:
- Reconstruct the reviewed combo's actual behavior: where it searched, which media it inspected, what each capability returned, how evidence moved between modules, how it reasoned, and why it produced its answer.
- {"No predecessor exists, so do not invent a comparison. Fully explain the unique current best's data flow and decision." if initial_reference_mode else "Compare the candidate with the supplied pre-evolution current-best trace step by step: methods, calls, parameters, evidence handoff, and final decision. Do not merely restate candidate outputs."}
- You may inspect the annotated evidence windows and the video clips, structure records, or local transcripts that the candidate actually visited.
- Inspecting a candidate-visited clip is allowed only to explain misleading evidence, an unsupported tool return, or why that clip does not support the authoritative answer.
- Do not search arbitrary new intervals. Do not use candidate-visited clips to challenge the authoritative answer or annotated evidence windows.
- Internal ACTION names are protocol details. In final JSON, use capability-level descriptions such as "candidate-clip video review," "candidate-clip structure-store review," or "trajectory evidence review."

Output `ACTION: FINISH`, followed by JSON:
```json
{{
  "student_path": {{
    "visited_windows": ["Time window actually visited by the candidate"],
    "tool_sequence": ["Candidate capability calls in execution order"],
    "reference_tool_sequence": ["Reference capability calls in order; empty if no reference exists"],
    "observed_evidence": ["Evidence actually obtained by the candidate trace"],
    "misleading_signals": ["Misleading retrieval, structure-building, perception, or reasoning signal"],
    "answer_status": "correct / incorrect",
    "evidence_quality": "strong / medium / weak / misleading",
    "stability": "stable / fragile / overfit_risk",
    "student_channel_usage": {{
      "used_channels": ["visual / audio / subtitle / ocr / structured_retrieval / reasoning"],
      "missing_channel": ["Channel needed for the correct solution but not used by the candidate"],
      "misused_channel": ["Channel whose evidence the candidate interpreted incorrectly"],
      "evidence": "Trace or capability-result evidence for the candidate's actual channel use"
    }}
  }}
}}
```"""
        student = self._run_teacher_phase(
            phase_name="student_path",
            system_content=system_content,
            user_prompt=student_prompt,
            start_sec=start_sec,
            end_sec=end_sec,
            time_reference=time_reference,
            evidence_windows=evidence_windows,
            student_windows=student_windows,
            max_rounds=7,
            allow_tools=True,
        )
        react_trace.extend(student.get("react_trace", []))

        divergence_prompt = common_context + f"""You now perform Phase 3: Divergence Diagnosis.

Gold Path:
        {_review_prompt_json(gold, max_chars=18000)}

Student Path:
        {_review_prompt_json(student, max_chars=18000)}

Goal:
- Tools are no longer called.
- {"Compare the Gold Path with the sole initial reference. Identify the earliest deviation from the correct solution without claiming candidate changes." if initial_reference_mode else "Compare the Gold Path, the pre-evolution reference, and the candidate. Identify the earliest candidate deviation or explain why the new mechanism succeeds."}
- Report multi-module attribution, `fault_steps`, the complete `failure_chain`, and concrete repair mechanisms without hard-coding task answers.
- In `evolved_micro` mode, also assess `evidence_quality`, `stability`, `overfit_signal`, and `preserve_signal`.
- `evolution_info.expected_outcome_role` is a label determined by the true answer change; `causal_effect_point.outcome_role` must be consistent with it.
- Emit a complete `causal_effect_point` for each question. Use `stable_no_failure` only when the answer is correct and no risk is evidenced. Otherwise cite a verifiable failure or invalid point from the real trajectory.
- For a wrong, regressed, or unimproved result, emit at least one `minimal_failure_point`: the earliest repairable deviation, not a general suggestion. `trajectory_step_index` is the numbered tool step in the trajectory; `evidence_event_ids` may cite only supplied IDs; `active_window` must be the candidate's actual access window. A thinking or memory failure with no capability event may use an empty ID list, but it must still bind any relevant tool step.
- For a structure artifact, final reasoning decision, or memory state, use `trace_anchor_kind=structure_artifact`, `reasoning_decision`, or `memory_state` and set `trajectory_step_index=0`. Otherwise use `trace_anchor_kind=tool_execution` with the real tool step.
- `capability_needs` only lists the runtime capabilities that the MFP has triggered or is required for counterfactual verification (such as ocr/asr/vlm/structured_retrieval); when there is an evidence event, it must be consistent with the event capability, and provider/asset failures cannot be written as a repairable mechanism.
- Before emitting JSON, verify every MFP against the Student Path. Copy the exact
  `trajectory_step_index`, `active_window`, `evidence_event_ids`, and capability from
  the same trajectory row. Never cite Gold Path events, Teacher tool calls, unrelated
  steps, or inferred event IDs. For `structure_artifact`, `reasoning_decision`, or
  `memory_state`, set `trajectory_step_index` to 0 and do not fabricate tool events.
  If the Student Path cannot support the anchor, do not claim an MFP; record the
  evidence gap in `fault_evidence`.
- If "evolution information" contains iteration_change / previous_run / current_run:
  1. First determine what changes have occurred in the answers, correctness, evidence, and tool paths from the previous round to the current round;
  2. Even if you answer the current question correctly, you must explain the reasons for the change and stability;
  3. Give an explicit change_cause for fixed/regression/changed_wrong_to_wrong/changed_correct_to_correct/error_changed.

Output ACTION: FINISH and give JSON:
```json
{{
  "divergence_diagnosis": {{
    "root_cause": "One sentence root cause",
    "why_student_answer_failed": "Why the student's answer failed",
    "why_gold_answer_holds": "Why the correct answer holds",
    "channel_misalignment": {{
      "exists": true,
      "description": "Whether the correct solution is misaligned with the information channel used by students",
      "suggested_module": "thinking / memory / video_structuring / localization / perception / none"
    }}
  }},
  "fault_steps": [
    {{      "step": "Step N, student trajectory actions/parameters or reasoning decisions",
      "fault_module": "thinking / memory / video_structuring / localization / perception",
      "fault_type": "Short error type",
      "is_root_cause": true,
      "evidence": "Concrete evidence citing Gold Path / Student Path"
    }}
  ],
  "failure_chain": [
    {{"module": "thinking / memory / video_structuring / localization / perception",
      "role": "root_cause / contributing / downstream_unable_to_recover / not_observed",
      "trace_anchor_kind": "tool_execution / structure_artifact / reasoning_decision / memory_state",
      "trajectory_step_index": 1,
      "active_window": [0.0, 1.0],
      "rationale": "How this module actually causes or amplifies errors in the complete trajectory",
      "blocked_by": "Only fill in the upstream module that blocks it when role=not_observed"
    }}  ],
  "fault_module": "thinking / memory / video_structuring / localization / perception / none",
  "fault_type": "Main cause type",
  "fault_evidence": "Overall attribution summary",
  "module_design_issue": "Module design issue",
  "answer_status": "correct / incorrect",
  "evidence_quality": "strong / medium / weak / misleading",
  "stability": "stable / fragile / overfit_risk",
  "overfit_signal": "Empty if none",
  "preserve_signal": "Empty if none",
  "iteration_change_analysis": {{    "change_type": "fixed / regression / changed_wrong_to_wrong / changed_correct_to_correct / unchanged_wrong / unchanged_correct / error_changed / not_applicable",
    "what_changed": "What specifically changed from the last round to the current round",
    "change_cause": "The most likely cause of the change",
    "evidence_delta": "Difference between two rounds of evidence/retrieval/tool paths",
    "stability_risk": "Stable / fragile / degradation risk / not applicable"
  }},
  "next_evolution_hint": "Single question level guidance for the next round of evolution",
  "causal_effect_point": {{    "outcome_role": "repair_evidence / candidate_correction_evidence / regression_guard / unchanged_failure_evidence / stable_no_failure",
    "candidate_method_delta": "When a reference exists, state the candidate's observed change relative to current best. Otherwise write 'no predecessor comparison; this is the current execution behavior' and do not invent a delta",
    "effect_verdict": "helpful / harmful / no_effect / blocked_upstream / blocked_downstream / not_triggered / stable_no_failure",
    "step": "The candidate trajectory step that took effect or failed",
    "method_or_tool": "Competency-level methods or tools must not be written as Teacher internal tool names",
    "observed_behavior": "Actual returns or actual decisions in the trajectory",
    "verification": "How does the Gold/current-best/candidate tripartite comparison prove the above judgment",
    "causal_scope": "target_module / upstream_module / downstream_module / cross_module"
  }},
  "minimal_failure_points": [
    {{"mfp_id": "mfp_1",
      "root_module": "thinking / memory / video_structuring / localization / perception",
      "failure_kind": "For example ocr_character_loss / retrieval_miss / evidence_not_consumed / option_mapping_error",
      "trace_anchor_kind": "tool_execution / structure_artifact / reasoning_decision / memory_state",
      "trajectory_step_index": 1,
      "evidence_event_ids": ["ev_0001"],
      "capability_needs": ["ocr"],
      "active_window": [0.0, 1.0],
      "observed_output": "Error/insufficient evidence actually returned or parsed by this step",
      "expected_evidence": "The necessary evidence properties supported by Gold Path in the same restricted interval",
      "downstream_effect": "Which reserved module how to consume this output and cause an error",
      "counterfactual_success": "The falsifiable evidence state that candidates must generate/deliver under the same input"
    }}
  ],
  "improvement": {{    "target_module": "thinking / memory / video_structuring / localization / perception",
    "action": "Specific improvement measures",
    "expected_gain": "expected gain"
  }},
  "concrete_repair_actions": [
    {{      "module": "thinking / memory / video_structuring / localization / perception",
      "failure_link": "root_cause / contributing / downstream_unable_to_recover",
      "current_observed_failure": "Cite specific facts about this question's trajectory, module output, or capability event",
      "repair_mechanism": "Common behaviors that should be added within the module or at module handovers; videos, answers or question exceptions cannot be written",
      "input_output_or_handoff": "Canonical status that needs to be read, retained, verified or passed",
      "falsifiable_runtime_signal": "Observable status/event/output in similar runs after repair",
      "generalization_scope": "Applicable evidence/reasoning patterns, not single-question patches"
    }}
  ]
}}
```"""
        divergence = self._run_teacher_phase(
            phase_name="divergence_diagnosis",
            system_content=system_content,
            user_prompt=divergence_prompt,
            start_sec=start_sec,
            end_sec=end_sec,
            time_reference=time_reference,
            evidence_windows=evidence_windows,
            student_windows=student_windows,
            max_rounds=5,
            allow_tools=False,
        )
        react_trace.extend(divergence.get("react_trace", []))

        result = dict(divergence)
        result["gold_path"] = gold.get("gold_path", gold)
        result["student_path"] = student.get("student_path", student)
        result.setdefault("divergence_diagnosis", divergence.get("divergence_diagnosis", {}))
        result.setdefault("fault_steps", [])
        result.setdefault("fault_module", result.get("fault_module", "unknown"))
        result.setdefault("fault_type", result.get("fault_type", ""))
        result.setdefault("fault_evidence", result.get("fault_evidence", result.get("key_evidence_summary", "")))
        result.setdefault("minimal_failure_points", [])
        result.setdefault("concrete_repair_actions", [])
        result.setdefault("key_evidence_summary", result.get("fault_evidence", ""))
        result.setdefault("iteration_change_analysis", result.get("iteration_change_analysis", {}))
        if not result.get("divergence_diagnosis") and (
            result.get("fault_steps") or result.get("fault_evidence") or result.get("key_evidence_summary")
        ):
            result["divergence_diagnosis"] = {
                "fault_module": result.get("fault_module", "unknown"),
                "fault_type": result.get("fault_type", ""),
                "fault_steps": result.get("fault_steps", []),
                "fault_evidence": result.get("fault_evidence", result.get("key_evidence_summary", "")),
                "improvement": result.get("improvement", {}),
            }
        result.setdefault("success", True)
        result["react_trace"] = react_trace
        result["time_reference"] = time_reference
        result["question"] = question
        result["gt_answer"] = gt_answer
        result["agent_answer"] = agent_answer
        result["gold_evidence_coverage"] = gold_video_evidence
        return _sanitize_teacher_output_terms(result, preserve_trace=False)

    def _run_teacher_phase(
        self,
        phase_name: str,
        system_content: str,
        user_prompt: str,
        start_sec: float,
        end_sec: float,
        time_reference: str,
        evidence_windows: list[tuple[float, float]] | None,
        student_windows: list,
        max_rounds: int,
        allow_tools: bool,
    ) -> dict:
        """Run one bounded Teacher ReAct phase and return parsed JSON."""
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]
        tool_call_re = re.compile(
            r'TOOL_CALL:\s*(\w+)\s*\(([^)]*)\)', re.IGNORECASE
        )
        react_trace = []
        last_content = ""
        last_parse_error = ""

        for round_idx in range(max_rounds):
            print(f"     [{phase_name} Round {round_idx + 1}/{max_rounds}]")
            if round_idx == max_rounds - 1:
                messages.append({
                    "role": "user",
                    "content": (
                        "This is the last round of this stage. Please ACTION: FINISH immediately and output JSON at this stage, and do not call the tool again."
                        "JSON must be completely closed; if the content is too long, compress the text and reduce array items, giving priority to ensuring that JSON can be parsed."
                    ),
                })
            try:
                resp = self.llm_client.chat.completions.create(
                    model=TEACHER_LLM_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=_phase_max_tokens(phase_name),
                    extra_body=dict(TEACHER_LLM_REQUEST_OPTIONS),
                )
                content = _extract_content(resp)
            except Exception as e:
                return _sanitize_teacher_output_terms({
                    "success": False,
                    "phase": phase_name,
                    "error": str(e),
                    "react_trace": react_trace,
                }, preserve_trace=False)

            if not content:
                messages.append({"role": "user", "content": "Go ahead and output THOUGHT and ACTION strictly."})
                continue

            messages.append({"role": "assistant", "content": content})
            last_content = content

            if "FINISH" in content.upper() or "```json" in content:
                parsed = _robust_json_extract(content)
                if parsed:
                    parsed["success"] = True
                    parsed["phase"] = phase_name
                    trace = react_trace + [{
                        "phase": phase_name,
                        "round": round_idx + 1,
                        "action": "FINISH",
                        "thought": content,
                    }]
                    parsed["react_trace"] = trace
                    return _sanitize_teacher_output_terms(parsed, preserve_trace=False)
                last_parse_error = "FINISH/JSON detected but JSON was not parseable, likely truncated or malformed"
                react_trace.append({
                    "phase": phase_name,
                    "round": round_idx + 1,
                    "action": "PARSE_FAILED",
                    "error": last_parse_error,
                    "thought": content[-2000:],
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "The previous output contains FINISH/JSON, but the JSON cannot be parsed and may be truncated or incompletely formatted."
                        "Stop explaining and do not call more tools. Output only ACTION: FINISH followed by complete, compact JSON that json.loads can parse."
                        "Compressed field contents, up to 3 items per array, must contain the top-level fields required by this stage."
                    ),
                })
                continue

            m = tool_call_re.search(content)
            if m:
                tool_name = m.group(1).lower()
                raw_args = m.group(2).strip()
                if not allow_tools:
                    obs = "Tool calls are not allowed at this stage. Please output the final JSON based on Gold Path and Student Path."
                else:
                    windows = [] if phase_name == "gold_path" else (student_windows or [])
                    obs = self._dispatch_tool(
                        tool_name,
                        raw_args,
                        start_sec,
                        end_sec,
                        time_reference,
                        evidence_windows=evidence_windows,
                        student_windows=windows,
                        phase_name=phase_name,
                    )
                print(f"     {phase_name}: {tool_name}({raw_args}) -> {len(obs)} chars")
                react_trace.append({
                    "phase": phase_name,
                    "round": round_idx + 1,
                    "action": "TOOL_CALL",
                    "tool_call": f"{tool_name}({raw_args})",
                    "observation": obs[:800] + ("...[truncated]" if len(obs) > 800 else ""),
                    "thought": content,
                })
                messages.append({
                    "role": "user",
                    "content": "The tool returns the result (the long original text has been retained in the review trace, the following is a bounded excerpt): \\n"
                    + str(_review_prompt_value(obs, key="observation")),
                })
            else:
                react_trace.append({
                    "phase": phase_name,
                    "round": round_idx + 1,
                    "action": "THINK",
                    "thought": content,
                })
                messages.append({"role": "user", "content": "Please continue with this stage of analysis, or ACTION: FINISH to output JSON."})

        return _sanitize_teacher_output_terms({
            "success": False,
            "phase": phase_name,
            "error": last_parse_error or "The stage reaches the maximum number of rounds and still does not output parsable JSON.",
            "last_output": _sanitize_teacher_output_terms(last_content[-4000:]),
            "react_trace": react_trace,
        }, preserve_trace=False)

    # ==================================================================
    # ReAct main loop
    # ==================================================================


    # ==================================================================
    # Tool distribution
    # ==================================================================

    def _dispatch_tool(self, tool_name: str, raw_args: str,
                       ref_start: float, ref_end: float,
                       time_reference: str, student_windows: list = None,
                       phase_name: str = "student_path",
                       evidence_windows: list[tuple[float, float]] | None = None) -> str:
        """Parse parameters and call corresponding tools."""
        tool_aliases = {
            "audit_video": "watch_video",
            "audit_detail_video": "detailed_watch",
            "audit_enhanced_video": "enhanced_watch",
            "audit_structure": "browse_struct_db",
            "audit_transcript": "read_transcript",
            "video_review": "watch_video",
            "detail_video_review": "detailed_watch",
            "fine_video_review": "detailed_watch",
            "enhanced_video_review": "enhanced_watch",
            "structure_review": "browse_struct_db",
            "transcript_review": "read_transcript",
            "audio_text_review": "read_transcript",
        }
        tool_name = tool_aliases.get(tool_name, tool_name)
        allow_student_windows = phase_name in (
            "student_path",
            "compare_evolution",
            "combined_review",
        )
        allowed_student_windows = (student_windows or []) if allow_student_windows else []
        evidence_windows = evidence_windows or [(ref_start, ref_end)]
        # First extract the optional query string (content within quotes), then parse the value from the remaining part
        q_match = re.search(r'["\'](.+?)["\']', raw_args)
        # Remove the query string and then extract the value to avoid mismatching of the decimal point in the query.
        args_no_str = re.sub(r'["\'].+?["\']', '', raw_args)
        nums = [float(x) for x in re.findall(r'\d+\.?\d*', args_no_str)]

        def _all_reference_windows(call):
            """Run a no-argument evidence request once for every labelled window.

            This makes the default action complete for disjoint annotations.
            A request with explicit bounds remains a single bounded request and
            is rejected if it attempts to bridge an unlabelled gap.
            """
            if nums or len(evidence_windows) == 1:
                return None
            observations = []
            for idx, (start, end) in enumerate(evidence_windows, 1):
                observations.append(
                    f"[Authority window {idx}/{len(evidence_windows)}: {start:.1f}s-{end:.1f}s]\n"
                    f"{call(start, end)}"
                )
            return "\n\n".join(observations)

        if tool_name == "watch_video":
            query = q_match.group(1) if q_match else None
            all_windows = _all_reference_windows(
                lambda start, end: self._watch_video(start, end, query=query)
            )
            if all_windows is not None:
                return all_windows
            s = nums[0] if len(nums) > 0 else ref_start
            e = nums[1] if len(nums) > 1 else ref_end
            clamped = self._clamp_to_evidence_or_student_window(
                s, e, ref_start, ref_end, time_reference,
                allowed_student_windows, tool_name, phase_name=phase_name,
                evidence_windows=evidence_windows)
            if isinstance(clamped, str):
                return clamped
            s, e = clamped
            return self._watch_video(s, e, query=query)

        elif tool_name == "detailed_watch":
            all_windows = _all_reference_windows(
                lambda start, end: self._detailed_watch(start, min(end, start + DETAIL_MAX_SEC))
            )
            if all_windows is not None:
                return all_windows
            s = nums[0] if len(nums) > 0 else ref_start
            e = nums[1] if len(nums) > 1 else ref_end
            clamped = self._clamp_to_evidence_or_student_window(
                s, e, ref_start, ref_end, time_reference,
                allowed_student_windows, tool_name, phase_name=phase_name,
                evidence_windows=evidence_windows)
            if isinstance(clamped, str):
                return clamped
            s, e = clamped
            # up to 5 seconds
            if e - s > DETAIL_MAX_SEC:
                e = s + DETAIL_MAX_SEC
            return self._detailed_watch(s, e)

        elif tool_name == "seek_forward":
            return self._reject_seek_tool(tool_name, time_reference, ref_start, ref_end)

        elif tool_name == "seek_backward":
            return self._reject_seek_tool(tool_name, time_reference, ref_start, ref_end)

        elif tool_name == "browse_struct_db":
            all_windows = _all_reference_windows(self._browse_struct_db)
            if all_windows is not None:
                return all_windows
            s = nums[0] if len(nums) > 0 else ref_start
            e = nums[1] if len(nums) > 1 else ref_end
            clamped = self._clamp_to_evidence_or_student_window(
                s, e, ref_start, ref_end, time_reference,
                allowed_student_windows, tool_name, phase_name=phase_name,
                evidence_windows=evidence_windows)
            if isinstance(clamped, str):
                return clamped
            s, e = clamped
            return self._browse_struct_db(s, e)

        elif tool_name == "enhanced_watch":
            query = q_match.group(1) if q_match else None
            all_windows = _all_reference_windows(
                lambda start, end: self._enhanced_watch(start, end, query=query)
            )
            if all_windows is not None:
                return all_windows
            s = nums[0] if len(nums) > 0 else ref_start
            e = nums[1] if len(nums) > 1 else ref_end
            clamped = self._clamp_to_evidence_or_student_window(
                s, e, ref_start, ref_end, time_reference,
                allowed_student_windows, tool_name, phase_name=phase_name,
                evidence_windows=evidence_windows)
            if isinstance(clamped, str):
                return clamped
            s, e = clamped
            return self._enhanced_watch(s, e, query=query)

        elif tool_name in ("read_transcript", "listen_audio"):
            all_windows = _all_reference_windows(self._read_transcript)
            if all_windows is not None:
                return all_windows
            s = nums[0] if len(nums) > 0 else ref_start
            e = nums[1] if len(nums) > 1 else ref_end
            clamped = self._clamp_to_evidence_or_student_window(
                s, e, ref_start, ref_end, time_reference,
                allowed_student_windows, tool_name, phase_name=phase_name,
                evidence_windows=evidence_windows)
            if isinstance(clamped, str):
                return clamped
            s, e = clamped
            return self._read_transcript(s, e)

        else:
            return f"Unknown review tool: {tool_name}"

    def _clamp_to_evidence_or_student_window(
        self,
        start_sec: float,
        end_sec: float,
        ref_start: float,
        ref_end: float,
        time_reference: str,
        student_windows: list,
        tool_name: str,
        phase_name: str = "student_path",
        evidence_windows: list[tuple[float, float]] | None = None,
    ):
        """Allow evidence-window inspection and student-visited-window inspection."""
        if phase_name not in ("student_path", "compare_evolution", "combined_review"):
            student_windows = []
        original_start, original_end = start_sec, end_sec

        def _overlap(a_start, a_end, b_start, b_end):
            s = max(a_start, b_start)
            e = min(a_end, b_end)
            return (s, e) if e - s > EVIDENCE_EPS else None

        evidence_windows = evidence_windows or [(ref_start, ref_end)]
        evidence_overlaps = [
            overlap for ref_window in evidence_windows
            if (overlap := _overlap(start_sec, end_sec, ref_window[0], ref_window[1]))
        ]
        if len(evidence_overlaps) == 1:
            evidence_overlap = evidence_overlaps[0]
            s, e = evidence_overlap
            if (
                abs(s - original_start) > EVIDENCE_EPS
                or abs(e - original_end) > EVIDENCE_EPS
            ):
                print(
                    f"       ⚠️ {tool_name} request {original_start:.1f}s-{original_end:.1f}s "
                    f"was clipped to the annotated evidence {time_reference}: {s:.1f}s-{e:.1f}s",
                    flush=True,
                )
            return s, e
        if len(evidence_overlaps) > 1:
            return (
                f"{tool_name} request denied: {original_start:.1f}s-{original_end:.1f}s spans multiple "
                f"disjoint evidence windows {_format_time_windows(evidence_windows)} and would include unannotated gaps. "
                "Request each window separately, or omit the time arguments to review every approved window in order."
            )

        for stu_start, stu_end in student_windows or []:
            stu_overlap = _overlap(start_sec, end_sec, stu_start, stu_end)
            if stu_overlap:
                s, e = stu_overlap
                if (
                    abs(s - original_start) > EVIDENCE_EPS
                    or abs(e - original_end) > EVIDENCE_EPS
                ):
                    print(
                        f"       ⚠️ {tool_name} request {original_start:.1f}s-{original_end:.1f}s "
                        f"was clipped to candidate-visited interval {stu_start:.1f}s-{stu_end:.1f}s: {s:.1f}s-{e:.1f}s",
                        flush=True,
                    )
                return s, e

        return (
            f"{tool_name} request denied in phase {phase_name}. Only approved fragments may be inspected. "
            f"Annotated evidence: {time_reference}; allowed windows: {_format_time_windows(evidence_windows)}. "
            f"The request {original_start:.1f}s-{original_end:.1f}s is outside the approved scope.\n"
            f"Candidate-visited windows allowed in Student Path: {_format_time_windows(student_windows or [])}.\n"
            "Gold Path must use annotated evidence only. Student Path may inspect candidate-visited media or structure records only to explain misleading or unsupported evidence."
        )

    def _reject_seek_tool(self, tool_name: str, time_reference: str,
                          ref_start: float, ref_end: float) -> str:
        """Reject out-of-window search tools in Teacher micro-review."""
        return (
            f"{tool_name} request denied: open-ended video search is not allowed. "
            f"Gold Path is restricted to {time_reference} ({ref_start:.1f}s-{ref_end:.1f}s). "
            "Student Path is restricted to segments actually visited by the candidate. "
            "Treat the answer and evidence annotation as authoritative; do not search for alternative events outside that scope. "
            "To diagnose the candidate, inspect only its recorded clips or structure-store entries."
        )

    # ==================================================================
    # Tool implementation
    # ==================================================================

    def _watch_video(self, start_sec: float, end_sec: float, query: str = None) -> str:
        """Review precisely the approved interval through frames plus ASR."""
        prompt = (
            f"Inspect only the original video's {start_sec:.2f}s-{end_sec:.2f}s interval.\n"
            + (f"Question: {query}\n" if query else "Describe the visuals, actions, visible text, sounds or dialogue, and their chronology.\n")
            + "Report only verifiable observations from this clip. State explicitly when something is not visible or audible. Answer in English."
        )
        return self._review_video_clip(start_sec, end_sec, prompt, TEACHER_ACTIVE_VLM_MODEL, "watch_video", 2048)

    def _read_transcript(self, start_sec: float, end_sec: float) -> str:
        """Read local transcript/ASR text for the requested segment if available."""
        candidate_dirs = [
            os.path.join(self.workspace_dir, "transcripts"),
            os.path.join(self.workspace_dir, "asr"),
            os.path.join(self.workspace_dir, "audio_transcripts"),
        ] + list(self.structure_dirs)
        candidate_files = []
        for directory in candidate_dirs:
            if not os.path.isdir(directory):
                continue
            for suffix in (".jsonl", ".json", ".txt", ".srt", ".vtt"):
                for stem in (f"{self.video_id}", f"{self.video_id}_transcript"):
                    path = os.path.join(directory, f"{stem}{suffix}")
                    if os.path.exists(path):
                        candidate_files.append(path)
                if suffix in (".jsonl", ".json"):
                    candidate_files.extend(
                        sorted(glob.glob(os.path.join(directory, f"{self.video_id}_*{suffix}")))
                    )

        if not candidate_files:
            return (
                "No local transcript or ASR artifact was found for this video. "
                "Audio claims therefore require evidence from the configured bounded ASR review; "
                "images, OCR, structure records, and trajectories are not substitutes for audio."
            )

        snippets = []
        for path in dict.fromkeys(candidate_files):
            try:
                if path.endswith(".jsonl"):
                    snippets.extend(self._read_transcript_jsonl(path, start_sec, end_sec))
                elif path.endswith(".json"):
                    snippets.extend(self._read_transcript_json(path, start_sec, end_sec))
                else:
                    snippets.extend(self._read_transcript_text(path, start_sec, end_sec))
            except Exception as exc:
                snippets.append(f"[Failed to read {os.path.basename(path)}: {exc}]")
            if len("\n".join(snippets)) > 5000:
                break

        text = "\n".join(s for s in snippets if s).strip()
        if not text:
            return (
                f"A transcript artifact exists, but it contains no text matching "
                f"{start_sec:.1f}s-{end_sec:.1f}s."
            )
        return f"Native transcript/ASR snippet for {start_sec:.1f}s-{end_sec:.1f}s:\n{text[:6000]}"

    def _read_transcript_jsonl(self, path: str, start_sec: float, end_sec: float) -> list:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = row.get("start") or row.get("start_sec") or row.get("timestamp_start")
                e = row.get("end") or row.get("end_sec") or row.get("timestamp_end")
                if s is not None and e is not None:
                    try:
                        s, e = float(s), float(e)
                    except (TypeError, ValueError):
                        s, e = None, None
                if s is not None and e is not None and (e < start_sec or s > end_sec):
                    continue
                text = row.get("text") or row.get("transcript") or row.get("caption") or row.get("content") or ""
                if text:
                    out.append(f"- {s if s is not None else '?'}-{e if e is not None else '?'}s: {text}")
        return out

    def _read_transcript_json(self, path: str, start_sec: float, end_sec: float) -> list:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            rows = payload.get("segments") or payload.get("transcript") or payload.get("items") or []
            if isinstance(rows, str):
                return [rows[:6000]]
        else:
            rows = payload
        if not isinstance(rows, list):
            return []
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            s = row.get("start") or row.get("start_sec")
            e = row.get("end") or row.get("end_sec")
            try:
                s = float(s) if s is not None else None
                e = float(e) if e is not None else None
            except (TypeError, ValueError):
                s, e = None, None
            if s is not None and e is not None and (e < start_sec or s > end_sec):
                continue
            text = row.get("text") or row.get("transcript") or row.get("caption") or ""
            if text:
                out.append(f"- {s if s is not None else '?'}-{e if e is not None else '?'}s: {text}")
        return out

    def _read_transcript_text(self, path: str, start_sec: float, end_sec: float) -> list:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read(6000)
        return [f"[{os.path.basename(path)}: unsegmented text]\n{text}"] if text.strip() else []

    def _enhanced_watch(self, start_sec: float, end_sec: float, query: str = None) -> str:
        """A second bounded frame+ASR review with a detail-focused prompt."""
        prompt = (
            f"Review only the original video interval {start_sec:.2f}s-{end_sec:.2f}s.\n"
            + (f"Address this review question: {query}\n" if query else "Inspect brief movements, background details, text, visual changes, and audio cues.\n")
            + "Report only the evidence that can be supported by the fragment, indicating relative times when possible. Answer in English."
        )
        result = self._review_video_clip(start_sec, end_sec, prompt, TEACHER_ACTIVE_ENHANCED_VLM_MODEL, "enhanced_watch", 3000)
        return f"[enhanced_watch model: {TEACHER_ACTIVE_ENHANCED_VLM_MODEL}]\n{result}" if not result.startswith("enhanced_watch failed") else result

    def _observe_complete_time_reference(
        self,
        evidence_windows: list[tuple[float, float]],
        question: str,
    ) -> dict:
        """Observe every labelled interval through selected VLM frames plus ASR.

        A multi-interval annotation is an ordered set, not one broad range.
        Each member gets its own timestamped visual samples and bounded ASR
        request, so neither visual nor audio evidence can cross an unlabelled
        gap.  The returned per-window coverage record is persisted in the
        micro review and rejects a Gold Path with missing visual evidence.
        """
        window_text = _format_time_windows(evidence_windows)
        prompt = (
            "Below are timestamped frames and per-segment audio transcripts for every annotated evidence window. "
            "The windows may be disjoint; no media from unannotated gaps is included.\n"
            f"Annotated windows: {window_text}.\n"
            f"Question: {question}\n"
            "Review each window in order. Report the supported visual, action, text, and audio evidence, and state whether that window contributes relevant evidence. "
            "If an observation is unclear or inaudible, say so explicitly. Do not infer information from outside the supplied windows. Answer in English."
        )
        coverage = self._review_media_windows(
            evidence_windows=evidence_windows,
            prompt=prompt,
            tool_name="gold_time_reference_video",
            max_tokens=3000,
        )
        return coverage

    def _review_video_clip(self, start_sec: float, end_sec: float, prompt: str,
                           model: str, tool_name: str, max_tokens: int) -> str:
        """Review one bounded interval through the formal frame+ASR route."""
        if model != TEACHER_ACTIVE_VLM_MODEL:
            return f"{tool_name} failed: unregistered review VLM model {model!r}."
        result = self._review_media_windows(
            evidence_windows=[(float(start_sec), float(end_sec))],
            prompt=prompt,
            tool_name=tool_name,
            max_tokens=max_tokens,
            detail=(tool_name in {"detailed_watch", "enhanced_watch"}),
        )
        return str(result.get("observation") or result.get("error") or f"{tool_name}: provider returned an empty response.")

    def _review_media_windows(self, evidence_windows: list[tuple[float, float]],
                              prompt: str, tool_name: str, max_tokens: int,
                              detail: bool = False) -> dict:
        """Run selected VLM frame evidence and ASR evidence per window.

        This is deliberately not a video upload adapter.  It materializes
        timestamped frames and WAV audio for each approved interval, submits
        them through the formal execution runtime adapters, and preserves the
        individual success/failure state in the returned coverage record.
        """
        normalized = []
        for item in evidence_windows or []:
            try:
                start, end = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if end > start:
                normalized.append((start, end))
        coverage = {
            "success": False,
            "model": TEACHER_ACTIVE_VLM_MODEL,
            "asr_model": TEACHER_ASR_MODEL,
            "visual_profile_id": DEFAULT_PROFILE_IDS["vlm"],
            "asr_profile_id": DEFAULT_PROFILE_IDS["asr"],
            "transport": "time_reference_frames_and_audio_segments",
            "time_reference_windows": [list(item) for item in normalized],
            "coverage_complete": False,
            "window_evidence": [],
        }
        if not normalized:
            coverage["error"] = "time_reference has no valid interval"
            return coverage
        try:
            runtime_utils = self._review_runtime_utils()
        except Exception as exc:
            coverage["error"] = f"{tool_name} failed: unable to load the configured VLM/ASR adapter: {exc}"
            return coverage

        observations = []
        visual_failures = []
        for index, (start_sec, end_sec) in enumerate(normalized, 1):
            frame_budget = (
                REVIEW_DETAIL_FRAMES_PER_WINDOW if detail
                else min(self._watch_frame_budget(start_sec, end_sec), REVIEW_FRAMES_PER_WINDOW)
            )
            frame_paths, timestamps = self._sample_frame_paths(
                start_sec, end_sec, max_frames=max(1, frame_budget)
            )
            record = {
                "window_index": index,
                "time_range": [start_sec, end_sec],
                "frame_timestamps": [round(item, 3) for item in timestamps],
                "frame_count": len(frame_paths),
                "visual_status": "not_called",
                "asr_status": "not_called",
            }

            audio_text, audio_status = self._transcribe_review_window(
                runtime_utils, start_sec, end_sec
            )
            record["asr_status"] = audio_status
            record["asr_transcript"] = audio_text

            if not frame_paths:
                record["visual_status"] = "frame_extraction_failed"
                visual_failures.append(f"{start_sec:.2f}s-{end_sec:.2f}s: failed to extract frame")
                coverage["window_evidence"].append(record)
                continue

            timestamp_text = ", ".join(f"{item:.2f}s" for item in timestamps)
            visual_prompt = (
                f"{prompt}\n\n"
                f"Current review window {index}/{len(normalized)}: {start_sec:.2f}s-{end_sec:.2f}s.\n"
                f"The following frames are arranged in chronological order, corresponding to timestamps: [{timestamp_text}].\n"
                f"ASR status for the same window: {audio_status}.\n"
                f"ASR transcript for the same window: {audio_text[:5000]}\n"
                "Use only these timestamped frames and this window's transcript. "
                "Do not transfer evidence across windows or infer unsampled content."
            )
            print(
                            f"       {tool_name} [{start_sec:.2f}s-{end_sec:.2f}s] "
                f"{len(frame_paths)} frames with same-window ASR",
                flush=True,
            )
            visual_text = runtime_utils.call_vlm(
                visual_prompt,
                frame_paths,
                model_name=TEACHER_ACTIVE_VLM_MODEL,
                temperature=0.1,
                max_tokens=max_tokens,
                profile_id=DEFAULT_PROFILE_IDS["vlm"],
            )
            record["visual_observation"] = visual_text
            if self._is_vlm_failure(visual_text):
                record["visual_status"] = "api_error"
                visual_failures.append(f"{start_sec:.2f}s-{end_sec:.2f}s: {visual_text}")
            else:
                record["visual_status"] = "ok"
            coverage["window_evidence"].append(record)
            observations.append(
                f"[Authority window {index}/{len(normalized)}: {start_sec:.2f}s-{end_sec:.2f}s]\n"
                f"ASR({audio_status}): {audio_text}\n"
                f"Visual observation: {visual_text}"
            )

        coverage["coverage_complete"] = not visual_failures and len(coverage["window_evidence"]) == len(normalized)
        if not coverage["coverage_complete"]:
            coverage["error"] = f"{tool_name} lacks complete visual evidence: " + " | ".join(visual_failures)
            return coverage
        coverage["success"] = True
        coverage["observation"] = "\n\n".join(observations)
        return coverage

    @staticmethod
    def _is_vlm_failure(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        return (not normalized) or normalized.startswith((
            "error calling vlm api", "vlm returned empty", "vlm error:",
        ))

    def _sample_frame_paths(self, start_sec: float, end_sec: float,
                            max_frames: int) -> tuple[list[str], list[float]]:
        """Extract chronological frame files only inside one review interval."""
        duration = max(0.0, end_sec - start_sec)
        if duration <= 0:
            return [], []
        count = max(1, min(int(max_frames), max(1, int(duration) + 1)))
        step = duration / count
        paths, timestamps = [], []
        for index in range(count):
            timestamp = start_sec + (index + 0.5) * step
            frame_path = self._extract_frame(timestamp)
            if frame_path:
                paths.append(frame_path)
                timestamps.append(timestamp)
        return paths, timestamps

    def _transcribe_review_window(self, runtime_utils, start_sec: float,
                                  end_sec: float) -> tuple[str, str]:
        """Extract and transcribe only one review window's audio."""
        audio_path = os.path.join(
            self.cache_dir, f"review_asr_{start_sec:.3f}_{end_sec:.3f}.wav"
        )
        if not self._extract_review_audio(start_sec, end_sec, audio_path):
            return "[Audio is unavailable for this interval.]", "unavailable"
        transcript = runtime_utils.call_asr(
            audio_path,
            model_name=TEACHER_ASR_MODEL,
            profile_id=DEFAULT_PROFILE_IDS["asr"],
        )
        if str(transcript).startswith("Error calling ASR API"):
            return str(transcript), "api_error"
        return str(transcript or "[NO SPEECH DETECTED]"), "ok"

    def _extract_review_audio(self, start_sec: float, end_sec: float,
                              audio_path: str) -> bool:
        """Materialize a WAV from exactly one bounded review window."""
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 44:
            return True
        try:
            completed = subprocess.run(
                [
                    self._ffmpeg_binary(), "-y", "-ss", str(start_sec),
                    "-to", str(end_sec), "-i", self.video_path,
                    "-map", "0:a:0?", "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", audio_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=False,
            )
            return completed.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 44
        except Exception:
            return False


    def _watch_frame_budget(self, start_sec: float, end_sec: float) -> int:
        """Bound VLM image payloads for long authority intervals."""
        duration = max(0.0, end_sec - start_sec)
        if duration > LONG_WATCH_THRESHOLD_SEC:
            return LONG_WATCH_MAX_FRAMES
        return WATCH_MAX_FRAMES

    def _detailed_watch(self, start_sec: float, end_sec: float) -> str:
        """Dense bounded frame+ASR review for a short detail interval."""
        duration = end_sec - start_sec
        if duration > DETAIL_MAX_SEC:
            return (
                f"Error: detailed_watch accepts intervals up to {DETAIL_MAX_SEC:.0f} seconds. "
                f"The requested interval is {duration:.1f} seconds "
                f"({start_sec:.1f}s-{end_sec:.1f}s). For example, call "
                f"detailed_watch({start_sec:.1f}, {start_sec + DETAIL_MAX_SEC:.1f})."
            )
        prompt = (
            f"Inspect only the original video interval {start_sec:.2f}s-{end_sec:.2f}s.\n"
            "Describe the visible evidence moment by moment:\n"
            f"1. Screen content and action sequence\n"
            f"2. All text appearing on the screen (accurate reading)\n"
            f"3. Character action details\n"
            f"4. Any numbers, logos, or items\n"
            f"Answer in English and be as precise as possible."
        )

        return self._review_video_clip(start_sec, end_sec, prompt, TEACHER_ACTIVE_VLM_MODEL, "detailed_watch", 2048)

    def _browse_struct_db(self, start_sec: float, end_sec: float) -> str:
        """Inspect structured-video records for the requested interval.

        Matching records appear first, followed by remaining records in time
        order. The result exposes canonical narration, screen text, ASR, and
        subject registry fields so the reviewer can assess structure quality.
        """
        jsonl_path = ""
        checked = []
        for struct_dir in self.structure_dirs:
            pattern = os.path.join(struct_dir, f"{self.video_id}_*.jsonl")
            candidates = sorted(glob.glob(pattern))
            checked.append(pattern)
            for candidate in candidates:
                if os.path.exists(candidate):
                    jsonl_path = candidate
                    break
            if jsonl_path:
                break

        if not jsonl_path:
            return "Structured database file does not exist, checked:" + ", ".join(checked[:6])

        # Streaming filters entries that overlap with [start_sec, end_sec] to avoid reading large jsonl in its entirety.
        overlapping = []
        total_count = 0
        first_start = None
        last_end = None
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                e_start = e.get("start_sec", 0)
                e_end = e.get("end_sec", 0)
                total_count += 1
                if first_start is None:
                    first_start = e_start
                last_end = e_end
                if e_start < end_sec and e_end > start_sec:
                    overlapping.append(e)
                if len(overlapping) >= 120:
                    # Keep counting would require scanning the whole file.
                    # For Teacher context, a lower-bound total and coverage are
                    # enough and avoid large jsonl full reads.
                    pass

        # Sort by start_sec
        overlapping.sort(key=lambda x: x.get("start_sec", 0))

        if not overlapping:
            coverage = (
                f", spanning approximately {float(first_start or 0):.0f}s-{float(last_end or 0):.0f}s"
                if total_count else ""
            )
            return (
                f"No structure record overlaps {start_sec:.0f}s-{end_sec:.0f}s. "
                f"Scanned {total_count} record(s){coverage}."
            )

        # Formatted output
        print(f"       structure_review [{start_sec:.0f}s-{end_sec:.0f}s] matched {len(overlapping)} record(s)", flush=True)

        result_parts = [
            f"## Structure records [{start_sec:.0f}s - {end_sec:.0f}s]",
            f"Matched {len(overlapping)} of {total_count} scanned record(s).\n"
        ]

        for i, e in enumerate(overlapping):
            e_start = e.get("start_sec", 0)
            e_end = e.get("end_sec", 0)
            embedded = self._coerce_structured_json_field(
                e.get("multimodal_narration", e.get("raw_multimodal_narration", "")),
                expected="dict",
            )
            desc = (
                e.get("description")
                or embedded.get("description")
                or e.get("multimodal_narration")
                or "(no description)"
            )
            screen_text = e.get("screen_text") or embedded.get("screen_text") or []
            asr_text = e.get("asr_text") or embedded.get("transcript_summary") or ""
            registry_raw = e.get("subject_registry", embedded.get("subject_registry", {}))
            registry = self._coerce_structured_json_field(registry_raw, expected="dict")

            # Truncate overly long descriptions (retain key information)
            if len(desc) > 600:
                desc = desc[:600] + "...[truncated]"

            part = f"### [{e_start:.0f}s - {e_end:.0f}s]\n"
            part += f"**Description:** {desc}\n"

            if screen_text:
                part += f"**Screen Text:** {self._short_struct_text(screen_text, 500)}\n"
            if asr_text:
                part += f"**ASR / Transcript:** {self._short_struct_text(asr_text, 500)}\n"

            if registry:
                part += "**Subject Registry:**\n"
                for subj_key, subj_info in registry.items():
                    if isinstance(subj_info, dict):
                        name = subj_info.get("name", subj_key)
                        appearance = subj_info.get("appearance", "")
                        actions = subj_info.get("actions", "")
                        part += f"  - {name}: {self._short_struct_text(appearance, 150)}"
                        if actions:
                            part += f" | Actions: {self._short_struct_text(actions, 150)}"
                        part += "\n"
                    else:
                        part += f"  - {subj_key}: {self._short_struct_text(subj_info, 220)}\n"
            elif registry_raw:
                part += (
                    "**Subject Registry (unparsed text):** "
                    f"{self._short_struct_text(registry_raw, 500)}\n"
                )

            result_parts.append(part)

        result = "\n".join(result_parts)
        # Total output does not exceed 4000 characters to avoid context explosion
        if len(result) > 4000:
            result = result[:4000] + f"\\n\\n...[Output truncated; {len(overlapping)} matching records]"

        return result

    @staticmethod
    def _coerce_structured_json_field(value, expected: str = "dict"):
        """Normalize structured JSONL values for review rendering.

        Native dict/list fields remain preferred. Fenced or serialized values
        are decoded when present so review diagnostics can report the actual
        producer output.
        """
        if isinstance(value, str):
            text = value.strip()
            # Generated MetaVideoAgent records commonly store a JSON evidence card
            # inside a Markdown ```json fence.  Strip only the outer fence;
            # a malformed payload remains plain text rather than being
            # silently fabricated into an empty object.
            if text.startswith("```") and text.endswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1]).strip() if len(lines) >= 2 else ""
            if text and text[0] in "[{":
                try:
                    value = json.loads(text)
                except Exception:
                    # Preserve malformed text only for callers that explicitly
                    # accept a scalar. Typed callers receive an empty container.
                    value = text
            else:
                value = text
        if expected == "dict":
            return value if isinstance(value, dict) else {}
        if expected == "list":
            return value if isinstance(value, list) else []
        return value

    @staticmethod
    def _short_struct_text(value, limit: int = 180) -> str:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value or "")
        return text[:limit] + ("...[truncated]" if len(text) > limit else "")


    # ==================================================================
    # Final attribution fallback
    # ==================================================================


    # ==================================================================
    # Low-level tools
    # ==================================================================

    @staticmethod
    def _ffmpeg_binary() -> str:
        """Resolve the ffmpeg binary used for bounded review audio extraction."""
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def _extract_frame(self, timestamp: float) -> str:
        """Extract a single frame from the video and cache it to disk."""
        fname = f"frame_{timestamp:.2f}.jpg"
        fpath = os.path.join(self.cache_dir, fname)

        if os.path.exists(fpath):
            return fpath

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()

        if ret:
            h, w = frame.shape[:2]
            if max(h, w) > 384:
                scale = 384.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            cv2.imwrite(fpath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return fpath
        return ""

    # ==================================================================
    # Result build
    # ==================================================================

    def _fail(self, reason: str) -> dict:
        return {
            "error": reason,
            "key_evidence_summary": reason,
            "fault_module": "unknown",
            "fault_type": "",
            "fault_steps": [],
            "fault_evidence": reason,
            "module_design_issue": "",
            "success": False,
        }

    # ==================================================================
    # evolutionary contrast model
    # ==================================================================

    def compare_evolution(self, question: str, gt_answer: str, time_reference: str,
                          baseline_trajectory: list, evolved_trajectory: list,
                          reference_answer: str = "", evolved_answer: str = "",
                          evolution_info: dict = None) -> dict:
        """Compare reference and candidate trajectories and attribute their differences.

        Args:
            question: question text (including options)
            gt_answer: Correct answer letter
            time_reference: Annotated evidence interval.
            baseline_trajectory: Reference trajectory.
            evolved_trajectory: Candidate trajectory.
            reference_answer: Reference answer.
            evolved_answer: Candidate answer.
            evolution_info: Candidate metadata and mechanism context.

        Returns:
            dict: contains diff_analysis, regression_cause, evolution_issue, suggestion
        """
        evidence_windows = _parse_time_reference_windows(time_reference)
        if not evidence_windows:
            return {"success": False, "error": "No time_reference", "time_reference": time_reference}
        start_sec, end_sec = evidence_windows[0]

        reference_letter = _extract_answer_letter(reference_answer)
        evolved_letter = _extract_answer_letter(evolved_answer)

        baseline_traj_text = _format_trajectory(baseline_trajectory or [])
        evolved_traj_text = _format_trajectory(evolved_trajectory or [])
        baseline_windows = _extract_student_time_windows(baseline_trajectory or [])
        evolved_windows = _extract_student_time_windows(evolved_trajectory or [])
        compare_student_windows = baseline_windows + evolved_windows

        evo_info_text = ""
        if evolution_info:
            evo_info_text = f"""## Evolution information
- Target module: {evolution_info.get('target_module', '?')}
- Candidate module: {evolution_info.get('module_name', '?')}
- Strategy: {evolution_info.get('strategy', '?')}
"""

        # Match execution-layer semantics, including labelled free-text answers.
        reference_correct = _answer_is_correct(reference_answer, gt_answer)
        evolved_correct = _answer_is_correct(evolved_answer, gt_answer)

        if reference_correct and not evolved_correct:
            change_type = "regression"
        elif not reference_correct and evolved_correct:
            change_type = "correction"
        elif reference_correct and evolved_correct:
            change_type = "stable_correct"
        else:
            change_type = "stable_incorrect"

        init_msg = f"""## Evolution comparison

- Change type: {change_type}
- Correct answer: {gt_answer}
- Reference answer: {reference_letter or 'no answer'} ({'correct' if reference_correct else 'incorrect'})
- Candidate answer: {evolved_letter or 'no answer'} ({'correct' if evolved_correct else 'incorrect'})
- Annotated evidence interval: {time_reference}

{evo_info_text}
## Question

{question}

## Reference trajectory

{baseline_traj_text}

Accessed windows: {_format_time_windows(baseline_windows)}

## Candidate trajectory

{evolved_traj_text}

Accessed windows: {_format_time_windows(evolved_windows)}

Compare the trajectories and determine:
1. Which tool calls, parameters, observations, or reasoning steps changed?
2. For a regression, which change caused it and what side effect was introduced?
3. For a correction, which observed failure did the candidate resolve?
4. What does this result imply for the next evolution decision?"""

        compare_prompt = COMPARE_EVOLUTION_PROMPT.replace(
            "{model_registry}", self._model_registry)

        messages = [
            {"role": "system", "content": compare_prompt},
            {"role": "user", "content": init_msg},
        ]

        tool_call_re = re.compile(
            r'TOOL_CALL:\s*(\w+)\s*\(([^)]*)\)', re.IGNORECASE
        )

        tool_calls_used = 0
        for round_idx in range(MAX_ROUNDS):
            print(f"     [Compare Round {round_idx + 1}/{MAX_ROUNDS}] {time_reference}")

            try:
                resp = self.llm_client.chat.completions.create(
                    model=TEACHER_LLM_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=3000,
                    extra_body=dict(TEACHER_LLM_REQUEST_OPTIONS),
                )
                content = _extract_content(resp)
            except Exception as e:
                print(f"     ❌ LLM call failed: {e}")
                return {"success": False, "error": str(e)}

            if not content:
                messages.append({"role": "user", "content": "Please continue to analyze."})
                continue

            messages.append({"role": "assistant", "content": content})

            # Check if FINISH
            if "FINISH" in content.upper() and "```json" in content:
                result = _robust_json_extract(content)
                if result:
                    result["success"] = True
                    result["change_type"] = change_type
                    result["time_reference"] = time_reference
                    return _sanitize_teacher_output_terms(result, preserve_trace=False)

            # Check tool calls
            m = tool_call_re.search(content)
            if m:
                tool_name, raw_args = m.group(1).lower(), m.group(2).strip()
                if tool_calls_used >= MAX_TOOL_CALLS:
                    messages.append({"role": "user", "content":
                        f"Tool call budget exhausted ({MAX_TOOL_CALLS}/{MAX_TOOL_CALLS}). Please ACTION: FINISH now and output JSON."})
                    continue
                tool_calls_used += 1
                obs = self._dispatch_tool(
                    tool_name, raw_args, start_sec or 0, end_sec or 0, time_reference,
                    student_windows=compare_student_windows,
                    phase_name="compare_evolution",
                    evidence_windows=evidence_windows)
                print(f"     {tool_name}({raw_args}) -> {len(obs)} chars")
                messages.append({
                    "role": "user",
                    "content": "The full tool result is retained in the review trace. Bounded excerpt:\\n"
                    + str(_review_prompt_value(obs, key="observation")),
                })
            else:
                # No tools and no FINISH → urge
                messages.append({"role": "user", "content":
                    "Please output your analysis conclusion. Format: THOUGHT + ACTION: FINISH + JSON."})

        return {"success": False, "error": "The analysis-round limit was reached without a final result.",
                "change_type": change_type, "time_reference": time_reference}

    def review_evolved_trajectory(self, question: str, gt_answer: str,
                                  time_reference: str, evolved_trajectory: list,
                                  evolved_answer: str = "",
                                  evolution_info: dict = None,
                                  baseline_review: dict = None,
                                  reference_trajectory: list = None,
                                  reference_answer: str = "",
                                  capability_events: list = None) -> dict:
        """Review one evolved rollout without re-running baseline micro-review.

        Unlike solve(), this reviews both correct and incorrect evolved answers.
        Correct answers are checked for evidence quality, stability, and
        overfit-risk instead of being skipped.
        """
        evolved_correct = _answer_is_correct(evolved_answer, gt_answer)
        result = self.review_trajectory_three_phase(
            question=question,
            gt_answer=gt_answer,
            time_reference=time_reference,
            trajectory=evolved_trajectory or [],
            agent_answer=evolved_answer,
            mode="evolved_micro",
            baseline_review=baseline_review,
            evolution_info=evolution_info,
            reference_trajectory=reference_trajectory,
            reference_answer=reference_answer,
            capability_events=capability_events,
        )
        result["time_reference"] = time_reference
        result["question"] = question
        result["gt_answer"] = gt_answer
        result["evolved_answer"] = evolved_answer
        result["evolved_correct"] = evolved_correct
        result.setdefault("answer_status", "correct" if evolved_correct else "incorrect")
        result.setdefault("evidence_quality", "weak")
        result.setdefault("stability", "fragile" if evolved_correct else "unstable")
        return result

    # ==================================================================
    # Batch evolution comparison
    # ==================================================================



    def summarize_post_evolution(self, payload: dict) -> dict:
        """Summarize baseline cached reviews + evolved reviews + deltas."""
        def request_macro_json(prompt_text: str, max_tokens: int) -> tuple[dict, str]:
            """Request a bounded macro review, preferring provider JSON mode.

            Some reasoning-model responses contain valid analysis followed by
            an incomplete JSON block. Request JSON mode first, then retry without
            `response_format` when the selected provider does not implement it.
            """
            failures = []
            for strict_json in (True, False):
                kwargs = {
                    "model": TEACHER_LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                    "extra_body": dict(TEACHER_LLM_REQUEST_OPTIONS),
                }
                if strict_json:
                    kwargs["response_format"] = {"type": "json_object"}
                try:
                    resp = self.llm_client.chat.completions.create(**kwargs)
                    content = _extract_content(resp)
                    result = _robust_json_extract(content)
                    if result:
                        return result, ""
                    failures.append(
                        f"strict_json={strict_json}: unparseable response: {content[-600:]}"
                    )
                except Exception as exc:
                    failures.append(f"strict_json={strict_json}: {exc}")
            return {}, " | ".join(failures)[-1800:]

        macro_ledger = payload.get("macro_evidence_ledger") or {
            "review_semantics": (
                (payload.get("candidate_context", {}).get("evolution_info", {}) or {}).get("review_mode", "")
            ),
            "all_question_inventory": [
                {
                    "task_ref": item.get("task_id") or item.get("time_reference", ""),
                    "outcome": item.get("current_best_outcome") or item.get("change_type", ""),
                }
                for item in (payload.get("evolved_micro_reviews", []) or [])
                if isinstance(item, dict)
            ],
            "delta_summary": {},
        }
        review_mode = str(macro_ledger.get("review_semantics") or "")
        initial_reference_mode = review_mode in {
            "unique_initial_current_best", "initial_reference_single_baseline",
        }
        # Do not slice the first JSON bytes.  The ledger is intentionally a
        # complete per-question causal record built by the runner, so a macro
        # conclusion is grounded in all reviewed training questions.
        macro_material = {
            "metrics": payload.get("metrics", {}),
            "reviewed_execution_context": payload.get("reviewed_execution_context", {}),
            "candidate_context": payload.get("candidate_context", {}),
            "current_best_context": payload.get("current_best_context", {}),
            "evaluation_audit": payload.get("evaluation_audit", {}),
            "unchanged_failure_summary": payload.get("unchanged_failure_summary", {}),
            "retention_review": payload.get("retention_review", {}),
            "observed_distribution_profile": payload.get("observed_distribution_profile", {}),
            "macro_evidence_ledger": macro_ledger,
        }
        digest = json.dumps(macro_material, ensure_ascii=False, indent=2, default=str)
        semantics_instruction = (
            "This is a health review of the initial current best. No predecessor or candidate delta exists, "
            "so do not use repair, regression, or candidate-retention semantics. Focus on cross-question "
            "failure chains, module-handoff defects, and verifiable improvement opportunities."
            if initial_reference_mode else
            "This round includes an explicit current-best/candidate comparison. Ground every claim in "
            "ledger trajectories and explain gains, regressions, and unchanged failures relative to the current best."
        )
        reviewed_rows_label = (
            "current_best_micro_reviews (complete per-question reviews of the unique current best)"
            if initial_reference_mode else
            "evolved_micro_reviews (complete per-question reviews of this round's candidate)"
        )
        prompt = f"""You are the macro trajectory reviewer for MetaVideoAgent.

Generate a post-evolution macro review from the material below.

- Treat the baseline per-question reviews as read-only completed evidence.
- Review `{reviewed_rows_label}`.
- {"No delta reviews exist because there is no predecessor." if initial_reference_mode else "Delta reviews contain only corrections and regressions; use them to explain what changed and why."}
- The macro review supplies evidence to DiagnosisAgent. Do not choose the next module, prescribe code changes, or emit planning fields such as `recommended_evolution_decision`, `implementation_contract`, or `evolution_hypothesis`.
- {semantics_instruction}
- {"Set candidate retention to `not_applicable_no_predecessor`, with empty correction and regression references." if initial_reference_mode else "Candidate-retention fields may report observed evidence but must not replace later diagnosis decisions."}
- Evaluate cost and convergence from `evaluation_audit` and `metrics.cost_audit`. Flag repeated, non-convergent, or unnecessarily expensive structure retrieval, visual inspection, or VLM calls. When an issue code is `cost_explosion`, `non_convergent_tool_loop`, `expensive_wrong_answer`, or `expensive_correction`, require preservation of accuracy gains while reducing duplicate or ineffective work.
- `unchanged_failure_summary` describes still-wrong questions. Identify distinct failure groups and their evidenced module responsibility; do not report only a count.
- Every `*_refs` entry must be a real `task_id` or `time_reference` from the material. Never put generic mechanism descriptions in reference fields. High or medium module responsibility requires at least one real evidence reference and a concise rationale.
- Keep stable fixes, fragile fixes, regressions, unchanged failures, and cost risks mutually distinguishable. Do not place one correction or regression in conflicting buckets.
- Keep output compact: at most five items per list, at most five entries per `*_refs`, and at most two sentences per module-responsibility rationale. Prefer representative evidence over exhaustive enumeration.
- `observed_distribution_profile` contains five-frame training-video observations and associated questions/options. It never contains answers, evidence annotations, task traces, or inferred genre labels. Describe observable channel properties and failure patterns without naming or guessing video genres.
- Output both `distribution_evidence_profile` and `distribution_level_failure_pattern`.
- Do not expose internal Teacher or historical runtime tool names as future callable tools. Use capability-level descriptions; the active combo's registry determines actual runtime names.

## Material JSON
{digest}

Output JSON:
```json
{{  "overall_verdict": "accept / reject / conditional",
  "verdict_reason": "One sentence explanation",
  "current_best_failure_summary": {{"scope": "unique_current_best / candidate_comparison",
    "overall_failure_chain": ["Cross-topic failure chain described in module handover sequence"],
    "dominant_observed_defects": ["Must be based on specific defect facts of all ledgers"],
    "evidence_refs": ["real task_id/time_reference"]
  }},
  "bundle_level_improvement_opportunities": [
    {{      "modules": ["One or more modules that can be linked"],
      "observed_pattern": "Current behavioral flaws that can be reviewed across topics",
      "evidence_refs": ["real refs"],
      "handoff_or_state_to_repair": "Status that needs to be retained/transmitted/verified",
      "generic_repair_direction": "General mechanism direction, not single question or hard-coded answer",
      "falsifiable_runtime_signal": "Success signal observable by future probe/smoke"
    }}  ],
  "fix_mechanisms": ["Positive mechanisms brought about by evolution"],
  "regression_mechanisms": ["Degradation/overfitting mechanisms introduced by evolution"],
  "cost_and_efficiency_assessment": {{    "cost_verdict": "efficient / acceptable / expensive_but_useful / wasteful",
    "cost_issue_codes": ["Issue codes from evaluation_audit.issue_summary"],
    "accuracy_cost_tradeoff": "Describe whether the improvement is worth the current cost",
    "next_round_cost_target": "finalize / preserve_accuracy_reduce_cost / repair_expensive_wrong_answers / no_cost_action_needed"
  }},
  "preserve_constraints": ["Effective mechanisms or positive examples that must be preserved in the next round"],
  "avoid_repeating": ["Invalid strategies that must be avoided in the next round"],
  "distribution_evidence_profile": {{    "observed_dominant_channels": ["Information channels seen from observed_distribution_profile and micro-reviews"],
    "channel_misalignment": ["current_best/candidate takes the calculation or retrieval as evidence of the wrong channel"],
    "evidence_interval_pattern": "Short event/long interval summary/mixed",
    "no_human_type_label_used": true
  }},
  "distribution_level_failure_pattern": [
    "Information channel-level failure modes that recur across videos/cross-topics cannot write artificial type tags"
  ],
  "candidate_retention_evidence": {{    "candidate_should_be_preserved_as_evidence": true,
    "stable_correction_refs": ["Stable correction task_id/time_reference"],
    "fragile_correction_refs": ["Maybe fix the problem accidentally task_id/time_reference"],
    "regression_refs": ["regression question task_id/time_reference"],
    "cost_risk_refs": ["High cost or invalid call problem task_id/time_reference"],
    "evidence_only_reason": "Only explain the evidence and do not make the next round of module decisions"
  }},
  "unchanged_failure_analysis": {{
    "taxonomy": [
      "evidence_absent",
      "wrong_temporal_localization",
      "evidence_retrieved_not_used",
      "vlm_perception_miss",
      "option_mapping_error",
      "counting_or_aggregation_error"
    ],
    "counts": {{"wrong_temporal_localization": 0}},
    "refs_by_category": {{"wrong_temporal_localization": []}},
    "module_implications": [
      {{"category": "Error type", "target_module": "Suggested module", "reason": "Why this type of error points to this module"}}
    ]
  }},
  "module_responsibility": {{
    "video_structuring": {{"level": "high / medium / low", "evidence_refs": [], "reason": "basis for liability determination"}},
    "localization": {{"level": "high / medium / low", "evidence_refs": [], "reason": "basis for liability determination"}},
    "perception": {{"level": "high / medium / low", "evidence_refs": [], "reason": "basis for liability determination"}},
    "thinking": {{"level": "high / medium / low", "evidence_refs": [], "reason": "basis for liability determination"}},
    "memory": {{"level": "high / medium / low", "evidence_refs": [], "reason": "basis for liability determination"}}
  }},
  "diagnosis_evidence_seed": {{    "summary": "Summary of evidence that can be injected into the DiagnosisAgent",
    "correction_refs": ["Correction task_id/time_reference"],
    "regression_refs": ["regression question task_id/time_reference"],
    "unchanged_failure_refs_by_category": {{"Error type": ["task_id/time_reference"]}},
    "cost_issue_refs": ["High cost or invalid call issue task_id/time_reference"]
  }}
}}
```"""
        result, primary_error = request_macro_json(prompt, 20000)
        if result:
            result["success"] = True
            return _sanitize_teacher_output_terms(result, preserve_trace=False)
        compact_prompt = f"""You are the macro trajectory reviewer for MetaVideoAgent.

The previous macro-review response could not be parsed. Return one valid JSON object from the compact evidence below.
Requirements:
- Only output JSON, no Markdown.
- Do not expose Teacher audit-token names or runtime implementation names. Use capability-level descriptions.
- Include `module_responsibility`, `diagnosis_evidence_seed`, and `distribution_level_failure_pattern`.
- All refs must come verbatim from task_id/time_reference in compact material; high/medium
  module responsibility must have at least one `evidence_ref`. Distinguish repairs,
  regressions, unchanged failures, and cost risks; never invent refs.
- Use at most five items in each list, including ref lists. Keep the JSON compact.
- Do not emit next-round planning fields such as `recommended_evolution_decision`,
  `implementation_contract`, or `evolution_hypothesis`.

Complete mechanism ledger (review all rows, not only the first few):
{digest}

JSON schema:
{{  "overall_verdict": "accept / reject / conditional",
  "verdict_reason": "One sentence explanation",
  "current_best_failure_summary": {{"scope": "", "overall_failure_chain": [], "dominant_observed_defects": [], "evidence_refs": []}},
  "bundle_level_improvement_opportunities": [{{"modules": [], "observed_pattern": "", "evidence_refs": [], "handoff_or_state_to_repair": "", "generic_repair_direction": "", "falsifiable_runtime_signal": ""}}],
  "candidate_retention_evidence": {{
    "candidate_should_be_preserved_as_evidence": true,
    "stable_correction_refs": [],
    "fragile_correction_refs": [],
    "regression_refs": [],
    "cost_risk_refs": [],
    "evidence_only_reason": ""
  }},
  "distribution_level_failure_pattern": ["Cross-topic failure pattern"],
  "unchanged_failure_analysis": {{
    "taxonomy": [],
    "counts": {{}},
    "refs_by_category": {{}},
    "module_implications": [
      {{"category": "Error type", "target_module": "Suggested module", "reason": "Based on"}}
    ]
  }},
  "module_responsibility": {{
    "video_structuring": {{"level": "high / medium / low", "evidence_refs": [], "reason": ""}},
    "localization": {{"level": "high / medium / low", "evidence_refs": [], "reason": ""}},
    "perception": {{"level": "high / medium / low", "evidence_refs": [], "reason": ""}},
    "thinking": {{"level": "high / medium / low", "evidence_refs": [], "reason": ""}},
    "memory": {{"level": "high / medium / low", "evidence_refs": [], "reason": ""}}
  }},
  "diagnosis_evidence_seed": {{
    "summary": "",
    "correction_refs": [],
    "regression_refs": [],
    "unchanged_failure_refs_by_category": {{}},
    "cost_issue_refs": []
  }}
}}"""
        result, compact_error = request_macro_json(compact_prompt, 20000)
        if result:
            result["success"] = True
            result["compact_retry"] = True
            return _sanitize_teacher_output_terms(result, preserve_trace=False)
        return {
            "success": False,
            "error": "Unable to parse post evolution macro comment JSON",
            "primary_request_error": primary_error,
            "compact_retry_error": compact_error,
        }
