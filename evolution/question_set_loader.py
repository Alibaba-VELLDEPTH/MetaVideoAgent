"""Question-set loading helpers for isolated MetaVideoAgent runs.

The evolution layer must not implicitly sweep every dataset in the repository
when a MetaVideoAgent or split manifest is provided. This module centralizes the
safe operational fields from distribution manifests and returns sanitized tasks
with stable task_id values.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

try:
    from .distribution_spec import load_distribution_spec
    from .task_identity import make_task_id
    from .task_sanitizer import sanitize_task_for_evolution
except ImportError:
    from distribution_spec import load_distribution_spec
    from task_identity import make_task_id
    from task_sanitizer import sanitize_task_for_evolution


ALL_VIDEO_ALIASES = {"", "ALL", "all", "multi_video"}


def _valid_evidence_interval(value) -> bool:
    """Return whether a task declares at least one positive time interval."""
    pairs = []
    if isinstance(value, dict):
        pairs = [(value.get("start", value.get("start_sec")),
                  value.get("end", value.get("end_sec")))]
    elif isinstance(value, (list, tuple)):
        if len(value) >= 2 and not isinstance(value[0], (list, tuple, dict)):
            pairs = [(value[0], value[1])]
        else:
            for item in value:
                if isinstance(item, dict):
                    pairs.append((item.get("start", item.get("start_sec")),
                                  item.get("end", item.get("end_sec"))))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    pairs.append((item[0], item[1]))
    elif isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        pairs = list(zip(numbers[0::2], numbers[1::2]))
    for start, end in pairs:
        try:
            if float(end) > float(start) >= 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def require_evolution_task_contract(tasks: Sequence[dict], *, require_evidence: bool = True) -> None:
    """Validate evaluation supervision, including Teacher evidence when requested."""
    issues = []
    for index, task in enumerate(tasks):
        missing = []
        if not str(task.get("video_id") or "").strip():
            missing.append("video_id")
        if not str(task.get("question") or "").strip():
            missing.append("question")
        if not task.get("choices"):
            missing.append("choices/options")
        if not str(task.get("gt_answer") or task.get("answer") or "").strip():
            missing.append("gt_answer/answer")
        if require_evidence and not _valid_evidence_interval(task.get("time_reference")):
            missing.append("valid time_reference")
        if missing:
            issues.append(
                f"row {index + 1} ({task.get('task_id', 'unknown')}): " + ", ".join(missing)
            )
    if issues:
        preview = "; ".join(issues[:5])
        suffix = f"; and {len(issues) - 5} more" if len(issues) > 5 else ""
        raise ValueError(
            "Evaluation rows must provide video_id, question, choices/options, and "
            "ground-truth answer; evolution rows also require annotated evidence "
            "intervals: " + preview + suffix
        )


def _default_dataset_dir(workspace: str) -> str:
    return os.path.join(workspace, "..", "datasets")


def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _dataset_files_from_dir(path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(path, "*.jsonl")))


def _dataset_files_from_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        values = data
    elif isinstance(data, dict):
        values = (
            data.get("datasets")
            or data.get("files")
            or data.get("dataset_files")
            or data.get("paths")
            or []
        )
    else:
        values = []
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for item in values:
        value = item.get("path") if isinstance(item, dict) else item
        if not isinstance(value, str) or not value:
            continue
        out.append(value if os.path.isabs(value) else os.path.normpath(os.path.join(base, value)))
    return out


def dataset_files_from_path(path: str) -> List[str]:
    """Resolve a split path into jsonl dataset files.

    Supported inputs:
    - directory containing ``*.jsonl``
    - one ``*.jsonl`` file
    - JSON manifest listing dataset files under datasets/files/dataset_files/paths
    """
    if not path:
        return []
    if os.path.isdir(path):
        return _dataset_files_from_dir(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"question split path not found: {path}")
    if path.endswith(".jsonl"):
        return [path]
    if path.endswith(".json"):
        return _dataset_files_from_json(path)
    raise ValueError(f"unsupported question split path: {path}")


def resolve_manifest_split_paths(distribution_manifest: str,
                                 preferred_splits: Sequence[str]) -> Tuple[List[str], dict]:
    if not distribution_manifest:
        return [], {}
    spec = load_distribution_spec(distribution_manifest)
    splits = spec.get("splits", {}) or {}
    for name in preferred_splits:
        split_path = splits.get(name)
        if split_path:
            return dataset_files_from_path(split_path), spec
    return [], spec


def discover_dataset_files(workspace: str, video_id: str = "ALL",
                           distribution_manifest: str = "",
                           preferred_splits: Sequence[str] = ("test", "validation", "val", "train")) -> Tuple[List[str], dict]:
    files, spec = resolve_manifest_split_paths(distribution_manifest, preferred_splits)
    if files:
        return files, spec

    dataset_dir = _default_dataset_dir(workspace)
    if video_id not in ALL_VIDEO_ALIASES:
        files = [
            os.path.join(dataset_dir, f"{vid.strip()}.jsonl")
            for vid in str(video_id).split(",")
            if vid.strip()
        ]
    else:
        files = _dataset_files_from_dir(dataset_dir)
    missing = [path for path in files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"missing dataset files: {missing}")
    return files, spec


def load_questions(workspace: str, video_id: str = "ALL",
                   distribution_manifest: str = "",
                   preferred_splits: Sequence[str] = ("test", "validation", "val", "train")) -> Tuple[List[dict], List[str], dict]:
    files, spec = discover_dataset_files(workspace, video_id, distribution_manifest, preferred_splits)
    requested_video_ids = {
        vid.strip()
        for vid in str(video_id or "").split(",")
        if vid.strip() and vid.strip() not in ALL_VIDEO_ALIASES
    }
    questions: List[dict] = []
    video_ids = []
    for path in files:
        video_id_from_file = os.path.splitext(os.path.basename(path))[0]
        for raw in _read_jsonl(path):
            raw_video_id = raw.get("video_id") or video_id_from_file
            if requested_video_ids and raw_video_id not in requested_video_ids:
                continue
            if raw_video_id not in video_ids:
                video_ids.append(raw_video_id)
            task = sanitize_task_for_evolution(raw)
            task["video_id"] = task.get("video_id") or raw_video_id
            if not task.get("gt_answer"):
                task["gt_answer"] = (
                    task.get("answer")
                    or task.get("correct_answer")
                    or raw.get("gt_answer")
                    or raw.get("answer")
                    or raw.get("correct_answer")
                    or ""
                )
            if raw.get("task_id"):
                task["task_id"] = str(raw["task_id"])
            else:
                task["task_id"] = make_task_id(
                    task.get("video_id", raw_video_id),
                    task.get("time_reference", ""),
                    task.get("question", ""),
                )
            questions.append(task)
    if tuple(preferred_splits) == ("train",):
        require_evolution_task_contract(questions)
    return questions, video_ids, spec


def select_questions_streaming(
        workspace: str, *, video_id: str = "ALL", distribution_manifest: str = "",
        preferred_splits: Sequence[str] = ("test", "validation", "val", "train"),
        max_questions: int = 1,
        selection_key: Callable[[dict], tuple] | None = None,
) -> Tuple[List[dict], List[str], dict]:
    """Select a small deterministic subset without materializing a split JSONL.

    This is intended for engineering smoke, where reading every task into a
    list merely to execute one or two cases is wasteful.  The files are still
    scanned line-by-line so the selected set is stable regardless of source
    ordering, but memory is bounded by ``max_questions`` per video.  The
    returned tasks deliberately omit ground-truth answers: smoke validates a
    runnable trajectory, not answer correctness.
    """
    limit = max(1, int(max_questions or 1))
    files, spec = discover_dataset_files(
        workspace, video_id, distribution_manifest, preferred_splits,
    )
    requested_video_ids = {
        vid.strip()
        for vid in str(video_id or "").split(",")
        if vid.strip() and vid.strip() not in ALL_VIDEO_ALIASES
    }
    key_fn = selection_key or (lambda task: (str(task.get("task_id") or ""),))
    candidates_by_video: Dict[str, List[dict]] = {}
    seen_video_ids: List[str] = []
    for path in files:
        video_id_from_file = os.path.splitext(os.path.basename(path))[0]
        for raw in _read_jsonl(path):
            raw_video_id = raw.get("video_id") or video_id_from_file
            if requested_video_ids and raw_video_id not in requested_video_ids:
                continue
            task = sanitize_task_for_evolution(raw)
            task["video_id"] = task.get("video_id") or raw_video_id
            if raw.get("task_id"):
                task["task_id"] = str(raw["task_id"])
            else:
                task["task_id"] = make_task_id(
                    task.get("video_id", raw_video_id),
                    task.get("time_reference", ""),
                    task.get("question", ""),
                )
            # A behavioral smoke must never receive answer supervision.
            for key in ("gt_answer", "answer", "correct_answer"):
                task.pop(key, None)
            if raw_video_id not in candidates_by_video:
                candidates_by_video[raw_video_id] = []
                seen_video_ids.append(raw_video_id)
            bucket = candidates_by_video[raw_video_id]
            bucket.append(task)
            bucket.sort(key=key_fn)
            del bucket[limit:]

    # Match the smoke policy: cover distinct videos first, then fill remaining
    # cases from the globally cheapest retained candidates.
    selected: List[dict] = []
    for task in sorted((items[0] for items in candidates_by_video.values() if items), key=key_fn):
        if len(selected) >= limit:
            break
        selected.append(task)
    selected_ids = {id(task) for task in selected}
    for task in sorted(
            (item for items in candidates_by_video.values() for item in items), key=key_fn):
        if len(selected) >= limit:
            break
        if id(task) not in selected_ids:
            selected.append(task)
            selected_ids.add(id(task))
    return selected, seen_video_ids, spec


def questions_by_ref(workspace: str, video_id: str = "ALL",
                     distribution_manifest: str = "",
                     preferred_splits: Sequence[str] = ("test", "validation", "val", "train")) -> Tuple[Dict[str, dict], Dict[str, str], List[str], dict]:
    questions, video_ids, spec = load_questions(
        workspace,
        video_id=video_id,
        distribution_manifest=distribution_manifest,
        preferred_splits=preferred_splits,
    )
    by_ref: Dict[str, dict] = {}
    question_to_ref: Dict[str, str] = {}
    for task in questions:
        ref = task.get("task_id", "")
        if ref:
            by_ref[ref] = task
        q = task.get("question", "")
        if q:
            question_to_ref[q] = ref
    return by_ref, question_to_ref, video_ids, spec
