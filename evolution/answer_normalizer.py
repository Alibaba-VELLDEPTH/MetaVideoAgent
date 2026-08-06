"""Answer normalization and judging helpers for heterogeneous QA formats.

The execution layer should keep a stable final-answer outlet, but benchmarks
still differ: some store gold answers as option labels, some as option text,
and some datasets store gold text under ``task_meta.answer`` while
choices may span A-G or beyond.  This module centralizes answer parsing so
evolution reports, probes, and reviews do not depend on stale A-D assumptions.
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, Iterable, List, Tuple

LETTERS = string.ascii_uppercase


def canonical_text(value: Any) -> str:
    """Normalize text for equality checks without semantic expansion."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_choices(task_or_meta: Dict[str, Any] | None = None,
                    choices: Iterable[Any] | None = None) -> List[str]:
    if choices is not None:
        return [str(item).strip() for item in choices if str(item).strip()]
    task_or_meta = task_or_meta or {}
    meta = task_or_meta.get("task_meta") if isinstance(task_or_meta.get("task_meta"), dict) else {}
    for source in (task_or_meta, meta):
        for key in ("choices", "options"):
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, dict):
                ordered = []
                for label in LETTERS:
                    if label in value:
                        ordered.append(str(value[label]).strip())
                    elif label.lower() in value:
                        ordered.append(str(value[label.lower()]).strip())
                if ordered:
                    return [item for item in ordered if item]
    return []


def extract_gold_answer(task_or_meta: Dict[str, Any] | None = None,
                        explicit_gt: Any = None) -> str:
    if explicit_gt not in (None, ""):
        return str(explicit_gt).strip()
    task_or_meta = task_or_meta or {}
    meta = task_or_meta.get("task_meta") if isinstance(task_or_meta.get("task_meta"), dict) else {}
    # Result rows use top-level ``answer`` for the model prediction while the
    # official answer lives in task_meta.answer.  Plain task rows may use
    # top-level answer as gold.  Prefer nested task metadata whenever present.
    sources = (meta, task_or_meta) if meta else (task_or_meta,)
    for source in sources:
        if not isinstance(source, dict):
            continue
        keys = ("gt_answer", "answer", "gold_answer", "correct_answer")
        if source is task_or_meta and meta:
            keys = ("gt_answer", "gold_answer", "correct_answer")
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return str(value).strip()
    return ""


def parse_answer_labels(answer: Any) -> Tuple[List[str], str]:
    """Return explicit option labels and remaining text.

    Supports "A. foo", "(B) bar", "A; C", "Answer: G", etc.  The regex is
    intentionally anchored around answer-like spans to avoid scanning arbitrary
    option lists in long observations.
    """
    text = str(answer or "").strip()
    if not text:
        return [], ""

    labels: List[str] = []
    remainder = text
    # Do not treat the leading article in an option surface (for example
    # ``A gun`` or ``a bat``) as option label ``A``.  A bare label is still
    # accepted, as are explicitly-labelled forms such as ``A. foo`` and
    # ``Answer: A foo``.  The previous case-insensitive whitespace branch
    # conflated ordinary answer text with labels and could mark two different
    # options as equal solely because both started with "A".
    first = re.match(
        r"^\s*(?:(?P<prefix>[Aa]nswer\s*(?:is|:)?\s*))?\(?([A-Z])\)?"
        r"(?:\s*[\).:：-]\s*(.*)|$)",
        text,
    )
    if not first and re.match(r"^\s*[Aa]nswer\s*(?:is|:)?\s*\(?[A-Z]\)?\s+", text):
        # An explicit "Answer: A explanation" remains unambiguous even
        # without punctuation after the label.
        first = re.match(
            r"^\s*(?:[Aa]nswer\s*(?:is|:)?\s*)\(?([A-Z])\)?\s+(.*)$",
            text,
    )
    if first:
        # The two accepted patterns above use different group layouts.
        if first.re.groups >= 3:
            label, remainder_group = first.group(2), first.group(3)
        else:
            label, remainder_group = first.group(1), first.group(2)
        labels.append(label.upper())
        remainder = (remainder_group or "").strip()

    multi_prefix = re.match(
        r"^\s*(?:answer\s*(?:is|:)?\s*)?((?:\(?[A-Z]\)?\s*(?:[,;/&]|and|\s)\s*){1,}\(?[A-Z]\)?)\s*$",
        text,
    )
    if multi_prefix:
        labels = [m.group(1).upper() for m in re.finditer(r"\(?([A-Z])\)?", multi_prefix.group(1), re.I)]
        remainder = ""

    # Also support multiple labelled fragments: "A. cup; C. book".
    labelled_fragments = re.findall(r"(?:^|[;,]\s*)\(?([A-Z])\)?\s*[\).:：-]\s*([^;,]+)", text)
    if labelled_fragments:
        labels = []
        fragments = []
        for label, fragment in labelled_fragments:
            label = label.upper()
            if label not in labels:
                labels.append(label)
            fragments.append(fragment.strip())
        remainder = "; ".join(fragments)

    return labels, remainder


def labels_to_texts(labels: Iterable[str], choices: List[str]) -> List[str]:
    texts = []
    for label in labels:
        idx = LETTERS.find(str(label or "").upper())
        if 0 <= idx < len(choices):
            texts.append(choices[idx])
    return texts


def normalize_answer(answer: Any,
                     *,
                     question: str = "",
                     choices: Iterable[Any] | None = None,
                     task: Dict[str, Any] | None = None) -> Dict[str, Any]:
    choice_list = extract_choices(task, choices=choices)
    raw = str(answer or "").strip()
    labels, remainder = parse_answer_labels(raw)
    selected_texts = labels_to_texts(labels, choice_list)
    normalized_text = remainder or raw
    if selected_texts and (not remainder or canonical_text(remainder) == canonical_text(selected_texts[0])):
        normalized_text = "; ".join(selected_texts)

    answer_type = "choice" if labels or selected_texts else "free_text"
    if len(labels) > 1 or len(selected_texts) > 1:
        answer_type = "multi_choice"
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s+\w+)?", canonical_text(normalized_text).replace(" ", " "), re.I):
        answer_type = "count" if not labels else answer_type

    return {
        "raw_answer": raw,
        "normalized_answer": normalized_text,
        "canonical_answer": canonical_text(normalized_text),
        "selected_labels": labels,
        "selected_texts": selected_texts,
        "answer_type": answer_type,
        "choices": choice_list,
        "question": question or ((task or {}).get("question") if isinstance(task, dict) else ""),
    }


def _gold_labels_and_text(gold: str, choices: List[str]) -> Tuple[List[str], str]:
    labels, remainder = parse_answer_labels(gold)
    if labels:
        texts = labels_to_texts(labels, choices)
        return labels, "; ".join(texts) if texts else remainder
    matched_labels = []
    for idx, choice in enumerate(choices):
        # Choice tasks are evaluated by the dataset's exact option surface,
        # not by a lossy text normalizer.  In particular, the former
        # ASCII-only canonicalization can collapse non-Latin options to empty
        # strings and consequently treat every label as correct.
        if str(choice).strip() == str(gold).strip():
            matched_labels.append(LETTERS[idx])
    return matched_labels, gold


def judge_answer(prediction: Any,
                 gold: Any = None,
                 *,
                 task: Dict[str, Any] | None = None,
                 choices: Iterable[Any] | None = None) -> Dict[str, Any]:
    task = task or {}
    choice_list = extract_choices(task, choices=choices)
    gold_text = extract_gold_answer(task, explicit_gt=gold)
    pred_norm = normalize_answer(prediction, choices=choice_list, task=task)
    gold_labels, gold_norm_text = _gold_labels_and_text(gold_text, choice_list)
    pred_labels = pred_norm.get("selected_labels") or []
    pred_texts = pred_norm.get("selected_texts") or []

    correct = False
    reason = "no_gold_or_prediction"
    if gold_text and str(prediction or "").strip():
        if gold_labels and pred_labels:
            correct = set(pred_labels) == set(gold_labels)
            reason = "label_match" if correct else "label_mismatch"
        # A model may return the exact option surface without its label.  This
        # is an exact comparison too; do not convert multilingual text into a
        # heuristic canonical form after the model has answered.
        exact_prediction = str(prediction).strip()
        exact_texts = [str(text).strip() for text in pred_texts]
        if not correct and str(gold_text).strip() in [exact_prediction, *exact_texts]:
            correct = True
            reason = "exact_text_match"
        # Without a choice list, a labelled textual response such as
        # ``D. black`` can only be compared through its parsed answer text.
        # This path is intentionally unavailable for multiple-choice tasks,
        # where option labels are authoritative and ambiguous leading words
        # (``A gun``) must remain ordinary option text.
        if (
            not correct
            and not choice_list
            and pred_norm.get("canonical_answer")
            and pred_norm.get("canonical_answer") == canonical_text(gold_text)
        ):
            correct = True
            reason = "normalized_text_match"

    return {
        "is_correct": bool(correct),
        "reason": reason,
        "gold_answer": gold_text,
        "gold_labels": gold_labels,
        "gold_normalized_answer": gold_norm_text or gold_text,
        "prediction": pred_norm,
    }


def is_answer_parseable(answer: Any,
                        *,
                        task: Dict[str, Any] | None = None,
                        choices: Iterable[Any] | None = None) -> bool:
    norm = normalize_answer(answer, task=task, choices=choices)
    raw = norm.get("raw_answer", "")
    if not raw:
        return False
    choice_list = norm.get("choices") or []
    if choice_list:
        if norm.get("selected_labels") or norm.get("selected_texts"):
            return True
        return any(str(choice).strip() == raw for choice in choice_list)
    return bool(raw)
