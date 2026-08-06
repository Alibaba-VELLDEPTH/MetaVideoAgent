#!/usr/bin/env python3
"""Validate the integrity and protocol invariants of a VA-EvoBench package."""

import argparse
import hashlib
import json
import math
from pathlib import Path

QUESTION_FIELDS = {
    "instance_id",
    "source_dataset",
    "distribution",
    "split",
    "video_id",
    "duration_sec",
    "question_id",
    "question",
    "choices",
    "answer",
    "correct_choice",
    "source_domain",
    "question_category",
    "time_reference",
}
VIDEO_FIELDS = {
    "source_dataset",
    "distribution",
    "split",
    "video_id",
    "overview_timestamps_sec",
}


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_question(row, split, location):
    require(set(row) == QUESTION_FIELDS, f"{location}: unexpected schema")
    require(row["source_dataset"] == "CG-Bench", f"{location}: source mismatch")
    require(row["split"] == split, f"{location}: split mismatch")
    expected_id = f"{row['distribution']}/{row['video_id']}/{row['question_id']}"
    require(row["instance_id"] == expected_id, f"{location}: instance_id mismatch")
    require(isinstance(row["duration_sec"], (int, float)), f"{location}: bad duration")
    require(row["duration_sec"] > 0, f"{location}: non-positive duration")
    require(isinstance(row["question"], str) and row["question"], f"{location}: empty question")
    require(isinstance(row["choices"], list) and len(row["choices"]) >= 2, f"{location}: bad choices")
    require(all(isinstance(choice, str) and choice for choice in row["choices"]), f"{location}: empty choice")
    require(len(set(row["choices"])) == len(row["choices"]), f"{location}: duplicate choices")
    letter = row["correct_choice"]
    require(isinstance(letter, str) and len(letter) == 1, f"{location}: bad answer letter")
    answer_index = ord(letter) - ord("A")
    require(0 <= answer_index < len(row["choices"]), f"{location}: answer letter out of range")
    require(row["choices"][answer_index] == row["answer"], f"{location}: answer mismatch")
    intervals = row["time_reference"]
    require(isinstance(intervals, list) and intervals, f"{location}: empty time_reference")
    for interval in intervals:
        require(isinstance(interval, list) and len(interval) == 2, f"{location}: bad interval")
        require(all(isinstance(x, (int, float)) and math.isfinite(x) for x in interval), f"{location}: non-numeric interval")
        require(0 <= interval[0] <= interval[1] <= row["duration_sec"], f"{location}: interval out of bounds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="VA-EvoBench package root",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("benchmark") == "VA-EvoBench", "unexpected benchmark name")
    require(manifest.get("version") == "1.0.0", "unexpected release version")
    require(
        manifest.get("package_type") == "public_benchmark_release",
        "release manifest is not a public benchmark release",
    )
    require(
        manifest.get("metavideoagent_runtime_integration") is False,
        "benchmark must remain independent from the MetaVideoAgent runtime",
    )
    require((root / "LICENSE").is_file(), "missing LICENSE")
    require((root / "NOTICE.md").is_file(), "missing NOTICE.md")

    criteria = json.loads((root / "selection_criteria.json").read_text(encoding="utf-8"))
    require(criteria.get("benchmark") == manifest["benchmark"], "selection benchmark mismatch")
    require(criteria.get("version") == manifest["version"], "selection version mismatch")
    declared_distributions = {
        item.get("id") for item in criteria.get("distributions", [])
        if isinstance(item, dict)
    }

    questions = {}
    seen_instances = set()
    video_to_split = {}
    video_to_distribution = {}
    video_duration = {}
    observed_counts = {}

    for split in ("evolution", "held_out"):
        path = root / "data" / f"{split}.jsonl"
        rows = load_jsonl(path)
        questions[split] = rows
        for line_number, row in enumerate(rows, 1):
            location = f"{path}:{line_number}"
            validate_question(row, split, location)
            require(row["instance_id"] not in seen_instances, f"{location}: duplicate instance_id")
            seen_instances.add(row["instance_id"])
            video_id = row["video_id"]
            require(video_to_split.get(video_id, split) == split, f"{location}: cross-split video leakage")
            require(video_to_distribution.get(video_id, row["distribution"]) == row["distribution"], f"{location}: cross-distribution video reuse")
            require(video_duration.get(video_id, row["duration_sec"]) == row["duration_sec"], f"{location}: inconsistent duration")
            video_to_split[video_id] = split
            video_to_distribution[video_id] = row["distribution"]
            video_duration[video_id] = row["duration_sec"]
        observed_counts[split] = {
            "videos": len({row["video_id"] for row in rows}),
            "questions": len(rows),
        }

    require(observed_counts == manifest["counts"], "manifest total counts do not match data")

    video_rows = load_jsonl(root / "data" / "videos.jsonl")
    seen_video_rows = set()
    for line_number, row in enumerate(video_rows, 1):
        location = f"data/videos.jsonl:{line_number}"
        require(set(row) == VIDEO_FIELDS, f"{location}: unexpected schema")
        require(row["source_dataset"] == "CG-Bench", f"{location}: source mismatch")
        video_id = row["video_id"]
        require(video_id in video_to_split, f"{location}: video has no questions")
        require(row["split"] == video_to_split[video_id], f"{location}: split mismatch")
        require(row["distribution"] == video_to_distribution[video_id], f"{location}: distribution mismatch")
        require(video_id not in seen_video_rows, f"{location}: duplicate video row")
        seen_video_rows.add(video_id)
        timestamps = row["overview_timestamps_sec"]
        require(isinstance(timestamps, list) and len(timestamps) == 10, f"{location}: expected ten timestamps")
        require(timestamps == sorted(timestamps) and len(set(timestamps)) == 10, f"{location}: timestamps are not unique and ordered")
        require(all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= video_duration[video_id] for x in timestamps), f"{location}: timestamp out of bounds")
    require(seen_video_rows == set(video_to_split), "video manifest membership mismatch")

    distribution_counts = {}
    for distribution in sorted(set(video_to_distribution.values())):
        distribution_counts[distribution] = {}
        for split in ("evolution", "held_out"):
            split_rows = [row for row in questions[split] if row["distribution"] == distribution]
            distribution_counts[distribution][split] = {
                "videos": len({row["video_id"] for row in split_rows}),
                "questions": len(split_rows),
            }
    require(distribution_counts == manifest["distribution_counts"], "manifest distribution counts do not match data")
    require(
        declared_distributions == set(distribution_counts),
        "selection criteria distributions do not match the data",
    )

    for relative_path, expected_hash in manifest["data_files"].items():
        require(sha256(root / relative_path) == expected_hash, f"hash mismatch: {relative_path}")

    print(json.dumps({
        "status": "valid",
        "questions": sum(item["questions"] for item in observed_counts.values()),
        "videos": len(video_to_split),
        "splits": observed_counts,
        "distributions": len(distribution_counts),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
