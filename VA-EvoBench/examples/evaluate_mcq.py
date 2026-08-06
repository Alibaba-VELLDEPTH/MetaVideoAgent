#!/usr/bin/env python3
"""Evaluate complete VA-EvoBench multiple-choice predictions."""

import argparse
import collections
import json


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Report per-distribution, macro, and micro VA-EvoBench accuracy."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    annotations = read_jsonl(args.annotations)
    predictions = read_jsonl(args.predictions)
    if not annotations:
        raise ValueError("Annotation file is empty")

    expected = {}
    for row in annotations:
        instance_id = row["instance_id"]
        if instance_id in expected:
            raise ValueError(f"Duplicate annotation: {instance_id}")
        expected[instance_id] = row

    submitted = {}
    for row in predictions:
        instance_id = row.get("instance_id")
        prediction = row.get("prediction")
        if instance_id not in expected:
            raise ValueError(f"Unknown instance_id: {instance_id}")
        choices = expected[instance_id]["choices"]
        valid = [chr(ord("A") + index) for index in range(len(choices))]
        if prediction not in valid:
            raise ValueError(
                f"Prediction for {instance_id} must be one of {valid}; got {prediction!r}"
            )
        if instance_id in submitted:
            raise ValueError(f"Duplicate prediction: {instance_id}")
        submitted[instance_id] = prediction

    missing = sorted(set(expected) - set(submitted))
    if missing:
        raise ValueError(
            f"Missing {len(missing)} predictions; first missing ID: {missing[0]}"
        )

    by_distribution = collections.defaultdict(lambda: {"correct": 0, "total": 0})
    for instance_id, annotation in expected.items():
        distribution = annotation["distribution"]
        by_distribution[distribution]["total"] += 1
        by_distribution[distribution]["correct"] += int(
            submitted[instance_id] == annotation["correct_choice"]
        )

    results = {}
    for distribution in sorted(by_distribution):
        counts = by_distribution[distribution]
        results[distribution] = {
            **counts,
            "accuracy_percent": 100.0 * counts["correct"] / counts["total"],
        }

    correct = sum(item["correct"] for item in results.values())
    total = sum(item["total"] for item in results.values())
    macro = sum(item["accuracy_percent"] for item in results.values()) / len(results)
    output = {
        "per_distribution": results,
        "macro_accuracy_percent": macro,
        "micro": {
            "correct": correct,
            "total": total,
            "accuracy_percent": 100.0 * correct / total,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

