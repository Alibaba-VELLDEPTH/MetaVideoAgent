# VA-EvoBench

VA-EvoBench contains the annotations and split manifests used to evaluate
distribution-adaptive video-agent evolution. It is distributed as a standalone
benchmark within the MetaVideoAgent source repository and is not imported or
consumed automatically by the MetaVideoAgent package. VA-EvoBench is
derived from CG-Bench and organizes 130 source videos into eight target video
distributions. Every distribution has video-disjoint `evolution` and
`held_out` splits. The package contains 356 evolution questions and 867
held-out questions used in the accompanying MetaVideoAgent evaluation.

## Package contents

- `data/evolution.jsonl`: evolution-split questions and source annotations.
- `data/held_out.jsonl`: held-out questions and source annotations.
- `data/videos.jsonl`: video-level distribution and split membership, plus the
  timestamps used for global manual review.
- `selection_criteria.json`: distribution definitions and curation protocol.
- `release_manifest.json`: version, schema, counts, and data-file hashes.
- `DATASET_CARD.md`: provenance, curation, intended use, limitations, and
  annotation-access rules.
- `NOTICE.md`: upstream rights, access conditions, and privacy notice.
- `LICENSE`: Apache License 2.0 for original VA-EvoBench contributions, subject
  to the third-party scope stated in `NOTICE.md`.
- `scripts/validate_release.py`: dependency-free package validator.
- `examples/evaluate_mcq.py`: dependency-free multiple-choice evaluator.
- `SHA256SUMS`: integrity hashes for every release file except itself.

No video, frame, transcript, model output, execution trajectory, local path,
credential, or author-identifying artifact is included.

## Validate the package

From the package root, run:

```bash
python scripts/validate_release.py
sha256sum -c SHA256SUMS
```

The validator checks JSON syntax, field types, answer/choice consistency,
temporal bounds, unique instance identifiers, video-disjoint splits, manifest
counts, video membership, and data-file hashes.

## Question schema

Each question record contains:

- `instance_id`: stable identifier in the form
  `<distribution>/<video_id>/<question_id>`;
- `source_dataset`: always `CG-Bench`;
- `distribution`: one of the eight distribution identifiers;
- `split`: `evolution` or `held_out`;
- `video_id` and `duration_sec`;
- `question_id`, `question`, and ordered `choices`;
- `answer` and `correct_choice`, where `A` denotes `choices[0]`, `B`
  denotes `choices[1]`, and so forth;
- `source_domain` and `question_category` from the source annotations; and
- `time_reference`, a non-empty list of source-annotated evidence intervals
  `[start_sec, end_sec]` in seconds.

`data/videos.jsonl` contains `source_dataset`, `distribution`, `split`,
`video_id`, and ten monotonically increasing `overview_timestamps_sec` values.

## Annotation-access protocol

The JSONL files contain labels and temporal references so that users can
reproduce and audit the reported evaluation. Their presence does not make them valid
inputs to the evolving agent.

- A Student execution receives only the video, question, and answer options.
- For the evolution split, `answer` and `time_reference` may be revealed only
  to post-execution Teacher review after a Student trajectory is complete.
- Held-out labels, temporal references, predictions, and scores must remain
  inaccessible to profiling, research, diagnosis, code generation, repair,
  candidate promotion, stopping, rollback, and checkpoint selection.
- The primary result is the last-iteration agent, not a checkpoint selected by
  held-out performance.

These rules are part of the benchmark protocol. Systems using held-out
annotations during adaptation are not comparable to compliant VA-EvoBench
results.

## Media access

This directory does not redistribute videos. Obtain the original CG-Bench media
through the [official gated CG-Bench dataset](https://huggingface.co/datasets/CG-Bench/CG-Bench), accept all
upstream access conditions, and resolve records by `video_id`. Upstream media
and annotations remain governed by the CG-Bench terms summarized in
`NOTICE.md`.

## Evaluate predictions

Predictions are JSONL records containing `instance_id` and one uppercase option
letter in `prediction`:

```json
{"instance_id":"interview/0coUiuG5gG4/434","prediction":"D"}
```

Evaluate a complete split with:

```bash
python examples/evaluate_mcq.py \
  --annotations data/held_out.jsonl \
  --predictions predictions.jsonl
```

The evaluator reports correct counts and accuracy for each target distribution,
the macro average over the eight distributions (the primary aggregate metric),
and the question-weighted micro average.

## Relationship to MetaVideoAgent

VA-EvoBench is a separately usable research artifact colocated with the
MetaVideoAgent implementation. No MetaVideoAgent runtime, initialization, or
evolution entry point discovers this directory implicitly. To use it with any
system, explicitly prepare the upstream videos and select the desired JSONL
split under the benchmark's annotation-access protocol.

## Licensing and citation

MetaVideoAgent's original VA-EvoBench curation metadata and utility scripts are
released under the repository's Apache License 2.0. Selected CG-Bench
annotations and all upstream media remain governed by the CG-Bench license,
gated-access agreement, and original media rights described in `NOTICE.md`.
Users should cite both VA-EvoBench/MetaVideoAgent and CG-Bench when publishing
results based on this benchmark.
