#!/usr/bin/env python3
"""Public read-only held-out reporting interface for a frozen MetaVideoAgent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evolution.full_eval_runner import main as full_eval_main  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen MetaVideoAgent on the held-out test split."
    )
    parser.add_argument("--distribution-manifest", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--candidate-bundle", required=True)
    parser.add_argument("--train-diagnostic-report", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--structure-workers", type=int, default=4)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    forwarded = [
        "--workspace", args.workspace,
        "--distribution-manifest", args.distribution_manifest,
        "--candidate-run", args.candidate_run,
        "--candidate-bundle", args.candidate_bundle,
        "--train-diagnostic-report", args.train_diagnostic_report,
        "--question-split", "test",
        "--concurrency", str(max(1, args.concurrency)),
        "--structure-workers", str(max(1, args.structure_workers)),
    ]
    if args.api_key:
        forwarded.extend(["--api-key", args.api_key])
    if args.base_url:
        forwarded.extend(["--base-url", args.base_url])
    if args.overwrite:
        forwarded.append("--overwrite")
    if args.check_only:
        forwarded.append("--check-only")
    sys.argv = [sys.argv[0], *forwarded]
    return full_eval_main()


if __name__ == "__main__":
    raise SystemExit(main())
