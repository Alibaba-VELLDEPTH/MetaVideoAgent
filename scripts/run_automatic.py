#!/usr/bin/env python3
"""Public CLI for the primary Codex-authored multi-round workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from execution.run_evolution import main as evolution_main  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evolve MetaVideoAgent automatically with Codex."
    )
    parser.add_argument("--distribution-manifest", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--auto-rounds", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--first-n", type=int, default=0)
    parser.add_argument("--structure-workers", type=int, default=4)
    parser.add_argument(
        "--codex-cli",
        default="",
        help="Codex executable or a wrapper implementing the same exec command contract.",
    )
    parser.add_argument(
        "--research-codex-cli",
        default="",
        help="Optional compatible executable used for the research stage.",
    )
    parser.add_argument("--codex-model", default="")
    parser.add_argument("--codex-reasoning-effort", default="")
    parser.add_argument("--api-key", default="", help="Provider key; prefer METAVIDEOAGENT_API_KEY.")
    parser.add_argument("--base-url", default="", help="Provider base URL; prefer METAVIDEOAGENT_BASE_URL.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser


def _append_value(argv: list[str], flag: str, value: object) -> None:
    if value not in (None, ""):
        argv.extend([flag, str(value)])


def main() -> int:
    args = _parser().parse_args()
    forwarded = [
        "--distribution-manifest", args.distribution_manifest,
        "--workspace", args.workspace,
        "--stage", "automatic",
        "--auto-rounds", str(max(1, args.auto_rounds)),
        "--concurrency", str(max(1, args.concurrency)),
        "--first-n", str(max(0, args.first_n)),
        "--structure-workers", str(max(1, args.structure_workers)),
    ]
    for flag, value in (
        ("--run-id", args.run_id), ("--output-root", args.output_root),
        ("--codex-cli", args.codex_cli),
        ("--research-codex-cli", args.research_codex_cli),
        ("--codex-model", args.codex_model),
        ("--codex-reasoning-effort", args.codex_reasoning_effort),
        ("--api-key", args.api_key), ("--base-url", args.base_url),
    ):
        _append_value(forwarded, flag, value)
    if args.check_only:
        forwarded.append("--check-only")
    if args.run:
        forwarded.append("--run")
    sys.argv = [sys.argv[0], *forwarded]
    return evolution_main()


if __name__ == "__main__":
    raise SystemExit(main())
