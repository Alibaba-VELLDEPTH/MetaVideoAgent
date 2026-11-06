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
    parser.add_argument("--initialization", choices=("observe", "map"), default="observe")
    parser.add_argument("--source-agent", default="", help="Source package to adapt with --initialization map.")
    parser.add_argument("--auto-rounds", type=int, default=None,
                        help="Total update-round budget; new runs default to 4 and resumptions keep the saved budget.")
    parser.add_argument("--iteration-index", type=int, default=0,
                        help="Iteration to continue; otherwise use the run's saved continuation point.")
    parser.add_argument("--restart-from", choices=("diagnosis", "research", "codegen", "smoke", "probe", "full_eval"), default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--first-n", type=int, default=0)
    parser.add_argument("--structure-workers", type=int, default=4)
    parser.add_argument("--repair-attempts", type=int, default=3,
                        help="Additional source-assembly or smoke repair attempts before probe.")
    parser.add_argument("--probe-repair-attempts", type=int, default=2,
                        help="Candidate continuation attempts after probe or engineering feedback.")
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
        "--concurrency", str(max(1, args.concurrency)),
        "--first-n", str(max(0, args.first_n)),
        "--structure-workers", str(max(1, args.structure_workers)),
        "--repair-attempts", str(max(0, args.repair_attempts)),
        "--probe-repair-attempts", str(max(0, args.probe_repair_attempts)),
    ]
    if args.auto_rounds is not None:
        forwarded.extend(["--auto-rounds", str(max(1, args.auto_rounds))])
    for flag, value in (
        ("--initialization", args.initialization), ("--source-agent", args.source_agent),
        ("--run-id", args.run_id), ("--output-root", args.output_root),
        ("--iteration-index", args.iteration_index), ("--restart-from", args.restart_from),
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
