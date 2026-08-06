"""MetaVideoAgent information-channel profile runner.

This runner is the explicit entry point for the first step of a MetaVideoAgent
evolution iteration: observe the training distribution and infer information
channels without using hidden human type labels.  It wraps the deterministic
``distribution_profiler`` so orchestration code can consistently refresh or
reuse a profile across iterations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict

try:
    from distribution_profiler import build_observed_distribution_profile
    from distribution_spec import load_distribution_spec
    from runtime_paths import metavideoagent_run_dir
    from video_distribution_observer import run_observer
except ImportError:  # pragma: no cover
    from .distribution_profiler import build_observed_distribution_profile
    from .distribution_spec import load_distribution_spec
    from .runtime_paths import metavideoagent_run_dir
    from .video_distribution_observer import run_observer


def read_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_runtime_provenance(distribution_manifest: str) -> Dict[str, Any]:
    """Return non-prompt provenance used to bind a profile to one train split."""
    if not distribution_manifest:
        return {}
    manifest_path = os.path.realpath(os.path.abspath(distribution_manifest))
    spec = load_distribution_spec(manifest_path)
    train_path = str((spec.get("splits") or {}).get("train") or "")
    return {
        "schema_version": 1,
        "distribution_manifest_path": manifest_path,
        "distribution_manifest_sha256": _sha256_file(manifest_path),
        "train_split_path": os.path.realpath(os.path.abspath(train_path)) if train_path else "",
        "train_split_sha256": _sha256_file(train_path) if train_path and os.path.isfile(train_path) else "",
    }


def profile_matches_manifest(profile: Dict[str, Any], distribution_manifest: str) -> tuple[bool, str]:
    """Fail closed when an explicit profile has no matching runtime provenance."""
    if (profile or {}).get("profile_source") != "five_frame_query_aware_train_distribution":
        return False, "profile_method_mismatch"
    if not distribution_manifest:
        return True, "no_manifest_requested"
    expected = profile_runtime_provenance(distribution_manifest)
    actual = (profile or {}).get("runtime_provenance") or {}
    if not actual:
        return False, "profile_missing_runtime_provenance"
    for key in (
        "schema_version", "distribution_manifest_path", "distribution_manifest_sha256",
        "train_split_path", "train_split_sha256",
    ):
        if actual.get(key) != expected.get(key):
            return False, f"profile_runtime_provenance_mismatch:{key}"
    return True, "profile_runtime_provenance_matches"


def default_profile_path(run_id: str = "", output_root: str = "",
                         filename: str = "channel_profile.json") -> str:
    if run_id:
        return os.path.join(metavideoagent_run_dir(run_id, output_root), "profiles", filename)
    return os.path.join(metavideoagent_run_dir("", output_root), "profiles", filename)


def compact_channel_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    hypothesis = profile.get("information_channel_hypothesis", {}) or {}
    raw = profile.get("raw_video_signal_profile", {}) or {}
    schema = profile.get("information_channel_schema", {}) or {}
    semantic_profile = profile.get("semantic_channel_observation_profile", {}) or {}
    semantic_plan = profile.get("semantic_channel_sampling_plan", {}) or {}
    return {
        "profile_path": profile.get("output_path", ""),
        "primary_observed_channel": hypothesis.get("primary_observed_channel", ""),
        "secondary_observed_channels": hypothesis.get("secondary_observed_channels", []),
        "channel_scores": hypothesis.get("channel_scores", {}),
        "information_channel_schema": schema,
        "dominant_information_channel": schema.get("dominant_information_channel", ""),
        "channel_confidence": schema.get("channel_confidence"),
        "raw_video_summary": raw.get("summary", {}) if isinstance(raw, dict) else {},
        "semantic_observation_summary": {
            "sampled_observations": semantic_profile.get("sampled_observations", 0),
            "source_available": semantic_profile.get("source_available", False),
            "density": semantic_profile.get("density", {}),
        },
        "semantic_sampling_plan_summary": {
            "sampled_videos": semantic_plan.get("sampled_videos", 0),
            "frames_per_video": semantic_plan.get("frames_per_video"),
            "coverage": semantic_plan.get("coverage"),
        },
        "sampling": profile.get("sampling", {}),
        "leakage_policy": profile.get("leakage_policy", ""),
    }


def build_channel_profile(
    workspace: str,
    distribution_manifest: str = "",
    output_path: str = "",
    run_id: str = "",
    output_root: str = "",
    include_execution_artifacts: bool = True,
    force: bool = False,
    semantic_observer_mode: str = "observe",
    semantic_observer_dir: str = "",
    semantic_video_budget: int = 0,
    semantic_observe_workers: int = 1,
    runtime_dir: str = "",
    vlm_model: str = "",
) -> Dict[str, Any]:
    """Build or reuse a prompt-safe distribution profile.

    The output contains the full observed distribution profile plus
    ``channel_profile_summary`` for prompts and reports.  ``semantic_observer_mode=none``
    and ``plan`` are local-only; ``observe`` asks the configured VLM to
    summarize five uniformly sampled frames with the associated query set.
    """
    output_path = output_path or default_profile_path(run_id, output_root)
    if output_path and os.path.exists(output_path) and not force:
        profile = read_json(output_path)
        matches, _reason = profile_matches_manifest(profile, distribution_manifest)
        if profile and matches:
            return profile

    spec = load_distribution_spec(distribution_manifest) if distribution_manifest else {}
    observer_profile = {}
    if semantic_observer_mode and semantic_observer_mode != "none":
        if semantic_observer_mode not in {"plan", "observe"}:
            raise ValueError("semantic_observer_mode must be none, plan, or observe")
        observer_dir = semantic_observer_dir or os.path.join(
            os.path.dirname(os.path.abspath(output_path)),
            "video_distribution_observer",
        )
        observer_profile = run_observer(
            workspace=workspace,
            distribution_manifest=distribution_manifest,
            output_dir=observer_dir,
            mode=semantic_observer_mode,
            video_budget=semantic_video_budget,
            runtime_dir=runtime_dir,
            vlm_model=vlm_model,
            observe_workers=semantic_observe_workers,
        )
        observations_path = observer_profile.get("observations_path", "")
        if observations_path:
            spec.setdefault("artifacts", {})["semantic_channel_observations"] = observations_path
    profile = build_observed_distribution_profile(
        workspace,
        distribution_spec=spec,
        output_path=output_path,
        include_execution_artifacts=include_execution_artifacts,
    )
    if observer_profile:
        profile["video_distribution_observer_profile"] = observer_profile
    profile["output_path"] = os.path.abspath(output_path)
    # This provenance is persisted for runtime validation only; the compact
    # prompt-facing profile summary intentionally never contains local paths
    # or fingerprints.
    profile["runtime_provenance"] = profile_runtime_provenance(distribution_manifest)
    profile["artifact_type"] = "metavideoagent_channel_profile"
    profile["schema_version"] = 1
    profile["channel_profile_summary"] = compact_channel_summary(profile)
    write_json(output_path, profile)
    summary_path = os.path.join(os.path.dirname(output_path), "channel_profile_summary.json")
    write_json(summary_path, profile["channel_profile_summary"])
    profile["channel_profile_summary_path"] = summary_path
    write_json(output_path, profile)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/reuse MetaVideoAgent information-channel profile")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--distribution-manifest", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--semantic-observer-mode", choices=("none", "plan", "observe"), default="observe",
                        help="Build five-frame overview observations; observe calls the configured VLM.")
    parser.add_argument("--semantic-observer-dir", default="")
    parser.add_argument("--semantic-video-budget", type=int, default=0)
    parser.add_argument("--semantic-observe-workers", type=int, default=1)
    parser.add_argument("--runtime-dir", default="")
    parser.add_argument("--vlm-model", default="")
    args = parser.parse_args()

    profile = build_channel_profile(
        workspace=os.path.abspath(args.workspace),
        distribution_manifest=args.distribution_manifest,
        output_path=args.output,
        run_id=args.run_id or f"channel_profile_{int(time.time())}",
        output_root=args.output_root,
        include_execution_artifacts=False,
        force=args.force,
        semantic_observer_mode=args.semantic_observer_mode,
        semantic_observer_dir=args.semantic_observer_dir,
        semantic_video_budget=args.semantic_video_budget,
        semantic_observe_workers=args.semantic_observe_workers,
        runtime_dir=args.runtime_dir,
        vlm_model=args.vlm_model,
    )
    print("CHANNEL_PROFILE_DONE")
    print(f"profile_path={profile.get('output_path')}")
    print(json.dumps(profile.get("channel_profile_summary", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
