"""Runtime capability-profile registry shared by generated modules and adapters.

Profiles are declarative and contain no secrets.  A profile identifies one
provider/model/transport combination; generated code may select only a profile
listed in the capability manual and must still call ``runtime_evidence``.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_configured_profile_path = os.environ.get("METAVIDEOAGENT_CAPABILITY_PROFILES", "").strip()
PROFILE_MANUAL_PATH = (
    Path(_configured_profile_path).expanduser().resolve()
    if _configured_profile_path
    else PROJECT_ROOT / "evolution" / "agent_docs" / "capabilities" / "runtime_capability_profiles.json"
)

DEFAULT_PROFILE_IDS = {
    "llm": os.environ.get("METAVIDEOAGENT_LLM_PROFILE", "paper.llm.reasoning"),
    "vlm": os.environ.get("METAVIDEOAGENT_VLM_PROFILE", "paper.vlm.default"),
    "asr": os.environ.get("METAVIDEOAGENT_ASR_PROFILE", "paper.asr.default"),
    "ocr": os.environ.get("METAVIDEOAGENT_OCR_PROFILE", "paper.ocr.default"),
    "embedding": os.environ.get("METAVIDEOAGENT_EMBEDDING_PROFILE", "paper.embedding.default"),
}
LLM_ROLE_PROFILE_IDS = {
    "execution": DEFAULT_PROFILE_IDS["llm"],
    "evolution": os.environ.get("METAVIDEOAGENT_EVOLUTION_LLM_PROFILE", "paper.llm.evolution"),
    "review": os.environ.get(
        "METAVIDEOAGENT_REVIEW_LLM_PROFILE",
        os.environ.get("METAVIDEOAGENT_EVOLUTION_LLM_PROFILE", "paper.llm.evolution"),
    ),
    "post_decision": os.environ.get(
        "METAVIDEOAGENT_POST_DECISION_LLM_PROFILE",
        os.environ.get("METAVIDEOAGENT_EVOLUTION_LLM_PROFILE", "paper.llm.evolution"),
    ),
}
DEFAULT_VLM_PROFILE_ID = DEFAULT_PROFILE_IDS["vlm"]


class CapabilityProfileError(ValueError):
    """Raised when code requests a profile that is absent or incompatible."""


@lru_cache(maxsize=1)
def _manual() -> Dict[str, Any]:
    with PROFILE_MANUAL_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), list):
        raise CapabilityProfileError(f"Invalid capability manual: {PROFILE_MANUAL_PATH}")
    return payload




def iter_profiles(*, capability: str | None = None) -> Iterable[Dict[str, Any]]:
    for profile in _manual().get("profiles", []):
        if not isinstance(profile, dict):
            continue
        if capability and profile.get("capability") != capability:
            continue
        yield dict(profile)




def get_profile(profile_id: str, *, capability: str | None = None) -> Dict[str, Any]:
    wanted = str(profile_id or "").strip()
    for profile in iter_profiles():
        if profile.get("profile_id") != wanted:
            continue
        if capability and profile.get("capability") != capability:
            raise CapabilityProfileError(
                f"Profile {wanted!r} has capability {profile.get('capability')!r}, expected {capability!r}"
            )
        return profile
    raise CapabilityProfileError(f"Unknown runtime capability profile: {wanted!r}")


def resolve_profile(capability: str, profile_id: str | None = None) -> Dict[str, Any]:
    requested = profile_id or DEFAULT_PROFILE_IDS.get(capability, "")
    return get_profile(requested, capability=capability)


def resolve_llm_role(role: str) -> Dict[str, Any]:
    """Resolve an orchestration LLM role through the same profile registry."""
    requested = LLM_ROLE_PROFILE_IDS.get(str(role or "").strip())
    if not requested:
        raise CapabilityProfileError(f"Unknown LLM role: {role!r}")
    return get_profile(requested, capability="llm")


def first_env(names: Iterable[str]) -> str:
    for name in names or []:
        value = os.environ.get(str(name), "").strip()
        if value:
            return value
    return ""


def provider_config(profile: Dict[str, Any]) -> Dict[str, str]:
    return {
        "api_key": first_env(profile.get("api_key_env", [])),
        "base_url": first_env(profile.get("base_url_env", [])) or str(profile.get("default_base_url") or ""),
    }


