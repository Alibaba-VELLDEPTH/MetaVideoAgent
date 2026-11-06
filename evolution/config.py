"""Shared model, provider and review-concurrency configuration."""

from __future__ import annotations

import os

try:
    from capability_registry import (
        LLM_ROLE_PROFILE_IDS,
        provider_config,
        resolve_llm_role,
        resolve_profile,
    )
except ImportError:  # pragma: no cover - package import
    from execution.action_runtime.capability_registry import (
        LLM_ROLE_PROFILE_IDS,
        provider_config,
        resolve_llm_role,
        resolve_profile,
    )


_EVOLUTION_LLM_PROFILE = resolve_llm_role("evolution")
_EVOLUTION_PROVIDER = provider_config(_EVOLUTION_LLM_PROFILE)
_REVIEW_LLM_PROFILE = resolve_llm_role("review")
_REVIEW_PROVIDER = provider_config(_REVIEW_LLM_PROFILE)
_POST_DECISION_LLM_PROFILE = resolve_llm_role("post_decision")
_POST_DECISION_PROVIDER = provider_config(_POST_DECISION_LLM_PROFILE)

EVOLUTION_LLM_API_KEY = _EVOLUTION_PROVIDER["api_key"]
EVOLUTION_LLM_BASE_URL = _EVOLUTION_PROVIDER["base_url"]
EVOLUTION_LLM_MODEL = str(_EVOLUTION_LLM_PROFILE.get("model_id") or "")
EVOLUTION_LLM_REQUEST_OPTIONS = dict(_EVOLUTION_LLM_PROFILE.get("request_options") or {})

TEACHER_MULTIMODAL_MODEL = str(resolve_profile("vlm").get("model_id") or "")
TEACHER_ENHANCED_VLM_MODEL = TEACHER_MULTIMODAL_MODEL
TEACHER_ASR_MODEL = str(resolve_profile("asr").get("model_id") or "")
TEACHER_LLM_MODEL = str(_REVIEW_LLM_PROFILE.get("model_id") or "")
TEACHER_LLM_API_KEY = _REVIEW_PROVIDER["api_key"]
TEACHER_LLM_BASE_URL = _REVIEW_PROVIDER["base_url"]
TEACHER_LLM_REQUEST_OPTIONS = dict(_REVIEW_LLM_PROFILE.get("request_options") or {})
POST_DECISION_LLM_PROFILE_ID = LLM_ROLE_PROFILE_IDS["post_decision"]
POST_DECISION_LLM_MODEL = str(_POST_DECISION_LLM_PROFILE.get("model_id") or "")
POST_DECISION_LLM_API_KEY = _POST_DECISION_PROVIDER["api_key"]
POST_DECISION_LLM_BASE_URL = _POST_DECISION_PROVIDER["base_url"]
POST_DECISION_LLM_REQUEST_OPTIONS = dict(
    _POST_DECISION_LLM_PROFILE.get("request_options") or {}
)
TEACHER_MAX_FRAMES = int(os.getenv("TEACHER_MAX_FRAMES", "180") or "180")

META_VLM_MAX_CONCURRENT = int(os.getenv("META_VLM_MAX_CONCURRENT", "16") or "16")
TEACHER_REVIEW_MAX_CONCURRENT = int(
    os.getenv("TEACHER_REVIEW_MAX_CONCURRENT", "16") or "16"
)
