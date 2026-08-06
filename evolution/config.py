"""Shared model, provider, concurrency, and output configuration."""

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

METAVIDEOAGENT_API_KEY = os.getenv("METAVIDEOAGENT_API_KEY", "")
METAVIDEOAGENT_BASE_URL = os.getenv("METAVIDEOAGENT_BASE_URL", "")

EVOLUTION_LLM_PROFILE_ID = LLM_ROLE_PROFILE_IDS["evolution"]
EVOLUTION_LLM_API_KEY = _EVOLUTION_PROVIDER["api_key"]
EVOLUTION_LLM_BASE_URL = _EVOLUTION_PROVIDER["base_url"]
EVOLUTION_LLM_MODEL = str(_EVOLUTION_LLM_PROFILE.get("model_id") or "")
EVOLUTION_LLM_REQUEST_OPTIONS = dict(_EVOLUTION_LLM_PROFILE.get("request_options") or {})
EVOLUTION_MODEL = EVOLUTION_LLM_MODEL
EVOLUTION_TEMPERATURE = 0.7

METAVIDEOAGENT_VLM_MODEL = str(resolve_profile("vlm").get("model_id") or "")
METAVIDEOAGENT_EMBEDDING_MODEL = str(resolve_profile("embedding").get("model_id") or "")
METAVIDEOAGENT_ASR_MODEL = str(resolve_profile("asr").get("model_id") or "")
METAVIDEOAGENT_OCR_MODEL = str(resolve_profile("ocr").get("model_id") or "")

TEACHER_MULTIMODAL_MODEL = METAVIDEOAGENT_VLM_MODEL
TEACHER_ENHANCED_VLM_MODEL = METAVIDEOAGENT_VLM_MODEL
TEACHER_ASR_MODEL = METAVIDEOAGENT_ASR_MODEL
TEACHER_LLM_PROFILE_ID = LLM_ROLE_PROFILE_IDS["review"]
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

META_LLM_MAX_CONCURRENT = int(os.getenv("META_LLM_MAX_CONCURRENT", "8") or "8")
META_VLM_MAX_CONCURRENT = int(os.getenv("META_VLM_MAX_CONCURRENT", "16") or "16")
TEACHER_REVIEW_MAX_CONCURRENT = int(
    os.getenv("TEACHER_REVIEW_MAX_CONCURRENT", "16") or "16"
)

try:
    from runtime_paths import default_workspace

    DEFAULT_WORKSPACE = default_workspace()
except Exception:
    DEFAULT_WORKSPACE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "execution", "workspace")
    )

TRAJECTORY_DIR = "trajectories"
EVOLUTION_HISTORY_DIR = "evolution_history"

API_CONCURRENCY = {
    "embedding": {"max_concurrent": 15, "batch_size": 10},
    "llm": {"max_concurrent": META_LLM_MAX_CONCURRENT},
    "vlm": {"max_concurrent": META_VLM_MAX_CONCURRENT},
    "asr": {"max_concurrent": 6},
    "ocr": {"max_concurrent": 4},
}

BUILD_DB_CONCURRENCY_HINT = """
Use ThreadPoolExecutor for independent embedding calls in addStructure().
Collect texts first, respect a maximum of {embedding_concurrent} workers, and
write embeddings to the vector database in batches.
""".strip().format(
    embedding_concurrent=API_CONCURRENCY["embedding"]["max_concurrent"]
)

MODEL_REGISTRY = {
    "ocr": {
        "active_model": METAVIDEOAGENT_OCR_MODEL,
        "candidates": [METAVIDEOAGENT_OCR_MODEL],
        "category": "ocr",
        "temperature": 0.0,
    },
    "asr": {
        "active_model": METAVIDEOAGENT_ASR_MODEL,
        "candidates": [METAVIDEOAGENT_ASR_MODEL],
        "category": "asr",
        "temperature": 0.0,
    },
    "vlm": {
        "active_model": METAVIDEOAGENT_VLM_MODEL,
        "candidates": [METAVIDEOAGENT_VLM_MODEL],
        "category": "vlm",
        "temperature": 0.1,
    },
    "embedding": {
        "active_model": METAVIDEOAGENT_EMBEDDING_MODEL,
        "candidates": [METAVIDEOAGENT_EMBEDDING_MODEL],
        "category": "embedding",
        "temperature": 0.0,
    },
    "llm": {
        "active_model": str(resolve_profile("llm").get("model_id") or ""),
        "candidates": [str(resolve_profile("llm").get("model_id") or "")],
        "category": "llm",
        "temperature": 0.1,
    },
}
