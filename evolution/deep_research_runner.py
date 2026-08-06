"""Codex-backed deep research for MetaVideoAgent evolution.

The first MetaVideoAgent stage and later evolution rounds need source-grounded
design context before code generation.  This runner asks Codex to do online
research and produce a five-module design brief, then validates that the output
contains verifiable sources and concrete designs for all five execution modules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List

try:
    from channel_profile_runner import compact_channel_summary
    from bundle_contract import normalize_modules
    from codex_runtime_settings import codex_exec_model_args, resolve_codex_runtime_settings
    from coding_agent_runtime import coding_agent_environment, resolve_coding_agent_executable
    from diagnosis_execution_brief import (
        model_handoff_issues,
        validate_bundle_execution_brief,
    )
    from runtime_capability_context import DEFAULT_PROFILE_IDS, build_runtime_capability_context
    from runtime_paths import metavideoagent_run_dir
except ImportError:  # pragma: no cover
    from .channel_profile_runner import compact_channel_summary
    from .bundle_contract import normalize_modules
    from .codex_runtime_settings import codex_exec_model_args, resolve_codex_runtime_settings
    from .coding_agent_runtime import coding_agent_environment, resolve_coding_agent_executable
    from .diagnosis_execution_brief import (
        model_handoff_issues,
        validate_bundle_execution_brief,
    )
    from .runtime_capability_context import DEFAULT_PROFILE_IDS, build_runtime_capability_context
    from .runtime_paths import metavideoagent_run_dir


MODULES = ["video_structuring", "localization", "perception", "memory", "thinking"]
DEFAULT_YEAR_MIN = 2024
RESEARCH_BUCKETS = ["high_confidence_designs", "hypotheses_to_test", "avoid_or_low_priority"]
ACADEMIC_OR_OFFICIAL_DOMAINS = (
    "arxiv.org",
    "openaccess.thecvf.com",
    "aclanthology.org",
    "proceedings.neurips.cc",
    "openreview.net",
    "proceedings.mlr.press",
    "ieeexplore.ieee.org",
    "dl.acm.org",
)
IMPLEMENTATION_OR_MODEL_DOMAINS = (
    "github.com",
    "huggingface.co",
)

NON_RUNTIME_TOOL_TERMS = {
    "DIAG_VIDEO_AUDIT": "Video review capability",
    "DIAG_DETAIL_VIDEO_AUDIT": "Detailed video review capability",
    "AUDIT_VIDEO": "Video review capability",
    "AUDIT_DETAIL_VIDEO": "Detailed video review capability",
    "AUDIT_ENHANCED_VIDEO": "Enhance video review capabilities",
    "AUDIT_STRUCTURE": "Structure library review capability",
    "AUDIT_TRANSCRIPT": "Subtitle/audio text review capabilities",
    "watch_video": "Video review capability",
    "video_review": "Video review capability",
    "teacher_video_verification": "Video review capability",
    "detailed_watch": "Detailed video review capability",
    "detail_video_review": "Detailed video review capability",
    "fine_video_review": "Detailed video review capability",
    "teacher_detailed_video_verification": "Detailed video review capability",
    "enhanced_watch": "Enhance video review capabilities",
    "enhanced_video_review": "Enhance video review capabilities",
    "teacher_enhanced_video_verification": "Enhance video review capabilities",
    "browse_struct_db": "Structure library review capability",
    "structure_review": "Structure library review capability",
    "teacher_structure_db_audit": "Structure library review capability",
    "read_transcript": "Subtitle/audio text review capabilities",
    "transcript_review": "Subtitle/audio text review capabilities",
    "teacher_transcript_audit": "Subtitle/audio text review capabilities",
    "listen_audio": "Subtitle/audio text review capabilities",
    "frame_inspect": "visual observation ability",
    "single_frame_inspect": "visual observation ability",
    "inspect_frames": "visual observation ability",
    "runtime_visual_observation_tool": "visual observation ability",
    "runtime_visual_observation_capability": "visual observation ability",
    "clip_search": "Positioning/structure search capabilities",
    "frame_clip_search": "Positioning/structure search capabilities",
    "search_evidence": "Positioning/structure search capabilities",
    "runtime_localization_or_structure_retrieval_tool": "Positioning/structure search capabilities",
    "runtime_localization_or_structure_retrieval_capability": "Positioning/structure search capabilities",
    "runtime_evidence_retrieval_capability": "Positioning/structure search capabilities",
    "global_browse": "Global structure browsing capabilities",
    "runtime_broad_structure_retrieval_tool": "Global structure browsing capabilities",
    "runtime_broad_structure_retrieval_capability": "Global structure browsing capabilities",
    "audio_analysis": "Audio/subtitle understanding ability",
    "runtime_audio_or_transcript_evidence_tool": "Audio/subtitle understanding ability",
    "runtime_audio_or_transcript_capability": "Audio/subtitle understanding ability",
}


def sanitize_tool_terms(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: sanitize_tool_terms(v)
            for k, v in value.items()
            if k != "react_trace"
        }
    if isinstance(value, list):
        return [sanitize_tool_terms(v) for v in value]
    if not isinstance(value, str):
        return value
    text = value
    for old, new in sorted(NON_RUNTIME_TOOL_TERMS.items(), key=lambda item: -len(item[0])):
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return text


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


def write_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def default_research_dir(run_id: str = "", output_root: str = "") -> str:
    return os.path.join(metavideoagent_run_dir(run_id, output_root), "deep_research")


def _channel_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    return profile.get("channel_profile_summary") or compact_channel_summary(profile)


def _target_modules(diagnosis: Dict[str, Any]) -> list[str]:
    """Return the bundle direction from a validated later-round brief."""
    if not diagnosis:
        return []
    if diagnosis.get("artifact_type") != "diagnosis_execution_brief":
        raise RuntimeError("later-round research requires a diagnosis execution brief")
    modules = (diagnosis.get("execution_policy") or {}).get("target_modules") or []
    try:
        return normalize_modules(modules)
    except ValueError as exc:
        raise RuntimeError(f"diagnosis execution brief has an invalid target_modules direction: {exc}") from exc


def _failure_summary(diagnosis: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    del review
    if not diagnosis:
        return {}
    task = diagnosis.get("implementation_task", {}) or {}
    plan = diagnosis.get("evolution_plan", {}) or {}
    issues = model_handoff_issues(task, path="implementation_task")
    issues.extend(model_handoff_issues(plan, path="evolution_plan"))
    if issues:
        raise RuntimeError(
            "refusing execution brief with machine-only evidence in research handoff: "
            + "; ".join(issues)
        )
    return {
        "diagnosis_target_modules": _target_modules(diagnosis),
        "failure_chain": task.get("failure_chain", ""),
        "evolution_mechanism": task.get("evolution_mechanism", ""),
        "handoff_contracts": task.get("handoff_contracts", []),
        "diagnosis_evolution_hint": task.get("goal", ""),
        "allowed_runtime_capabilities": task.get("runtime_capabilities", []),
    }


def build_research_brief(profile: Dict[str, Any],
                         diagnosis: Dict[str, Any] | None = None,
                         review: Dict[str, Any] | None = None,
                         queries: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
    """Build either initial-design research or bundle-evolution research."""
    summary = _channel_summary(profile)
    diagnosis = diagnosis or {}
    review = review or {}
    target_modules = _target_modules(diagnosis)
    failure = sanitize_tool_terms(_failure_summary(diagnosis, review))
    later_round = bool(target_modules)
    if later_round:
        objective = (
            "Research implementation methods for the diagnosis-selected bundle "
            "mechanism, responsibility boundaries, and producer/consumer handoffs. "
            "Do not prescribe task-specific behavior or an edit whitelist."
        )
        research_questions = [
            "Which concrete methods implement the diagnosed evolution mechanism without task-specific rules?",
            "What state, stopping, verification, and fallback logic should the connected modules add?",
            "Which producer/consumer contracts must remain stable across the bundle?",
            "What smoke and probe failure modes should Codex guard against while coding?",
        ]
    else:
        objective = (
            "Design an initial five-module MetaVideoAgent baseline for this five-frame, "
            "query-aware distribution before any fixed baseline bundle exists."
        )
        research_questions = [
            "Which information channel should be indexed first for this distribution?",
            "How should long-video localization retrieve candidate evidence?",
            "What should perception verify after retrieval?",
            "What memory representation should preserve evidence across steps?",
            "What thinking/control policy should coordinate the five modules?",
        ]
    return {
        "objective": objective,
        "stage": "later_round_bundle_evolution" if later_round else "initial_baseline_design",
        "target_modules": target_modules,
        "video_distribution_observation": summary,
        "distribution_records": profile.get("distribution_records", []),
        "engineering_constraints": {
            "must_use_five_modules": True,
            "modules": MODULES,
            "codex_may_adapt_changed_modules_within_bundle": later_round,
            "must_be_implementable_by_codex": True,
            "must_fit_existing_execution_layer_interfaces": True,
            "avoid_answer_leakage": True,
            "initial_stage_forbidden_inputs": [
                "questions", "answers", "evidence intervals",
                "human type/category labels", "task trajectories",
            ],
            "later_round_rule": (
                "Research supplies the diagnosed evolution mechanism and boundary constraints, "
                "not a per-file edit whitelist. Codex may adapt related modules inside the "
                "five-module bundle while preserving unaffected behavior."
            ),
            "initial_execution_lifecycle": (
                "For initial-baseline research, the execution-layer structure builder owns "
                "bounded media sampling and model calls. Generated addStructure methods only "
                "persist or index supplied records. Use registered runtime adapters rather "
                "than provider clients or local model toolboxes. The configured ASR profile is "
                f"{DEFAULT_PROFILE_IDS['asr']}; the configured VLM profile is "
                f"{DEFAULT_PROFILE_IDS['vlm']}."
            ),
        },
        "research_questions": research_questions,
        "allowed_runtime_capabilities": failure.get("allowed_runtime_capabilities", []),
        "seed_queries_not_binding": queries or [],
        "failure_context_if_any": failure,
        "runtime_tool_name_policy": {
            "teacher_tools_are_not_runtime_tools": True,
            "do_not_recommend_concrete_tool_names_from_teacher_rollouts": True,
            "use_capability_language_until_codex_receives_the_current_tool_registry": True,
        },
    }


def _strip_json_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def _extract_json_object(text: str) -> Dict[str, Any]:
    value = _strip_json_fence(text)
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(value[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def build_research_queries(profile: Dict[str, Any],
                           diagnosis: Dict[str, Any] | None = None,
                           review: Dict[str, Any] | None = None,
                           max_queries: int = 3) -> List[Dict[str, str]]:
    """Build distribution queries or diagnosis-conditioned bundle queries."""
    summary = _channel_summary(profile)
    primary = summary.get("primary_observed_channel") or "mixed_or_unknown"
    diagnosis = diagnosis or {}
    target_modules = _target_modules(diagnosis)
    failure = _failure_summary(diagnosis, review or {})

    channel_queries = {
        "speech_or_audio": [
            "audio-first long video question answering temporal grounding",
            "speech transcript indexed memory for long video QA",
        ],
        "screen_text_or_subtitle": [
            "OCR subtitle aware long video QA retrieval",
            "screen text verification for multimodal video question answering",
        ],
        "visual_action": [
            "event-centric temporal grounding for long video question answering",
            "entity action state tracking for video QA",
        ],
        "visual_object_detail": [
            "fine-grained object attribute verification for video question answering",
            "object-centric video memory retrieval",
        ],
        "long_context_reasoning": [
            "long video narrative reasoning graph memory",
            "multi-hop long video question answering planning",
        ],
        "mixed_or_unknown": [
            "long video QA hierarchical retrieval multimodal agent",
            "multimodal evidence memory for long video question answering",
        ],
    }
    candidates = list(channel_queries.get(primary, channel_queries["mixed_or_unknown"]))
    if target_modules:
        mechanism = " ".join([
            str(failure.get("failure_chain") or ""),
            str(failure.get("evolution_mechanism") or ""),
        ])
        mechanism = re.sub(r"[^\w\s-]", " ", mechanism)
        mechanism = " ".join(mechanism.split())[:220]
        module_text = " ".join(target_modules)
        candidates = [
            (
                "long video question answering five-module bundle "
                f"{module_text} producer consumer handoff {mechanism}"
            ).strip(),
            *[
                f"five-module video agent handoff compatibility {query}"
                for query in candidates
            ],
        ]

    out = []
    seen = set()
    for query in candidates:
        query = " ".join(str(query).split())
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        out.append({
            "query": query,
            "target_modules": list(target_modules),
            "reason": (
                f"primary_observed_channel={primary}; "
                f"diagnosis_targets={','.join(target_modules) or 'initial_design'}"
            ),
        })
        if len(out) >= max_queries:
            break
    return out


def _valid_source_url(url: str) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    parsed = urllib.parse.urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _source_domain_class(url: str) -> str:
    parsed = urllib.parse.urlparse((url or "").strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if any(host == domain or host.endswith("." + domain) for domain in ACADEMIC_OR_OFFICIAL_DOMAINS):
        return "academic_or_official"
    if any(host == domain or host.endswith("." + domain) for domain in IMPLEMENTATION_OR_MODEL_DOMAINS):
        return "implementation_or_model"
    return "other"


def _source_reachable(url: str, timeout: int = 8) -> bool:
    if not _valid_source_url(url):
        return False
    headers = {"User-Agent": "MetaVideoAgent-DeepResearch/1.0"}
    for method in ("HEAD", "GET"):
        try:
            request = urllib.request.Request(url, method=method, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if 200 <= int(status) < 400:
                    return True
        except Exception:
            continue
    return False


def validate_design_brief(brief: Dict[str, Any],
                          min_sources: int = 2,
                          verify_source_urls: bool = False,
                          source_timeout: int = 8,
                          expected_target_modules: List[str] | None = None,
                          allowed_runtime_capabilities: List[Dict[str, Any]] | None = None,
                          required_bundle_contract: Dict[str, Any] | None = None) -> Dict[str, Any]:
    issues: List[str] = []
    if not isinstance(brief, dict) or not brief:
        return {"passed": False, "issues": ["missing design_brief"], "valid_source_count": 0}
    if brief.get("research_complete") is not True:
        issues.append("research_complete must be explicitly true for a formal research brief")

    sources = brief.get("sources_used") or []
    if not isinstance(sources, list):
        issues.append("sources_used must be a list")
        sources = []
    valid_sources = []
    unreachable_sources = []
    source_domain_counts = {
        "academic_or_official": 0,
        "implementation_or_model": 0,
        "other": 0,
    }
    for source in sources:
        if not isinstance(source, dict):
            continue
        title = str(source.get("title") or source.get("name") or "").strip()
        url = str(source.get("url") or source.get("link") or "").strip()
        used_for = str(source.get("used_for") or "").strip()
        if title and used_for and _valid_source_url(url):
            if verify_source_urls and not _source_reachable(url, timeout=source_timeout):
                unreachable_sources.append({
                    "title": title,
                    "url": url,
                    "used_for": used_for,
                })
            else:
                valid_sources.append(source)
                source_domain_counts[_source_domain_class(url)] += 1
    if len(valid_sources) < min_sources:
        issues.append(f"need at least {min_sources} verifiable sources with title/url/used_for")
    if valid_sources and not (
        source_domain_counts["academic_or_official"]
        or source_domain_counts["implementation_or_model"]
    ):
        issues.append(
            "need at least one academic/official/implementation source "
            "(arXiv/CVF/ACL/NeurIPS/OpenReview/GitHub/HuggingFace/etc.)"
        )
    valid_source_titles = {
        str(source.get("title") or source.get("name") or "").strip().lower()
        for source in valid_sources
    }

    research_plan = brief.get("research_plan") or {}
    if not isinstance(research_plan, dict):
        issues.append("research_plan must be an object")
        research_plan = {}
    if not str(research_plan.get("problem_understanding") or "").strip():
        issues.append("research_plan.problem_understanding is empty")
    source_strategy = research_plan.get("source_strategy") or []
    if not isinstance(source_strategy, list) or not [s for s in source_strategy if str(s).strip()]:
        issues.append("research_plan.source_strategy is empty")

    target_modules = list(expected_target_modules or brief.get("target_modules") or [])
    if target_modules:
        if list(brief.get("target_modules") or []) != target_modules:
            issues.append("target_modules must match the diagnosis execution brief")

        # Later-round research supports one adaptive module-set mechanism
        # rather than an initial-baseline redesign.
        design = brief.get("module_set_design") or {}
        if not isinstance(design, dict):
            issues.append("module_set_design must be an object")
            design = {}
        for field in ("goal", "mechanism"):
            if not str(design.get(field) or "").strip():
                issues.append(f"module_set_design.{field} is empty")
        modules = design.get("candidate_module_roles") or {}
        if not isinstance(modules, dict) or not [key for key in modules if key in MODULES]:
            issues.append("module_set_design.candidate_module_roles needs at least one module")
        handoffs = design.get("handoff_contracts") or []
        if not isinstance(handoffs, list) or not handoffs:
            issues.append("module_set_design.handoff_contracts is empty")
        implementation_plan = brief.get("implementation_plan") or []
        if not isinstance(implementation_plan, list) or not [p for p in implementation_plan if str(p).strip()]:
            issues.append("implementation_plan is empty")
        required_bundle_contract = required_bundle_contract or {}
        policy = required_bundle_contract.get("execution_policy") or {}
        task_contract = required_bundle_contract.get("implementation_task") or {}
        expected_modules = {
            str(module) for module in (policy.get("target_modules") or [])
            if str(module) in MODULES
        }
        actual_modules = {
            str(module) for module in modules
            if str(module) in MODULES and str(modules.get(module) or "").strip()
        }
        if expected_modules and not expected_modules.issubset(actual_modules):
            issues.append(
                "module_set_design.candidate_module_roles must include every "
                "diagnosis target modules: " + ", ".join(sorted(expected_modules - actual_modules))
            )
        expected_handoffs = {
            (str(item.get("producer") or ""), str(item.get("consumer") or ""))
            for item in (task_contract.get("handoff_contracts") or [])
            if isinstance(item, dict) and str(item.get("producer") or "") and str(item.get("consumer") or "")
        }
        actual_handoffs = {
            (str(item.get("producer") or ""), str(item.get("consumer") or ""))
            for item in handoffs if isinstance(item, dict)
            and str(item.get("producer") or "") and str(item.get("consumer") or "")
            and str(item.get("contract") or item.get("output_protocol") or "").strip()
        }
        if expected_handoffs and not expected_handoffs.issubset(actual_handoffs):
            missing = [f"{producer}->{consumer}" for producer, consumer in sorted(expected_handoffs - actual_handoffs)]
            issues.append(
                "module_set_design.handoff_contracts must cover diagnosis handoffs: "
                + ", ".join(missing)
            )

        mappings = brief.get("runtime_capability_mapping")
        if allowed_runtime_capabilities is not None and not isinstance(mappings, list):
            issues.append("runtime_capability_mapping must be a list")
            mappings = []
        if allowed_runtime_capabilities is not None:
            # One allowed entry may name several equivalent public helpers.
            def helper_identities(runtime_call: Any) -> set[str]:
                return set(re.findall(
                    r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_.]*)\s*\(",
                    str(runtime_call or ""),
                ))
            allowed_helpers = {}
            for allowed in allowed_runtime_capabilities:
                if not isinstance(allowed, dict):
                    continue
                key = (str(allowed.get("capability") or ""), str(allowed.get("profile_id") or ""))
                identities = helper_identities(allowed.get("runtime_call"))
                if key[0] and identities:
                    allowed_helpers.setdefault(key, set()).update(identities)
            for item in mappings or []:
                if not isinstance(item, dict):
                    issues.append("runtime_capability_mapping entries must be objects")
                    continue
                key = (str(item.get("capability") or ""), str(item.get("profile_id") or ""))
                identities = helper_identities(item.get("runtime_call"))
                if not identities or not identities.issubset(allowed_helpers.get(key, set())):
                    issues.append("runtime_capability_mapping uses a capability not allowed by execution brief")

        # Source-backed research must justify every diagnosed role and handoff,
        # rather than merely provide unrelated bibliography entries.
        source_mappings = brief.get("functional_change_research") or []
        if not isinstance(source_mappings, list):
            issues.append("functional_change_research must be a list")
            source_mappings = []
        expected_components = {f"module:{module}" for module in expected_modules}
        expected_components.update(
            f"handoff:{producer}->{consumer}" for producer, consumer in expected_handoffs
        )
        mapped_components = set()
        for item in source_mappings:
            if not isinstance(item, dict):
                issues.append("functional_change_research entries must be objects")
                continue
            component = str(item.get("component") or item.get("functional_change") or "").strip()
            mapped_components.add(component)
            for field in ("method", "implementation_guidance", "source_support"):
                if not str(item.get(field) or "").strip():
                    issues.append(f"functional_change_research.{component or '<missing>'}.{field} is empty")
            if valid_source_titles:
                # One component can legitimately draw on several sources.  Do
                # not force Codex to discard that provenance merely because it
                # uses a semicolon/newline-separated citation list; every
                # cited title must still exactly match a verified source.
                source_support = str(item.get("source_support") or "").strip()
                cited_titles = [
                    title.strip().lower()
                    for title in re.split(r"[;\n]+", source_support)
                    if title.strip()
                ]
                if not cited_titles or any(title not in valid_source_titles for title in cited_titles):
                    issues.append(
                        f"functional_change_research.{component or '<missing>'}.source_support "
                        "must reference one or more validated source titles"
                    )
        if expected_components - mapped_components:
            issues.append(
                "functional_change_research must cover diagnosis bundle components: "
                + ", ".join(sorted(expected_components - mapped_components))
            )
    else:
        for module in MODULES:
            key = f"{module}_design"
            item = brief.get(key)
            if not isinstance(item, dict):
                issues.append(f"missing {key}")
                continue
            goal = str(item.get("goal") or "").strip()
            mechanisms = item.get("mechanisms") or []
            if not goal:
                issues.append(f"{key}.goal is empty")
            if not isinstance(mechanisms, list) or not [m for m in mechanisms if str(m).strip()]:
                issues.append(f"{key}.mechanisms is empty")

        module_source_map = brief.get("module_source_map") or {}
        if not isinstance(module_source_map, dict):
            issues.append("module_source_map must be an object")
            module_source_map = {}
        for module in MODULES:
            mappings = module_source_map.get(module) or []
            if not isinstance(mappings, list) or not mappings:
                issues.append(f"module_source_map.{module} is empty")
                continue
            mapped_to_valid_source = False
            for mapping in mappings:
                if not isinstance(mapping, dict):
                    continue
                source = str(mapping.get("source") or "").strip()
                method = str(mapping.get("method_idea") or "").strip()
                hint = str(mapping.get("implementation_hint") or "").strip()
                if not method:
                    issues.append(f"module_source_map.{module}.method_idea is empty")
                if not hint:
                    issues.append(f"module_source_map.{module}.implementation_hint is empty")
                source_l = source.lower()
                if source_l and any(source_l == title or source_l in title or title in source_l for title in valid_source_titles):
                    mapped_to_valid_source = True
            if valid_source_titles and not mapped_to_valid_source:
                issues.append(f"module_source_map.{module} does not reference a validated source title")

        design_spec = brief.get("baseline_combo_design_spec") or {}
        if not isinstance(design_spec, dict):
            issues.append("baseline_combo_design_spec must be an object")
            design_spec = {}
        if not str(design_spec.get("overall_strategy") or "").strip():
            issues.append("baseline_combo_design_spec.overall_strategy is empty")
        for module in MODULES:
            spec = design_spec.get(module)
            if not isinstance(spec, dict):
                issues.append(f"baseline_combo_design_spec.{module} is missing")
                continue
            if not str(spec.get("class_goal") or spec.get("goal") or "").strip():
                issues.append(f"baseline_combo_design_spec.{module}.class_goal is empty")
            mechanisms = spec.get("core_mechanisms") or spec.get("mechanisms") or []
            if not isinstance(mechanisms, list) or not [m for m in mechanisms if str(m).strip()]:
                issues.append(f"baseline_combo_design_spec.{module}.core_mechanisms is empty")

    for bucket in RESEARCH_BUCKETS:
        value = brief.get(bucket)
        if not isinstance(value, list):
            issues.append(f"{bucket} must be a list")
    if isinstance(brief.get("high_confidence_designs"), list) and not brief.get("high_confidence_designs"):
        issues.append("high_confidence_designs is empty")

    if not str(brief.get("codex_instructions") or "").strip():
        issues.append("codex_instructions is empty")
    if target_modules:
        if not str(brief.get("diagnosis_support_summary") or "").strip():
            issues.append("diagnosis_support_summary is empty")
    elif not str(brief.get("recommended_architecture") or "").strip():
        issues.append("recommended_architecture is empty")
    return {
        "passed": not issues,
        "issues": issues,
        "valid_source_count": len(valid_sources),
        "min_sources": min_sources,
        "source_url_verification": bool(verify_source_urls),
        "source_domain_counts": source_domain_counts,
        "unreachable_sources": unreachable_sources,
        "warnings": [
            f"source URL not reachable: {item['url']}"
            for item in unreachable_sources
        ],
    }


def _build_codex_prompt(profile: Dict[str, Any],
                        diagnosis: Dict[str, Any],
                        review: Dict[str, Any],
                        queries: List[Dict[str, str]],
                        output_path: str,
                        year_min: int,
                        min_sources: int) -> str:
    target_modules = _target_modules(diagnosis)
    if target_modules:

        schema_text = (
            "{\n"
            '  "research_complete": true,\n'
            '  "research_plan": {"problem_understanding": "...", "key_uncertainties": ["..."], "source_strategy": ["..."]},\n'
            f'  "target_modules": {json.dumps(target_modules)},\n'
            '  "diagnosis_support_summary": "how sources support the diagnosis without changing its outer direction",\n'
            '  "executed_searches": [{"query": "...", "source_target": "academic|implementation|benchmark|model", "why": "...", "top_sources_considered": ["..."]}],\n'
            '  "rejected_sources": [{"title": "...", "url": "https://...", "reason": "..."}],\n'
            '  "module_set_design": {"goal": "...", "mechanism": "...", "candidate_module_roles": {"localization": "..."}, "handoff_contracts": [{"producer": "...", "consumer": "...", "contract": "..."}]},\n'
            '  "implementation_plan": ["concrete coding steps; inner Codex feedback may adapt related modules"],\n'
            '  "runtime_capability_mapping": [{"capability": "only an allowed capability", "profile_id": "exact allowed profile_id", "runtime_call": "registered runtime helper identity", "use": "..."}],\n'
            '  "functional_change_research": [{"component": "module:localization|handoff:localization->perception", "method": "source-backed method", "implementation_guidance": "general implementation guidance", "source_support": "exact source title"}],\n'
            '  "high_confidence_designs": [{"claim": "...", "why": "...", "module": "...", "source": "..."}],\n'
            '  "hypotheses_to_test": [{"claim": "...", "probe": "...", "risk": "..."}],\n'
            '  "avoid_or_low_priority": [{"idea": "...", "reason": "..."}],\n'
            '  "codex_instructions": "implementation guidance for the first bundle candidate; later probe feedback may revise related modules",\n'
            '  "sources_used": [{"title": "...", "url": "https://...", "used_for": "..."}],\n'
            '  "risks": ["..."], "source_quality": "strong|medium|weak"\n'
            "}\n"
        )
        task_line = (
            "Task: research the diagnosis-selected five-module bundle mechanism. "
            "This is an OUTER-ROUND direction, not an edit whitelist: explain responsibility boundaries and "
            "producer→consumer handoffs, while allowing the later inner Codex loop to adapt related modules."
        )
        schema_rule = (
            "- Output the bundle schema with at least one target module role; include additional roles only when the mechanism requires them.\n"
            "- candidate_module_roles must include every diagnosis target module, and handoff_contracts must include every diagnosis handoff.\n"
            "- functional_change_research must contain one source-backed entry for each target module and diagnosis handoff, using component values `module:<name>` or `handoff:<producer>-><consumer>`.\n"
            "- runtime_capability_mapping may use only the capability/profile/helper triples in allowed_runtime_capabilities.\n"
            "- Do not include a per-file edit whitelist.\n"
        )
    else:
        schema_text = (
            "{\n"
            '  "research_complete": true,\n'
            '  "research_plan": {"problem_understanding": "...", "key_uncertainties": ["..."], "source_strategy": ["..."]},\n'
            '  "executed_searches": [{"query": "...", "source_target": "academic|implementation|benchmark|model", "why": "...", "top_sources_considered": ["..."]}],\n'
            '  "rejected_sources": [{"title": "...", "url": "https://...", "reason": "..."}],\n'
            '  "distribution_interpretation": "...",\n'
            '  "recommended_architecture": "...",\n'
            '  "module_source_map": {\n'
            '    "video_structuring": [{"source": "exact source title", "method_idea": "...", "why_relevant": "...", "implementation_hint": "..."}],\n'
            '    "localization": [{"source": "exact source title", "method_idea": "...", "why_relevant": "...", "implementation_hint": "..."}],\n'
            '    "perception": [{"source": "exact source title", "method_idea": "...", "why_relevant": "...", "implementation_hint": "..."}],\n'
            '    "memory": [{"source": "exact source title", "method_idea": "...", "why_relevant": "...", "implementation_hint": "..."}],\n'
            '    "thinking": [{"source": "exact source title", "method_idea": "...", "why_relevant": "...", "implementation_hint": "..."}]\n'
            '  },\n'
            '  "baseline_combo_design_spec": {\n'
            '    "overall_strategy": "...",\n'
            '    "video_structuring": {"class_goal": "...", "core_mechanisms": ["..."], "implementation_notes": ["..."]},\n'
            '    "localization": {"class_goal": "...", "core_mechanisms": ["..."], "implementation_notes": ["..."]},\n'
            '    "perception": {"class_goal": "...", "core_mechanisms": ["..."], "implementation_notes": ["..."]},\n'
            '    "memory": {"class_goal": "...", "core_mechanisms": ["..."], "implementation_notes": ["..."]},\n'
            '    "thinking": {"class_goal": "...", "core_mechanisms": ["..."], "implementation_notes": ["..."]}\n'
            '  },\n'
            '  "high_confidence_designs": [{"claim": "...", "why": "...", "module": "...", "source": "..."}],\n'
            '  "hypotheses_to_test": [{"claim": "...", "probe": "...", "risk": "..."}],\n'
            '  "avoid_or_low_priority": [{"idea": "...", "reason": "..."}],\n'
            '  "video_structuring_design": {"goal": "...", "mechanisms": ["..."], "source_support": ["..."]},\n'
            '  "localization_design": {"goal": "...", "mechanisms": ["..."], "source_support": ["..."]},\n'
            '  "perception_design": {"goal": "...", "mechanisms": ["..."], "source_support": ["..."]},\n'
            '  "memory_design": {"goal": "...", "mechanisms": ["..."], "source_support": ["..."]},\n'
            '  "thinking_design": {"goal": "...", "mechanisms": ["..."], "source_support": ["..."]},\n'
            '  "codex_instructions": "specific instructions for the later Codex code-writing agent",\n'
            '  "sources_used": [{"title": "...", "url": "https://...", "used_for": "..."}],\n'
            '  "risks": ["..."],\n'
            '  "source_quality": "strong|medium|weak"\n'
            "}\n"
        )
        task_line = (
            "Task: solve an initial-baseline engineering research problem, not a generic literature-survey task. "
            "Given the research_brief, use online research to decide how a five-module MetaVideoAgent should be designed for this observed video distribution."
        )
        schema_rule = "- Answer concrete designs for all five modules: video_structuring, localization, perception, memory, thinking.\n"
    payload = {
        "research_brief": build_research_brief(profile, diagnosis, review, queries),
        "runtime_capability_manual": build_runtime_capability_context(),
        "required_modules": MODULES,
        "year_min": year_min,
        "output_path": output_path,
        "min_verifiable_sources": min_sources,
    }
    return (
        "You are the Deep Research agent for MetaVideoAgent target-distribution evolution.\n\n"
        f"{task_line} "
        "You must use verifiable public sources such as papers, project pages, GitHub repositories, HuggingFace model/dataset pages, or benchmark documentation. "
        "Do not rely only on model memory. If you cannot access the internet or cannot provide verifiable URLs, set research_complete=false.\n\n"
        "Hard constraints:\n"
        "- Do not use human type/category/question_type/sub_category/domain labels.\n"
        "- Initial-baseline research may use only five-frame query-aware profile evidence and public sources.\n"
        "- For an initial baseline, treat the execution-layer builder as the owner of build-time media sampling/model calls; "
        "do not prescribe model calls inside video_structuring.addStructure. "
        "Only the registered runtime capability profiles supplied in the input are available.\n"
        "- Later-round research may also use diagnosis/review/full-eval/probe context provided in the input.\n"
        f"{schema_rule}"
        "- You may choose and refine your own searches. The provided seed queries are starting points, not a complete search plan.\n"
        "- Prefer academic and implementation-grade sources: arXiv, CVF, ACL Anthology, NeurIPS proceedings, OpenReview, official benchmark pages, GitHub repos, and HuggingFace pages.\n"
        "- Explicitly map methods from sources to implementation hints.\n"
        "- Separate high-confidence designs from hypotheses that still require probe/full-eval validation.\n"
        "- The runtime_capability_manual in the input is the implementation boundary for later Codex codegen. "
        "Prefer designs that can be expressed through those existing runtime helpers. If research suggests ASR, OCR, "
        "VLM or embedding tools, map the method to the documented profile_id, runtime call path, and fallback policy.\n"
        "- For later-round briefs, allowed_runtime_capabilities (including profile_id) is a binding subset selected by "
        "the validated diagnosis handoff. Do not introduce a provider/model outside that subset or a new runtime helper. "
        "For a later-round bundle, describe only module changes required by its diagnosed mechanism; do not turn "
        "the outer direction into a fixed edit whitelist.\n"
        "- For later-round briefs, functional_change_research must map every diagnosis_evolution_design.functional_changes item to a source-backed implementation method; do not replace the diagnosis design with a new one.\n"
        "- Every source in sources_used must have title, url, and used_for.\n"
        "- Use at least the requested number of verifiable sources.\n"
        "- Write JSON to the output_path and also print the same JSON to stdout.\n\n"
        "Required JSON schema:\n"
        f"{schema_text}\n"
        "Input JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _is_codex_runtime_environment_error(stderr: str) -> bool:
    text = stderr or ""
    markers = (
        "failed to open state db",
        "attempt to write a readonly database",
        "failed to initialize in-process app-server client",
        "Read-only file system",
    )
    return any(marker in text for marker in markers)


def _codex_runtime_environment_error_message(stderr: str) -> str:
    return (
        "Codex CLI failed before research because its runtime state directory is "
        "not writable. Run this stage in an environment where the Codex state "
        "directory is initialized and writable. Raw stderr:\n"
        + (stderr or "")[-3000:]
    )


def _run_codex_research(profile: Dict[str, Any],
                        diagnosis: Dict[str, Any],
                        review: Dict[str, Any],
                        queries: List[Dict[str, str]],
                        *,
                        output_dir: str,
                        timeout: int = 900,
                        codex_cli: str = "",
                        year_min: int = DEFAULT_YEAR_MIN,
                        min_sources: int = 2) -> Dict[str, Any]:
    settings = resolve_codex_runtime_settings()
    try:
        codex_bin = resolve_coding_agent_executable(codex_cli)
    except FileNotFoundError:
        codex_bin = ""
    output_path = os.path.join(output_dir, "codex_deep_research_output.json")
    prompt = _build_codex_prompt(profile, diagnosis, review, queries, output_path, year_min, min_sources)
    prompt_path = os.path.join(output_dir, "codex_deep_research_prompt.txt")
    stdout_path = os.path.join(output_dir, "codex_deep_research_stdout.txt")
    stderr_path = os.path.join(output_dir, "codex_deep_research_stderr.txt")
    write_text(prompt_path, prompt)
    if not codex_bin:
        return {
            "ok": False,
            "provider": "codex",
            "error": "Codex CLI not found; install it on PATH or pass --codex-cli",
            "prompt_path": prompt_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "design_brief": {},
            "validation": {"passed": False, "issues": ["codex CLI not found"]},
        }
    cmd = [
        codex_bin,
        "--search",
        "exec",
        *codex_exec_model_args(settings),
        "--sandbox",
        "workspace-write",
        "-C",
        os.path.abspath(output_dir),
        "--skip-git-repo-check",
        "-",
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=os.path.abspath(output_dir),
            text=True,
            input=prompt,
            capture_output=True,
            timeout=timeout,
            env=coding_agent_environment({
                "DEEP_RESEARCH_OUTPUT": output_path,
                "CODEX_MODEL": settings["model"],
                "CODEX_REASONING_EFFORT": settings["reasoning_effort"],
            }),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        write_text(stdout_path, stdout)
        write_text(stderr_path, stderr)
        return {
            "ok": False,
            "provider": "codex",
            "error": f"codex deep_research timeout > {timeout}s",
            "elapsed_sec": round(time.time() - started, 2),
            "prompt_path": prompt_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "design_brief": {},
            "validation": {"passed": False, "issues": ["codex timeout"]},
        }
    write_text(stdout_path, stdout)
    write_text(stderr_path, stderr)
    if _is_codex_runtime_environment_error(stderr):
        return {
            "ok": False,
            "provider": "codex",
            "returncode": proc.returncode,
            "error": _codex_runtime_environment_error_message(stderr),
            "elapsed_sec": round(time.time() - started, 2),
            "prompt_path": prompt_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "output_path": output_path,
            "cmd": cmd,
            "design_brief": {},
            "validation": {
                "passed": False,
                "issues": ["codex runtime environment is not writable"],
            },
        }
    brief = read_json(output_path) if os.path.exists(output_path) else {}
    if not brief:
        brief = _extract_json_object(stdout)
    if isinstance(brief, dict) and isinstance(brief.get("design_brief"), dict):
        brief = brief["design_brief"]
    validation = validate_design_brief(
        brief,
        min_sources=min_sources,
        verify_source_urls=True,
        expected_target_modules=_target_modules(diagnosis),
        allowed_runtime_capabilities=(
            ((diagnosis.get("implementation_task", {}) or {}).get("runtime_capabilities", []))
            if diagnosis.get("artifact_type") == "diagnosis_execution_brief" else None
        ),
        required_bundle_contract=(
            diagnosis if diagnosis.get("artifact_type") == "diagnosis_execution_brief" else None
        ),
    )
    research_complete = brief.get("research_complete") is True and validation.get("passed")
    return {
        "ok": proc.returncode == 0 and research_complete,
        "provider": "codex",
        "returncode": proc.returncode,
        "error": "" if proc.returncode == 0 and research_complete else (stderr[-1200:] or "codex output failed validation"),
        "elapsed_sec": round(time.time() - started, 2),
        "prompt_path": prompt_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "output_path": output_path,
        "cmd": cmd,
        "codex_runtime_settings": settings,
        "design_brief": brief,
        "validation": validation,
    }


def build_codex_research_summary(queries: List[Dict[str, str]],
                                 profile: Dict[str, Any],
                                 diagnosis: Dict[str, Any] | None = None,
                                 review: Dict[str, Any] | None = None,
                                 output_dir: str = "",
                                 timeout: int = 900,
                                 codex_cli: str = "",
                                 year_min: int = DEFAULT_YEAR_MIN,
                                 min_sources: int = 2) -> Dict[str, Any]:
    record = _run_codex_research(
        profile,
        diagnosis or {},
        review or {},
        queries,
        output_dir=output_dir,
        timeout=timeout,
        codex_cli=codex_cli,
        year_min=year_min,
        min_sources=min_sources,
    )
    design_brief = record.get("design_brief") or {}
    validation = record.get("validation") or {}
    complete = bool(record.get("ok") and validation.get("passed"))
    summary_text = render_markdown_summary(
        channel=(_channel_summary(profile).get("primary_observed_channel") or "mixed_or_unknown"),
        queries=queries,
        design_brief=design_brief,
        codex_stdout=read_text(record.get("stdout_path", "")),
        validation=validation,
        note=(
            "Codex deep research completed with verifiable sources and target-module implementation guidance."
            if complete else
            "Codex deep research failed validation. Do not use this for formal baseline/evolution until rerun succeeds."
        ),
    )
    return {
        "mode": "online",
        "search_provider": "codex",
        "online_success": complete,
        "research_complete": complete,
        "formal_research": complete,
        "failure_reasons": [] if complete else [{
            "provider": "codex",
            "error": record.get("error", ""),
            "validation_issues": validation.get("issues", []),
        }],
        "channel_context": _channel_summary(profile),
        "failure_context": _failure_summary(diagnosis or {}, review or {}),
        "queries": queries,
        "source_records": [record],
        "codex_record": record,
        "design_brief": design_brief,
        "validation": validation,
        "summary_text": summary_text,
    }


def read_text(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_markdown_summary(channel: str,
                            queries: List[Dict[str, str]],
                            design_brief: Dict[str, Any],
                            codex_stdout: str,
                            validation: Dict[str, Any],
                            note: str = "") -> str:
    lines = ["# Deep Research Summary", ""]
    if note:
        lines.append(note)
        lines.append("")
    lines.append(f"- observed_primary_channel: `{channel}`")
    lines.append(f"- validation_passed: `{validation.get('passed')}`")
    if validation.get("issues"):
        lines.append("- validation_issues:")
        for issue in validation.get("issues", [])[:10]:
            lines.append(f"  - {issue}")
    lines.append("- research_queries:")
    for q in queries:
        targets = ", ".join(q.get("target_modules") or []) or "initial design"
        lines.append(f"  - `{targets}`: {q.get('query')} ({q.get('reason')})")
    lines.append("")
    lines.append("## Verifiable Sources")
    sources = design_brief.get("sources_used") or []
    if not sources:
        lines.append("- No validated sources.")
    for source in sources:
        lines.append(f"- {source.get('title')}: {source.get('url')} ({source.get('used_for')})")
    lines.append("")
    plan = design_brief.get("research_plan") or {}
    if plan:
        lines.append("## Research Plan")
        lines.append(f"- problem_understanding: {plan.get('problem_understanding', '')}")
        uncertainties = plan.get("key_uncertainties") or []
        if uncertainties:
            lines.append("- key_uncertainties:")
            for item in uncertainties[:8]:
                lines.append(f"  - {item}")
        strategy = plan.get("source_strategy") or []
        if strategy:
            lines.append("- source_strategy:")
            for item in strategy[:8]:
                lines.append(f"  - {item}")
        lines.append("")
    lines.append("## Five-Module Design Brief")
    target_modules = list(design_brief.get("target_modules") or [])
    module_set_design = design_brief.get("module_set_design") or {}
    if target_modules and isinstance(module_set_design, dict):
        lines[-1] = "## Bundle Evolution Design"
        lines.append(f"- target_modules: `{', '.join(target_modules)}`")
        lines.append(f"- diagnosis_support_summary: {design_brief.get('diagnosis_support_summary', '')}")
        lines.append(f"- goal: {module_set_design.get('goal', '')}")
        lines.append(f"- mechanism: {module_set_design.get('mechanism', '')}")
        roles = module_set_design.get("candidate_module_roles") or {}
        if isinstance(roles, dict):
            for module, role in roles.items():
                if module in MODULES and str(role).strip():
                    lines.append(f"- **{module}**: {role}")
        handoffs = module_set_design.get("handoff_contracts") or []
        if handoffs:
            lines.append("- handoff_contracts:")
            for handoff in handoffs[:10]:
                if isinstance(handoff, dict):
                    lines.append(
                        f"  - {handoff.get('producer', '')} -> {handoff.get('consumer', '')}: "
                        f"{handoff.get('contract', '')}"
                    )
    else:
        lines.append(f"- distribution_interpretation: {design_brief.get('distribution_interpretation', '')}")
        lines.append(f"- recommended_architecture: {design_brief.get('recommended_architecture', '')}")
        for module in MODULES:
            item = design_brief.get(f"{module}_design") or {}
            lines.append(f"- **{module}**: {item.get('goal', '')}")
            for mechanism in item.get("mechanisms", [])[:5]:
                if mechanism:
                    lines.append(f"  - {str(mechanism)[:900]}")
    module_map = design_brief.get("module_source_map") or {}
    if module_map:
        lines.append("")
        lines.append("## Method To Module Map")
        for module in MODULES:
            lines.append(f"### {module}")
            mappings = module_map.get(module) or []
            if not mappings:
                lines.append("- No mapping.")
                continue
            for mapping in mappings[:5]:
                if not isinstance(mapping, dict):
                    continue
                lines.append(
                    "- "
                    f"source={mapping.get('source', '')}; "
                    f"method={mapping.get('method_idea', '')}; "
                    f"hint={mapping.get('implementation_hint', '')}"
                )
    design_spec = design_brief.get("baseline_combo_design_spec") or {}
    if design_spec:
        lines.append("")
        lines.append("## Baseline Combo Design Spec")
        lines.append(f"- overall_strategy: {design_spec.get('overall_strategy', '')}")
        for module in MODULES:
            spec = design_spec.get(module) or {}
            lines.append(f"- **{module}**: {spec.get('class_goal') or spec.get('goal') or ''}")
            for mechanism in (spec.get("core_mechanisms") or spec.get("mechanisms") or [])[:5]:
                lines.append(f"  - {mechanism}")
    if any(design_brief.get(bucket) for bucket in RESEARCH_BUCKETS):
        lines.append("")
        lines.append("## Confidence And Test Buckets")
        for bucket in RESEARCH_BUCKETS:
            value = design_brief.get(bucket) or []
            lines.append(f"### {bucket}")
            if not value:
                lines.append("- None.")
                continue
            for item in value[:8]:
                if isinstance(item, dict):
                    lines.append("- " + "; ".join(f"{k}={v}" for k, v in item.items()))
                else:
                    lines.append(f"- {item}")
    if design_brief.get("codex_instructions"):
        lines.append("")
        lines.append("## Codex Instructions")
        lines.append(str(design_brief.get("codex_instructions")))
    if codex_stdout and not design_brief:
        lines.append("")
        lines.append("## Raw Codex Output Preview")
        lines.append(codex_stdout[:3000])
    lines.append("")
    lines.append("Use this research as design evidence; success still requires probe/full-eval validation.")
    return "\n".join(lines)


def run_deep_research(
    *,
    workspace: str,
    channel_profile_path: str,
    output_dir: str = "",
    run_id: str = "",
    output_root: str = "",
    diagnosis_path: str = "",
    review_path: str = "",
    max_queries: int = 3,
    search_timeout: int = 900,
    codex_cli: str = "",
    year_min: int = DEFAULT_YEAR_MIN,
    min_sources: int = 2,
) -> Dict[str, Any]:
    profile = read_json(channel_profile_path)
    diagnosis = read_json(diagnosis_path)
    review = read_json(review_path)
    if diagnosis.get("artifact_type") == "diagnosis_execution_brief":
        issues = validate_bundle_execution_brief(diagnosis)
        if issues:
            raise RuntimeError(
                "refusing diagnosis-conditioned deep research with invalid execution brief: "
                + "; ".join(issues)
            )
    if not output_dir:
        output_dir = default_research_dir(run_id or f"deep_research_{int(time.time())}", output_root)
    os.makedirs(output_dir, exist_ok=True)
    queries = build_research_queries(profile, diagnosis, review, max_queries=max_queries)

    result = build_codex_research_summary(
        queries,
        profile,
        diagnosis,
        review,
        output_dir=output_dir,
        timeout=search_timeout,
        codex_cli=codex_cli,
        year_min=year_min,
        min_sources=min_sources,
    )

    result.update({
        "schema_version": 1,
        "created_at": int(time.time()),
        "workspace": os.path.abspath(workspace),
        "channel_profile_path": os.path.abspath(channel_profile_path) if channel_profile_path else "",
        "diagnosis_path": os.path.abspath(diagnosis_path) if diagnosis_path else "",
        "review_path": os.path.abspath(review_path) if review_path else "",
        "search_provider": "codex",
        "research_complete": bool(result.get("research_complete")),
        "formal_research": bool(result.get("formal_research")),
        "search_timeout": search_timeout,
        "year_min": year_min,
        "min_sources": min_sources,
        "target_modules": _target_modules(diagnosis),
        "leakage_policy": (
            "Research queries are derived from the five-frame, query-aware profile "
            "and, after the initial reference exists, diagnosis/review/full-eval/probe "
            "artifacts. Human type/category labels are not used."
        ),
    })
    report_path = os.path.join(output_dir, "deep_research_report.json")
    summary_path = os.path.join(output_dir, "deep_research_summary.md")
    design_brief_path = os.path.join(output_dir, "deep_research_design_brief.json")
    result["report_path"] = report_path
    result["summary_path"] = summary_path
    result["design_brief_path"] = design_brief_path
    write_json(design_brief_path, result.get("design_brief", {}))
    write_json(report_path, result)
    write_text(summary_path, result["summary_text"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Codex-backed deep-research context for target-distribution evolution"
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--channel-profile", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--diagnosis", default="")
    parser.add_argument("--review", default="")
    parser.add_argument("--max-queries", type=int, default=3)
    parser.add_argument("--search-timeout", type=int, default=900)
    parser.add_argument(
        "--codex-cli",
        default="",
        help="Codex executable or a wrapper implementing the same command contract.",
    )
    parser.add_argument("--year-min", type=int, default=DEFAULT_YEAR_MIN)
    parser.add_argument("--min-sources", type=int, default=2)
    args = parser.parse_args()

    result = run_deep_research(
        workspace=args.workspace,
        channel_profile_path=args.channel_profile,
        output_dir=args.output_dir,
        run_id=args.run_id,
        output_root=args.output_root,
        diagnosis_path=args.diagnosis,
        review_path=args.review,
        max_queries=args.max_queries,
        search_timeout=args.search_timeout,
        codex_cli=args.codex_cli,
        year_min=args.year_min,
        min_sources=args.min_sources,
    )
    print("DEEP_RESEARCH_DONE")
    print(f"research_complete={result.get('research_complete')}")
    print(f"formal_research={result.get('formal_research')}")
    print(f"report_path={result.get('report_path')}")
    print(f"summary_path={result.get('summary_path')}")
    print(f"design_brief_path={result.get('design_brief_path')}")
    return 0 if result.get("research_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
