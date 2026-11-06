"""Research source-backed designs and implementation tasks for MetaVideoAgent."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List

if not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution.bundle_contract import normalize_modules
from evolution.channel_profile_runner import compact_channel_summary
from evolution.codex_runtime_settings import codex_exec_model_args, resolve_codex_runtime_settings
from evolution.coding_agent_runtime import coding_agent_environment, resolve_coding_agent_executable
from evolution.diagnosis_execution_brief import validate_bundle_execution_brief
from evolution.runtime_capability_context import (
    DEFAULT_PROFILE_IDS,
    build_runtime_capability_context,
)
from evolution.runtime_paths import metavideoagent_run_dir

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
    """Pass the actual investigation forward, including evidence and open work."""
    if not diagnosis:
        return {}
    task = diagnosis.get("implementation_task") or {}
    return {
        "diagnosis_target_modules": _target_modules(diagnosis),
        "summary": diagnosis.get("summary", ""),
        "analysis": diagnosis.get("analysis", ""),
        "open_questions": diagnosis.get("open_questions", []),
        "next_task": diagnosis.get("next_task", {}),
        "evolution_design": diagnosis.get("evolution_design", {}),
        "failure_chain": task.get("failure_chain", ""),
        "evolution_mechanism": task.get("evolution_mechanism", ""),
        "handoff_contracts": task.get("handoff_contracts", []),
        "diagnosis_evolution_hint": task.get("goal", ""),
        "working_materials": diagnosis.get("working_materials", {}),
        "review": review,
        "history": diagnosis.get("iter_context", {}),
        "allowed_runtime_capabilities": build_runtime_capability_context().get("profiles", []),
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
    failure = _failure_summary(diagnosis, review)
    later_round = bool(target_modules)
    if later_round:
        objective = (
            "Research implementation methods for the diagnosis-selected bundle "
            "mechanism, responsibility boundaries, and producer/consumer handoffs. "
            "Ground the implementation in reusable runtime behavior."
        )
        research_questions = list((diagnosis.get("research_plan") or {}).get("questions") or []) + list(diagnosis.get("open_questions") or []) + [
            "Which concrete methods implement the diagnosed evolution mechanism without task-specific rules?",
            "What state, stopping, verification, and fallback logic should the connected modules add?",
            "Which producer/consumer contracts must remain stable across the bundle?",
            "What smoke and probe failure modes should Codex guard against while coding?",
        ]
    else:
        objective = (
            "Design an initial five-module MetaVideoAgent baseline for this five-frame, "
            "query-aware distribution."
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
                "gold answers", "annotated evidence intervals",
                "human type/category labels", "task trajectories",
            ],
            "later_round_rule": (
                "Research develops the diagnosed mechanism and boundary contracts. "
                "Codex may adapt related modules when implementation or execution "
                "evidence supports the change, while preserving unaffected behavior."
            ),
            "initial_execution_lifecycle": (
                "For initial-baseline research, the execution-layer structure builder owns "
                "bounded media sampling and model calls. Generated addStructure methods only "
                "persist or index supplied records. Use registered runtime adapters rather "
                "than provider clients or local model toolboxes. The configured ASR profile is "
                f"{DEFAULT_PROFILE_IDS['asr']}; the configured VLM profile is "
                f"{DEFAULT_PROFILE_IDS['vlm']}. An optional buildStructure(env) hook may "
                "own a custom bounded build and must persist canonical records; addStructure "
                "itself remains a persistence/indexing operation."
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
                          expected_target_modules: List[str] | None = None) -> Dict[str, Any]:
    """Check source-backed findings and the next implementation task."""
    issues: List[str] = []
    if not isinstance(brief, dict) or not brief:
        return {"passed": False, "issues": ["missing design_brief"], "valid_source_count": 0}
    if brief.get("research_complete") is not True:
        issues.append("research is incomplete; resolve or report the remaining questions")
    sources = brief.get("sources_used") or []
    valid_sources = []
    unreachable_sources = []
    domains = {"academic_or_official": 0, "implementation_or_model": 0, "other": 0}
    for source in sources if isinstance(sources, list) else []:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or source.get("link") or "")
        if not (source.get("title") or source.get("name")) or not source.get("used_for") or not _valid_source_url(url):
            continue
        valid_sources.append(source)
        domains[_source_domain_class(url)] += 1
        # A transient HTTP failure is not evidence that a source is fabricated.
        # Retain it as a warning; Codex must inspect the sources during research.
        if verify_source_urls and not _source_reachable(url, timeout=source_timeout):
            unreachable_sources.append(source)
    if len(valid_sources) < min_sources:
        issues.append(f"need at least {min_sources} sources with title, public URL and implementation relevance")
    source_titles = {
        str(source.get("title") or source.get("name")).strip().casefold()
        for source in valid_sources
    }

    def linked_source(title):
        return isinstance(title, str) and title.strip().casefold() in source_titles

    if expected_target_modules:
        design = brief.get("module_set_design") or {}
        for field in ("goal", "mechanism"):
            if not isinstance(design, dict) or not str(design.get(field) or "").strip():
                issues.append(f"module_set_design.{field} must explain the implementation approach")
        try:
            normalize_modules(brief.get("target_modules") or expected_target_modules)
        except ValueError as exc:
            issues.append(str(exc))
        findings = brief.get("functional_change_research") or []
        if not isinstance(findings, list) or not findings:
            issues.append("functional_change_research needs source-backed implementation findings")
        else:
            for index, finding in enumerate(findings):
                if not isinstance(finding, dict) or not all(
                    isinstance(finding.get(field), str) and finding[field].strip()
                    for field in ("component", "method", "implementation_guidance")
                ):
                    issues.append(f"functional_change_research[{index}] needs a component, method and implementation_guidance")
                    continue
                support = finding.get("source_support") or []
                if not isinstance(support, list) or not support or not all(linked_source(title) for title in support):
                    issues.append(f"functional_change_research[{index}].source_support must cite sources_used titles")
    else:
        spec = brief.get("baseline_combo_design_spec") or {}
        if not isinstance(spec, dict) or not spec.get("overall_strategy"):
            issues.append("baseline_combo_design_spec.overall_strategy is missing")
        for module in MODULES:
            design = spec.get(module) if isinstance(spec, dict) else None
            if not isinstance(design, dict) or not (design.get("class_goal") or design.get("goal")):
                issues.append(f"baseline_combo_design_spec.{module} needs a concrete responsibility")
            elif not (design.get("core_mechanisms") or design.get("mechanisms")):
                issues.append(f"baseline_combo_design_spec.{module} needs an implementation mechanism")
            if isinstance(design, dict):
                support = design.get("source_support") or []
                if not isinstance(support, list) or not support or not all(
                    isinstance(item, dict) and linked_source(item.get("source"))
                    and all(isinstance(item.get(field), str) and item[field].strip()
                            for field in ("method_idea", "implementation_hint"))
                    for item in support
                ):
                    issues.append(f"baseline_combo_design_spec.{module}.source_support needs a cited method and implementation_hint")
    next_task = brief.get("next_task") or {}
    if (not isinstance(next_task, dict) or not str(next_task.get("objective") or "").strip()
            or not isinstance(next_task.get("implementation_steps"), list)
            or not next_task["implementation_steps"]
            or not all(isinstance(step, str) and step.strip() for step in next_task["implementation_steps"])):
        issues.append("next_task must describe an objective and implementation_steps")
    return {
        "passed": not issues, "issues": issues,
        "valid_source_count": len(valid_sources), "min_sources": min_sources,
        "source_url_verification": bool(verify_source_urls),
        "source_domain_counts": domains, "unreachable_sources": unreachable_sources,
        "warnings": [f"source URL was not reachable during this check: {item.get('url') or item.get('link')}"
                     for item in unreachable_sources],
    }


def _build_codex_prompt(profile: Dict[str, Any],
                        diagnosis: Dict[str, Any],
                        review: Dict[str, Any],
                        queries: List[Dict[str, str]],
                        output_path: str,
                        year_min: int,
                        min_sources: int) -> str:
    target_modules = _target_modules(diagnosis)
    schema = {
        "research_complete": True,
        "research_plan": {"problem_understanding": "...", "key_uncertainties": ["..."], "source_strategy": ["..."]},
        "executed_searches": [{"query": "...", "source_target": "academic|implementation|benchmark|model",
                               "why": "...", "top_sources_considered": ["..."]}],
        "rejected_sources": [{"title": "...", "url": "https://...", "reason": "..."}],
        "analysis": "Evidence, findings, alternatives, implementation reasoning and tradeoffs",
        "open_questions": ["Unresolved implementation or evidence questions"],
        "next_task": {"stage": "codegen", "objective": "...",
                      "implementation_steps": ["Concrete behavioral changes"], "checks": ["Observable smoke/probe checks"]},
        "high_confidence_designs": [{"claim": "...", "why": "...", "module": "...", "source": "..."}],
        "hypotheses_to_test": [{"claim": "...", "probe": "...", "risk": "..."}],
        "avoid_or_low_priority": [{"idea": "...", "reason": "..."}],
        "sources_used": [{"title": "...", "url": "https://...", "used_for": "..."}],
        "risks": ["..."],
        "source_quality": "strong|medium|weak",
    }
    if target_modules:
        schema.update({
            "target_modules": target_modules,
            "diagnosis_support_summary": "Findings that support or qualify the causal diagnosis",
            "module_set_design": {
                "goal": "...", "mechanism": "...", "candidate_module_roles": {module: "..." for module in target_modules},
                "handoff_contracts": [{"producer": "...", "consumer": "...", "contract": "..."}],
            },
            "runtime_capability_mapping": [{"capability": "...", "profile_id": "registered profile ID",
                                            "runtime_call": "bounded adapter", "use": "..."}],
            "functional_change_research": [{"component": "module or handoff", "method": "source-backed method",
                                            "implementation_guidance": "...", "source_support": ["source title"]}],
        })
        task = (
            "Investigate the diagnosed evolution mechanism. Explain the affected responsibilities, "
            "producer/consumer handoffs and implementation tradeoffs. Codex may adapt related "
            "modules when implementation or execution evidence supports the change."
        )
    else:
        schema.update({
            "distribution_interpretation": "...",
            "baseline_combo_design_spec": {
                "overall_strategy": "...",
                **{module: {
                    "class_goal": "...", "core_mechanisms": ["..."], "implementation_notes": ["..."],
                    "source_support": [{"source": "exact source title", "method_idea": "...",
                                        "why_relevant": "...", "implementation_hint": "..."}],
                } for module in MODULES},
            },
        })
        task = (
            "Research the initial five-module design for the observed video distribution. "
            "For each module, connect its responsibility, mechanism and implementation guidance "
            "to the relevant public sources."
        )
    payload = {
        "research_brief": build_research_brief(profile, diagnosis, review, queries),
        "runtime_capability_manual": build_runtime_capability_context(),
        "required_modules": MODULES,
        "year_min": year_min,
        "output_path": output_path,
        "min_verifiable_sources": min_sources,
    }
    return (
        "You are the Deep Research stage of MetaVideoAgent.\n\n"
        f"{task}\n\n"
        "Research requirements:\n"
        "- Investigate public papers, official documentation and implementation repositories. "
        "Use at least min_verifiable_sources relevant sources with verifiable URLs, "
        "and explain each source's relevance. If access or evidence is insufficient, "
        "set research_complete=false and describe the remaining work.\n"
        "- Initial research uses the five-frame query-aware distribution profile and public sources. "
        "Keep answer labels, annotated evidence intervals and human category labels outside this stage. "
        "Later research may also use the supplied training review, diagnosis, evaluation and probe evidence.\n"
        "- Read upstream analysis, current source, cited training evidence and open questions in "
        "working_materials when they affect the design. Preserve trace and source references.\n"
        "- Resolve the implementation questions and compare alternatives. When evidence contradicts "
        "the diagnosis, explain the consequence in analysis and open_questions.\n"
        "- Distinguish source-supported designs from hypotheses that require execution. Explain the "
        "behavioral change, downstream consumption, fallback behavior, cost and observable checks.\n"
        "- Use the configured runtime capability inventory. Map model/media operations to registered "
        "profiles and bounded adapters; Teacher audit tools belong to review. Select capabilities that "
        "close the diagnosed information gap, including an empty capability plan when none is needed.\n"
        "- The generic execution builder performs build-time media sampling and model calls. "
        "A custom buildStructure(env) may implement a bounded build with registered adapters. "
        "addStructure persists and indexes canonical evidence or media references.\n"
        "- Seed queries are starting points; refine searches as findings develop. Prefer research "
        "papers, official projects, benchmark documentation and implementation-grade sources.\n"
        "- Record each module design once, with its supporting sources. Keep substantive reasoning "
        "in analysis and the actionable implementation handoff in next_task.\n"
        "- Write the complete JSON to output_path. Return a short completion summary and remaining work.\n\n"
        "Output schema:\n" + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\nInput JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
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
    lines = ["# Deep Research", ""]
    if note:
        lines.extend([note, ""])
    lines.extend([
        f"- Observed primary channel: {channel}",
        f"- Research complete: {validation.get('passed')}",
        "",
    ])

    def section(title: str, value) -> None:
        if value:
            content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
            lines.extend([f"## {title}", "", content, ""])

    section("Validation issues", validation.get("issues"))
    section("Analysis", design_brief.get("analysis"))
    section("Distribution interpretation", design_brief.get("distribution_interpretation"))
    section("Research plan", design_brief.get("research_plan"))
    section("Seed queries", queries)
    section("Executed searches", design_brief.get("executed_searches"))
    section("Sources considered but not used", design_brief.get("rejected_sources"))

    target_modules = design_brief.get("target_modules") or []
    if target_modules:
        section("Target modules", target_modules)
        section("Diagnosis findings", design_brief.get("diagnosis_support_summary"))
        section("Evolution design", design_brief.get("module_set_design"))
        section("Implementation methods", design_brief.get("functional_change_research"))
        section("Runtime capability plan", design_brief.get("runtime_capability_mapping"))
    else:
        design = design_brief.get("baseline_combo_design_spec") or {}
        section("Overall strategy", design.get("overall_strategy"))
        for module in MODULES:
            section(module, design.get(module))

    section("Sources", design_brief.get("sources_used"))
    for bucket in RESEARCH_BUCKETS:
        section(bucket.replace("_", " ").capitalize(), design_brief.get(bucket))
    section("Risks", design_brief.get("risks"))
    section("Open questions", design_brief.get("open_questions"))
    section("Next implementation task", design_brief.get("next_task"))
    if codex_stdout and not design_brief:
        section("Research command output", codex_stdout[:3000])
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
    if (diagnosis.get("iter_context") or {}).get("iter_dir"):
        from evolution.iteration_workspace import IterationWorkspace
        diagnosis["iter_context"]["stage_context"] = IterationWorkspace(diagnosis["iter_context"]["iter_dir"]).context("research")
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
