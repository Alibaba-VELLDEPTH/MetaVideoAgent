"""Contracts for self-contained five-module MetaVideoAgent candidate bundles."""

from __future__ import annotations

import hashlib
import json
from typing import Any

MODULE_TYPES = (
    "video_structuring", "localization", "perception", "memory", "thinking",
)
COMBO_SLOT = {
    "video_structuring": "video_structuring",
    "localization": "localization",
    "perception": "perception",
    "memory": "memory",
    "thinking": "thinking",
}


def normalize_modules(value: Any, *, allow_empty: bool = False) -> list[str]:
    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in values:
        module = str(item or "").strip()
        if module and module not in result:
            result.append(module)
    if not allow_empty and not result:
        raise ValueError("target_modules must be a non-empty list")
    invalid = [item for item in result if item not in MODULE_TYPES]
    if invalid:
        raise ValueError(f"unsupported module types: {invalid}")
    return result


def bundle_fingerprint(bundle: dict) -> str:
    """Fingerprint code/interface facts, never task data or output paths."""
    bundle = bundle if isinstance(bundle, dict) else {}
    modules = bundle.get("modules") or {}
    if isinstance(modules, list):
        modules = {str(item.get("module_type") or ""): item for item in modules if isinstance(item, dict)}
    payload = {
        "changed_modules": sorted(normalize_modules(bundle.get("changed_modules") or list(modules), allow_empty=True)),
        "combo": bundle.get("combo") or {},
        "modules": {
            key: {
                "name": str((value or {}).get("name") or ""),
                "code": str((value or {}).get("code") or ""),
                "source_file": str((value or {}).get("source_file") or ""),
            }
            for key, value in sorted(modules.items()) if isinstance(value, dict)
        },
        "handoff_contracts": bundle.get("handoff_contracts") or [],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "bundle_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def validate_handoff_contracts(contracts: Any, *, changed_modules: list[str]) -> list[str]:
    issues: list[str] = []
    if not isinstance(contracts, list):
        return ["handoff_contracts must be a list"]
    if changed_modules and not contracts:
        return ["handoff_contracts must describe at least one affected bundle boundary"]
    seen = set()
    for index, item in enumerate(contracts):
        prefix = f"handoff_contracts[{index}]"
        if not isinstance(item, dict):
            issues.append(f"{prefix} must be an object")
            continue
        producer = str(item.get("producer") or "")
        consumer = str(item.get("consumer") or "")
        if producer not in MODULE_TYPES or consumer not in MODULE_TYPES:
            issues.append(f"{prefix} producer/consumer must be valid modules")
        if producer == consumer:
            issues.append(f"{prefix} producer and consumer must differ")
        key = (producer, consumer, tuple(item.get("required_fields") or []))
        if key in seen:
            issues.append(f"{prefix} duplicates an earlier handoff")
        seen.add(key)
        fields = item.get("required_fields")
        if not isinstance(fields, list) or not all(str(field).strip() for field in fields):
            issues.append(f"{prefix}.required_fields must be a non-empty list of names")
        for field in ("output_protocol", "consumer_behavior", "fallback"):
            if not str(item.get(field) or "").strip():
                issues.append(f"{prefix}.{field} is empty")
        # A contract must be relevant to an actual changed edge.  It may have
        # one preserved endpoint, but cannot describe two untouched modules.
        if producer not in changed_modules and consumer not in changed_modules:
            issues.append(f"{prefix} has no changed endpoint")
    return issues


def validate_candidate_bundle(bundle: Any, *, require_source: bool = True) -> list[str]:
    """Validate persisted bundle shape before static/runtime validation."""
    if not isinstance(bundle, dict):
        return ["candidate bundle must be an object"]
    issues: list[str] = []
    if bundle.get("artifact_type") != "candidate_bundle":
        issues.append("artifact_type must be candidate_bundle")
    if int(bundle.get("schema_version", 0) or 0) != 1:
        issues.append("candidate bundle schema_version must be 1")
    try:
        changed = normalize_modules(bundle.get("changed_modules"))
    except ValueError as exc:
        return [str(exc)]
    retained = bundle.get("retained_modules") or []
    try:
        retained = normalize_modules(retained, allow_empty=True)
    except ValueError as exc:
        issues.append(str(exc))
        retained = []
    if set(changed) & set(retained):
        issues.append("changed_modules and retained_modules overlap")
    if set(changed) | set(retained) != set(MODULE_TYPES):
        issues.append("changed_modules plus retained_modules must explicitly cover all five modules")
    combo = bundle.get("combo") or {}
    missing_slots = [slot for slot in COMBO_SLOT.values() if not str(combo.get(slot) or "").strip()]
    if missing_slots:
        issues.append(f"combo is missing required slots: {missing_slots}")
    modules = bundle.get("modules") or {}
    if not isinstance(modules, dict):
        return issues + ["modules must map module_type to module records"]
    # The artifact is self-contained: retained modules are copied from the
    # immutable current-best bundle, while changed modules carry the new code.
    # This prevents a later replay from silently resolving names through a
    # mutable archive.
    for module in MODULE_TYPES:
        item = modules.get(module)
        if not isinstance(item, dict):
            issues.append(f"modules.{module} is required for every bundle module")
            continue
        if str(item.get("module_type") or module) != module:
            issues.append(f"modules.{module}.module_type disagrees with its key")
        for field in ("name", "source_file", "code"):
            if require_source and not str(item.get(field) or "").strip():
                issues.append(f"modules.{module}.{field} is empty")
        slot = COMBO_SLOT[module]
        if str(combo.get(slot) or "") != str(item.get("name") or ""):
            issues.append(f"combo.{slot} must name modules.{module}.name")
    issues.extend(validate_handoff_contracts(bundle.get("handoff_contracts") or [], changed_modules=changed))
    spec = bundle.get("implementation_spec") or {}
    if spec.get("contract_mode") == "adaptive_bundle":
        if spec.get("module_protocol_version") != "metavideoagent_module_protocol":
            issues.append("adaptive bundle must declare module_protocol_version=metavideoagent_module_protocol")
        responsibilities = spec.get("module_responsibilities") or {}
        if not isinstance(responsibilities, dict):
            issues.append("adaptive bundle implementation_spec.module_responsibilities must be an object")
        else:
            for module in MODULE_TYPES:
                item = responsibilities.get(module)
                if not isinstance(item, dict):
                    issues.append(f"adaptive bundle responsibility for {module} is missing")
                    continue
                for field in ("responsibility", "input_invariant", "output_invariant", "fallback"):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"adaptive bundle responsibility {module}.{field} is empty")
    declared = str(bundle.get("bundle_fingerprint") or "")
    if declared and declared != bundle_fingerprint(bundle):
        issues.append("bundle_fingerprint does not match candidate contents")
    return issues


def make_candidate_bundle(*, combo: dict, modules: dict, changed_modules: list[str],
                          handoff_contracts: list[dict], rationale: dict | None = None,
                          implementation_spec: dict | None = None) -> dict:
    changed = normalize_modules(changed_modules)
    bundle = {
        "artifact_type": "candidate_bundle",
        "schema_version": 1,
        "combo": dict(combo or {}),
        "changed_modules": changed,
        "retained_modules": [module for module in MODULE_TYPES if module not in changed],
        "modules": dict(modules or {}),
        "handoff_contracts": list(handoff_contracts or []),
        "change_rationale": dict(rationale or {}),
        "implementation_spec": dict(implementation_spec or {}),
    }
    bundle["bundle_fingerprint"] = bundle_fingerprint(bundle)
    return bundle
