"""Isolated runtime assembly and evaluation for MetaVideoAgent bundles.

This module validates five-module bundle manifests, injects their classes into
the execution runtime, runs smoke or multi-question evaluations, and records
reference-versus-candidate evidence in a run-scoped sandbox.
"""

import glob
import json
import math
import os
import re
import shutil
import signal
import sys
import time
import traceback as tb
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
from runtime_paths import (
    default_workspace,
    runtime_dir,
    use_runtime_path,
)

_runtime_dir = use_runtime_path()
from answer_normalizer import (
    extract_gold_answer,
    is_answer_parseable,
    judge_answer,
)
from combo_contract import require_complete_agent_combo
from evidence_contract import validate_canonical_structure_record
from task_identity import (
    attach_task_identity,
    identity_from_entry,
    make_task_id,
)

BUNDLE_MODULE_ORDER = (
    "video_structuring", "localization", "perception", "memory", "thinking",
)
BUNDLE_COMBO_KEY = {module_type: module_type for module_type in BUNDLE_MODULE_ORDER}


def _ensure_runtime_path_for_workspace(workspace_dir: str) -> None:
    """Keep sandbox imports aligned with the workspace under evaluation."""
    try:
        workspace_abs = os.path.abspath(workspace_dir or "")
    except Exception:
        workspace_abs = ""
    active_workspace_abs = os.path.abspath(default_workspace())
    if workspace_abs != active_workspace_abs and not workspace_abs.startswith(active_workspace_abs + os.sep):
        return

    runtime_abs = os.path.abspath(runtime_dir())
    os.environ["METAVIDEOAGENT_RUNTIME_DIR"] = runtime_abs
    if runtime_abs in sys.path:
        sys.path.remove(runtime_abs)
    sys.path.insert(0, runtime_abs)

    runtime_module_names = (
        "run_video_qa",
        "agent",
        "module_map",
        "utils",
        "video_structuring_modules",
        "thinking_module",
        "memory_modules",
        "localization_modules",
        "perception_modules",
    )
    for name in runtime_module_names:
        module = sys.modules.get(name)
        module_file = os.path.abspath(getattr(module, "__file__", "") or "") if module else ""
        if module_file and not module_file.startswith(runtime_abs):
            sys.modules.pop(name, None)


def _runtime_report_path(filename: str = "") -> str:
    """Return the active runtime path used in machine-readable repair reports."""
    relative_runtime = os.path.relpath(runtime_dir(), os.path.dirname(_current_dir))
    relative_runtime = relative_runtime.replace(os.sep, "/")
    return f"{relative_runtime}/{filename}" if filename else relative_runtime



def _extract_time_reference_bounds(time_reference) -> tuple:
    starts = []
    ends = []
    if isinstance(time_reference, (int, float)):
        starts.append(float(time_reference))
        ends.append(float(time_reference))
    elif isinstance(time_reference, str):
        nums = re.findall(r"\d+(?:\.\d+)?", time_reference)
        if nums:
            starts.append(float(nums[0]))
            ends.append(float(nums[-1]))
    elif isinstance(time_reference, list):
        for item in time_reference:
            if isinstance(item, (int, float)):
                starts.append(float(item))
                ends.append(float(item))
            elif isinstance(item, list) and item:
                starts.append(float(item[0]))
                ends.append(float(item[-1]))
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def _extract_time_reference_windows(time_reference) -> list:
    """Parse exact referenced intervals without adding probe-only padding."""
    windows = []

    def _add(start, end):
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            return
        if end < start:
            start, end = end, start
        if end > start:
            windows.append((start, end))

    if isinstance(time_reference, (int, float)):
        point = float(time_reference)
        _add(point, point + 0.001)
    elif isinstance(time_reference, (list, tuple)):
        if len(time_reference) >= 2 and all(isinstance(item, (int, float)) for item in time_reference[:2]):
            _add(time_reference[0], time_reference[1])
        else:
            for item in time_reference:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    _add(item[0], item[1])
                elif isinstance(item, (int, float)):
                    point = float(item)
                    _add(point, point + 0.001)
    elif isinstance(time_reference, str):
        # Covers common forms such as ``[[910, 933]]``, ``910-933`` and
        # ``910s to 933s`` while retaining multiple disjoint intervals.
        pairs = re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:s)?\s*(?:,|~|–|—|-|to)\s*(\d+(?:\.\d+)?)\s*(?:s)?",
            time_reference,
            flags=re.IGNORECASE,
        )
        for start, end in pairs:
            _add(start, end)
        if not pairs:
            bounds = _extract_time_reference_bounds(time_reference)
            if bounds:
                _add(*bounds)

    windows.sort()
    merged = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(round(start, 3), round(end, 3)) for start, end in merged]


def _record_intersects_windows(record: dict, windows: list) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        start = float(record.get("start_sec", 0.0) or 0.0)
        end = float(record.get("end_sec", start) or start)
    except (TypeError, ValueError):
        return False
    if end < start:
        start, end = end, start
    return any(min(end, window_end) >= max(start, window_start) for window_start, window_end in windows)


def _restrict_sandbox_structure_artifact(structuring, windows: list) -> dict:
    """Physically restrict a sandbox-only JSONL/Chroma artifact to probe media.

    This is deliberately below module code: generated modules that read
    ``struct_db.db_path`` directly, call ``_load_all_records()``, or use a
    vector collection all see the same time-reference-only artifact.
    """
    db_path = getattr(structuring, "db_path", "")
    if not db_path or not os.path.isfile(db_path):
        raise RuntimeError("Probe structure artifact is missing before active-window restriction.")
    temp_path = f"{db_path}.probe-window.tmp"
    total = kept = 0
    try:
        with open(db_path, "r", encoding="utf-8") as source, open(temp_path, "w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _record_intersects_windows(record, windows):
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
        os.replace(temp_path, db_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if getattr(structuring, "requires_vector_db", False):
        if not hasattr(structuring, "rebuild_chroma_from_jsonl"):
            raise RuntimeError("Probe structure uses a vector DB without JSONL rebuild support.")
        structuring.rebuild_chroma_from_jsonl(force=True, dedupe=True)
    return {"total_records": total, "visible_records": kept}


class _ProbeWindowRestrictedStructure:
    """Proxy enforcing exact probe windows for all standard structure access."""

    def __init__(self, delegate, windows: list):
        self._delegate = delegate
        self._windows = [tuple(item) for item in windows]

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def _load_all_records(self) -> list:
        loader = getattr(self._delegate, "_load_all_records", None)
        if not callable(loader):
            return []
        return [record for record in loader() if _record_intersects_windows(record, self._windows)]

    def retrieveStructure(self, **kwargs) -> str:
        requested_start = float(kwargs.get("start_sec", 0.0) or 0.0)
        requested_end = float(kwargs.get("end_sec", 999999.0) or 999999.0)
        if requested_end < requested_start:
            requested_start, requested_end = requested_end, requested_start
        visible_ranges = [
            (max(requested_start, start), min(requested_end, end))
            for start, end in self._windows
            if min(requested_end, end) >= max(requested_start, start)
        ]
        if not visible_ranges:
            return "No structure records available inside the active probe windows."

        outputs = []
        for start, end in visible_ranges:
            bounded = dict(kwargs)
            bounded["start_sec"] = start
            bounded["end_sec"] = end
            try:
                output = self._delegate.retrieveStructure(**bounded)
            except TypeError as exc:
                raise RuntimeError(
                    "Structure module does not accept bounded retrieval required by probe."
                ) from exc
            if output:
                outputs.append(str(output))
        return "\n\n".join(outputs) or "No structure records available inside the active probe windows."


def _install_probe_structure_boundary(agent, windows: list) -> None:
    """Expose one restricted structure view to every execution-layer module."""
    restricted = _ProbeWindowRestrictedStructure(agent.structuring, windows)
    agent.env.struct_db = restricted
    for name in ("localization", "perception"):
        module = getattr(agent, name, None)
        if module is not None:
            module.struct_db = restricted


def _merged_time_reference_windows(tasks: list, padding: float = 20.0) -> list:
    windows = []
    for task in tasks or []:
        bounds = _extract_time_reference_bounds((task or {}).get("time_reference", ""))
        if not bounds:
            continue
        start, end = bounds
        windows.append((max(0.0, float(start) - padding), max(1.0, float(end) + padding)))
    windows.sort()
    merged = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [[round(s, 3), round(e, 3)] for s, e in merged if e > s]


@contextmanager
def _sandbox_file_lock(lock_path: str):
    """Serialize sandbox Chroma/JSONL initialization across worker threads."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        try:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


# =====================================================================
# LLM calling tool
# =====================================================================



def _wall_clock_timeout(seconds: float, label: str):
    """Raise TimeoutError if a smoke-only operation exceeds a wall-clock limit."""
    timeout = int(float(seconds or 0))
    if timeout <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"{label} exceeded {timeout}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_remaining = signal.alarm(0)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_remaining:
            signal.alarm(old_remaining)



def create_module_class(module_dict: dict):
    code = module_dict.get("code", "")
    class_name = module_dict.get("name", "UnknownModule")
    import inspect as _inspect
    import re as _re
    import typing

    import utils as _utils
    global_ns = {
        "json": json, "os": os, "re": _re, "inspect": _inspect, "utils": _utils,
        "List": typing.List, "Dict": typing.Dict,
        "Optional": typing.Optional, "Tuple": typing.Tuple, "Any": typing.Any,
        "Union": typing.Union, "call_llm": None,
        "uuid": __import__("uuid"),
    }
    # Bundle code is dynamically injected.  Its base classes must therefore
    # come from a stable ABI module, not from the writable candidate source
    # files bearing the same import names.
    from runtime_bases import (
        LocalizationBase,
        PerceptionBase,
        ThinkingBase,
        VideoStructuringBase,
        WorkMemoryBase,
    )
    global_ns.update({
        "VideoStructuringBase": VideoStructuringBase,
        "LocalizationBase": LocalizationBase,
        "PerceptionBase": PerceptionBase,
        "WorkMemoryBase": WorkMemoryBase,
        "ThinkingBase": ThinkingBase,
    })
    base_imports = (
        "video_structuring_modules", "localization_modules", "perception_modules",
        "memory_modules", "thinking_module",
    )
    # A generated subclass may retain the ABI witness import.  Remove only an
    # exact one-name base import; mixed imports are rejected instead of silently
    # changing candidate semantics.
    sanitized = []
    for line in str(code or "").splitlines():
        match = re.match(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\s+([A-Za-z_]\w*)\s*$", line)
        if match and match.group(1) in base_imports and match.group(2).endswith("Base"):
            continue
        sanitized.append(line)
    code = "\n".join(sanitized) + ("\n" if sanitized else "")
    # IMPORTANT: Do not use detached local_ns!
    # In Python exec, top-level assignments (such as DEFAULT_STRUCTURING_CONFIG={...}) enter local_ns,
    # But the name lookup in the class method only checks global_ns. After separation, the class method cannot see the module level variables.
    # Solution: Use global_ns directly and also as locals, so that all definitions enter the same namespace.
    exec(code, global_ns)
    if class_name in global_ns:
        return global_ns[class_name]
    for name, obj in global_ns.items():
        if isinstance(obj, type) and name != "object" and name not in ("json", "re"):
            return obj
    raise RuntimeError(f"Unable to find class in generated code: {class_name}")



def _get_module_map_key_for_type(module_type: str) -> str:
    """module_type → key used in module_map"""
    return {
        "video_structuring": "structuring", "thinking": "thinking",
        "memory": "memory",
        "localization": "localization", "perception": "perception",
    }.get(module_type, module_type)


def _redirect_structuring_to_sandbox(structuring, sandbox_dir: str,
                                     reset: bool = False,
                                     copy_existing: bool = True,
                                     reference_structure_dir: str = "") -> None:
    """Route evolved video-structuring storage to a sandbox-local directory.

    The sandbox must not mutate baseline video_structure files. For probe
    validation, however, rebuilding the whole video structure is too expensive,
    so this function copies the current JSONL/ChromaDB into sandbox-local paths
    and lets the evolved retrieval code run against that isolated copy.
    """
    if not structuring or not sandbox_dir:
        return

    source_db_dir = getattr(structuring, "db_dir", "")
    source_db_path = getattr(structuring, "db_path", "")
    # Later-round non-structuring probes must use the current-best structure
    # artifact, not whichever mutable workspace DB happens to be present.
    # The caller must identify the attested artifact explicitly. A process-wide
    # environment value is unsafe for concurrent candidate evaluations.
    if reference_structure_dir and os.path.isdir(reference_structure_dir):
        candidate = os.path.join(
            reference_structure_dir,
            os.path.basename(source_db_path) if source_db_path else "",
        )
        if candidate and os.path.exists(candidate):
            source_db_dir = reference_structure_dir
            source_db_path = candidate
            print("   [Sandbox] Reuse current-best structure artifact")
        elif copy_existing:
            expected = os.path.basename(source_db_path) if source_db_path else "<unknown>.jsonl"
            available = []
            try:
                available = sorted(
                    name for name in os.listdir(reference_structure_dir)
                    if name.endswith(".jsonl")
                )
            except OSError:
                pass
            raise RuntimeError(
                "Current-best structure artifact does not contain the JSONL "
                f"required for video {getattr(structuring, 'video_id', '')}: "
                f"expected={expected}, artifact_dir={reference_structure_dir}, "
                f"available={available[:12]}"
            )
    source_chroma_path = os.path.join(source_db_dir, "chroma_db") if source_db_dir else ""

    sandbox_struct_dir = os.path.join(sandbox_dir, "video_structure")
    os.makedirs(sandbox_struct_dir, exist_ok=True)

    sandbox_db_path = os.path.join(
        sandbox_struct_dir,
        f"{structuring.video_id}_{structuring.struct_type}.jsonl",
    )
    if os.environ.get("SANDBOX_STRUCTURE_CHROMA_PER_VIDEO", "").lower() in ("1", "true", "yes"):
        chroma_dir_name = f"{structuring.video_id}_{structuring.struct_type}_chroma_db"
    else:
        chroma_dir_name = "chroma_db"
    sandbox_chroma_path = os.path.join(sandbox_struct_dir, chroma_dir_name)
    lock_path = os.path.join(sandbox_struct_dir, ".sandbox_structure.lock")

    with _sandbox_file_lock(lock_path):
        resume_structure_build = (
            os.environ.get("VIDEO_STRUCT_RESUME_BUILD", "").lower()
            in ("1", "true", "yes")
        )
        if reset and resume_structure_build and os.path.exists(sandbox_db_path):
            reset = False
            print(
                "   ↩️ [Sandbox] resume mode: retain the existing isolation structure JSONL,"
                "Complete missing fragments by library builder"
            )
        if reset and os.path.exists(sandbox_db_path):
            os.remove(sandbox_db_path)
        if reset and os.path.isdir(sandbox_chroma_path):
            shutil.rmtree(sandbox_chroma_path)

        if copy_existing and source_db_path and os.path.exists(source_db_path) and not os.path.exists(sandbox_db_path):
            shutil.copy2(source_db_path, sandbox_db_path)
            print(f"   [Sandbox] Copied structure JSONL into isolation: {os.path.basename(sandbox_db_path)}")
        if copy_existing and source_chroma_path and os.path.isdir(source_chroma_path) and not os.path.isdir(sandbox_chroma_path):
            try:
                shutil.copytree(source_chroma_path, sandbox_chroma_path)
                print("   [Sandbox] Copy ChromaDB to quarantine directory")
            except Exception as e:
                print(f"   ⚠️ [Sandbox] ChromaDB copy failed; it will be rebuilt from JSONL when needed: {e}")

        structuring.db_dir = sandbox_struct_dir
        structuring.db_path = sandbox_db_path

        if getattr(structuring, "requires_vector_db", False):
            import chromadb
            structuring.chroma_client = chromadb.PersistentClient(path=sandbox_chroma_path)
            collection_name = f"{structuring.video_id}_{structuring.struct_type}".replace("-", "_")[:63]
            structuring.collection = structuring.chroma_client.get_or_create_collection(name=collection_name)


def _rebuild_sandbox_chroma_if_requested(structuring) -> None:
    if not getattr(structuring, "requires_vector_db", False):
        return
    if not hasattr(structuring, "rebuild_chroma_from_jsonl"):
        return
    force_rebuild = os.environ.get("VIDEO_STRUCT_FORCE_CHROMA_REBUILD", "").lower() in ("1", "true", "yes")
    if force_rebuild:
        print("   [Sandbox] Force JSONL deduplication and rebuild ChromaDB collection")
        structuring.rebuild_chroma_from_jsonl(force=True, dedupe=True)
    else:
        structuring.rebuild_chroma_from_jsonl()


def _full_structure_build_disabled() -> bool:
    """Default to no full VLM structure rebuild in Codex smoke/probe sandboxes."""
    return os.environ.get("SANDBOX_DISABLE_FULL_STRUCTURE_BUILD", "1").lower() not in (
        "0", "false", "no"
    )


def _allow_isolated_full_structure_rebuild() -> bool:
    return os.environ.get("SANDBOX_ALLOW_FULL_STRUCTURE_REBUILD", "").lower() in (
        "1", "true", "yes"
    )


# =====================================================================
# Phase 1: candidate compatibility check and runtime injection
# =====================================================================

def _inject_bundle_module(module_dict: dict, module_type: str, module_map_refs: dict,
                          diagnostics: list | None = None) -> (bool, str, dict):
    """Inject one record while assembling a complete candidate bundle.
    Returns: (success, module_name, custom_config_or_None)
    """
    map_key = _get_module_map_key_for_type(module_type)
    if not map_key or map_key not in module_map_refs:
        return False, "", None

    target_map = module_map_refs[map_key]
    module_name = module_dict.get("name", "")
    try:
        cls = create_module_class(module_dict)

        target_map[module_name] = cls
        print(f"   [Sandbox] Injected {module_type}: {module_name}")
        return True, module_name, None
    except Exception as e:
        print(f"   ⚠️ [Sandbox] Failed to inject {module_name}: {e}")
        if diagnostics is not None:
            diagnostics.append({
                "module": module_type,
                "class_name": module_name,
                "exception_type": type(e).__name__,
                "error": str(e),
                "traceback": tb.format_exc()[-12000:],
            })
        return False, module_name, None


def _bundle_module_records(bundle: dict) -> list[dict]:
    """Return bundle modules in the runtime's list representation.

    Initial code generation writes a list while resumed bundle evolution keeps
    modules keyed by type.  Both are storage representations of the same
    five-module manifest; runtime smoke must not treat either representation
    as a source-code error.
    """
    raw_modules = (bundle or {}).get("modules") if isinstance(bundle, dict) else None
    if isinstance(raw_modules, list):
        return [item for item in raw_modules if isinstance(item, dict)]
    if isinstance(raw_modules, dict):
        return [
            dict(item, module_type=str(module_type))
            for module_type, item in raw_modules.items()
            if isinstance(item, dict)
        ]
    return []



def validate_bundle_assembly(bundle: dict) -> dict:
    """Validate only the serializable five-module assembly manifest.

    This deliberately does not parse, import, execute, or inspect candidate
    source.  Candidate executability is established exclusively by the same
    dynamic injection path used by task-free/runtime smoke.
    """
    issues = []
    raw_modules = bundle.get("modules") if isinstance(bundle, dict) else None
    modules = _bundle_module_records(bundle)
    if not isinstance(raw_modules, (list, dict)):
        issues.append("modules must be a list or an object keyed by module_type")

    by_type = {}
    duplicate_types = []
    duplicate_names = []
    seen_names = set()
    for module in modules:
        if not isinstance(module, dict):
            issues.append("module entry must be an object")
            continue
        module_type = module.get("module_type", "")
        name = module.get("name", "")
        if module_type in by_type:
            duplicate_types.append(module_type)
        by_type[module_type] = module
        if name:
            if name in seen_names:
                duplicate_names.append(name)
            seen_names.add(name)
        else:
            issues.append(f"module {module_type or '<unknown>'} is missing name")
        if not str(module.get("code") or "").strip():
            issues.append(f"module {module_type or '<unknown>'} is missing code")

    for module_type in BUNDLE_MODULE_ORDER:
        if module_type not in by_type:
            issues.append(f"missing module: {module_type}")
    for module_type in sorted(set(duplicate_types)):
        issues.append(f"duplicate module_type: {module_type}")
    for name in sorted(set(duplicate_names)):
        issues.append(f"duplicate module name: {name}")

    combo = bundle.get("combo") or {}
    if not isinstance(combo, dict):
        combo = {}
        issues.append("combo must be an object")
    for module_type in BUNDLE_MODULE_ORDER:
        combo_key = BUNDLE_COMBO_KEY[module_type]
        module = by_type.get(module_type) or {}
        expected_name = module.get("name", "")
        combo_name = combo.get(combo_key, "")
        if not combo_name:
            issues.append(f"missing combo key: {combo_key}")
        elif expected_name and combo_name != expected_name:
            issues.append(
                f"combo {combo_key}={combo_name!r} does not match "
                f"{module_type} module name {expected_name!r}"
            )

    passed = not issues
    return {
        "passed": passed,
        "issues": issues,
        "validation_mode": "assembly_only_no_source_introspection",
        "combo": combo,
        "module_names": {
            module_type: (by_type.get(module_type) or {}).get("name", "")
            for module_type in BUNDLE_MODULE_ORDER
        },
    }


def inject_module_bundle(bundle: dict, module_map_refs: dict):
    """Inject all modules through the real runtime path and return diagnostics.

    Returns:
        (success, agent_combo, custom_configs, validation)
    """
    validation = validate_bundle_assembly(bundle)
    if not validation.get("passed"):
        return False, {}, {}, validation

    by_type = {
        module.get("module_type"): module
        for module in _bundle_module_records(bundle)
        if isinstance(module, dict)
    }
    agent_combo = {}
    custom_configs = {}
    injected = {}
    injection_diagnostics = []
    for module_type in BUNDLE_MODULE_ORDER:
        module = by_type[module_type]
        ok, module_name, cc = _inject_bundle_module(
            module, module_type, module_map_refs, diagnostics=injection_diagnostics,
        )
        if not ok:
            validation.setdefault("issues", []).append(
                f"runtime injection failed for {module_type}: {module_name}"
            )
            validation["runtime_injection_errors"] = injection_diagnostics
            validation["passed"] = False
            return False, {}, custom_configs, validation
        agent_combo[BUNDLE_COMBO_KEY[module_type]] = module_name
        injected[module_type] = module_name
        if cc:
            custom_configs[module_name] = cc

    # This is a dynamic assembly fact, not source introspection: these are the
    # exact classes that the Agent will construct.  Neutral ``*Base`` classes
    # are protocol infrastructure and must never become a selected bundle slot.
    # Checking here makes initial smoke, later smoke, probe and full eval share
    # the same protection against an extractor/manifest selecting a helper.
    map_key_for_type = {
        "video_structuring": "structuring", "thinking": "thinking",
        "memory": "memory", "localization": "localization", "perception": "perception",
    }
    effective_modules = {}
    for module_type in BUNDLE_MODULE_ORDER:
        module_name = str(agent_combo.get(BUNDLE_COMBO_KEY[module_type]) or "")
        registry = module_map_refs.get(map_key_for_type[module_type], {})
        cls = registry.get(module_name) if isinstance(registry, dict) else None
        tool_surface = sorted(
            name[5:] for name in dir(cls)
            if name.startswith("tool_") and callable(getattr(cls, name, None))
        ) if isinstance(cls, type) else []
        effective_modules[module_type] = {
            "selected_class": module_name,
            "injected_class": getattr(cls, "__name__", ""),
            "tool_surface": tool_surface,
        }
        if not isinstance(cls, type):
            validation.setdefault("issues", []).append(
                f"runtime injection produced no class for {module_type}={module_name!r}"
            )
        elif module_name.endswith("Base") or getattr(cls, "__name__", "").endswith("Base"):
            validation.setdefault("issues", []).append(
                f"runtime combo selected infrastructure class for {module_type}: {module_name!r}"
            )
    validation["effective_modules"] = effective_modules
    if validation.get("issues"):
        validation["passed"] = False
        return False, {}, custom_configs, validation

    validation["injected_modules"] = injected
    return True, agent_combo, custom_configs, validation


def preflight_bundle_runtime(bundle: dict) -> dict:
    """Run the smallest useful real-runtime admission check for a bundle.

    This is intentionally *not* a synthetic agent run.  It checks the exact
    code path used by the execution layer to compile, inject, and construct
    the five modules, but does not fabricate a question, media window, tool
    call, structure record, or model response.  Real smoke remains the sole
    behavioral acceptance test.
    """
    # Reassert the selected runtime before importing the module registry.
    _ensure_runtime_path_for_workspace(default_workspace())
    validation = validate_bundle_assembly(bundle)
    result = {
        "artifact_type": "bundle_runtime_preflight", "schema_version": 1,
        "stage": "runtime_preflight",
        "passed": False,
        "validation_mode": "compile_inject_construct_real_runtime",
        "assembly": validation,
        "compile": {},
        "injection": {},
        "construction": {},
        "issues": [],
        "falsifiable_success": [
            "all five module sources compile",
            "inject_module_bundle succeeds against the real MetaVideoAgent module maps",
            "the real execution-layer constructor arguments instantiate all five modules",
        ],
    }
    if not validation.get("passed"):
        result["issues"] = list(validation.get("issues") or [])
        return result

    records = {
        item.get("module_type"): item
        for item in _bundle_module_records(bundle)
        if isinstance(item, dict)
    }
    compile_issues = []
    for module_type in BUNDLE_MODULE_ORDER:
        record = records.get(module_type) or {}
        try:
            compile(str(record.get("code") or ""), f"<bundle:{module_type}>", "exec")
            result["compile"][module_type] = {"passed": True}
        except (SyntaxError, ValueError, TypeError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            result["compile"][module_type] = {"passed": False, "error": message}
            compile_issues.append(f"{module_type} source does not compile: {message}")
    if compile_issues:
        result["issues"] = compile_issues
        return result

    import tempfile

    import module_map

    module_map_refs = {
        "structuring": module_map.STRUCTURING_MAP,
        "thinking": module_map.THINKING_MAP,
        "memory": module_map.WORK_MEMORY_MAP,
        "localization": module_map.LOCALIZATION_MAP,
        "perception": module_map.PERCEPTION_MAP,
    }
    backups = {key: dict(value) for key, value in module_map_refs.items()}
    try:
        ok, combo, custom_configs, injection = inject_module_bundle(bundle, module_map_refs)
        result["injection"] = injection
        if not ok:
            result["issues"] = list(injection.get("issues") or ["runtime bundle injection failed"])
            return result

        with tempfile.TemporaryDirectory(prefix="metavideoagent_bundle_preflight_") as temp_dir:
            class PreflightEnv:
                workspace_dir = temp_dir
                raw_video_path = ""
                frames_dir = ""
                video_length_secs = 0.0
                struct_db = None

                @staticmethod
                def get_active_time_windows():
                    return []

                @staticmethod
                def get_frame_at_timestamp(_timestamp):
                    return ""

            def config_for(combo_key: str):
                return (custom_configs or {}).get((combo or {}).get(combo_key, ""))

            constructors = (
                ("video_structuring", lambda env: module_map.STRUCTURING_MAP[combo["video_structuring"]](
                    temp_dir, "runtime_preflight", custom_config=config_for("video_structuring")
                )),
                ("memory", lambda env: module_map.WORK_MEMORY_MAP[combo["memory"]](
                    temp_dir, "runtime_preflight", "runtime preflight", custom_config=config_for("memory")
                )),
                ("thinking", lambda env: module_map.THINKING_MAP[combo["thinking"]](
                    env, custom_config=config_for("thinking")
                )),
                ("localization", lambda env: module_map.LOCALIZATION_MAP[combo["localization"]](
                    env, struct_db=env.struct_db, custom_config=config_for("localization")
                )),
                ("perception", lambda env: module_map.PERCEPTION_MAP[combo["perception"]](
                    env, struct_db=env.struct_db, custom_config=config_for("perception")
                )),
            )
            env = PreflightEnv()
            for module_type, construct in constructors:
                try:
                    instance = construct(env)
                    if module_type == "video_structuring":
                        env.struct_db = instance
                    result["construction"][module_type] = {"passed": True}
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    result["construction"][module_type] = {"passed": False, "error": message}
                    result["issues"].append(
                        f"{module_type} real-runtime construction failed: {message}"
                    )
    finally:
        for key, backup in backups.items():
            module_map_refs[key].clear()
            module_map_refs[key].update(backup)

    result["passed"] = not result["issues"]
    return result


def _shorten_text(value, limit: int = 1200) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _time_ranges_in_text(text: str) -> list:
    ranges = []
    for match in re.finditer(
        r"\[(?:From\s+)?(\d+(?:\.\d+)?)s?\s*(?:-|–|to|,)\s*(\d+(?:\.\d+)?)s?\]",
        str(text or ""),
        re.IGNORECASE,
    ):
        try:
            start = float(match.group(1))
            end = float(match.group(2))
        except ValueError:
            continue
        if 0 <= start <= end:
            ranges.append([start, end])
    return ranges


def _looks_like_video_evidence(text: str) -> bool:
    value = str(text or "")
    low = value.lower()
    if not value.strip():
        return False
    if _time_ranges_in_text(value):
        return True
    markers = (
        "visual", "frame", "video clip", "clip", "narration", "asr",
        "transcript", "ocr", "subtitle", "audio", "screen text",
        "timestamp", "start_sec", "end_sec", "modalities",
    )
    return len(value) >= 160 and any(marker in low for marker in markers)


def _is_errorish_text(text: str) -> bool:
    low = str(text or "").lower()
    return any(
        marker in low
        for marker in (
            "execution error", "traceback", "llm error", "api error",
            "connection error", "error code:", "module not found",
            "no valid candidate windows", "missing dependency variable",
            "tool '", "not found.",
        )
    )


def _is_guess_or_failure_answer(answer: str) -> bool:
    low = str(answer or "").strip().lower()
    if not low:
        return True
    failure_markers = (
        "task failed", "error:", "unable to answer", "cannot answer",
        "insufficient evidence", "no concrete evidence", "forced to make",
        "educated guess", "generic guess", "we have no", "without any confirmed",
        "option or statement aligned with", "strongest supported answer is the option",
        "strongest supported answer is the statement", "based on the verified localized evidence",
    )
    return any(marker in low for marker in failure_markers)


def _is_uncertain_answer(answer: str) -> bool:
    low = str(answer or "").strip().lower()
    return low in {
        "uncertain", "unknown", "not sure", "cannot determine",
        "insufficient verified evidence to answer.",
        "insufficient verified evidence to answer",
    }


def _grounded_uncertainty_summary(row: dict, summary: dict) -> str:
    """Return diagnostic context for an uncertainty terminal output.

    A grounded conflict can still be useful for the engineering failure report,
    but it never makes ``uncertain`` an acceptable smoke answer.  The current
    smoke contract requires a concrete, evaluator-consumable payload.answer.
    """
    answer = summary.get("final_answer", "")
    if not _is_uncertain_answer(answer):
        return ""
    tool_steps = [
        step for step in summary.get("steps", [])
        if step.get("step_type") == "tool_execution"
    ]
    if not any(step.get("looks_like_video_evidence") and not step.get("errorish") for step in tool_steps):
        return ""
    trajectory = row.get("trajectory") if isinstance(row.get("trajectory"), list) else []
    finish_payload = {}
    for step in reversed(trajectory):
        if step.get("step_type") == "reasoning" and step.get("decision") == "finish":
            payload = step.get("action_input")
            if isinstance(payload, dict):
                finish_payload = payload
            break
    evidence_summary = str(finish_payload.get("evidence_summary") or "")
    if len(evidence_summary.strip()) < 20:
        return ""
    low = evidence_summary.lower()
    concrete_markers = (
        "but", "conflict", "not among", "no matching option", "mismatch",
        "observed", "takes", "count", "evidence", "visible", "verified",
    )
    has_number = bool(re.search(r"\b\d+\b", evidence_summary))
    if has_number and any(marker in low for marker in concrete_markers):
        return evidence_summary
    absence_or_mismatch_markers = (
        "no ", "not visible", "not shown", "not mentioned", "none",
        "not found", "instead", "but no", "no matching", "mismatch",
    )
    target_markers = (
        "bottle", "glue", "pink", "object", "item", "person", "text",
        "action", "event",
    )
    if (
        any(marker in low for marker in absence_or_mismatch_markers)
        and any(marker in low for marker in target_markers)
    ):
        return evidence_summary
    # A synthesis call may intentionally return an uncertainty sentence without
    # copying lengthy, conflicting frame observations into its compact summary.
    # Treat it as grounded only when the actual successful visual observations
    # make two or more incompatible explicit answer/count claims. This is an
    # engineering-smoke allowance, not an accuracy decision.
    explicit_claims = set()
    for step in trajectory:
        if step.get("step_type") != "tool_execution" or step.get("execution_status") != "ok":
            continue
        observation = str(step.get("observation") or "")
        for match in re.finditer(
            r"(?:answer\s*[::]|number\s+of[^.\n]{0,100}?\s+is)\s*\**\s*(\d+)",
            observation,
            flags=re.IGNORECASE,
        ):
            explicit_claims.add(match.group(1))
    if len(explicit_claims) >= 2:
        return (
            "Successful bounded visual verification produced conflicting explicit "
            f"count claims ({', '.join(sorted(explicit_claims, key=int))}); "
            "the final uncertainty is grounded rather than a no-evidence fallback."
        )
    return ""


def _plan_params_from_step(step: dict) -> list:
    payloads = []
    if not isinstance(step, dict):
        return payloads
    if step.get("step_type") == "planning":
        plan = step.get("plan_output") or []
        if isinstance(plan, list):
            for item in plan:
                if isinstance(item, dict):
                    payloads.append({
                        "tool": item.get("target_tool", ""),
                        "params": item.get("tool_params") or {},
                        "instruction": item.get("instruction", ""),
                    })
    action_input = step.get("action_input")
    if isinstance(action_input, dict):
        plan = action_input.get("plan")
        if isinstance(plan, list):
            for item in plan:
                if isinstance(item, dict):
                    payloads.append({
                        "tool": item.get("target_tool", ""),
                        "params": item.get("tool_params") or {},
                        "instruction": item.get("instruction", ""),
                    })
    return payloads


def _summarize_smoke_trajectory(row: dict) -> dict:
    """Build a behavior summary for smoke checks.

    The returned object keeps raw question/final_answer for local deterministic
    checks and persisted debugging. Before sending to an LLM/codegen judge, call
    `_judge_safe_smoke_summary()` to remove task-specific content.
    """
    trajectory = row.get("trajectory") or []
    steps = []
    tool_steps = []
    planning_steps = []
    reasoning_steps = []
    for idx, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue
        stype = step.get("step_type", "")
        if stype == "tool_execution":
            obs = str(step.get("observation", "") or "")
            item = {
                "index": idx,
                "step_type": stype,
                "tool": step.get("action") or step.get("tool") or "",
                "thought": _shorten_text(step.get("thought", ""), 240),
                "observation_preview": _shorten_text(obs, 900),
                "observation_chars": len(obs),
                "time_ranges_found": _time_ranges_in_text(obs)[:8],
                "looks_like_video_evidence": _looks_like_video_evidence(obs),
                "errorish": _is_errorish_text(obs),
            }
            tool_steps.append(item)
            steps.append(item)
        elif stype == "planning":
            plan = _plan_params_from_step(step)
            item = {
                "index": idx,
                "step_type": stype,
                "plan": [
                    {
                        "tool": p.get("tool", ""),
                        "params": p.get("params", {}),
                        "instruction": _shorten_text(p.get("instruction", ""), 220),
                    }
                    for p in plan
                ],
            }
            planning_steps.append(item)
            steps.append(item)
        elif stype == "reasoning":
            item = {
                "index": idx,
                "step_type": stype,
                "decision": step.get("decision", ""),
                "thought": _shorten_text(step.get("thought", ""), 600),
                "action_input_preview": _shorten_text(step.get("action_input", ""), 400),
            }
            reasoning_steps.append(item)
            steps.append(item)
        elif stype in ("final_sparse_evidence", "sparse_evidence_state"):
            item = {
                "index": idx,
                "step_type": stype,
                "accepted_count": len(step.get("accepted") or []),
                "pending_count": len(step.get("pending") or []),
                "rejected_count": len(step.get("rejected") or []),
            }
            steps.append(item)
    answer = (
        row.get("final_agent_answer")
        or row.get("answer")
        or row.get("final_answer")
        or ""
    )
    return {
        "question": row.get("question") or (row.get("task_meta") or {}).get("question", ""),
        "video_id": row.get("video_id", ""),
        "final_answer": _shorten_text(answer, 500),
        "final_answer_present": bool(str(answer or "").strip()),
        "final_answer_looks_like_guess_or_failure": _is_guess_or_failure_answer(answer),
        "trajectory_length": len(trajectory),
        "tool_step_count": len(tool_steps),
        "planning_step_count": len(planning_steps),
        "reasoning_step_count": len(reasoning_steps),
        "steps": steps[-18:],
    }


def _judge_safe_smoke_summary(summary: dict) -> dict:
    """Remove task-specific content before LLM/codegen smoke judging."""
    safe = json.loads(json.dumps(summary or {}, ensure_ascii=False))
    question = str(safe.pop("question", "") or "")
    final_answer = str(safe.pop("final_answer", "") or "")
    safe.pop("video_id", None)

    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str):
            text = value
            if question:
                text = text.replace(question, "<redacted_smoke_question>")
            if final_answer:
                text = text.replace(final_answer, "<redacted_smoke_final_answer>")
            return text
        return value

    return scrub(safe)



def _detect_smoke_dataflow_issues(summary: dict) -> list:
    issues = []
    steps = summary.get("steps") or []
    seen_video_evidence = False
    repeated_empty_verification = 0
    dataflow_break_reported = False
    for step in steps:
        if step.get("step_type") == "tool_execution":
            if step.get("looks_like_video_evidence") and not step.get("errorish"):
                seen_video_evidence = True
            obs = str(step.get("observation_preview") or "").lower()
            if (
                "no valid candidate windows" in obs
                or "no candidate windows" in obs
                or "missing dependency variable" in obs
            ):
                repeated_empty_verification += 1
        elif step.get("step_type") == "planning":
            for plan in step.get("plan") or []:
                params = plan.get("params") or {}
                empty_window_param = any(
                    (
                        isinstance(value, list) and len(value) == 0
                        or value in ("", None)
                    )
                    and any(token in str(key).lower() for token in ("window", "range", "candidate", "time"))
                    for key, value in params.items()
                )
                if seen_video_evidence and empty_window_param:
                    if not dataflow_break_reported:
                        issues.append({
                            "issue_type": "smoke_dataflow_break",
                            "severity": "critical",
                            "description": (
                                "A previous tool produced video evidence/candidate timing information, "
                                "but a later plan passed an empty candidate/time/window parameter."
                            ),
                            "repair_hint": (
                                "Make the thinking module export structured variables from retrieval "
                                "observations or parse candidate windows directly, then pass non-empty "
                                "candidate windows to verification/perception tools."
                            ),
                        })
                        dataflow_break_reported = True
                    break
    if repeated_empty_verification >= 2:
        issues.append({
            "issue_type": "smoke_degenerate_empty_verification_loop",
            "severity": "critical",
            "description": "The agent repeatedly called verification with empty/missing candidate evidence.",
            "repair_hint": (
                "Stop repeating the same ineffective verification call; either retrieve narrower "
                "evidence, extract variables, or finish only after grounded evidence is available."
            ),
        })
    return issues


def _normalized_window_pairs(value) -> list[tuple[float, float]]:
    pairs = []
    for item in value or []:
        try:
            if isinstance(item, dict):
                start = float(item.get("start_sec", item.get("start")))
                end = float(item.get("end_sec", item.get("end")))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start, end = float(item[0]), float(item[1])
            else:
                continue
        except (TypeError, ValueError):
            continue
        if end > start:
            pairs.append((start, end))
    return pairs


def _localized_media_scope_issues(row: dict) -> list:
    """Detect a range handoff that reaches a tool but not its media adapter.

    This is deliberately task-free: it compares only standard module result
    windows with capability-event timestamps/segments recorded in the same
    tool execution.  It catches the otherwise silent full-video fallback that
    can make a candidate look runnable while defeating localization.
    """
    trajectory = row.get("trajectory") if isinstance(row, dict) else []
    events = row.get("capability_events") if isinstance(row, dict) else []
    event_by_id = {
        str(event.get("evidence_event_id")): event
        for event in events or []
        if isinstance(event, dict) and event.get("evidence_event_id")
    }
    issues = []
    for step in trajectory or []:
        if not isinstance(step, dict) or step.get("step_type") != "tool_execution":
            continue
        if str(step.get("producer_module") or "") != "perception":
            continue
        result = step.get("module_result") if isinstance(step.get("module_result"), dict) else {}
        requested = _normalized_window_pairs(result.get("time_ranges"))
        if not requested:
            continue
        linked_events = [
            event_by_id.get(str(event_id))
            for event_id in (step.get("evidence_event_ids") or [])
        ]
        media_events = [
            event for event in linked_events
            if isinstance(event, dict)
            and event.get("capability") in {"vlm", "asr", "ocr"}
            and event.get("event") in {"inference", "transcription"}
            and event.get("status") in {"ok", "no_speech", "no_content"}
        ]
        observed = 0
        overlapping = 0
        for event in media_events:
            try:
                if event.get("timestamp_sec") is not None:
                    start = end = float(event["timestamp_sec"])
                else:
                    start = float(event.get("start_sec"))
                    end = float(event.get("end_sec"))
            except (TypeError, ValueError):
                continue
            observed += 1
            if any(start <= hi and end >= lo for lo, hi in requested):
                overlapping += 1
        if observed and not overlapping:
            issues.append({
                "issue_type": "smoke_localized_media_scope_violation",
                "severity": "critical",
                "description": (
                    "A perception tool received non-empty localized windows, but every linked "
                    "successful media capability event fell outside those windows."
                ),
                "repair_hint": (
                    "Use runtime_evidence.inspect_time_ranges/transcribe_time_ranges (or the "
                    "documented scoped-media API) with the perception input ranges; do not call an "
                    "active-window adapter on the unmodified environment."
                ),
                "requested_window_count": len(requested),
                "observed_media_event_count": observed,
                "overlapping_media_event_count": overlapping,
                "producer_module": "perception",
            })
    return issues


def _basic_behavior_smoke_assessment(row: dict) -> dict:
    summary = _summarize_smoke_trajectory(row)
    answer = summary.get("final_answer", "")
    task_for_answer = row.get("task_meta") if isinstance(row.get("task_meta"), dict) else row
    tool_steps = [
        step for step in summary.get("steps", [])
        if step.get("step_type") == "tool_execution"
    ]
    evidence_steps = [
        step for step in tool_steps
        if step.get("looks_like_video_evidence") and not step.get("errorish")
    ]
    error_steps = [step for step in tool_steps if step.get("errorish")]
    issues = []
    advisories = []
    if not answer.strip():
        issues.append({
            "issue_type": "smoke_no_final_answer",
            "severity": "critical",
            "description": "The agent did not produce a final answer.",
            "repair_hint": "Ensure the thinking module can terminate with a final answer after bounded evidence gathering.",
        })
    elif _is_uncertain_answer(answer) or _is_guess_or_failure_answer(answer):
        issues.append({
            "issue_type": "smoke_non_answer_terminal_fallback",
            "severity": "critical",
            "description": (
                "The agent terminated with an uncertainty, failure, or generic fallback token instead "
                "of a concrete task answer."
            ),
            "repair_hint": (
                "Repair the producer-to-thinking evidence path and final answer serialization. "
                "A finish payload must contain one concrete answer in the requested response form; "
                "confidence/evidence_summary may express uncertainty but must not replace answer."
            ),
        })
    elif not is_answer_parseable(answer, task=task_for_answer):
        issues.append({
            "issue_type": "smoke_final_answer_not_parseable",
            "severity": "critical",
            "description": (
                "The final answer cannot be consumed by the unified evaluator's requested response schema."
            ),
            "repair_hint": (
                "Preserve a concrete final answer in payload['answer'] and serialize it in the "
                "runtime question's requested response form."
            ),
        })
    if not tool_steps:
        issues.append({
            "issue_type": "smoke_no_tool_execution",
            "severity": "critical",
            "description": "The agent answered without executing any localization/perception/video-grounding tool.",
            "repair_hint": "Add at least one video-grounded tool call before finalizing.",
        })
    if not evidence_steps:
        issues.append({
            "issue_type": "smoke_no_video_grounding",
            "severity": "critical",
            "description": "No tool observation looked like substantive video evidence.",
            "repair_hint": "Ensure localization/perception tools return non-empty video, audio, OCR, or timestamped evidence.",
        })
    issues.extend(_detect_smoke_dataflow_issues(summary))
    issues.extend(_localized_media_scope_issues(row))
    if error_steps and len(error_steps) == len(tool_steps):
        issues.append({
            "issue_type": "smoke_all_tools_error",
            "severity": "critical",
            "description": "Every tool execution in the smoke trajectory returned an error-like observation.",
            "repair_hint": "Fix tool names, signatures, parameters, and runtime dependencies before full evaluation.",
        })
    passed = not any(issue.get("severity") in ("critical", "high") for issue in issues)
    return {
        "passed": passed,
        "judge_type": "deterministic_behavior_prefilter",
        "issues": issues,
        "advisories": advisories,
        "grounded_uncertainty_summary": _grounded_uncertainty_summary(row, summary),
        "summary": summary,
        "repair_instruction": " ".join(
            issue.get("repair_hint", "") for issue in issues[:4]
        ).strip(),
    }


def behavioral_smoke_assessment(result_rows: list, combo: dict) -> dict:
    if not result_rows:
        return {
            "passed": False,
            "stage": "behavioral_smoke",
            "judge_type": "no_result_rows",
            "verdict": "fail_engineering_error",
            "reason": "Smoke execution produced no result rows.",
            "repair_instruction": "Fix execution so one task produces a trajectory and final answer.",
        }
    row = result_rows[0]
    deterministic = _basic_behavior_smoke_assessment(row)
    # Smoke repair is an engineering contract, not a second task-solving
    # prompt. An LLM judge can turn a valid runtime violation into a
    # task-specific prescription. Keep the complete no-label trace for Codex
    # inspection, but route solely on deterministic producer/consumer/event
    # facts.
    del combo
    passed = bool(deterministic.get("passed"))
    redacted_summary = _judge_safe_smoke_summary(deterministic.get("summary", {}))
    return {
        "passed": passed,
        "stage": "behavioral_smoke",
        "deterministic_prefilter": {
            key: value for key, value in deterministic.items()
            if key != "summary"
        },
        "sanitized_summary": redacted_summary,
        "verdict": "pass" if passed else "fail_engineering_error",
        "reason": " ".join(
            str(issue.get("description") or "")
            for issue in deterministic.get("issues", [])[:4]
        ).strip(),
        "repair_instruction": deterministic.get("repair_instruction", ""),
        "note": (
            "Behavioral smoke checks runtime viability only; it does not judge answer correctness."
        ),
    }


_ENGINEERING_SMOKE_SUPERVISION_KEYS = {
    "answer", "gt_answer", "gold_answer", "final_answer", "final_agent_answer",
    "is_correct", "correct", "correctness", "score", "reward",
}


def _redact_engineering_smoke_supervision(value, candidate_answer: str = ""):
    """Remove answer supervision without collapsing the execution trace."""
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted_answer_supervision>"
                if str(key).lower() in _ENGINEERING_SMOKE_SUPERVISION_KEYS
                else _redact_engineering_smoke_supervision(item, candidate_answer)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_engineering_smoke_supervision(item, candidate_answer) for item in value]
    if isinstance(value, str) and candidate_answer:
        return value.replace(candidate_answer, "<redacted_candidate_answer>")
    return value


def build_engineering_smoke_execution_trace(result_rows: list) -> dict:
    """Keep the complete smoke dataflow trace while withholding all labels.

    This artifact is for Codex engineering repair only. It does not contain a
    gold answer, candidate final answer, correctness bit or score, and is never
    used by probe/full-eval scoring. Unlike a compact failure contract it keeps
    every trajectory step and capability event, so Codex can inspect the real
    producer→consumer break rather than guess from a prose summary.
    """
    rows = []
    for row in result_rows or []:
        if not isinstance(row, dict):
            continue
        task_meta = row.get("task_meta") if isinstance(row.get("task_meta"), dict) else {}
        request = {
            "question": row.get("question", ""),
            "choices": task_meta.get("choices", row.get("choices", [])),
            "time_reference": row.get("time_reference", []),
        }
        candidate_answer = str(row.get("final_agent_answer") or row.get("answer") or "")
        rows.append({
            "runtime_request": _redact_engineering_smoke_supervision(request, candidate_answer),
            "trajectory": _redact_engineering_smoke_supervision(
                row.get("trajectory") or [], candidate_answer,
            ),
            "capability_events": _redact_engineering_smoke_supervision(
                row.get("capability_events") or [], candidate_answer,
            ),
            "runtime_trace": _redact_engineering_smoke_supervision(
                row.get("runtime_trace") or [], candidate_answer,
            ),
            "runtime_error": _redact_engineering_smoke_supervision(
                row.get("error") or "", candidate_answer,
            ),
        })
    return {
        "artifact_type": "engineering_smoke_execution_trace", "schema_version": 1,
        "trace_policy": (
            "complete runtime steps, model/tool runtime trace, and capability events with answer supervision removed; "
            "not an accuracy or correctness artifact"
        ),
        "rows": rows,
    }


def _bundle_text_for_quality(bundle: dict, combo: dict) -> str:
    """Compact bundle text used to infer the candidate's own evidence contract."""
    payload = {
        "combo": combo or bundle.get("combo") or {},
        "design_answers": bundle.get("design_answers", {}),
        "observed_channel_summary": bundle.get("observed_channel_summary", {}),
        "module_names": [
            module.get("name", "")
            for module in (bundle.get("modules") or [])
            if isinstance(module, dict)
        ],
        "module_thoughts": [
            module.get("thought", "")
            for module in (bundle.get("modules") or [])
            if isinstance(module, dict)
        ],
    }
    # Include channel/dataflow identifiers without giving the checker full code.
    for module in bundle.get("modules") or []:
        if not isinstance(module, dict):
            continue
        code = module.get("code", "") or ""
        identifiers = re.findall(
            r"\b(?:audio|asr|speech|transcript|subtitle|ocr|visual|frame|"
            r"scene|object|action|entity|timeline|timestamp|retrieve|search|"
            r"candidate|window|evidence)\w*\b",
            code,
            re.I,
        )
        if identifiers:
            payload.setdefault("code_channel_identifiers", []).extend(sorted(set(identifiers))[:80])
    return json.dumps(payload, ensure_ascii=False).lower()


EVIDENCE_CHANNEL_MARKERS = {
    "visual": (
        "visual", "vision", "frame", "image", "scene", "object", "action",
        "motion", "color", "appearance", "visual observation", "fine-grained visual detail",
    ),
    "text_ocr": (
        "ocr", "subtitle", "caption", "screen text", "onscreen text",
        "on-screen text", "visible text", "text region",
    ),
    "audio": (
        "audio", "asr", "speech", "spoken", "dialogue", "voice",
        "transcript", "narration", "sound",
    ),
    "temporal": (
        "time", "timestamp", "start_sec", "end_sec", "timeline", "window",
        "segment", "clip", "interval", "candidate window",
    ),
    "entity_relation": (
        "entity", "subject", "character", "person", "actor", "object",
        "relation", "participant", "registry", "track",
    ),
    "retrieval": (
        "retrieve", "retrieval", "search", "top_k", "embedding",
        "candidate", "rank", "rerank", "similarity",
    ),
}


def _infer_required_evidence_channels(bundle: dict, combo: dict) -> list:
    """Infer channels the generated bundle itself claims to use.

    This intentionally does not encode a project preference such as
    audio-first. It only turns the bundle's own design/code language into a
    generic evidence contract for smoke artifacts.
    """
    text = _bundle_text_for_quality(bundle, combo)
    required = []
    for channel, markers in EVIDENCE_CHANNEL_MARKERS.items():
        if any(marker in text for marker in markers):
            required.append(channel)
    return required


def _read_jsonl_limited(path: str, max_rows: int = 200) -> list:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_rows:
                    break
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"_raw": line[:2000], "_json_error": True})
    except FileNotFoundError:
        return []
    return rows


def _row_has_nonmissing_channel(text: str, channel: str) -> bool:
    value = str(text or "")
    low = value.lower()
    markers = EVIDENCE_CHANNEL_MARKERS.get(channel, ())
    if not markers or not any(marker in low for marker in markers):
        return False
    placeholder_patterns = (
        r"\[missing\]",
        r"\bmissing\b",
        r"\bnone\b",
        r"\bnull\b",
        r"not available",
        r"unavailable",
        r"no evidence",
        r"empty response",
        r"api error",
        r"parse failed",
    )
    marker_contexts = []
    for marker in markers:
        for match in re.finditer(re.escape(marker), low):
            start = max(0, match.start() - 80)
            end = min(len(value), match.end() + 220)
            marker_contexts.append(value[start:end])
    if not marker_contexts:
        return False
    for context in marker_contexts[:8]:
        if not any(re.search(pattern, context, re.I) for pattern in placeholder_patterns):
            # A marker with surrounding substantive text is enough for smoke.
            if len(re.sub(r"[\W_]+", "", context)) >= 24:
                return True
    return False


def _candidate_consumable_structure_text(row: dict) -> str:
    """Return only the normalized fields that generated consumers can read.

    A raw ``multimodal_narration`` blob may be rich while a generated
    ``addStructure`` implementation has erased every field used by retrieval.
    Quality checks must report that as a dataflow failure rather than count the
    raw blob as successful downstream evidence.
    """
    if not isinstance(row, dict):
        return ""
    if validate_canonical_structure_record(row):
        return ""
    canonical = row.get("canonical_evidence")
    retrieval_document = str(row.get("retrieval_document") or "")
    view = {
        "canonical_evidence": canonical,
        "retrieval_document": retrieval_document,
        "visual_entities": row.get("visual_entities", []),
        "visual_actions": row.get("visual_actions", []),
        "screen_text": row.get("screen_text", []),
        "asr_text": row.get("asr_text", ""),
        "media_assets": row.get("media_assets", []),
    }
    return json.dumps(view, ensure_ascii=False)


def _summarize_smoke_structure_artifacts(files: list, rows: list, required_channels: list) -> dict:
    key_counts = {}
    placeholder_rows = 0
    channel_hits = {channel: 0 for channel in EVIDENCE_CHANNEL_MARKERS}
    required_channel_hits = {channel: 0 for channel in required_channels}
    sample_rows = []
    total_chars = 0
    canonical_record_count = 0
    canonical_raw_narration_count = 0
    canonical_contract_issue_count = 0
    canonical_contract_issues = {}
    for row in rows:
        if isinstance(row, dict):
            for key in row:
                key_counts[str(key)] = key_counts.get(str(key), 0) + 1
        text = json.dumps(row, ensure_ascii=False)
        consumable_text = _candidate_consumable_structure_text(row)
        canonical = row.get("canonical_evidence") if isinstance(row, dict) else {}
        contract_issues = validate_canonical_structure_record(row)
        if not contract_issues:
            canonical_record_count += 1
            if str(canonical.get("raw_multimodal_narration") or "").strip():
                canonical_raw_narration_count += 1
        else:
            canonical_contract_issue_count += len(contract_issues)
            for issue in contract_issues:
                canonical_contract_issues[issue] = canonical_contract_issues.get(issue, 0) + 1
        total_chars += len(text)
        if len(sample_rows) < 3:
            sample_rows.append(_shorten_text(text, 900))
        if re.search(r"\[missing\]|\bmissing\b|\bnone\b|\bnull\b|api error|parse failed", text, re.I):
            placeholder_rows += 1
        for channel in channel_hits:
            if _row_has_nonmissing_channel(consumable_text, channel):
                channel_hits[channel] += 1
        for channel in required_channels:
            if _row_has_nonmissing_channel(consumable_text, channel):
                required_channel_hits[channel] += 1
    row_count = len(rows)
    return {
        "structure_files": files[:5],
        "row_count": row_count,
        "sampled_rows": row_count,
        "top_level_key_counts": dict(sorted(key_counts.items(), key=lambda item: (-item[1], item[0]))[:30]),
        "placeholder_rows": placeholder_rows,
        "placeholder_ratio": round(placeholder_rows / max(1, row_count), 3),
        "average_row_chars": round(total_chars / max(1, row_count), 1),
        "canonical_record_count": canonical_record_count,
        "canonical_record_missing_count": max(0, row_count - canonical_record_count),
        "canonical_raw_narration_count": canonical_raw_narration_count,
        "canonical_contract_issue_count": canonical_contract_issue_count,
        "canonical_contract_issues": canonical_contract_issues,
        "channel_hits": channel_hits,
        "required_channel_hits": required_channel_hits,
        "sample_rows": sample_rows,
    }


def _inspect_smoke_structure_quality(sandbox_dir: str,
                                     video_id: str,
                                     bundle: dict,
                                     combo: dict) -> dict:
    """Summarize smoke-built artifacts against the bundle's own generic contract."""
    required_channels = _infer_required_evidence_channels(bundle, combo)
    quality = {
        "checked": False,
        "stage": "structure_artifact_smoke",
        "required_evidence_channels": required_channels,
        "structure_files": [],
        "row_count": 0,
        "sampled_rows": 0,
        "artifact_summary": {},
        "issues": [],
        "advisories": [],
    }
    struct_dir = os.path.join(sandbox_dir, "video_structure")
    # Strict time-reference smoke owns a structure artifact per question so
    # concurrent/mixed-video runs cannot share a mutable structure DB.  Its
    # evaluator therefore writes below ``question_runs/<task>/video_structure``
    # rather than directly under the suite root.  This quality check is called
    # for exactly one smoke question; accept that nested artifact only when it
    # unambiguously belongs to the requested video.  Do not recursively scan
    # arbitrary artifacts, which could otherwise mask a missing current-task
    # structure library with an unrelated video's output.
    if not os.path.isdir(struct_dir):
        nested_candidates = sorted(glob.glob(os.path.join(
            sandbox_dir,
            "question_runs",
            "*",
            "video_structure",
        )))
        matching_candidates = [
            candidate for candidate in nested_candidates
            if glob.glob(os.path.join(candidate, f"{video_id}_*.jsonl"))
        ]
        if len(matching_candidates) == 1:
            struct_dir = matching_candidates[0]
            quality["structure_dir"] = struct_dir
    if not video_id or not os.path.isdir(struct_dir):
        quality["issues"].append({
            "issue_type": "structure_artifact_missing",
            "severity": "critical",
            "description": "Smoke run did not create an isolated video_structure directory.",
            "fix_hint": "Ensure video_structuring.addStructure writes JSONL/Chroma artifacts under the sandbox structure directory.",
            "affected_file": "generated video_structuring module",
        })
        return quality

    files = sorted(glob.glob(os.path.join(struct_dir, f"{video_id}_*.jsonl")))
    quality["structure_files"] = files[:5]
    if not files:
        quality["issues"].append({
            "issue_type": "structure_jsonl_missing",
            "severity": "critical",
            "description": "Smoke run completed but no per-video structure JSONL file was found.",
            "fix_hint": "Fix video_structuring.addStructure so it persists segment evidence for retrieval in the sandbox.",
            "affected_file": "generated video_structuring module",
        })
        return quality

    quality["checked"] = True
    all_rows = []
    for path in files[:3]:
        rows = _read_jsonl_limited(path, max_rows=200)
        all_rows.extend(rows)
    quality["row_count"] = len(all_rows)
    quality["sampled_rows"] = len(all_rows)
    if not all_rows:
        quality["issues"].append({
            "issue_type": "structure_jsonl_empty",
            "severity": "critical",
            "description": "Smoke structure JSONL exists but contains no segment rows.",
            "fix_hint": "Fix video_structuring.addStructure so bounded smoke rebuild still emits segment evidence rows.",
            "affected_file": "generated video_structuring module",
        })
        return quality

    artifact_summary = _summarize_smoke_structure_artifacts(
        files,
        all_rows,
        required_channels,
    )
    quality["artifact_summary"] = artifact_summary
    if artifact_summary.get("canonical_record_missing_count", 0) > 0:
        quality["issues"].append({
            "issue_type": "structure_canonical_record_missing",
            "severity": "critical",
            "description": (
                "Smoke structure rows contain raw builder output but do not expose "
                "the canonical fields consumed by localization/retrieval."
            ),
            "fix_hint": (
                "Preserve multimodal_narration through video_structuring.addStructure "
                "and call the neutral base persistence path; do not replace it with "
                "empty derived fields."
            ),
            "affected_file": "generated video_structuring module",
            "missing_rows": artifact_summary.get("canonical_record_missing_count", 0),
            "total_occurrences": artifact_summary.get("canonical_record_missing_count", 0),
            "affected_questions": 1,
            "total_questions": 1,
        })
    absent_required = [
        channel for channel, hit_count in artifact_summary.get("required_channel_hits", {}).items()
        if hit_count <= 0
    ]
    # A smoke case is one unlabeled video slice, not a coverage benchmark for
    # every optional modality mentioned anywhere in bundle code.  Require at
    # least one claimed evidence channel to make it through the persisted
    # producer→consumer handoff; record absent optional channels as audit
    # context rather than rejecting a working non-text/non-audio smoke case.
    if absent_required and len(absent_required) == len(required_channels):
        quality["issues"].append({
            "issue_type": "structure_required_evidence_channel_absent",
            "severity": "critical",
            "description": (
                "The generated bundle's own design/code claims to rely on evidence channel(s) "
                f"{absent_required}, but the smoke-built structure artifacts contain no "
                "non-missing retrievable evidence for those channel(s)."
            ),
            "fix_hint": (
                "Repair the generated modules so the structuring/retrieval artifacts preserve "
                "the evidence channels that the bundle itself uses. This may require fixing "
                "API output parsing, fallback serialization, field names, or tool dataflow. "
                "Do not hard-code task content."
            ),
            "affected_file": "generated initial baseline bundle",
            "missing_channels": absent_required,
            "total_occurrences": len(absent_required),
            "affected_questions": 1,
            "total_questions": 1,
        })
    elif absent_required:
        quality["advisories"].append({
            "issue_type": "structure_optional_evidence_channel_absent",
            "severity": "advisory",
            "description": (
                "This smoke slice did not contain every optional evidence channel "
                f"mentioned by the bundle: {absent_required}."
            ),
            "missing_channels": absent_required,
        })
    return quality


def _compose_smoke_repair_instruction(runtime_issues: list, behavior: dict) -> str:
    parts = []
    if behavior.get("repair_instruction"):
        parts.append(str(behavior.get("repair_instruction")))
    for issue in runtime_issues[:5]:
        hint = issue.get("fix_hint") or issue.get("repair_hint") or issue.get("description")
        if hint and hint not in parts:
            parts.append(str(hint))
    return " ".join(part.strip() for part in parts if part and str(part).strip())


def _smoke_observation_object(step: dict) -> dict:
    """Best-effort structured output view; never raises during smoke auditing."""
    try:
        value = json.loads(str(step.get("observation") or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def build_initial_smoke_failure_contract(result_rows: list, bundle: dict,
                                         structure_quality: dict | None = None) -> dict:
    """Derive a task-free producer/consumer repair contract from smoke.

    This deliberately records runtime facts rather than an LLM diagnosis.  It
    lets Codex distinguish a missing producer output from a consumer that
    incorrectly treats an optional/empty field as completed downstream work.
    Additional contract kinds can be added without changing the repair prompt.
    """
    row = result_rows[0] if result_rows else {}
    trajectory = row.get("trajectory") if isinstance(row, dict) else []
    trajectory = trajectory if isinstance(trajectory, list) else []
    tool_steps = [step for step in trajectory if isinstance(step, dict) and step.get("step_type") == "tool_execution"]
    retrievals = []
    perceptions = []
    for step in tool_steps:
        payload = _smoke_observation_object(step)
        windows = payload.get("candidate_windows") if isinstance(payload.get("candidate_windows"), list) else []
        valid_windows = []
        for item in windows:
            if isinstance(item, dict) and item.get("start_sec") is not None and item.get("end_sec") is not None:
                valid_windows.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                valid_windows.append(item)
        fact = {
            "tool": str(step.get("action") or ""),
            "producer_module": str(step.get("producer_module") or ""),
            "execution_status": str(step.get("execution_status") or ""),
            "output_protocol": str(step.get("output_protocol") or ""),
            "candidate_window_count": len(valid_windows),
            "output_keys": sorted(str(key) for key in payload.keys())[:24],
        }
        if valid_windows:
            retrievals.append(fact)
        if str(step.get("producer_module") or "") == "perception":
            perceptions.append(fact)
    finished = any(
        isinstance(step, dict)
        and step.get("step_type") == "reasoning"
        and step.get("decision") == "finish"
        for step in trajectory
    )
    required_profile = ""
    if isinstance(bundle.get("runtime_policy"), dict):
        required_profile = str(bundle["runtime_policy"].get("required_vlm_profile_id") or "")

    artifact = (structure_quality or {}).get("artifact_summary") if isinstance(structure_quality, dict) else {}
    canonical_missing = int((artifact or {}).get("canonical_record_missing_count", 0) or 0)
    if canonical_missing:
        return {
            "artifact_type": "initial_smoke_failure_contract", "schema_version": 1,
            "contract_kind": "canonical_structure_handoff_lost",
            "selection_basis": "smoke_structure_artifact_and_trajectory",
            "target_module": "video_structuring",
            "producer": {
                "module": "video_structuring",
                "input_protocol": "start_sec,end_sec,multimodal_narration",
                "missing_rows": canonical_missing,
            },
            "consumer": {
                "module": "localization",
                "missing_handoff": "canonical_evidence/retrieval_document",
                "observed_candidate_window_count": sum(item["candidate_window_count"] for item in retrievals),
            },
            "state_semantics": {
                "raw_builder_payload": "multimodal_narration, when supplied, is evidence rather than an optional display field",
                "canonical_record": "must preserve supplied narration or another time-bounded evidence source and expose retrievable normalized fields",
                "empty_record": "must be unavailable, never a successful candidate card",
            },
            "required_transition": {
                "must_call": "video_structuring.addStructure -> neutral base persistence with canonical time-bounded evidence",
                "must_not": "overwrite supplied builder output or alternative media/audio/text evidence with empty placeholders",
            },
            "capability_requirement": {
                "capability": "vlm",
                "required_profile_id": required_profile,
                "required_event": False,
            },
            "falsifiable_success": [
                "Every smoke structure row contains canonical_evidence with supplied raw narration or another non-empty time-bounded evidence source.",
                "Every persisted canonical row contains a non-empty retrieval_document.",
                "Localization reports unavailable rather than ok when no canonical evidence is available.",
            ],
        }

    scope_issues = _localized_media_scope_issues(row)
    if scope_issues:
        issue = scope_issues[0]
        return {
            "artifact_type": "initial_smoke_failure_contract", "schema_version": 1,
            "contract_kind": "localized_media_scope_not_consumed",
            "selection_basis": "same_tool_perception_result_and_capability_events",
            "target_module": "perception",
            "producer": {
                "module": "localization_or_thinking_handoff",
                "output": "non_empty_time_ranges",
                "requested_window_count": issue.get("requested_window_count", 0),
            },
            "consumer": {
                "module": "perception",
                "missing_handoff": "perception input time_ranges -> actual media capability scope",
                "observed_media_event_count": issue.get("observed_media_event_count", 0),
                "overlapping_media_event_count": issue.get("overlapping_media_event_count", 0),
            },
            "state_semantics": {
                "localized_time_ranges": "executable media selectors, not metadata retained only in a result envelope",
                "actual_media_scope": "every successful VLM/OCR/ASR event issued for that perception call must intersect the supplied ranges",
            },
            "required_transition": {
                "must_call": "documented range-scoped runtime_evidence adapter with the perception tool's supplied ranges",
                "must_not": "call an active-window adapter with the unmodified full-video environment after receiving localized ranges",
            },
            "capability_requirement": {
                "capability": "localized_media",
                "required_profile_id": required_profile,
                "required_event": True,
            },
            "falsifiable_success": [
                "Perception receives non-empty normalized time_ranges.",
                "At least one linked successful media capability event intersects those ranges.",
                "The returned perception observation records the actual scoped ranges or explicit unavailable state.",
            ],
        }

    if retrievals and not any(item["execution_status"] == "ok" for item in perceptions):
        producer = retrievals[-1]
        return {
            "artifact_type": "initial_smoke_failure_contract", "schema_version": 1,
            "contract_kind": "producer_consumer_handoff_missing",
            "selection_basis": "deterministic_smoke_trajectory",
            "target_module": "thinking",
            "producer": producer,
            "consumer": {
                "module": "thinking",
                "observed_terminal_decision": "finish" if finished else "non_perception_progress",
                "missing_handoff": "candidate_windows -> perception verification tool",
            },
            "state_semantics": {
                "candidate_windows": "non-empty retrieval output is pending evidence, not verified evidence",
                "optional_visual_fields": "missing, empty, null, or retrieval-origin visual fields must not satisfy visual verification",
                "verified_visual_evidence": "requires a successful perception tool output with non-empty visual evidence or visual summary",
            },
            "required_transition": {
                "when": "candidate_window_count > 0 and no successful perception output exists",
                "must_call": "a perception tool using the retrieved candidate_windows",
                "must_not": "finish or synthesize from retrieval-only output",
            },
            "capability_requirement": {
                "capability": "vlm",
                "required_profile_id": required_profile,
                "required_event": True,
            },
            "falsifiable_success": [
                "A non-empty retrieval candidate_windows output is consumed by a perception tool.",
                "The perception tool returns a successful, non-empty bounded evidence object.",
                "A matching VLM capability event is recorded when the perception design uses VLM.",
                "Thinking consumes the perception output before its finish decision.",
            ],
        }

    # The execution layer calls thinking with ``force_finish=True`` on its
    # final bounded step.  If a candidate has already consumed successful
    # perception output but keeps issuing tools and never returns ``finish``,
    # the generic behavioural contract loses the actual repair target.  This
    # is entirely trajectory-derived: it does not inspect answer labels or
    # infer which answer should have been selected.
    successful_perceptions = [
        item for item in perceptions if item["execution_status"] == "ok"
    ]
    repeated_perception_tools = {}
    for item in successful_perceptions:
        tool = str(item.get("tool") or "perception_tool")
        repeated_perception_tools[tool] = repeated_perception_tools.get(tool, 0) + 1
    repeated_tool_count = max(repeated_perception_tools.values(), default=0)
    if not finished and successful_perceptions and repeated_tool_count >= 2:
        producer = successful_perceptions[-1]
        return {
            "artifact_type": "initial_smoke_failure_contract", "schema_version": 1,
            "contract_kind": "thinking_bounded_loop_without_finish",
            "selection_basis": "deterministic_repeated_successful_perception_and_absent_finish",
            "target_module": "thinking",
            "producer": {
                "module": str(producer.get("producer_module") or "perception"),
                "output_protocol": str(producer.get("output_protocol") or "bounded_evidence_object"),
                "successful_perception_calls": len(successful_perceptions),
                "repeated_tool_calls": repeated_tool_count,
            },
            "consumer": {
                "module": "thinking",
                "missing_handoff": "successful bounded perception output -> finish payload.answer",
                "terminal_state": "no finish decision before the bounded runtime loop ended",
            },
            "state_semantics": {
                "successful_perception": "an ok bounded perception result is usable downstream evidence, not an instruction to repeat the same verification indefinitely",
                "force_finish": "the final runtime thinking call is a termination boundary and must return a finish payload rather than another act plan",
                "answer": "payload.answer must be a concrete evaluator-consumable response; confidence and evidence summary may represent uncertainty but do not replace it",
            },
            "required_transition": {
                "must_call": "the bundle's bounded evidence-to-answer path and finish_payload after successful perception evidence",
                "must_not": "repeat a successful perception request without new state, or return act when force_finish is true",
            },
            "capability_requirement": {
                "capability": "none",
                "required_profile_id": "",
                "required_event": False,
            },
            "falsifiable_success": [
                "Thinking returns action finish after consuming successful perception evidence.",
                "The final bounded thinking call does not return an act plan.",
                "Finish payload contains one non-empty concrete answer parseable for the runtime response form.",
                "A successful perception tool is not repeated without a state-changing reason before finish.",
            ],
        }

    final_answer = str(
        row.get("final_agent_answer") or row.get("answer") or row.get("final_answer") or ""
    )
    final_task = row.get("task_meta") if isinstance(row.get("task_meta"), dict) else row
    non_answer = (
        _is_uncertain_answer(final_answer)
        or _is_guess_or_failure_answer(final_answer)
        or not is_answer_parseable(final_answer, task=final_task)
    )
    if finished and non_answer:
        return {
            "artifact_type": "initial_smoke_failure_contract", "schema_version": 1,
            "contract_kind": "non_answer_terminal_fallback",
            "selection_basis": "finish_payload_shape_and_prior_module_outputs",
            "target_module": "thinking",
            "producer": {
                "module": "localized_and_perception_evidence",
                "output": "structured module results available to runtime thinking context",
            },
            "consumer": {
                "module": "thinking",
                "missing_handoff": "usable prior evidence -> concrete payload.answer",
                "terminal_answer_class": "fallback_uncertainty_or_unparseable",
            },
            "state_semantics": {
                "answer": "payload.answer is a concrete evaluator-consumable response, never an uncertainty or failure sentinel",
                "uncertainty": "may be represented through confidence/evidence_summary but does not replace payload.answer",
            },
            "required_transition": {
                "must_call": "the bundle's own bounded evidence-to-answer path before finish",
                "must_not": "finish with an uncertainty/failure sentinel after successful tool execution",
            },
            "capability_requirement": {
                "capability": "none",
                "required_profile_id": "",
                "required_event": False,
            },
            "falsifiable_success": [
                "Finish payload contains a non-empty concrete answer.",
                "Answer is not an uncertainty/failure/generic fallback token.",
                "Answer is parseable for the runtime question's requested response form.",
            ],
        }
    return {
        "artifact_type": "initial_smoke_failure_contract", "schema_version": 1,
        "contract_kind": "generic_behavioral_smoke_failure",
        "selection_basis": "deterministic_smoke_trajectory",
        "target_module": "",
        "producer_tool_count": len(tool_steps),
        "perception_tool_count": len(perceptions),
        "falsifiable_success": [
            "Repair must remove the observed smoke failure without bypassing runtime adapters or media boundaries.",
            "A preserved consumer must consume the repaired target output before finish.",
        ],
    }


def build_smoke_exception_failure_contract(*, stage: str, error: Exception,
                                          bundle: dict, traceback_text: str = "") -> dict:
    """Create repairable feedback when smoke fails before a trajectory exists."""
    modules = []
    for item in _bundle_module_records(bundle):
        if isinstance(item, dict) and item.get("module_type"):
            modules.append(str(item["module_type"]))
    return {
        "artifact_type": "initial_smoke_failure_contract", "schema_version": 1,
        "contract_kind": "runtime_exception_before_complete_trajectory",
        "failure_stage": stage,
        "producer_state": {"producer": "initial_bundle_runtime", "state": "execution_started"},
        "consumer_state": {"consumer": "real_smoke", "state": "trajectory_not_completed"},
        "observed_exception": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback_text[-4000:],
        },
        "affected_modules": modules,
        "falsifiable_success": [
            "the same bundle completes real smoke without this exception",
            "the smoke trajectory reaches a concrete parseable payload.answer",
        ],
    }


def single_question_bundle_smoke_test(bundle: dict,
                                      task_sample,
                                      workspace_dir: str,
                                      sandbox_dir: str,
                                      combo_id: str = "",
                                      llm_model: str = "",
                                      force_rebuild: bool = False,
                                      isolate_structure: bool = True,
                                      runtime_time_reference_only: bool = False,
                                      reference_structure_dir: str = "") -> dict:
    """Run one real question with a dynamically injected five-module bundle.

    This is a runtime/interface smoke test only. It must not be interpreted as
    an accuracy proof and it never writes to baseline trajectories or global
    video_structure when `sandbox_dir` is isolated by the caller.
    """
    if not llm_model:
        from capability_registry import resolve_profile
        llm_model = str(resolve_profile("llm").get("model_id") or "")
    validation = validate_bundle_assembly(bundle)
    if not validation.get("passed"):
        contract = build_smoke_exception_failure_contract(
            stage="bundle_assembly", error=RuntimeError("bundle assembly is incomplete"), bundle=bundle,
        )
        return {
            "passed": False,
            "stage": "bundle_assembly",
            "validation": validation,
            "engineering_invalid": False,
            "error": "bundle assembly is incomplete",
            "failure_contract": contract,
            "engineering_execution_trace": {
                "artifact_type": "engineering_smoke_execution_trace", "schema_version": 1,
                "rows": [{"stage": "bundle_assembly", "validation": validation}],
            },
        }
    if isinstance(task_sample, dict):
        question = dict(task_sample)
    elif isinstance(task_sample, list) and task_sample:
        question = dict(task_sample[0])
    else:
        contract = build_smoke_exception_failure_contract(
            stage="select_question", error=RuntimeError("empty task_sample"), bundle=bundle,
        )
        return {
            "passed": False,
            "stage": "select_question",
            "validation": validation,
            "engineering_invalid": False,
            "error": "empty task_sample",
            "failure_contract": contract,
        }
    video_id = question.get("video_id", "")
    if not video_id:
        contract = build_smoke_exception_failure_contract(
            stage="select_question", error=RuntimeError("smoke question missing video_id"), bundle=bundle,
        )
        return {
            "passed": False,
            "stage": "select_question",
            "validation": validation,
            "engineering_invalid": False,
            "error": "smoke question missing video_id",
            "failure_contract": contract,
        }

    _ensure_runtime_path_for_workspace(workspace_dir)
    import module_map
    module_map_refs = {
        "structuring": module_map.STRUCTURING_MAP,
        "thinking": module_map.THINKING_MAP,
        "memory": module_map.WORK_MEMORY_MAP,
        "localization": module_map.LOCALIZATION_MAP,
        "perception": module_map.PERCEPTION_MAP,
    }
    backups = {key: dict(value) for key, value in module_map_refs.items()}
    old_model = os.environ.get("METAVIDEOAGENT_LLM_MODEL")
    if llm_model:
        os.environ["METAVIDEOAGENT_LLM_MODEL"] = llm_model
    os.makedirs(sandbox_dir, exist_ok=True)
    result_rows = []
    try:
        ok, agent_combo, custom_configs, injected_validation = inject_module_bundle(
            bundle,
            module_map_refs,
        )
        if not ok:
            contract = build_smoke_exception_failure_contract(
                stage="inject_module_bundle", error=RuntimeError("bundle injection failed"), bundle=bundle,
            )
            return {
                "passed": False,
                "stage": "inject_module_bundle",
                "validation": injected_validation,
                "engineering_invalid": False,
                "error": "bundle injection failed",
                "failure_contract": contract,
            }
        smoke_timeout = float(os.environ.get("CODEX_BUNDLE_SMOKE_TIMEOUT_SEC", "240") or 240)
        with _wall_clock_timeout(smoke_timeout, "bundle smoke execution"):
            result_rows = evaluate_combo_multi_video(
                workspace_dir,
                [question],
                agent_combo,
                combo_id or f"InitialBundleSmoke_{int(time.time())}",
                sandbox_dir,
                custom_configs=custom_configs,
                force_rebuild=force_rebuild,
                isolate_structure=isolate_structure,
                concurrency=1,
                reference_structure_dir=reference_structure_dir,
                runtime_time_reference_only=runtime_time_reference_only,
            )
        runtime_issues = detect_runtime_engineering_issues(result_rows)
        fatal_rows = []
        for row in result_rows:
            answer_text = str(row.get("answer", "") or "")
            if row.get("error") or answer_text.strip().startswith("ERROR:"):
                fatal_rows.append({
                    "task_id": row.get("task_id", ""),
                    "video_id": row.get("video_id", ""),
                    "error": row.get("error", "") or answer_text[:300],
                })
        if fatal_rows:
            runtime_issues.append({
                "issue_type": "single_question_runtime_error",
                "severity": "critical",
                "description": "Smoke test question returned a runtime ERROR row.",
                "fix_hint": "Fix the runtime dependency/API/module error before probe or full evaluation.",
                "affected_file": _runtime_report_path(),
                "total_occurrences": len(fatal_rows),
                "affected_questions": len(fatal_rows),
                "total_questions": len(result_rows),
                "examples": fatal_rows[:3],
            })
        structure_quality = _inspect_smoke_structure_quality(
            sandbox_dir,
            video_id,
            bundle,
            agent_combo,
        )
        runtime_issues.extend(structure_quality.get("issues") or [])
        behavior = behavioral_smoke_assessment(result_rows, agent_combo)
        failure_contract = build_initial_smoke_failure_contract(
            result_rows, bundle, structure_quality=structure_quality,
        )
        engineering_trace = build_engineering_smoke_execution_trace(result_rows)
        if not behavior.get("passed"):
            runtime_issues.append({
                "issue_type": "behavioral_smoke_failed",
                "severity": "critical",
                "description": (
                    "The generated five-module MetaVideoAgent did not complete a usable "
                    "video-grounded QA loop in the smoke trajectory."
                ),
                "fix_hint": behavior.get("repair_instruction", "") or behavior.get("reason", ""),
                "affected_file": "generated initial baseline bundle",
                "total_occurrences": 1,
                "affected_questions": 1,
                "total_questions": len(result_rows),
                "verdict": behavior.get("verdict", ""),
            })
        passed = bool(result_rows) and not runtime_issues and behavior.get("passed")
        repair_instruction = _compose_smoke_repair_instruction(runtime_issues, behavior)
        return {
            "passed": passed,
            "stage": "runtime_smoke",
            "validation": injected_validation,
            "combo": agent_combo,
            "sandbox_dir": sandbox_dir,
            "results_count": len(result_rows),
            "sample_task_id": (result_rows[0].get("task_id") if result_rows else question.get("task_id", "")),
            "sample_video_id": video_id,
            "runtime_engineering_issues": runtime_issues,
            "engineering_invalid": bool(runtime_issues),
            "artifact_quality": structure_quality,
            "structure_quality": structure_quality,
            "behavioral_smoke": behavior,
            "failure_contract": failure_contract,
            "engineering_execution_trace": engineering_trace,
            "repair_instruction": repair_instruction,
            "note": "Smoke test validates runtime compatibility only; it is not final effectiveness evidence.",
        }
    except Exception as exc:
        trace = tb.format_exc()[-4000:]
        return {
            "passed": False,
            "stage": "runtime_exception",
            "validation": validation,
            "sandbox_dir": sandbox_dir,
            "results_count": len(result_rows),
            "engineering_invalid": True,
            "error": str(exc),
            "traceback": trace,
            "failure_contract": build_smoke_exception_failure_contract(
                stage="runtime_exception", error=exc, bundle=bundle, traceback_text=trace,
            ),
        }
    finally:
        for key, backup in backups.items():
            module_map_refs[key].clear()
            module_map_refs[key].update(backup)
        if old_model is None:
            os.environ.pop("METAVIDEOAGENT_LLM_MODEL", None)
        else:
            os.environ["METAVIDEOAGENT_LLM_MODEL"] = old_model


def inject_evolved_modules(evolved: dict, module_map_refs: dict):
    """Inject generated module records into the corresponding runtime maps.

    Args:
        evolved: {"thinking": [...], "perception": [...], ...}
        module_map_refs: {
            "structuring": STRUCTURING_MAP,
            "thinking": THINKING_MAP,
            "memory": WORK_MEMORY_MAP,
            "localization": LOCALIZATION_MAP,
            "perception": PERCEPTION_MAP,
        }
    Returns:
        custom_configs: {module_name: custom_config_dict}
    """
    map_key_map = {
        "thinking": "thinking",
        "memory": "memory",
        "video_structuring": "structuring",
        "localization": "localization",
        "perception": "perception",
    }

    custom_configs = {}

    for module_type, modules in evolved.items():
        if not modules:
            continue

        map_key = map_key_map.get(module_type)
        if not map_key or map_key not in module_map_refs:
            print(f"[Sandbox] Unknown module type {module_type}; skipping injection")
            continue

        target_map = module_map_refs[map_key]
        latest = modules[-1]
        module_name = latest.get("name", "")
        try:
            cls = create_module_class(latest)

            target_map[module_name] = cls
            print(f"   [Sandbox] Injected {module_type}: {module_name}")

        except Exception as e:
            print(f"   ⚠️ [Sandbox] Failed to inject {module_name}: {e}")

    return custom_configs






def map_combo_to_agent_config(combo_cfg: dict) -> dict:
    mapping = {
        "video_structuring": "video_structuring", "thinking": "thinking",
        "memory": "memory",
        "localization": "localization", "perception": "perception",
    }
    agent_combo = {}
    for src_key, dst_key in mapping.items():
        val = combo_cfg.get(src_key, "")
        if not val:
            continue
        agent_combo[dst_key] = val
    return require_complete_agent_combo(agent_combo, "MetaVideoAgent sandbox combo")


# =====================================================================
# Judgment
# =====================================================================

def _answer_judgement(final_answer: str, task_or_gt=None) -> dict:
    if isinstance(task_or_gt, dict):
        return judge_answer(final_answer, task=task_or_gt)
    return judge_answer(final_answer, task={"gt_answer": task_or_gt or ""})


def _rule_based_judge(final_answer: str, gt_answer_or_task) -> bool:
    return bool(_answer_judgement(final_answer, gt_answer_or_task).get("is_correct"))


def _resolved_gt_answer(task: dict) -> str:
    return extract_gold_answer(task)


def _normalized_answer_payload(answer: str, task: dict) -> dict:
    judgement = _answer_judgement(answer, task)
    return {
        "answer_normalization": judgement.get("prediction", {}),
        "gold_answer": judgement.get("gold_answer", ""),
        "gold_normalized_answer": judgement.get("gold_normalized_answer", ""),
        "gold_labels": judgement.get("gold_labels", []),
        "judge_reason": judgement.get("reason", ""),
    }


def _agent_visible_task(task: dict) -> dict:
    """Return the only dataset fields that may cross into agent execution.

    Ground truth, evidence intervals, and human annotations remain in the
    evaluator-owned task object for *post hoc* scoring only.  This hardens the
    held-out-test boundary even if a future runtime module starts inspecting
    ``extra_info`` beyond the current choices formatter.
    """
    source = dict(task or {})
    visible = {
        key: source.get(key)
        for key in ("video_id", "task_id", "question", "choices")
        if key in source
    }
    visible["runtime_input_redaction"] = (
        "answer,time_reference,evidence,type/category fields are evaluator-only"
    )
    return visible


# =====================================================================
# Single question evaluation
# =====================================================================

def run_single_question(agent_cls, env_cls, build_fn, workspace_dir, video_id, combo, task,
                        sandbox_dir, custom_configs=None, force_rebuild=False,
                        isolate_structure=False, reference_structure_dir: str = "",
                        runtime_time_reference_only: bool = False):
    task = dict(task or {})
    task.setdefault("video_id", video_id)
    task.setdefault("task_id", make_task_id(
        task.get("video_id", video_id),
        task.get("time_reference", ""),
        task.get("question", ""),
    ))
    question = task.get("question", "")
    gt_answer = _resolved_gt_answer(task)
    structure_window_env_backup = None
    media_window_token = None
    runtime_windows = []
    visible_structure = None
    # Keep the full, credential-free provider/tool trace beside the existing
    # compact capability-event audit.  Initialise these before setup so an
    # early structure/environment exception still produces a well-formed
    # engineering row instead of masking the original failure with an
    # UnboundLocalError in the exception handler.
    evidence_token = None
    runtime_trace_token = None
    capability_events = []
    runtime_trace = []

    def _restore_structure_window_env():
        nonlocal structure_window_env_backup
        if structure_window_env_backup is None:
            return
        for key, value in structure_window_env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        structure_window_env_backup = None

    try:
        env = env_cls(workspace_dir=workspace_dir, video_id=video_id)
        if runtime_time_reference_only:
            runtime_windows = _extract_time_reference_windows(task.get("time_reference", ""))
            if not runtime_windows:
                raise RuntimeError(
                    "time_reference-only probe requires at least one valid time_reference interval; "
                    "refusing full-video fallback."
                )
            runtime_windows = env.set_active_time_windows(
                runtime_windows,
                scope="probe_time_reference_exact",
            )
            if not runtime_windows:
                raise RuntimeError(
                    "time_reference-only probe intervals are outside the video; refusing full-video fallback."
                )
            # This ContextVar is consumed by shared runtime helpers such as
            # utils.extract_audio_segment. It is task-local even with probe
            # worker concurrency, unlike process-global environment variables.
            import utils
            media_window_token = utils.set_active_media_windows(runtime_windows)

            # A structuring candidate must build only the supplied media slice.
            # Reused artifacts are separately filtered below. The active
            # ContextVar is consumed by the runtime structure builder, so this
            # remains task-local when isolated probe questions run in parallel.
            # Do not write process-global build-window variables.
            if force_rebuild and isolate_structure:
                structure_window_env_backup = None
        if (
            force_rebuild
            and isolate_structure
            and os.environ.get("SANDBOX_STRUCTURE_TIME_REF_ONLY", "").lower() in ("1", "true", "yes")
            and not runtime_time_reference_only
        ):
            bounds = _extract_time_reference_bounds(task.get("time_reference", ""))
            if bounds:
                start, end = bounds
                padding = float(os.environ.get("SANDBOX_STRUCTURE_TIME_REF_PADDING", "15") or 15)
                original_len = float(getattr(env, "video_length_secs", 0.0) or 0.0)
                window_start = max(0.0, float(start) - padding)
                capped_len = min(original_len, max(1.0, float(end) + padding))
                allowed_windows = [(window_start, capped_len)]
                windows_raw = os.environ.get("STRUCTURE_BUILD_WINDOWS_JSON", "").strip()
                merge_windows_enabled = (
                    os.environ.get("SANDBOX_MERGE_TIME_REF_WINDOWS", "").lower()
                    in ("1", "true", "yes")
                )
                if windows_raw and merge_windows_enabled:
                    try:
                        windows = json.loads(windows_raw)
                        parsed_windows = [
                            (float(item[0]), float(item[1]))
                            for item in windows
                            if isinstance(item, list) and len(item) >= 2
                        ]
                        if parsed_windows:
                            allowed_windows = parsed_windows
                        capped_len = min(
                            original_len,
                            max(end for _, end in allowed_windows),
                        )
                    except Exception:
                        pass
                else:
                    structure_window_env_backup = {
                        "STRUCTURE_BUILD_WINDOWS_JSON": os.environ.get("STRUCTURE_BUILD_WINDOWS_JSON"),
                        "STRUCTURE_BUILD_START_SEC": os.environ.get("STRUCTURE_BUILD_START_SEC"),
                        "STRUCTURE_BUILD_END_SEC": os.environ.get("STRUCTURE_BUILD_END_SEC"),
                    }
                    os.environ.pop("STRUCTURE_BUILD_WINDOWS_JSON", None)
                    os.environ["STRUCTURE_BUILD_START_SEC"] = f"{window_start:.3f}"
                    os.environ["STRUCTURE_BUILD_END_SEC"] = f"{capped_len:.3f}"
                setattr(env, "allowed_time_windows", allowed_windows)
                if capped_len < original_len:
                    env.video_length_secs = capped_len
                    print(
                        "   [Sandbox] time-ref-only structure window: "
                        f"{window_start:.1f}s - {capped_len:.1f}s of {original_len:.1f}s"
                    )
        effective_configs = dict(custom_configs or {})
        effective_configs["agent"] = dict(effective_configs.get("agent", {}))
        effective_configs["agent"]["sandbox_dir"] = sandbox_dir
        # The evaluator is the single writer for sandbox result JSONL rows.
        # Agent-internal experience logging writes a different nested schema to
        # the same *_sandbox.jsonl path, which corrupts full-eval row
        # counts and downstream task-id matching.
        agent = agent_cls(env, combo, custom_configs=effective_configs)
        allow_full_rebuild = (
            isolate_structure and force_rebuild and _allow_isolated_full_structure_rebuild()
        )
        require_full_rebuild = (
            isolate_structure
            and force_rebuild
            and os.environ.get("SANDBOX_REQUIRE_STRUCTURE_REBUILD", "").lower() in ("1", "true", "yes")
        )
        structure_eval_mode = (
            "full_rebuild" if allow_full_rebuild
            else ("reuse_existing_db" if isolate_structure else "baseline_workspace")
        )
        if require_full_rebuild and not allow_full_rebuild:
            raise RuntimeError(
                "This video_structuring candidate requires full structure rebuild, "
                "but SANDBOX_ALLOW_FULL_STRUCTURE_REBUILD is not enabled. "
                "Refusing to evaluate against copied baseline JSONL/ChromaDB."
            )
        if isolate_structure:
            _redirect_structuring_to_sandbox(
                agent.structuring,
                sandbox_dir=sandbox_dir,
                reset=force_rebuild,
                copy_existing=not allow_full_rebuild,
                reference_structure_dir=reference_structure_dir,
            )
        db_exists = os.path.exists(agent.structuring.db_path) and os.path.getsize(agent.structuring.db_path) > 0
        resume_structure_build = (
            isolate_structure
            and os.environ.get("VIDEO_STRUCT_RESUME_BUILD", "").lower() in ("1", "true", "yes")
            and os.environ.get("SANDBOX_ALLOW_STRUCTURE_RESUME_BUILD", "").lower() in ("1", "true", "yes")
        )
        if allow_full_rebuild:
            if db_exists:
                print(f"   [Sandbox] The isolated structure store will be rebuilt: {agent.structuring.db_path}")
            print(f"   [Sandbox] Rebuilding the structure store in isolation: {agent.structuring.db_path}")
            build_fn(env, agent.structuring)
            _rebuild_sandbox_chroma_if_requested(agent.structuring)
        elif db_exists and force_rebuild and isolate_structure:
            print(f"   ⏭️ [Sandbox] Reusing the isolated structure store: {agent.structuring.db_path}")
            _rebuild_sandbox_chroma_if_requested(agent.structuring)
        elif db_exists and force_rebuild:
            # Clean up the contents of the ChromaDB collection (without deleting the directory to avoid sqlite readonly errors)
            if hasattr(agent.structuring, 'requires_vector_db') and agent.structuring.requires_vector_db:
                if agent.structuring.collection is not None:
                    col_name = agent.structuring.collection.name
                    try:
                        agent.structuring.chroma_client.delete_collection(col_name)
                        agent.structuring.collection = agent.structuring.chroma_client.get_or_create_collection(name=col_name)
                        print(f"   [Sandbox] Cleared ChromaDB collection: {col_name}")
                    except Exception as e:
                        print(f"⚠️ [Sandbox] Failed to clear ChromaDB: {e}")
            if isolate_structure:
                build_fn(env, agent.structuring)
            else:
                # Rebuild ChromaDB from JSONL (no VLM required, just Embedding API)
                _rebuild_sandbox_chroma_if_requested(agent.structuring)
        elif db_exists and resume_structure_build:
            print(f"   ↩️ [Sandbox] Continuing the isolated structure build: {agent.structuring.db_path}")
            build_fn(env, agent.structuring)
            _rebuild_sandbox_chroma_if_requested(agent.structuring)
        elif db_exists:
            print(f"   ⏭️ [Sandbox] Structure store already exists; build skipped: {agent.structuring.db_path}")
            # If ChromaDB is empty (e.g. not rebuilt since last force_rebuild), supplementary rebuild
            _rebuild_sandbox_chroma_if_requested(agent.structuring)
        else:
            if isolate_structure and _full_structure_build_disabled():
                raise RuntimeError(
                    "Isolated structure DB is missing and full structure rebuild is disabled "
                    "(SANDBOX_DISABLE_FULL_STRUCTURE_BUILD=1)."
                )
            build_fn(env, agent.structuring)
        if runtime_time_reference_only:
            visible_structure = _restrict_sandbox_structure_artifact(
                agent.structuring,
                runtime_windows,
            )
            _install_probe_structure_boundary(agent, runtime_windows)
        import runtime_evidence
        evidence_token = runtime_evidence.begin_evidence_run()
        runtime_trace_token = runtime_evidence.begin_runtime_trace()
        try:
            answer, trajectory = agent.run(
                question,
                extra_info=_agent_visible_task(task),
            )
        finally:
            capability_events = runtime_evidence.get_evidence_events()
            runtime_trace = runtime_evidence.get_runtime_trace()
            runtime_evidence.reset_evidence_run(evidence_token)
            runtime_evidence.reset_runtime_trace(runtime_trace_token)
            evidence_token = None
            runtime_trace_token = None
        is_correct = _rule_based_judge(answer, task)
        steps = len([s for s in trajectory if s.get("step_type") == "tool_execution"])
        row = {
            "question": question,
            "gt_answer": gt_answer,
            "time_reference": task.get("time_reference", ""),
            "video_id": video_id,
            "task_id": task.get("task_id", ""),
            "task_meta": task,
            "answer": answer,
            "is_correct": is_correct,
            "steps": steps,
            "trajectory": trajectory,
            "structure_eval_mode": structure_eval_mode,
            "runtime_media_scope": "time_reference_exact" if runtime_time_reference_only else "full_video",
            "runtime_time_windows": [list(item) for item in runtime_windows],
            "visible_structure": visible_structure,
            "capability_events": capability_events,
            "runtime_trace": runtime_trace,
            # Exact per-task API ledger from the runtime.  It is attached to
            # every successful agent execution before JSONL persistence so
            # full-eval/probe audits never have to infer token usage from text.
            "cost": dict(getattr(agent, "last_cost_summary", {}) or {}),
        }
        row.update(_normalized_answer_payload(answer, task))
        _restore_structure_window_env()
        if media_window_token is not None:
            utils.reset_active_media_windows(media_window_token)
        return attach_task_identity(row, fallback_video_id=video_id)
    except Exception as e:
        if evidence_token is not None:
            try:
                import runtime_evidence
                capability_events = runtime_evidence.get_evidence_events()
                runtime_evidence.reset_evidence_run(evidence_token)
            except Exception:
                pass
        if runtime_trace_token is not None:
            try:
                import runtime_evidence
                runtime_trace = runtime_evidence.get_runtime_trace()
                runtime_evidence.reset_runtime_trace(runtime_trace_token)
            except Exception:
                pass
        print(f"   ❌ [Sandbox] Agent execution failed: {e}")
        tb.print_exc()
        row = {
            "question": question,
            "gt_answer": gt_answer,
            "time_reference": task.get("time_reference", ""),
            "video_id": video_id,
            "task_id": task.get("task_id", ""),
            "task_meta": task,
            "answer": f"ERROR: {str(e)}",
            "is_correct": False,
            "steps": 0,
            "trajectory": [],
            "error": str(e),
            "traceback": tb.format_exc(),
            "structure_eval_mode": locals().get("structure_eval_mode", "unknown"),
            "runtime_media_scope": "time_reference_exact" if runtime_time_reference_only else "full_video",
            "runtime_time_windows": [list(item) for item in runtime_windows],
            "visible_structure": visible_structure,
            "capability_events": capability_events,
            "runtime_trace": runtime_trace,
            "cost": {},
        }
        row.update(_normalized_answer_payload(row["answer"], task))
        _restore_structure_window_env()
        if media_window_token is not None:
            try:
                import utils
                utils.reset_active_media_windows(media_window_token)
            except Exception:
                pass
        return attach_task_identity(row, fallback_video_id=video_id)


# =====================================================================
# Comparative analysis
# =====================================================================

def _result_answer(row: dict) -> str:
    if not isinstance(row, dict):
        return ""
    return (
        row.get("answer")
        or row.get("final_answer")
        or row.get("final_agent_answer")
        or ""
    )


def build_reference_lookup(result_rows: list, reference_label: str = "reference") -> dict:
    """Build a comparison lookup from sandbox/full-eval result rows."""
    lookup = {}
    for row in result_rows or []:
        if not isinstance(row, dict):
            continue
        entry = attach_task_identity(dict(row), fallback_video_id=row.get("video_id", ""))
        ident = identity_from_entry(entry)
        task_id = ident.get("task_id", "")
        q = ident.get("question", "") or entry.get("question", "")
        if not task_id and not q:
            continue
        gt = _resolved_gt_answer(entry)
        ans = _result_answer(entry)
        judgement = _answer_judgement(ans, entry)
        traj = entry.get("trajectory", []) or []
        steps = entry.get("steps")
        if steps is None:
            steps = len([
                s for s in traj
                if isinstance(s, dict) and s.get("step_type") == "tool_execution"
            ])
        item = {
            "task_id": task_id,
            "video_id": ident.get("video_id", "") or entry.get("video_id", ""),
            "time_reference": ident.get("time_reference", "") or entry.get("time_reference", ""),
            "question": q,
            "gt_answer": gt,
            "answer": ans,
            "steps": int(steps or 0),
            "is_correct": bool(judgement.get("is_correct")),
            "answer_normalization": judgement.get("prediction", {}),
            "gold_normalized_answer": judgement.get("gold_normalized_answer", ""),
            "gold_labels": judgement.get("gold_labels", []),
            "judge_reason": judgement.get("reason", ""),
            "reference_label": reference_label,
        }
        if task_id:
            lookup[task_id] = item
        if q:
            lookup.setdefault(q, item)
    return lookup


def compare_with_reference(candidate_results: list, reference_lookup: dict,
                           *, require_reference: bool = True,
                           reference_label: str = "current_best",
                           reference_source: str = "explicit_reference_results") -> dict:
    """Compare candidate rows with an explicit, task-aligned reference set."""
    report = {"total_questions": len(candidate_results), "reference_correct": 0, "candidate_correct": 0,
              "corrections": [], "regressions": [], "efficiency_gains": [],
              "unchanged_correct": [], "unchanged_incorrect": [],
              "reference_total_steps": 0, "candidate_total_steps": 0,
              "structure_eval_modes": {},
              "reference_label": reference_label,
              "reference_source": reference_source,
              "comparison_target": {
                  "label": reference_label,
                  "source": reference_source,
              },
              "field_semantics": {
                  "reference_correct": "primary comparison target correctness",
                  "candidate_correct": "candidate correctness",
              },
              "missing_reference": [],
              "comparison_valid": True}
    for sr in candidate_results:
        sr = attach_task_identity(dict(sr), fallback_video_id=sr.get("video_id", ""))
        sr_answer = _result_answer(sr)
        sr_judgement = _answer_judgement(sr_answer, sr)
        sr["gt_answer"] = sr_judgement.get("gold_answer") or sr.get("gt_answer", "")
        sr["is_correct"] = bool(sr_judgement.get("is_correct"))
        sr.setdefault("answer_normalization", sr_judgement.get("prediction", {}))
        sr.setdefault("gold_normalized_answer", sr_judgement.get("gold_normalized_answer", ""))
        sr.setdefault("gold_labels", sr_judgement.get("gold_labels", []))
        sr.setdefault("judge_reason", sr_judgement.get("reason", ""))
        # Some real runtime rows contain a trajectory but no denormalized
        # ``steps`` field.  The comparison is used by promotion gating, so it
        # must derive the same bounded tool-call count as reference indexing
        # rather than raising or silently relying on a producer-specific row
        # shape.
        if sr.get("steps") is None:
            sr["steps"] = len([
                step for step in (sr.get("trajectory") or [])
                if isinstance(step, dict) and step.get("step_type") == "tool_execution"
            ])
        q = sr["question"]
        task_id = sr.get("task_id", "")
        structure_eval_mode = sr.get("structure_eval_mode", "unknown")
        report["structure_eval_modes"][structure_eval_mode] = (
            report["structure_eval_modes"].get(structure_eval_mode, 0) + 1
        )
        reference = reference_lookup.get(task_id) or reference_lookup.get(q)
        if not reference:
            report["missing_reference"].append({
                "question": q[:80],
                "time_reference": sr.get("time_reference", ""),
                "video_id": sr.get("video_id", ""),
                "task_id": task_id,
                "reason": f"missing {reference_label} trajectory",
            })
            if require_reference:
                report["comparison_valid"] = False
            if sr.get("is_correct"):
                report["candidate_correct"] += 1
            continue
        reference_correct = reference.get("is_correct", False)
        candidate_correct = sr["is_correct"]
        reference_steps = reference.get("steps", 0)
        candidate_steps = sr["steps"]
        report["reference_total_steps"] += reference_steps
        report["candidate_total_steps"] += candidate_steps
        if reference_correct:
            report["reference_correct"] += 1
        if candidate_correct:
            report["candidate_correct"] += 1
        entry = {"question": q[:80], "time_reference": sr.get("time_reference", ""),
                 "video_id": sr.get("video_id", ""), "task_id": task_id,
                 "reference_correct": reference_correct, "candidate_correct": candidate_correct,
                 "reference_steps": reference_steps, "candidate_steps": candidate_steps,
                 "reference_answer": reference.get("answer", "")[:100], "candidate_answer": _result_answer(sr)[:100],
                 "candidate_normalized_answer": (sr.get("answer_normalization") or {}).get("normalized_answer", ""),
                 "gt_answer": sr.get("gt_answer", ""),
                 "gold_normalized_answer": sr.get("gold_normalized_answer", ""),
                 "judge_reason": sr.get("judge_reason", ""),
                 "structure_eval_mode": structure_eval_mode}
        if not reference_correct and candidate_correct:
            report["corrections"].append(entry)
        elif reference_correct and not candidate_correct:
            report["regressions"].append(entry)
        elif reference_correct and candidate_correct:
            if candidate_steps < reference_steps:
                report["efficiency_gains"].append(entry)
            else:
                report["unchanged_correct"].append(entry)
        else:
            report["unchanged_incorrect"].append(entry)
    report["accuracy_delta"] = report["candidate_correct"] - report["reference_correct"]
    total = report["total_questions"]
    if total:
        report["reference_accuracy"] = report["reference_correct"] / total
        report["candidate_accuracy"] = report["candidate_correct"] / total
    report["steps_delta"] = report["candidate_total_steps"] - report["reference_total_steps"]
    report["reference_coverage"] = (
        (total - len(report["missing_reference"])) / total if total else 0
    )
    report["accuracy_delta_vs_reference"] = report["accuracy_delta"]
    report["steps_delta_vs_reference"] = report["steps_delta"]
    report["reference_metrics"] = {
        "label": reference_label,
        "source": reference_source,
        "correct": report["reference_correct"],
        "accuracy": report["reference_accuracy"],
        "total_steps": report["reference_total_steps"],
        "coverage": report["reference_coverage"],
        "missing": report["missing_reference"],
    }
    report["candidate_metrics"] = {
        "correct": report["candidate_correct"],
        "accuracy": report["candidate_accuracy"],
        "total_steps": report["candidate_total_steps"],
    }
    report["delta_metrics"] = {
        "accuracy_delta": report["accuracy_delta_vs_reference"],
        "steps_delta": report["steps_delta_vs_reference"],
        "corrections": len(report.get("corrections", []) or []),
        "regressions": len(report.get("regressions", []) or []),
    }
    report["is_effective"] = (
        report["comparison_valid"]
        and report["accuracy_delta"] > 0
        and len(report.get("corrections", []) or []) >= 1
    )
    return report


def evaluate_combo_multi_video(workspace_dir, questions, combo, combo_id,
                               sandbox_dir, custom_configs=None,
                               on_question_done=None, force_rebuild=False,
                               isolate_structure=False, concurrency: int = 1,
                               reference_structure_dir: str = "",
                               runtime_time_reference_only: bool = False,
                               force_rebuild_by_video: dict | None = None,
                               global_question_scheduler: bool = False):
    """Evaluate a mixed-video question set.

    A full evaluation with already materialized per-video structure libraries
    can schedule questions from all videos through one worker pool.  This
    avoids an idle tail where workers assigned to a completed video wait for
    every question of that video before the next video is allowed to start.
    Structure rebuilding deliberately remains video-grouped: builders use
    shared per-video artifacts and are not safe to launch across videos from
    this lightweight answer scheduler.
    """
    grouped = {}
    for task in questions:
        video_id = task.get("video_id", "")
        if not video_id:
            raise ValueError(f"Question missing video_id: {task.get('question', '')[:80]}")
        grouped.setdefault(video_id, []).append(task)

    all_results = []
    rebuild_requested = bool(force_rebuild) or any(
        bool(value) for value in (force_rebuild_by_video or {}).values()
    )
    # Exact time-reference probes rebuild into a per-question isolated
    # directory and carry their media scope through ContextVar. They are not
    # the shared per-video rebuild case, so they can safely use the global
    # question scheduler. Other rebuild modes remain video-grouped.
    isolated_scoped_rebuild = bool(
        force_rebuild and isolate_structure and runtime_time_reference_only
    )
    if global_question_scheduler and (not rebuild_requested or isolated_scoped_rebuild):
        workers = max(1, int(concurrency or 1))
        print(
            f"   ⚙️ [{combo_id}] Scheduling questions globally: {workers} workers, "
            f"{len(questions)} questions across {len(grouped)} videos "
            "(prebuilt structures or isolated scoped rebuilds)"
        )

        def _run_one(question_index, task):
            video_id = str(task["video_id"])
            # A singleton call bypasses evaluate_combo's per-video warm-up.
            # The caller has already materialized the exact isolated JSONL
            # structures, so each worker may immediately answer its task.
            rows = evaluate_combo(
                workspace_dir, video_id, combo, combo_id, [task], sandbox_dir,
                custom_configs=custom_configs,
                force_rebuild=bool(isolated_scoped_rebuild),
                isolate_structure=isolate_structure,
                concurrency=1,
                reference_structure_dir=reference_structure_dir,
                runtime_time_reference_only=runtime_time_reference_only,
            )
            if len(rows) != 1:
                raise RuntimeError(
                    f"global question scheduler expected one row for {video_id}, got {len(rows)}"
                )
            row = rows[0]
            row["_global_question_index"] = question_index
            return row

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_one, index, task): index
                for index, task in enumerate(questions)
            }
            for future in as_completed(futures):
                row = future.result()
                row.pop("_global_question_index", None)
                attach_task_identity(row, fallback_video_id=str(row.get("video_id") or ""))
                all_results.append(row)
                print(
                    f"      ✓ [global {len(all_results)}/{len(questions)}] "
                    f"{row.get('video_id', '?')} {row.get('time_reference', '?')}"
                )
                if on_question_done:
                    should_continue = on_question_done(
                        len(all_results) - 1, row, list(all_results)
                    )
                    if not should_continue:
                        for pending in futures:
                            pending.cancel()
                        break
        return all_results
    if global_question_scheduler and rebuild_requested:
        print(
            f"   ℹ️ [{combo_id}] Structural reconstruction detected; retained by video schedule,"
            "Avoid cross-video concurrent database construction conflicts"
        )

    for video_id in sorted(grouped):
        group = grouped[video_id]
        video_force_rebuild = bool(
            (force_rebuild_by_video or {}).get(video_id, force_rebuild)
        )
        count = len(group)
        print(f"   [{combo_id}] Video {video_id}: {count} question{'s' if count != 1 else ''}")
        results = evaluate_combo(
            workspace_dir, video_id, combo, combo_id, group, sandbox_dir,
            custom_configs=custom_configs,
            on_question_done=on_question_done,
            force_rebuild=video_force_rebuild,
            isolate_structure=isolate_structure,
            concurrency=concurrency,
            reference_structure_dir=reference_structure_dir,
            runtime_time_reference_only=runtime_time_reference_only,
        )
        for item in results:
            attach_task_identity(item, fallback_video_id=video_id)
        all_results.extend(results)
    return all_results


# =====================================================================
# Effective combination pool
# =====================================================================


RUNTIME_ENGINEERING_PATTERNS = [
    {
        "regex": r"FileNotFoundError|(?:video|media) (?:asset|file).*(?:missing|not found)",
        "issue_type": "media_asset_unavailable",
        "severity": "critical",
        "description": "One or more required media assets are unavailable.",
        "fix_hint": "Restore the declared media assets and rerun the same candidate.",
        "affected_file": "execution environment",
    },
    {
        "regex": r"(?:401|403).*(?:auth|credential)|unauthorized|invalid (?:api )?(?:key|token)",
        "issue_type": "provider_authentication_failure",
        "severity": "critical",
        "description": "A configured provider rejected authentication.",
        "fix_hint": "Correct the provider profile or credentials and rerun the same candidate.",
        "affected_file": "provider configuration",
    },
    {
        "regex": r"ConnectionError|connection refused|ECONNREFUSED",
        "issue_type": "provider_connection_failure",
        "severity": "high",
        "description": "A required provider connection failed.",
        "fix_hint": "Verify the configured endpoint and network, then rerun the same candidate.",
        "affected_file": "provider configuration",
    },
    {
        "regex": r"TimeoutError|timed out",
        "issue_type": "provider_timeout",
        "severity": "high",
        "description": "A required provider request timed out repeatedly.",
        "fix_hint": "Verify provider availability and timeout settings, then rerun the same candidate.",
        "affected_file": "provider configuration",
    },
    {
        "regex": r"(?:HTTP )?429|rate limit",
        "issue_type": "provider_rate_limit",
        "severity": "high",
        "description": "A required provider repeatedly rejected requests because of rate limits.",
        "fix_hint": "Adjust provider capacity or retry timing and rerun the same candidate.",
        "affected_file": "provider configuration",
    },
    {
        "regex": r"ModuleNotFoundError|ImportError|AttributeError|TypeError|NameError",
        "issue_type": "runtime_python_contract_failure",
        "severity": "critical",
        "description": "Candidate execution violated a Python import or runtime interface contract.",
        "fix_hint": "Repair the candidate bundle against the documented runtime ABI and rerun validation.",
        "affected_file": "candidate bundle",
    },
]


def detect_runtime_engineering_issues(results: list, threshold_ratio: float = 0.5) -> list:
    """Analyze sandbox results and detect systemic engineering problems.

    An issue is systemic when it affects at least ``threshold_ratio`` of the
    questions or occurs at least five times.

    Args:
        results: list of results returned by evaluate_combo
        threshold_ratio: The minimum impact ratio that is determined to be a systemic problem (default 50%)

    Returns:
        Detected engineering issues, sorted by severity.
    """
    if not results:
        return []

    issue_counts = {}

    def record(issue_type: str, info: dict, *, occurrences: int = 1) -> None:
        data = issue_counts.setdefault(issue_type, {
            "occurrences": 0, "affected_questions": 0, "info": info,
        })
        data["occurrences"] += max(1, int(occurrences))
        data["affected_questions"] += 1

    structured_infos = {
        "structured_capability_provider_failure": {
            "severity": "critical",
            "description": "A structured capability event reports a provider or asset failure.",
            "fix_hint": "Fix the provider, credentials, network, or media assets and rerun the same candidate.",
            "affected_file": "execution environment",
        },
        "structured_runtime_tool_contract_failure": {
            "severity": "critical",
            "description": "A structured tool-execution trace reports an interface or parameter failure.",
            "fix_hint": "Repair the candidate against the recorded tool name, parameters, and output protocol.",
            "affected_file": _runtime_report_path("agent.py"),
        },
    }

    def structured_failure_text(result: dict) -> str:
        """Collect only machine-designated failures for previous regex checks.

        Raw VLM/LLM evidence can legitimately contain words such as ``timeout``
        or ``rate limit`` while describing a UI.  It must never be treated as
        provider telemetry.  Capability events, explicit result errors, and
        non-OK tool/trace states are the sole admissible sources here.
        """
        parts = []
        explicit_error = str(result.get("error") or "")
        answer = str(result.get("answer") or "")
        if explicit_error:
            parts.append(explicit_error)
        if answer.strip().startswith("ERROR:"):
            parts.append(answer)
        for event in result.get("capability_events", []) or []:
            if not isinstance(event, dict):
                continue
            if str(event.get("status") or "") in {"api_error", "asset_error", "unavailable"}:
                parts.append(json.dumps({
                    "error": event.get("error") or event.get("error_message") or "",
                    "reason": event.get("reason") or "",
                    "status": event.get("status"),
                }, ensure_ascii=False))
        for step in result.get("trajectory", []) or []:
            if not isinstance(step, dict) or step.get("step_type") != "tool_execution":
                continue
            if str(step.get("execution_status") or "ok") != "ok":
                parts.append(json.dumps({
                    "execution_status": step.get("execution_status"),
                    "error_code": step.get("error_code"),
                    "observation": step.get("observation"),
                }, ensure_ascii=False))
        for trace in result.get("runtime_trace", []) or []:
            if not isinstance(trace, dict):
                continue
            if str(trace.get("status") or "") in {"api_error", "asset_error", "unavailable", "invalid", "error"}:
                parts.append(json.dumps({
                    "stage": trace.get("stage"), "status": trace.get("status"),
                    "result": trace.get("result"),
                }, ensure_ascii=False))
        return "\n".join(parts)

    for result in results:
        trajectory = result.get("trajectory", []) or []
        capability_events = result.get("capability_events", []) or []
        if any(
            isinstance(event, dict)
            and (event.get("status") in {"api_error", "asset_error"}
                 or (event.get("status") == "unavailable" and event.get("reason") not in {"", "no_active_time_windows"}))
            for event in capability_events
        ):
            record("structured_capability_provider_failure", structured_infos["structured_capability_provider_failure"])
        if any(
            isinstance(step, dict) and step.get("step_type") == "tool_execution"
            and str(step.get("execution_status") or "ok") != "ok"
            for step in trajectory
        ):
            record("structured_runtime_tool_contract_failure", structured_infos["structured_runtime_tool_contract_failure"])

        # Answers and successful raw model evidence are not engineering
        # diagnostics.  Search only failures whose status was explicitly
        # designated by the runtime/evaluator.
        full_text = structured_failure_text(result)

        for pat_info in RUNTIME_ENGINEERING_PATTERNS:
            count = len(re.findall(pat_info["regex"], full_text, re.IGNORECASE))
            if count > 0:
                key = pat_info["issue_type"]
                record(key, pat_info, occurrences=count)

    # Filter: Systemic problem = affects enough questions or occurs too many times in total. single
    # The failure of the provider request may be just a momentary network/server fluctuation; it is caused by the full-eval
    # Per-question retry processing, the entire round of completed evaluations must not be directly discarded.
    total = len(results)
    threshold = max(2, int(math.ceil(total * threshold_ratio)))

    detected = []
    for key, data in issue_counts.items():
        if data["affected_questions"] >= threshold or data["occurrences"] >= 5:
            detected.append({
                "issue_type": key,
                "severity": data["info"]["severity"],
                "description": data["info"]["description"],
                "fix_hint": data["info"]["fix_hint"],
                "affected_file": data["info"]["affected_file"],
                "total_occurrences": data["occurrences"],
                "affected_questions": data["affected_questions"],
                "total_questions": total,
            })

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    detected.sort(key=lambda x: severity_order.get(x["severity"], 99))
    return detected


def is_engineering_invalid_report(report: dict) -> bool:
    """Return True when a verification/full-eval report is an engineering/API invalid run."""
    if not isinstance(report, dict):
        return False
    if report.get("engineering_invalid") is True:
        return True
    if report.get("status") in ("engineering_abort", "preflight_failed"):
        return True
    if report.get("verdict") in ("engineering_blocked", "preflight_failed"):
        return True
    issues = (
        report.get("runtime_engineering_issues")
        or report.get("runtime_issues")
        or report.get("api_issues")
        or []
    )
    return bool(issues and any(
        (item.get("severity") in ("critical", "high"))
        or item.get("issue_type") in (
            "llm_api_connection_error",
            "api_auth_401",
            "api_connection_refused",
            "embedding_api_failure",
            "embedding_endpoint_mismatch",
        )
        for item in issues if isinstance(item, dict)
    ))


def _redact_sensitive(value):
    if isinstance(value, dict):
        return {k: _redact_sensitive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_sensitive(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-<redacted>", value)
    return value


def build_engineering_invalid_report(report: dict, source_path: str = "") -> dict:
    """Build a review-skip report for engineering/API invalid eval output."""
    issues = _redact_sensitive(
        report.get("runtime_engineering_issues")
        or report.get("runtime_issues")
        or report.get("api_issues")
        or []
    )
    issue_types = []
    for item in issues:
        if isinstance(item, dict) and item.get("issue_type") not in issue_types:
            issue_types.append(item.get("issue_type"))
    reason = _redact_sensitive(
        report.get("engineering_invalid_reason")
        or report.get("abort_reason")
        or report.get("verdict_reason")
        or "Engineering/API invalid evaluation result; do not review as evolution failure."
    )
    return {
        "engineering_invalid": True,
        "source_report_path": source_path,
        "reason": reason,
        "issue_types": issue_types,
        "issues": issues,
        "action": "delete_or_archive_invalid_eval_result_and_rerun_after_fix",
        "skip_post_evolution_review": True,
        "fix_guidance": {
            "llm_api_connection_error": "Network/API connection problem; do not count it as an evolution failure, delete the results of this evaluation and run again.",
            "api_auth_401": "Check whether the selected provider API key is expired and exported through environment variables.",
            "api_rate_limit": "Reduce concurrency or wait for the current limit to be restored before running again.",
            "api_timeout": "Reduce concurrency, increase timeout, or rerun; do not enter review diagnostics.",
            "embedding_endpoint_mismatch": "Confirm embedding endpoint and Chroma dimensions are consistent; rebuild index in isolated sandbox.",
            "embedding_api_failure": "Check the selected embedding profile key/endpoint and do not silently fall back.",
        },
    }


def write_engineering_invalid_artifacts(out_dir: str, report: dict,
                                        source_path: str = "") -> dict:
    """Write engineering invalid artifacts and return the generated report."""
    os.makedirs(out_dir, exist_ok=True)
    invalid = build_engineering_invalid_report(report, source_path=source_path)
    report_path = os.path.join(out_dir, "engineering_invalid_report.json")
    reason_path = os.path.join(out_dir, "failure_reason.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(invalid, f, ensure_ascii=False, indent=2)
    with open(reason_path, "w", encoding="utf-8") as f:
        f.write(invalid["reason"].strip() + "\n")
        f.write("Post-evolution Teacher review skipped; rerun evaluation after fixing API/engineering issue.\n")
    invalid["engineering_invalid_report_path"] = report_path
    invalid["failure_reason_path"] = reason_path
    return invalid


def apply_effectiveness_policy(report: dict,
                               *,
                               expected_total_questions: int = 0,
                               result_rows: list | None = None,
                               require_complete: bool = False,
                               scope: str = "") -> dict:
    """Normalize final effectiveness and invalid-run fields.

    Success is intentionally net-positive against the explicit reference:
    ``accuracy_delta > 0`` and at least one correction. Regressions are not a
    hard veto; they are retained for Teacher review and probe guards.
    Engineering/API failures or incomplete full-split evaluations are invalid
    runs, not negative evolution evidence.
    """
    if not isinstance(report, dict):
        report = {}
    expected = int(expected_total_questions or report.get("expected_total_questions") or 0)
    actual = int(report.get("total_questions") or 0)
    runtime_issues = detect_runtime_engineering_issues(result_rows or [])
    if runtime_issues:
        report["runtime_engineering_issues"] = runtime_issues

    missing_result_count = max(0, expected - actual) if expected else 0
    complete = True
    if require_complete and expected:
        complete = actual == expected and not missing_result_count
    report["evaluation_complete"] = complete
    report["expected_total_questions"] = expected or report.get("expected_total_questions", actual)
    report["missing_result_count"] = missing_result_count
    if scope:
        report["evaluation_scope"] = scope

    comparison_valid = bool(report.get("comparison_valid", True))
    delta = int(report.get("accuracy_delta", 0) or 0)
    corrections = len(report.get("corrections", []) or [])
    regressions = len(report.get("regressions", []) or [])

    engineering_invalid = is_engineering_invalid_report(report)
    invalid_reasons = []
    if engineering_invalid:
        invalid_reasons.append("runtime_engineering_or_api_issue")
    if require_complete and not complete:
        engineering_invalid = True
        invalid_reasons.append("incomplete_full_evaluation")
    if not comparison_valid:
        invalid_reasons.append("comparison_invalid")

    passed = (
        comparison_valid
        and not engineering_invalid
        and (not require_complete or complete)
        and delta > 0
        and corrections >= 1
    )
    if engineering_invalid:
        status = "engineering_invalid"
    elif not comparison_valid:
        status = "comparison_invalid"
    elif passed and regressions:
        status = "effective_with_regression_risk"
    elif passed:
        status = "effective"
    else:
        status = "not_effective"

    report["engineering_invalid"] = bool(engineering_invalid)
    if invalid_reasons:
        report["engineering_invalid_reason"] = ", ".join(dict.fromkeys(invalid_reasons))
    report["accepted"] = passed
    report["full_success"] = passed if scope.startswith("full") or require_complete else False
    report["reference_eligible"] = not engineering_invalid and comparison_valid
    report["status"] = status
    report["final_success_criteria"] = {
        "accuracy_delta_gt_0": delta > 0,
        "corrections_ge_1": corrections >= 1,
        "regressions_eq_0": regressions == 0,
        "regressions_allowed": True,
        "net_positive_delta_policy": True,
        "comparison_valid": comparison_valid,
        "evaluation_complete": complete,
        "engineering_invalid": bool(engineering_invalid),
        "expected_total_questions": expected,
        "actual_total_questions": actual,
        "passed": passed,
    }
    return report


# =====================================================================
# Full-eval structure prebuild
# =====================================================================

def prebuild_isolated_video_structures(workspace_dir, video_ids, combo,
                                       sandbox_dir, custom_configs=None) -> list[dict]:
    """Build every full-video isolated structure before any question answer.

    This creates a real phase boundary for formal full eval.  Each video uses
    the structuring module's bounded clip-level VLM worker pool, but no
    ``agent.run`` call is made until all requested libraries are materialized.
    """
    _ensure_runtime_path_for_workspace(workspace_dir)
    from agent import MetaVideoAgent
    from run_video_qa import VideoQAEnv, build_video_structure

    reports = []
    unique_video_ids = sorted({str(item or "") for item in video_ids if str(item or "")})
    for index, video_id in enumerate(unique_video_ids, start=1):
        started = time.time()
        try:
            env = VideoQAEnv(workspace_dir=workspace_dir, video_id=video_id)
            effective_configs = dict(custom_configs or {})
            effective_configs["agent"] = dict(effective_configs.get("agent", {}))
            effective_configs["agent"]["sandbox_dir"] = sandbox_dir
            agent = MetaVideoAgent(env, combo, custom_configs=effective_configs)
            # ``structure_workers`` is an explicit full-eval run setting.  A
            # generated structuring class may provide a conservative default
            # in its constructor, but it must not silently override this
            # runner-owned concurrency policy after the Agent has assembled.
            struct_name = str((combo or {}).get("video_structuring") or "")
            requested_physics = (
                (custom_configs or {}).get(struct_name, {}).get("physics", {})
                if isinstance((custom_configs or {}).get(struct_name, {}), dict) else {}
            )
            if isinstance(requested_physics, dict) and requested_physics:
                agent.structuring.config.setdefault("physics", {}).update(requested_physics)
                print(
                    "   ⚙️ [Structure] Apply full-eval runner physics:"
                    f"parallel_workers={agent.structuring.config['physics'].get('parallel_workers')}"
                )
            _redirect_structuring_to_sandbox(
                agent.structuring,
                sandbox_dir=sandbox_dir,
                reset=True,
                copy_existing=False,
            )
            print(f"   [{index}/{len(unique_video_ids)}] Prebuilding full-video structure store: {video_id}")
            build_video_structure(env, agent.structuring)
            _rebuild_sandbox_chroma_if_requested(agent.structuring)
            artifact = str(getattr(agent.structuring, "db_path", "") or "")
            if not os.path.isfile(artifact) or os.path.getsize(artifact) <= 0:
                raise RuntimeError("prebuild completed without a usable structure JSONL")
            reports.append({
                "video_id": video_id,
                "status": "ok",
                "structure_path": artifact,
                "elapsed_sec": round(time.time() - started, 3),
            })
        except Exception as exc:
            reports.append({
                "video_id": video_id,
                "status": "error",
                "error": str(exc),
                "elapsed_sec": round(time.time() - started, 3),
            })
            raise RuntimeError(
                f"full-eval structure prebuild failed for {video_id}: {exc}"
            ) from exc
    return reports


# =====================================================================
# Phase 2: Comprehensive testing
# =====================================================================

def evaluate_combo(workspace_dir, video_id, combo, combo_id, questions, sandbox_dir,
                   custom_configs=None, on_question_done=None, force_rebuild=False,
                   isolate_structure=False, concurrency: int = 1,
                   reference_structure_dir: str = "",
                   runtime_time_reference_only: bool = False):
    """    Run the Agent sandbox evaluation question by question.

    Args:
        on_question_done: optional callback (question_index, result, all_results_so_far) -> bool
                          Abort subsequent questions when returning False (used for runtime engineering problem monitoring)
        force_rebuild: Force rebuild the structure library (required when evolving modules)"""
    _ensure_runtime_path_for_workspace(workspace_dir)
    from agent import MetaVideoAgent
    from run_video_qa import VideoQAEnv, build_video_structure

    def _is_retryable_result(result: dict) -> bool:
        text = f"{result.get('answer', '')}\n{result.get('error', '')}"
        retry_markers = (
            "ERROR:",
            "readonly database",
            "database is locked",
            "temporarily unavailable",
            "Connection error",
            "Read timed out",
            "429",
            "rate limit",
        )
        return any(marker.lower() in text.lower() for marker in retry_markers)

    def _run_indexed(i, task):
        print(f"   [{combo_id}] question{i+1}/{len(questions)}: {task.get('question', '?')[:50]}...")
        max_attempts = max(1, int(os.environ.get("SANDBOX_QUESTION_MAX_RETRIES", "3") or "3"))
        # A strict scoped smoke must never retry by changing the structure
        # rebuild mode; a failed single trajectory is the engineering result.
        if runtime_time_reference_only:
            max_attempts = 1
        per_question_rebuild = (
            os.environ.get("SANDBOX_PER_QUESTION_STRUCTURE_REBUILD", "").lower()
            in ("1", "true", "yes")
        )
        # Exact time-reference structure builds use process-level builder
        # settings; each question therefore owns an isolated artifact.
        per_question_rebuild = per_question_rebuild or bool(
            runtime_time_reference_only and force_rebuild and isolate_structure
        )
        effective_sandbox_dir = sandbox_dir
        if isolate_structure and (per_question_rebuild or runtime_time_reference_only):
            raw_task_id = task.get("task_id") or make_task_id(
                task.get("video_id", video_id),
                task.get("time_reference", ""),
                task.get("question", ""),
            )
            safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_task_id))[:120]
            effective_sandbox_dir = os.path.join(
                sandbox_dir,
                "question_runs",
                f"{i+1:03d}_{safe_task_id}",
            )
        result = None
        for attempt in range(max_attempts):
            result = run_single_question(MetaVideoAgent, VideoQAEnv, build_video_structure,
                                         workspace_dir, video_id, combo, task, effective_sandbox_dir,
                                         custom_configs=custom_configs,
                                         force_rebuild=(
                                             force_rebuild
                                             and (per_question_rebuild or i == 0)
                                             and attempt == 0
                                         ),
                                         isolate_structure=isolate_structure,
                                         reference_structure_dir=reference_structure_dir,
                                         runtime_time_reference_only=runtime_time_reference_only)
            if not _is_retryable_result(result):
                break
            if attempt < max_attempts - 1:
                wait_s = min(20, 2 ** attempt)
                print(
                    f"      ⚠️ [{combo_id}] title{i+1} A retryable engineering error occurred,"
                    f"Prepare to try again{attempt + 2}/{max_attempts}: {str(result.get('answer') or result.get('error'))[:120]}"
                )
                time.sleep(wait_s)
        result["combo_id"] = combo_id
        result["combo"] = combo
        result["_question_index"] = i
        result["retry_attempts"] = max(0, attempt)
        return result

    def _append_result(result):
        os.makedirs(sandbox_dir, exist_ok=True)
        result_file = os.path.join(sandbox_dir, f"{video_id}_sandbox.jsonl")
        clean = dict(result)
        clean.pop("_question_index", None)
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    results = []
    concurrency = max(1, int(concurrency or 1))
    merged_window_env_backup = None
    if (
        force_rebuild
        and isolate_structure
        and os.environ.get("SANDBOX_MERGE_TIME_REF_WINDOWS", "").lower() in ("1", "true", "yes")
        and not runtime_time_reference_only
    ):
        padding = float(os.environ.get("SANDBOX_STRUCTURE_TIME_REF_PADDING", "20") or 20)
        merged_windows = _merged_time_reference_windows(questions, padding=padding)
        if merged_windows:
            merged_window_env_backup = {
                "STRUCTURE_BUILD_WINDOWS_JSON": os.environ.get("STRUCTURE_BUILD_WINDOWS_JSON"),
                "STRUCTURE_BUILD_START_SEC": os.environ.get("STRUCTURE_BUILD_START_SEC"),
                "STRUCTURE_BUILD_END_SEC": os.environ.get("STRUCTURE_BUILD_END_SEC"),
            }
            os.environ["STRUCTURE_BUILD_WINDOWS_JSON"] = json.dumps(merged_windows)
            os.environ.pop("STRUCTURE_BUILD_START_SEC", None)
            os.environ.pop("STRUCTURE_BUILD_END_SEC", None)
            preview = ", ".join(f"{s}-{e}s" for s, e in merged_windows[:6])
            suffix = " ..." if len(merged_windows) > 6 else ""
            print(f"   [{combo_id}] Merged probe build windows: {preview}{suffix}")

    def _restore_merged_window_env():
        nonlocal merged_window_env_backup
        if merged_window_env_backup is None:
            return
        for key, value in merged_window_env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        merged_window_env_backup = None

    if on_question_done and concurrency > 1:
        # The callback is compatible with the concurrent path below: completed
        # futures are audited in order of completion and pending work is
        # cancelled on a stop decision. Do not silently discard an explicit
        # probe/full-eval concurrency budget just because monitoring is on.
        print(
            f"   [{combo_id}] Runtime monitoring enabled; reserved{concurrency} concurrency,"
            "Cancel futures that have not yet started when aborting (requests already in flight will complete naturally)"
        )
    if (
        force_rebuild
        and isolate_structure
        and os.environ.get("SANDBOX_PER_QUESTION_STRUCTURE_REBUILD", "").lower()
        in ("1", "true", "yes")
        and concurrency > 1
    ):
        print(
            f"   ⚠️ [{combo_id}] Explicitly enabled process-level topic-by-topic structure reconstruction will cause problems."
            "Force serial execution to avoid incorrect results"
        )
        concurrency = 1
    if concurrency <= 1 or len(questions) <= 1:
        for i, task in enumerate(questions):
            result = _run_indexed(i, task)
            results.append(result)
            _append_result(result)

            # Runtime monitoring callbacks
            if on_question_done:
                clean_results = [
                    {k: v for k, v in item.items() if k != "_question_index"}
                    for item in results
                ]
                should_continue = on_question_done(
                    i,
                    {k: v for k, v in result.items() if k != "_question_index"},
                    clean_results,
                )
                if not should_continue:
                    print(f"\n[Runtime Monitor] Abort verification (completed{i+1}/{len(questions)} question)")
                    break
        results.sort(key=lambda r: r.get("_question_index", 9999))
        for item in results:
            item.pop("_question_index", None)
        _restore_merged_window_env()
        return results

    print(f"   ⚙️ [{combo_id}] Sandbox evaluation: {concurrency} workers, {len(questions)} question(s)")
    start_index = 0
    if isolate_structure or force_rebuild:
        print(f"   [{combo_id}] Preheat the isolation structure library serially first, and then concurrently issue the remaining questions.")
        first = _run_indexed(0, questions[0])
        results.append(first)
        _append_result(first)
        if on_question_done:
            should_continue = on_question_done(
                0,
                {k: v for k, v in first.items() if k != "_question_index"},
                [{k: v for k, v in first.items() if k != "_question_index"}],
            )
            if not should_continue:
                first.pop("_question_index", None)
                _restore_merged_window_env()
                return [first]
        start_index = 1

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_run_indexed, i, task): i
            for i, task in enumerate(questions[start_index:], start=start_index)
        }
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            results.append(result)
            _append_result(result)
            print(f"      ✓ [{len(results)}/{len(questions)}] {result.get('time_reference', '?')}")

            # Runtime monitoring callbacks
            if on_question_done:
                clean_results = [
                    {k: v for k, v in item.items() if k != "_question_index"}
                    for item in results
                ]
                should_continue = on_question_done(
                    i,
                    {k: v for k, v in result.items() if k != "_question_index"},
                    clean_results,
                )
                if not should_continue:
                    for pending in futures:
                        pending.cancel()
                    print(f"\n[Runtime Monitor] Abort verification (completed{len(results)}/{len(questions)} question)")
                    break

    results.sort(key=lambda r: r.get("_question_index", 9999))
    for item in results:
        item.pop("_question_index", None)
    _restore_merged_window_env()
    return results


# =====================================================================
# Sandbox main function
# =====================================================================
