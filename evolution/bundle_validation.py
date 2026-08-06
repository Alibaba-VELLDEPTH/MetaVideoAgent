"""Bundle validation, smoke execution, and source extraction helpers."""

import ast
import json
import os
import subprocess
import sys
import tempfile

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
METAVIDEOAGENT_PYTHON = os.environ.get("METAVIDEOAGENT_PYTHON") or sys.executable
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MODULE_SOURCE_FILES = {
    "video_structuring": ["execution/action_runtime/video_structuring_modules.py"],
    "perception": ["execution/action_runtime/perception_modules.py"],
    "localization": ["execution/action_runtime/localization_modules.py"],
    "thinking": ["execution/action_runtime/thinking_module.py"],
    "memory": ["execution/action_runtime/memory_modules.py"],
}

def bundle_smoke_test(bundle: dict, *, workspace_dir: str = "",
                      distribution_manifest: str = "", question_split: str = "train",
                      reference_results: str = "", reference_report: str = "",
                      reference_bundle: str = "", failure_contract: dict | None = None) -> dict:
    """Run one real Agent trajectory with all changed bundle classes injected.

    The smoke contract covers the producer-to-consumer chain within one
    complete five-module combination.
    """
    if not isinstance(bundle, dict) or not bundle.get("changed_modules"):
        return {"passed": False, "failure_kind": "candidate_contract",
                "error": "bundle smoke requires a candidate bundle with changed_modules"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False)
        bundle_path = handle.name
    cmd = [METAVIDEOAGENT_PYTHON, os.path.join(_CURRENT_DIR, "smoke_test_runner.py"),
           "--bundle-json", bundle_path, "--workspace", workspace_dir,
           "--question-split", question_split,
           "--failure-contract-json", json.dumps(failure_contract or {}, ensure_ascii=False)]
    if distribution_manifest:
        cmd.extend(["--distribution-manifest", distribution_manifest])
    if reference_results:
        cmd.extend(["--reference-results", reference_results])
    if reference_report:
        cmd.extend(["--reference-report", reference_report])
    if reference_bundle:
        cmd.extend(["--reference-bundle", reference_bundle])
    try:
        proc = subprocess.run(
            cmd, cwd=_PROJECT_ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=int(os.environ.get("CODEX_SMOKE_TIMEOUT_SEC", "300")),
            env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False,
        )
        logs = proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        logs = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return {"passed": False, "failure_kind": "candidate_contract",
                "error": "bundle smoke timed out", "logs": logs[-10000:]}
    finally:
        try:
            os.unlink(bundle_path)
        except OSError:
            pass
    result = {}
    for line in reversed(logs.splitlines()):
        if "[SMOKE:RESULT] " not in line:
            continue
        try:
            result = json.loads(line.split("[SMOKE:RESULT] ", 1)[1])
        except (IndexError, json.JSONDecodeError):
            pass
        break
    events = result.get("capability_events") or []
    execution_layer = any(
        isinstance(event, dict) and event.get("status") in {"api_error", "asset_error"}
        for event in events
    )
    if result.get("passed") and proc.returncode == 0:
        return {"passed": True, "failure_kind": "",
                "steps": result.get("steps", 0), "capability_events": events,
                "answer": result.get("answer", ""),
                "assembly_preflight": result.get("assembly_preflight") or {},
                "engineering_execution_trace": result.get("engineering_execution_trace") or {},
                "full_execution_trace_path": str(result.get("full_execution_trace_path") or ""),
                "logs": logs[-10000:]}
    return {
        "passed": False,
        "failure_kind": "execution_layer" if execution_layer else "candidate_contract",
        "execution_layer_rerun_required": execution_layer,
        "error": str(result.get("error") or f"bundle smoke runner exited {proc.returncode}"),
        "contract_issues": result.get("contract_issues", []),
        "answer": result.get("answer", ""),
        "assembly_preflight": result.get("assembly_preflight") or {},
        "capability_events": events,
        "engineering_execution_trace": result.get("engineering_execution_trace") or {},
        "full_execution_trace_path": str(result.get("full_execution_trace_path") or ""),
        "logs": logs[-10000:],
    }


def get_full_module_code(filepath: str, class_name: str) -> str:
    """Extract imports, globals, base classes, and the requested class."""
    if not os.path.exists(filepath):
        return ""

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""

    lines = content.split("\n")

    # Extract import statements
    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for i in range(node.lineno - 1, node.end_lineno if hasattr(node, 'end_lineno') else node.lineno):
                if i < len(lines):
                    imports.append(lines[i])

    # Extract module-level variable assignments (e.g. DEFAULT_STRUCTURING_CONFIG = {...}).
    # Do not extract the module registry (*_MAP), including the form ``MAP["Class"] = Class``
    # Subscript assignment. These statements reference other classes in the file or candidate classes that have not yet been defined; the candidate code will be
    # exec alone, while the dynamic injector registers the target class itself.
    global_vars = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            target_names = [
                target.id for target in node.targets
                if isinstance(target, ast.Name)
            ]
            map_subscript_assignment = any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id.endswith("_MAP")
                for target in node.targets
            )
            if any(name.endswith("_MAP") for name in target_names) or map_subscript_assignment:
                continue
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else node.lineno
            global_vars.append("\n".join(lines[start:end]))

    class_nodes = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }

    def _base_names(node):
        names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                names.append(base.id)
            elif isinstance(base, ast.Attribute):
                names.append(base.attr)
        return names

    def _collect_base_chain(name: str, seen=None) -> list:
        seen = seen or set()
        if name in seen:
            return []
        seen.add(name)
        node = class_nodes.get(name)
        if not node:
            return []
        chain = []
        for base_name in _base_names(node):
            chain.extend(_collect_base_chain(base_name, seen))
            if base_name in class_nodes:
                chain.append(base_name)
        return chain

    # Find the target class and its base classes
    target_class_code = ""
    base_class_codes = []

    target_node = class_nodes.get(class_name)
    if target_node:
        for base_name in _collect_base_chain(class_name):
            node = class_nodes.get(base_name)
            if not node:
                continue
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else len(lines)
            code = "\n".join(lines[start:end])
            if code and code not in base_class_codes:
                base_class_codes.append(code)

        start = target_node.lineno - 1
        end = target_node.end_lineno if hasattr(target_node, 'end_lineno') and target_node.end_lineno else len(lines)
        target_class_code = "\n".join(lines[start:end])

    # Assembling complete code
    parts = ["\n".join(imports)]
    if global_vars:
        parts.append("\n".join(global_vars))
    parts.extend(base_class_codes)
    if target_class_code:
        parts.append(target_class_code)

    return "\n\n".join(parts)
