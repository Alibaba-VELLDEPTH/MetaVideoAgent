"""Portable source and resource files carried by one five-module agent."""

from __future__ import annotations

import atexit
import base64
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

MODULE_TYPES = ("video_structuring", "localization", "perception", "memory", "thinking")


def module_records(bundle: dict) -> dict:
    modules = bundle.get("modules") or {}
    if isinstance(modules, dict):
        return modules
    return {item["module_type"]: item for item in modules}


def relative_source_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Agent source path must be relative: {value!r}")
    return str(path)


def bundle_files(bundle: dict) -> dict:
    """Module code is authoritative; support files contain only other files."""
    files = {relative_source_path(path): content for path, content in (bundle.get("support_files") or {}).items()}
    for record in module_records(bundle).values():
        path = relative_source_path(str(record["source_file"]))
        if path in files and files[path] != record["code"]:
            raise ValueError(f"Conflicting source for {path}")
        files[path] = record["code"]
    return files


def write_source_files(root: str | Path, files: dict) -> None:
    for relative, content in files.items():
        path = Path(root) / relative_source_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, dict) and content.get("encoding") == "base64":
            path.write_bytes(base64.b64decode(content["content"], validate=True))
        else:
            path.write_text(content, encoding="utf-8")


def read_source_file(path: Path):
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"encoding": "base64", "content": base64.b64encode(raw).decode("ascii")}


def capture_support_files(root: str | Path, module_paths: list[str]) -> dict:
    """Agent-owned helpers/resources live in agent/, separate from the SDK."""
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): read_source_file(path)
        for path in sorted((root / "agent").rglob("*"))
        if path.is_file() and not path.is_symlink()
        and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
        and path.relative_to(root).as_posix() not in module_paths
    }


def entrypoint_manifest(bundle: dict) -> dict:
    return {
        module: {"source_file": record["source_file"], "name": record["name"],
                 "resources": record.get("resources") or []}
        for module, record in module_records(bundle).items()
    }


def module_resource_inputs(bundle: dict, module: str) -> dict:
    """Files and constructor settings declared as inputs to one module.

    Each module lists the full helper/resource closure it uses. Shared files
    appear in every consuming module's list; agent-controller settings belong
    to thinking. Dependency installation metadata is not a module input.
    """
    record = module_records(bundle).get(module) or {}
    files = bundle_files(bundle)
    configs = bundle.get("custom_configs") or {}
    return {
        "resources": {path: files[path] for path in record.get("resources") or []},
        "config": configs.get(record.get("name")) or {},
        "agent_config": (configs.get("agent") or {}) if module == "thinking" else {},
    }


def write_agent_manifest(root: str | Path, bundle: dict) -> None:
    metadata = {key: bundle[key] for key in (
        "artifact_type", "schema_version", "mode", "design_answers", "observed_channel_summary",
        "deep_research_summary", "implementation_notes",
    ) if key in bundle}
    Path(root, "agent.json").write_text(json.dumps({
        **metadata,
        "modules": entrypoint_manifest(bundle),
        "dependencies": bundle.get("dependencies") or [],
        "custom_configs": bundle.get("custom_configs") or {},
    }, indent=2), encoding="utf-8")


def collect_agent_source(root: str | Path) -> dict:
    """Read explicit entrypoints and whole files from the current Agent only."""
    root = Path(root)
    try:
        try:
            manifest = json.loads((root / "agent.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read agent.json: {exc}") from exc
        if not isinstance(manifest, dict) or not isinstance(manifest.get("modules"), dict):
            raise ValueError("agent.json must contain the five explicit module entrypoints")
        records = {}
        for module in MODULE_TYPES:
            entry = manifest["modules"].get(module)
            if not isinstance(entry, dict) or not all(isinstance(entry.get(key), str) and entry[key]
                                                      for key in ("name", "source_file")):
                raise ValueError(f"agent.json.modules.{module} requires source_file and name")
            relative = relative_source_path(entry["source_file"])
            try:
                code = (root / relative).read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"Selected module source cannot be read: {relative}: {exc}") from exc
            resources = entry.get("resources", [])
            if not isinstance(resources, list) or not all(isinstance(item, str) for item in resources):
                raise ValueError(f"{module}.resources must list relative paths")
            records[module] = {"module_type": module, "name": entry["name"], "source_file": relative,
                               "code": code, "resources": sorted({relative_source_path(item) for item in resources})}
        result = {**manifest, "modules": records,
                  "combo": {module: record["name"] for module, record in records.items()},
                  "support_files": capture_support_files(root, [item["source_file"] for item in records.values()])}
        notes = root / "implementation.md"
        if notes.is_file():
            result["implementation_notes"] = notes.read_text(encoding="utf-8")
        files = bundle_files(result)
        for module, record in records.items():
            for path in record["resources"]:
                if path not in files:
                    raise ValueError(f"{module} declares missing resource: {path}")
        return result
    except (OSError, ValueError, UnicodeError) as exc:
        raise RuntimeError(f"Agent source assembly failed: {exc}") from exc


def copy_agent_source(source: str | Path, target: str | Path, known_files=()) -> None:
    """Copy the exact current draft, preserving missing files and invalid manifests.

    The SDK and execution data are not Agent source. The caller owns the target
    Agent directory; a missing source manifest remains missing for preflight.
    """
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        return
    target.mkdir(parents=True, exist_ok=True)
    paths = set(known_files)
    for root in (source, target):
        try:
            manifest = json.loads((root / "agent.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries = manifest.get("modules", {}) if isinstance(manifest, dict) else {}
        for entry in entries.values() if isinstance(entries, dict) else ():
            if isinstance(entry, dict) and isinstance(entry.get("source_file"), str):
                paths.add(entry["source_file"])
    if (target / "agent").is_dir():
        shutil.rmtree(target / "agent")
    if (source / "agent").is_dir():
        shutil.copytree(source / "agent", target / "agent",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    for relative in paths:
        try:
            relative = relative_source_path(relative)
        except (ValueError, TypeError):
            continue
        if relative.startswith("agent/"):
            continue
        destination = target / relative
        if (source / relative).is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
        else:
            destination.unlink(missing_ok=True)
    for name in ("agent.json", "implementation.md"):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
        else:
            (target / name).unlink(missing_ok=True)


def materialize_agent(bundle: dict, root: str | Path) -> None:
    """Seed a new iteration from its explicitly selected current-best bundle."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    write_source_files(root, bundle_files(bundle))
    write_agent_manifest(root, bundle)
    (root / "implementation.md").write_text(bundle.get("implementation_notes") or "", encoding="utf-8")


class AgentSourcePackage:
    """Give each candidate its own Python package and real resource paths.

    Package-relative helper imports remain isolated across reference/candidate
    execution, including imports inside methods and importlib.resources reads.
    The package lives for the process lifetime, like the injected classes.
    """

    def __init__(self, bundle: dict):
        directory = tempfile.TemporaryDirectory(prefix="metavideoagent_source_")
        atexit.register(directory.cleanup)
        self.root = Path(directory.name)
        self.name = "_mva_" + self.root.name
        write_source_files(self.root, bundle_files(bundle))
        self._namespace(self.name, self.root)

    @staticmethod
    def _namespace(name: str, path: Path) -> None:
        if name not in sys.modules:
            initializer = path / "__init__.py"
            if not initializer.exists():
                initializer.write_text("", encoding="utf-8")
            spec = importlib.util.spec_from_file_location(name, initializer)
            package = importlib.util.module_from_spec(spec)
            sys.modules[name] = package
            spec.loader.exec_module(package)

    def module_globals(self, relative: str, defaults: dict) -> dict:
        relative = relative_source_path(relative)
        parts = Path(relative).with_suffix("").parts
        for index in range(1, len(parts)):
            self._namespace(self.name + "." + ".".join(parts[:index]), self.root.joinpath(*parts[:index]))
        name = self.name + "." + ".".join(parts)
        existing = sys.modules.get(name)
        if existing is not None:
            return existing.__dict__
        spec = importlib.util.spec_from_file_location(name, self.root / relative)
        module = importlib.util.module_from_spec(spec)
        module.__dict__.update(defaults)
        sys.modules[name] = module
        return module.__dict__
