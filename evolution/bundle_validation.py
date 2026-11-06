"""Bundle assembly validation and real-runtime smoke execution."""

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

def bundle_smoke_test(bundle: dict, *, workspace_dir: str = "",
                      distribution_manifest: str = "", question_split: str = "train",
                      reference_results: str = "", reference_report: str = "",
                      failure_contract: dict | None = None) -> dict:
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
    needs = (bundle.get("implementation_spec") or {}).get("runtime_capabilities") or []
    profiles = {item["capability"]: item["profile_id"] for item in needs
                if isinstance(item, dict) and item.get("capability") and item.get("profile_id")}
    cmd.extend(["--expected-capabilities", ",".join(sorted(profiles)),
                "--expected-profiles-json", json.dumps(profiles)])
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
            payload = json.loads(line.split("[SMOKE:RESULT] ", 1)[1])
            if isinstance(payload, dict):
                result = payload
        except (IndexError, json.JSONDecodeError):
            pass
        break
    events = result.get("capability_events") or []
    execution_layer = (
        result.get("failure_kind") == "execution_layer"
        or result.get("execution_layer_rerun_required") is True
        or any(
            isinstance(event, dict) and event.get("status") in {"api_error", "asset_error"}
            for event in events
        )
    )
    passed = result.get("passed") is True and proc.returncode == 0
    result.update({
        "passed": passed,
        "failure_kind": "" if passed else "execution_layer" if execution_layer else "candidate_contract",
        "execution_layer_rerun_required": not passed and execution_layer,
        "logs": "\n".join(line for line in logs.splitlines() if "[SMOKE:RESULT] " not in line)[-10000:],
    })
    if not passed:
        result["error"] = str(result.get("error") or f"bundle smoke runner exited {proc.returncode}")
    return result


