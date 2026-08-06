"""Stable five-module ABI bases for dynamically injected MetaVideoAgent bundles.

Candidate bundles are data artifacts. They must never replace the runtime
modules that provide their base classes: doing so makes a normal import such
as ``from memory_modules import WorkMemoryBase`` import the candidate file
itself.  These neutral bases keep the execution ABI separate from generated
strategy code for smoke, repair, probe, and full evaluation alike.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

try:
    from evidence_contract import canonicalize_structure_input
    from module_protocol import make_module_result, normalize_time_ranges, render_result_for_context
except ImportError:  # pragma: no cover - package import
    from execution.action_runtime.evidence_contract import canonicalize_structure_input
    from execution.action_runtime.module_protocol import (
        make_module_result,
        normalize_time_ranges,
        render_result_for_context,
    )


class VideoStructuringBase:
    def __init__(self, workspace_dir, video_id, struct_type="", custom_config=None,
                 requires_vector_db=False):
        self.workspace_dir = str(workspace_dir or "")
        self.video_id = str(video_id or "")
        self.struct_type = str(struct_type or self.__class__.__name__)
        self.requires_vector_db = bool(requires_vector_db)
        physics = {"sample_interval": 10.0, "parallel_workers": 4,
                   "parallel_batch_delay": 0.0, "frames_per_clip": 4}
        api_tools = {}
        if isinstance(custom_config, dict):
            physics.update(custom_config.get("physics") or {})
            api_tools.update(custom_config.get("api_tools") or {})
        self.config = {"physics": physics, "api_tools": api_tools}
        root = os.path.join(self.workspace_dir, "video_structure")
        os.makedirs(root, exist_ok=True)
        self.db_path = os.path.join(root, "%s_%s.jsonl" % (self.video_id, self.struct_type))
        self._structure_cost_lock = threading.Lock()
        self._structure_cost_state = self._fresh_structure_cost_state()

    @staticmethod
    def _fresh_structure_cost_state():
        return {
            "vision_frame_inputs": 0, "frame_assets": set(),
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens_reported": 0,
            "llm_calls": 0, "vlm_calls": 0, "ocr_calls": 0, "asr_calls": 0,
            "embedding_calls": 0, "audio_seconds_submitted": 0.0,
            "api_calls_with_usage": 0, "api_calls_without_usage": 0,
            "api_call_ledger": [], "started_at": None, "latency_sec": 0.0,
        }

    def begin_structure_cost_capture(self):
        """Start a per-structure thread-safe provider ledger.

        Structure building precedes ``agent.run()``, whose answer ledger is
        intentionally reset.  This separate ledger makes those successful VLM
        and ASR calls auditable without changing execution behavior.
        """
        with self._structure_cost_lock:
            self._structure_cost_state = self._fresh_structure_cost_state()
            self._structure_cost_state["started_at"] = time.time()

    def finish_structure_cost_capture(self):
        with self._structure_cost_lock:
            started = self._structure_cost_state.get("started_at")
            if started is not None:
                self._structure_cost_state["latency_sec"] = max(0.0, time.time() - float(started))
                self._structure_cost_state["started_at"] = None

    def record_structure_api_call(self, response, *, call_type, profile=None,
                                  model_name="", frame_paths=None,
                                  audio_seconds=0.0, metadata=None):
        """Mirror one successful builder provider response into its own ledger."""
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        getter = usage.get if isinstance(usage, dict) else lambda key, default=0: getattr(usage, key, default)
        prompt = int(getter("prompt_tokens", 0) or 0) if usage else 0
        completion = int(getter("completion_tokens", 0) or 0) if usage else 0
        total = int(getter("total_tokens", 0) or 0) if usage else 0
        has_usage = bool(prompt or completion or total)
        paths = [str(path) for path in (frame_paths or []) if str(path)]
        details = dict(profile or {})
        meta = dict(metadata or {})
        with self._structure_cost_lock:
            state = self._structure_cost_state
            counter = f"{str(call_type or 'unknown')}_calls"
            state[counter] = int(state.get(counter, 0)) + 1
            if call_type in {"vlm", "ocr", "embedding"}:
                state["vision_frame_inputs"] += len(paths)
                state["frame_assets"].update(paths)
            if call_type == "asr":
                state["audio_seconds_submitted"] += max(0.0, float(audio_seconds or 0.0))
            state["prompt_tokens"] += prompt
            state["completion_tokens"] += completion
            state["total_tokens_reported"] += total
            state["api_calls_with_usage" if has_usage else "api_calls_without_usage"] += 1
            state["api_call_ledger"].append({
                "call_index": len(state["api_call_ledger"]) + 1,
                "phase": "structure_build", "capability": str(call_type or "unknown"),
                "profile_id": str(details.get("profile_id") or ""),
                "provider": str(details.get("provider") or ""),
                "model_id": str(model_name or details.get("model_id") or details.get("resolved_model_id") or ""),
                "prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": total, "usage_available": has_usage,
                "image_inputs_submitted": len(paths),
                "audio_seconds_submitted": round(max(0.0, float(audio_seconds or 0.0)), 4),
                "structure_window": list(meta.get("structure_window") or []),
            })

    def structure_cost_summary(self):
        with self._structure_cost_lock:
            state = self._structure_cost_state
            total_tokens = max(
                int(state.get("total_tokens_reported") or 0),
                int(state.get("prompt_tokens") or 0) + int(state.get("completion_tokens") or 0),
            )
            return {
                "accounting_scope": "structure_build",
                "frames_viewed": int(state["vision_frame_inputs"]),
                "vision_frame_inputs": int(state["vision_frame_inputs"]),
                "unique_frame_assets": len(state["frame_assets"]),
                "frame_asset_paths": sorted(state["frame_assets"]),
                "llm_calls": int(state.get("llm_calls") or 0),
                "vlm_calls": int(state.get("vlm_calls") or 0),
                "ocr_calls": int(state.get("ocr_calls") or 0),
                "asr_calls": int(state.get("asr_calls") or 0),
                "embedding_calls": int(state.get("embedding_calls") or 0),
                "api_successful_calls": len(state["api_call_ledger"]),
                "api_calls_with_usage": int(state["api_calls_with_usage"]),
                "api_calls_without_usage": int(state["api_calls_without_usage"]),
                "audio_seconds_submitted": round(float(state["audio_seconds_submitted"]), 4),
                "prompt_tokens": int(state["prompt_tokens"]),
                "completion_tokens": int(state["completion_tokens"]),
                "total_tokens": total_tokens,
                "latency_sec": round(float(state.get("latency_sec") or 0.0), 3),
                "api_call_ledger": [dict(item) for item in state["api_call_ledger"]],
            }

    def _get_api_params(self, tool_name):
        cfg = dict((self.config.get("api_tools") or {}).get(str(tool_name), {}) or {})
        if "active_model" in cfg and "model" not in cfg:
            cfg["model"] = cfg["active_model"]
        return cfg

    def addStructure(self, **kwargs):
        record = dict(kwargs or {})
        try:
            start, end = float(record.get("start_sec")), float(record.get("end_sec"))
        except (TypeError, ValueError):
            return
        if end <= start:
            return
        record["start_sec"], record["end_sec"] = start, end
        record = canonicalize_structure_input(record)
        with open(self.db_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def retrieveStructure(self, query="", start_sec=0.0, end_sec=None, top_k=5, **_kwargs):
        try:
            start = float(start_sec or 0.0)
            end = float("inf") if end_sec is None else float(end_sec)
        except (TypeError, ValueError):
            start, end = 0.0, float("inf")
        terms = {item.lower() for item in re.findall(r"[^\W_]+", str(query or ""), re.UNICODE) if len(item) > 2}
        rows = []
        try:
            with open(self.db_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        left, right = float(row.get("start_sec")), float(row.get("end_sec"))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
                    if right <= start or left >= end:
                        continue
                    document = str(row.get("retrieval_document") or row.get("asr_text") or row.get("multimodal_narration") or "")
                    score = sum(term in document.lower() for term in terms)
                    rows.append((score, left, right, document))
        except OSError:
            return ""
        rows.sort(key=lambda item: (-item[0], item[1]))
        try:
            limit = max(1, int(top_k))
        except (TypeError, ValueError):
            limit = 5
        return "\n".join(
            "%d. [%.3fs - %.3fs] score=%s:: %s" % (index, left, right, score, document)
            for index, (score, left, right, document) in enumerate(rows[:limit], 1)
        )


class _ToolModuleBase:
    producer_module = ""

    def __init__(self, env, struct_db=None, custom_config=None):
        self.env, self.struct_db = env, struct_db
        self.video_length_secs = float(getattr(env, "video_length_secs", 0.0) or 0.0)
        self.workspace_dir = str(getattr(env, "workspace_dir", "") or "")
        self.raw_video_path = str(getattr(env, "raw_video_path", "") or "")
        self.config = dict(custom_config or {})

    def execute(self, tool_name, params):
        func = getattr(self, "tool_" + str(tool_name), None)
        if not callable(func):
            return "Execution Error: Tool '%s' not found." % tool_name, "", {}
        try:
            result = func(**dict(params or {}))
            return json.dumps(result, ensure_ascii=False, default=str), "", {}
        except Exception as exc:
            return "Execution Error: %s" % exc, "", {}


class LocalizationBase(_ToolModuleBase):
    producer_module = "localization"

    def retrieve_structure(self, query="", **kwargs):
        if self.struct_db is None:
            return ""
        return self.struct_db.retrieveStructure(query=query, **kwargs)

    def make_localization_result(self, tool_name, *, status, time_ranges=None,
                                 structure_records=None, perception_requests=None,
                                 evidence=None, error="", request=None):
        return make_module_result(
            producer_module="localization", tool_name=tool_name, status=status,
            time_ranges=time_ranges, structure_records=structure_records,
            perception_requests=perception_requests, evidence=evidence,
            error=error, request=request,
        )


class PerceptionBase(_ToolModuleBase):
    producer_module = "perception"

    def _parse_time_ranges(self, value):
        return [[item["start_sec"], item["end_sec"]] for item in normalize_time_ranges(value)]

    def make_perception_result(self, tool_name, *, status, time_ranges=None,
                               evidence=None, error="", request=None):
        return make_module_result(
            producer_module="perception", tool_name=tool_name, status=status,
            time_ranges=time_ranges, evidence=evidence, error=error, request=request,
        )


class WorkMemoryBase:
    def __init__(self, workspace_dir, video_id, question, custom_config=None):
        self.workspace_dir, self.video_id, self.question = str(workspace_dir), str(video_id), str(question)
        self.config = dict(custom_config or {})
        self.combo, self.raw_history = {}, []

    def set_combo(self, combo):
        self.combo = dict(combo or {})

    def _log_reasoning(self, thought, decision, action_input):
        self.raw_history.append({"step_type": "reasoning", "timestamp": time.time(),
                                 "thought": str(thought or ""), "decision": str(decision or ""),
                                 "action_input": action_input})

    def _log_planning(self, plan, strategy_used=""):
        self.raw_history.append({"step_type": "planning", "timestamp": time.time(),
                                 "plan_output": plan, "strategy_used": str(strategy_used or "")})

    def _log_decision_provenance(self, final_answer, payload):
        producer_modules = sorted({
            str(item.get("producer_module") or "")
            for item in self.raw_history
            if item.get("step_type") == "tool_execution" and item.get("producer_module")
        })
        declared = [str(item) for item in (dict(payload or {}).get("decision_evidence_ids") or []) if str(item)]
        observed = {
            str(value) for item in self.raw_history if item.get("step_type") == "tool_execution"
            for value in (item.get("evidence_event_ids") or []) if str(value)
        }
        self.raw_history.append({
            "step_type": "decision_provenance", "timestamp": time.time(),
            "final_answer": str(final_answer or ""), "producer_modules": producer_modules,
            "declared_evidence_ids": declared,
            "declared_ids_known": not declared or set(declared).issubset(observed),
        })

    def update(self, *, thought, action, observation, execution_status="ok", error_code="",
               producer_module="", output_protocol="", output_contract_valid=None,
               evidence_event_ids=None, active_time_windows=None, tool_params_summary=None,
               tool_request=None, module_result=None):
        self.raw_history.append({
            "step_type": "tool_execution", "timestamp": time.time(), "thought": str(thought or ""),
            "action": str(action or ""), "observation": str(observation or ""),
            "execution_status": str(execution_status or ""), "error_code": str(error_code or ""),
            "producer_module": str(producer_module or ""), "output_protocol": str(output_protocol or ""),
            "output_contract_valid": output_contract_valid, "evidence_event_ids": list(evidence_event_ids or []),
            "active_time_windows": list(active_time_windows or []), "tool_params_summary": dict(tool_params_summary or {}),
            "tool_request": dict(tool_request or {}), "module_result": dict(module_result or {}),
        })

    def get_context(self):
        if not self.raw_history:
            return "No previous steps."
        return "\n".join(render_result_for_context(item.get("module_result") or {"observation": item.get("observation", "")}) for item in self.raw_history[-12:])

    def get_context_packet(self):
        return {"protocol_version": "metavideoagent_module_protocol", "kind": "memory_context",
                "recent_events": list(self.raw_history[-30:])}

    def export_trajectory(self):
        return list(self.raw_history)

    def save_session(self, final_answer, steps):
        return {"answer": str(final_answer or ""), "steps": int(steps or 0)}




class ThinkingBase:
    def __init__(self, env, custom_config=None):
        self.env, self.config, self._runtime_context = env, dict(custom_config or {}), {}

    def set_runtime_context(self, context):
        self._runtime_context = dict(context or {})

    def get_runtime_context(self):
        return dict(self._runtime_context)

    def get_recent_module_results(self):
        return list(self._runtime_context.get("recent_module_results") or [])

    def parse_runtime_options(self, question):
        text = str(question or "")
        matches = re.findall(r"(?:^|\n)\s*([A-Za-z])\s*[\).::]\s*([^\n]+)", text)
        return [{"label": label.upper(), "text": value.strip()} for label, value in matches]

    @staticmethod
    def extract_json_object(value):
        try:
            found = re.search(r"\{.*\}", str(value or ""), re.S)
            parsed = json.loads(found.group(0) if found else "{}")
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError, json.JSONDecodeError):
            return {}

    def finish_payload(self, answer, answer_type="unknown", confidence=0.0, evidence_summary=""):
        return {"answer": str(answer or ""), "answer_type": str(answer_type or "unknown"),
                "confidence": float(confidence or 0.0), "evidence_summary": str(evidence_summary or "")}

    def localization_request(self, tool_name, params, instruction="", export_vars=None):
        return {"subtask_id": "localization_%s" % tool_name, "target_tool": str(tool_name),
                "tool_params": dict(params or {}), "instruction": str(instruction or ""),
                "export_vars": dict(export_vars or {})}

    def perception_request(self, tool_name, params, instruction="", export_vars=None):
        return {"subtask_id": "perception_%s" % tool_name, "target_tool": str(tool_name),
                "tool_params": dict(params or {}), "instruction": str(instruction or ""),
                "export_vars": dict(export_vars or {})}
