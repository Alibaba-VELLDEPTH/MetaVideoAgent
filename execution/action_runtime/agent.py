import inspect

try:
    import utils
    from module_map import (
        AGENT_CONFIG,
        LOCALIZATION_MAP,
        PERCEPTION_MAP,
        STRUCTURING_MAP,
        THINKING_MAP,
        WORK_MEMORY_MAP,
    )
    from module_protocol import (
        PROTOCOL_VERSION,
        extract_export_values,
        make_thinking_context,
        make_tool_request,
        normalize_thinking_return,
        normalize_tool_result,
        render_result_for_context,
        result_is_usable,
    )
except ImportError:  # pragma: no cover - package import
    from . import utils
    from .module_map import (
        AGENT_CONFIG,
        LOCALIZATION_MAP,
        PERCEPTION_MAP,
        STRUCTURING_MAP,
        THINKING_MAP,
        WORK_MEMORY_MAP,
    )
    from .module_protocol import (
        PROTOCOL_VERSION,
        extract_export_values,
        make_thinking_context,
        make_tool_request,
        normalize_thinking_return,
        normalize_tool_result,
        render_result_for_context,
        result_is_usable,
    )

REQUIRED_COMBO_KEYS = ("video_structuring", "thinking", "memory", "localization", "perception")
RUNTIME_LOG_REDACT_KEYS = {
    "answer",
    "gt_answer",
    "ground_truth",
    "correct_answer",
    "label",
    "gold",
    "target",
    "time_reference",
    "question_time_reference",
    "evidence_interval",
    "evidence_intervals",
    "evidence_window",
    "evidence_windows",
    "question_type",
    "type",
    "category",
    "sub_category",
    "human_type",
    "human_category",
}


def _merge_cost_ledgers(structure_build: dict, answering: dict) -> dict:
    """Return one total ledger with auditable structure and answering splits."""
    build = dict(structure_build or {})
    answer = dict(answering or {})
    build_paths = set(build.pop("frame_asset_paths", []) or [])
    answer_paths = set(answer.pop("frame_asset_paths", []) or [])
    keys = (
        "vision_frame_inputs", "llm_calls", "vlm_calls", "ocr_calls", "asr_calls",
        "embedding_calls", "api_calls_with_usage", "api_calls_without_usage",
        "prompt_tokens", "completion_tokens", "total_tokens", "audio_seconds_submitted",
        "latency_sec",
    )
    merged = {key: sum(float(item.get(key, 0) or 0) for item in (build, answer)) for key in keys}
    for key in ("vision_frame_inputs", "llm_calls", "vlm_calls", "ocr_calls", "asr_calls",
                "embedding_calls", "api_calls_with_usage", "api_calls_without_usage",
                "prompt_tokens", "completion_tokens", "total_tokens"):
        merged[key] = int(merged[key])
    merged["frames_viewed"] = merged["vision_frame_inputs"]
    merged["unique_frame_assets"] = len(build_paths | answer_paths)
    merged["audio_seconds_submitted"] = round(merged["audio_seconds_submitted"], 4)
    merged["latency_sec"] = round(merged["latency_sec"], 3)
    ledger = []
    for phase, payload in (("structure_build", build), ("answering", answer)):
        for item in payload.get("api_call_ledger", []) or []:
            record = dict(item)
            record["phase"] = str(record.get("phase") or phase)
            record["call_index"] = len(ledger) + 1
            ledger.append(record)
    merged["api_successful_calls"] = len(ledger)
    merged["api_call_ledger"] = ledger
    merged["artifact_type"] = "metavideoagent_api_cost_ledger"
    merged["schema_version"] = 1
    merged["accounting_scope"] = "structure_build_plus_answering"
    merged["phase_summaries"] = {"structure_build": build, "answering": answer}
    return merged


def _require_combo_key(combo: dict, key: str, registry: dict) -> str:
    value = (combo or {}).get(key)
    if not value:
        raise RuntimeError(
            f"MetaVideoAgent combo missing required module slot {key!r}; "
            "refusing to use fixed default fallback."
        )
    if value not in registry:
        raise RuntimeError(
            f"MetaVideoAgent combo slot {key!r} references unregistered module {value!r}. "
            f"Available registered keys: {sorted(registry.keys())}"
        )
    return value


def _format_runtime_question(question: str, extra_info: dict = None) -> str:
    """Expose answer choices to the thinking module without leaking labels.

    Dataset choices are part of the user-facing QA task. They may arrive as a
    list like ["winds", "sunny"] or already-labelled text like
    "A. winds". The original question is still persisted unchanged elsewhere;
    this augmented string is only the runtime prompt seen by generated thinking
    modules.
    """
    if not isinstance(extra_info, dict):
        return question
    choices = extra_info.get("choices")
    if not choices:
        return question
    if isinstance(choices, str):
        raw_choices = [c.strip() for c in choices.replace("\r", "\n").split("\n") if c.strip()]
    elif isinstance(choices, (list, tuple)):
        raw_choices = [str(c).strip() for c in choices if str(c).strip()]
    else:
        return question
    if not raw_choices:
        return question

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for idx, choice in enumerate(raw_choices):
        # Preserve explicit labels such as "A. winds", "B) sunny", or
        # "C: rain"; otherwise add stable labels for generated code to parse.
        stripped = choice.strip()
        # A leading article in an answer (for example "A monkey riding a
        # horse") is not an option label.  Only preserve an *explicit*
        # label delimiter; otherwise every raw answer beginning with "A " is
        # silently made unlabelled and a later reasoning module cannot select
        # it by its runtime-visible option identity.
        if len(stripped) >= 2 and stripped[0].upper() in labels and stripped[1:2] in (".", ")", ":", "："):
            lines.append(stripped)
        else:
            label = labels[idx] if idx < len(labels) else str(idx + 1)
            lines.append(f"{label}. {stripped}")
    return question.rstrip() + "\n\nChoices:\n" + "\n".join(lines)


def _strip_runtime_choices_block(text: str) -> str:
    """Remove the runtime-only Choices block before evidence tools see it."""
    value = str(text or "")
    marker = "\n\nChoices:\n"
    if marker in value:
        return value.split(marker, 1)[0].rstrip()
    return value


def _sanitize_tool_params_for_evidence(params):
    """Keep answer-format options out of localization/perception observations.

    Generated thinking modules receive choices through ``runtime_question`` so
    they can format answers. Evidence tools should search/verify against the
    semantic question only; otherwise observations echo option text and weak
    generated answer extractors can mistake choices for visual evidence.
    """
    if isinstance(params, dict):
        clean = {}
        for key, value in params.items():
            if key in {"query", "question"} and isinstance(value, str):
                clean[key] = _strip_runtime_choices_block(value)
            else:
                clean[key] = _sanitize_tool_params_for_evidence(value)
        return clean
    if isinstance(params, list):
        return [_sanitize_tool_params_for_evidence(item) for item in params]
    return params


def _output_contract_for_module(module: str) -> str:
    return {
        "video_structuring": "structured_retrieval_records",
        "localization": "candidate_windows_or_time_ranges",
        "perception": "bounded_evidence_object",
        "memory": "bounded_working_memory_context",
        "thinking": "runtime_action_or_finish",
    }.get(module, "")


def _redact_runtime_log_meta(value):
    """Keep teacher-mode runtime logs from persisting answer/evidence labels."""
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            if str(key) in RUNTIME_LOG_REDACT_KEYS:
                continue
            safe[key] = _redact_runtime_log_meta(item)
        return safe
    if isinstance(value, list):
        return [_redact_runtime_log_meta(item) for item in value]
    return value




class MetaVideoAgent:
    """Coordinate the five generated modules for one MetaVideoAgent run."""
    def __init__(self, env, combo: dict, custom_configs: dict = None):
        self.env = env
        self.combo = combo
        self.video_id = env.video_id
        self.workspace_dir = env.workspace_dir
        self.config = AGENT_CONFIG.copy()
        missing = [key for key in REQUIRED_COMBO_KEYS if not combo.get(key)]
        if missing:
            raise RuntimeError(
                f"MetaVideoAgent combo must explicitly define all five module slots; missing {missing}"
            )

        if custom_configs and "agent" in custom_configs:
            self.config.update(custom_configs["agent"])

        print("[System] Assembling MetaVideoAgent...")
        print(f"   |> Combo: {combo}")

        def _cfg(module_key):
            if not custom_configs:
                return None
            return custom_configs.get(module_key)

        # 1. Persistent video structure
        struct_key = _require_combo_key(combo, 'video_structuring', STRUCTURING_MAP)
        struct_cls = STRUCTURING_MAP[struct_key]
        self.structuring = struct_cls(self.workspace_dir, self.video_id, custom_config=_cfg(struct_key))

        # Expose the active structure implementation to the remaining modules.
        self.env.struct_type = struct_key
        self.env.struct_db = self.structuring

        # 2. Thinking module
        thinking_key = _require_combo_key(combo, 'thinking', THINKING_MAP)
        thinking_cls = THINKING_MAP[thinking_key]
        self.thinker = thinking_cls(self.env, custom_config=_cfg(thinking_key))

        # 3. Localization and perception modules
        local_key = _require_combo_key(combo, 'localization', LOCALIZATION_MAP)
        local_cls = LOCALIZATION_MAP[local_key]
        self.localization = local_cls(self.env, struct_db=self.structuring, custom_config=_cfg(local_key))

        perc_key = _require_combo_key(combo, 'perception', PERCEPTION_MAP)
        perc_cls = PERCEPTION_MAP[perc_key]
        self.perception = perc_cls(self.env, struct_db=self.structuring, custom_config=_cfg(perc_key))

        self.work_memory = None

    def _extract_variables(self, module_result: dict, export_vars: dict) -> dict:
        """Export explicit protocol fields without a lossy second LLM call."""
        extracted = extract_export_values(module_result, export_vars)
        missing = [key for key in (export_vars or {}) if key not in extracted]
        if missing:
            print(f"   [Protocol Export] unavailable fields: {missing}")
        return extracted

    def _validate_and_cast_params(self, tool_name: str, params: dict) -> tuple:
        """Verify and convert parameter types"""
        func = getattr(self.localization, f"tool_{tool_name}", None) or getattr(self.perception, f"tool_{tool_name}", None)
        if not func:
            return False, params, f"Tool '{tool_name}' not found."

        sig = inspect.signature(func)
        casted_params = {}

        for k, v in params.items():
            if v is None:
                continue
            if k in sig.parameters:
                param_type = sig.parameters[k].annotation
                if param_type is float:
                    try:
                        casted_params[k] = float(str(v).replace('s', '').strip())
                    except (TypeError, ValueError):
                        return False, params, f"Parameter '{k}' expects float, got '{v}'."
                elif param_type is int:
                    try:
                        casted_params[k] = int(float(str(v).replace('s', '').strip()))
                    except (TypeError, ValueError):
                        return False, params, f"Parameter '{k}' expects int, got '{v}'."
                else:
                    casted_params[k] = v
            else:
                casted_params[k] = v

        return True, casted_params, ""

    def run(self, question: str, extra_info: dict = None) -> tuple:
        print(f"\n[Task] New question: {question}")
        runtime_question = _format_runtime_question(question, extra_info)

        # Cost statistics start
        utils.cost_tracker.reset()
        utils.cost_tracker.start_timer()

        self.task_variables = {}

        work_mem_key = _require_combo_key(self.combo, 'memory', WORK_MEMORY_MAP)
        work_mem_cls = WORK_MEMORY_MAP[work_mem_key]
        self.work_memory = work_mem_cls(self.workspace_dir, self.video_id, question)
        self.work_memory.set_combo(self.combo)

        step = 0
        final_answer = "Task failed to find answer within step limits."
        max_steps = self.config.get("max_steps", 10)
        pending_output_tools = []
        recent_module_results = []

        while step < max_steps:
            step += 1
            is_last_step = (step >= max_steps)
            print(f"\n[Loop {step}/{max_steps}] Thinker reasoning...")

            # Provide both the readable memory context and the lossless protocol
            # packet exposed through ThinkingBase.
            work_context = self.work_memory.get_context()
            memory_packet_getter = getattr(self.work_memory, "get_context_packet", None)
            memory_packet = (
                memory_packet_getter() if callable(memory_packet_getter) else {
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "memory_context",
                    "recent_events": list(getattr(self.work_memory, "raw_history", [])[-30:]),
                }
            )
            if work_context != "No previous steps.":
                try:
                    import runtime_evidence
                    runtime_evidence.mark_output_consumed(
                        consumer="thinking", producer_module="memory",
                        output_protocol="memory_context",
                        output_fields=list(memory_packet.keys()) if isinstance(memory_packet, dict) else [],
                    )
                except Exception:
                    pass
            if pending_output_tools:
                try:
                    import runtime_evidence
                    for pending in pending_output_tools:
                        runtime_evidence.mark_output_consumed(
                            consumer="thinking",
                            tool_name=pending["tool_name"],
                            producer_module=pending["producer_module"],
                            output_protocol=pending["output_protocol"],
                            contract_valid=pending["output_contract_valid"],
                            source_event_ids=pending.get("evidence_event_ids", []),
                        )
                except Exception:
                    pass
                pending_output_tools = []
            video_context = f"Video length: {self.env.video_length_secs}s; structure module: {self.combo.get('video_structuring')}."
            thinking_context = make_thinking_context(
                runtime_question=runtime_question,
                video_context={
                    "video_id": self.video_id,
                    "duration_sec": float(getattr(self.env, "video_length_secs", 0.0) or 0.0),
                    "structure_type": str(self.combo.get("video_structuring") or ""),
                },
                memory_context=memory_packet,
                recent_results=recent_module_results[-12:],
            )
            try:
                import runtime_evidence
                runtime_evidence.record_runtime_trace(
                    "thinking.context", status="ok", payload=thinking_context,
                )
            except Exception:
                pass
            context_setter = getattr(self.thinker, "set_runtime_context", None)
            if callable(context_setter):
                context_setter(thinking_context)

            # Unified thinking: one call determines act or finish
            thinking_return = self.thinker(
                runtime_question, video_context, work_context, force_finish=is_last_step
            )
            thought, action, payload, thinking_error = normalize_thinking_return(thinking_return)
            try:
                import runtime_evidence
                runtime_evidence.record_runtime_trace(
                    "thinking.output", status="invalid" if thinking_error else "ok",
                    payload={"thought": thought, "action": action, "payload": payload,
                             "force_finish": is_last_step},
                    result={"error": thinking_error},
                )
            except Exception:
                pass
            if thinking_error:
                print(f"   [Thinking Contract] {thinking_error}")
                self.work_memory._log_reasoning(
                    thought, "invalid", {"error": thinking_error, "raw_type": type(thinking_return).__name__}
                )
                continue

            # Recording thought processes into working memory
            self.work_memory._log_reasoning(thought, action,
                                            payload if isinstance(payload, dict) else {"plan": str(payload)[:200]})

            if action == "finish":
                answer = payload.get("answer", "") if isinstance(payload, dict) else str(payload)
                final_answer = answer
                provenance_logger = getattr(self.work_memory, "_log_decision_provenance", None)
                if callable(provenance_logger):
                    provenance_logger(final_answer, payload if isinstance(payload, dict) else {})
                print(f"   [Success] Answer: {final_answer}")
                break

            # action == "act" → execution uses the base normalizer so the
            # generated module can return either a list or its declared plan
            # shape without the runtime silently discarding it.
            normalizer = getattr(self.thinker, "normalize_plan", None)
            plan = normalizer(payload) if callable(normalizer) else payload
            plan = plan if isinstance(plan, list) else []
            if not plan:
                print("   [Warning] Thinker returned empty plan, retrying...")
                continue

            # Recording plans into working memory
            self.work_memory._log_planning(plan, "react")

            for i, subtask in enumerate(plan):
                if not isinstance(subtask, dict):
                    print(f"   [Warning] Subtask {i+1} is not a dict ({type(subtask).__name__}), skipping.")
                    continue
                tool_name = subtask.get('target_tool', '')
                params = dict(subtask.get('tool_params', {}) or {})
                instruction = subtask.get('instruction', 'General observation')
                export_vars = subtask.get('export_vars', {})

                print(f"   Execute subtask {i + 1}/{len(plan)}: {tool_name} | {instruction}")

                obs = ""
                circuit_break = False
                execution_status = "ok"
                error_code = ""
                producer_module = ""
                extracted = {}
                casted_params = {}
                module_result = {}
                tool_request = {}
                evidence_event_ids = []
                active_time_windows = []
                try:
                    import runtime_evidence
                    evidence_start = len(runtime_evidence.get_evidence_events())
                    getter = getattr(self.env, "get_active_time_windows", None)
                    active_time_windows = list(getter() if callable(getter) else [])
                except Exception:
                    evidence_start = 0

                # Variable dependency injection
                dependency_missing = False
                for k, v in list(params.items()):
                    if isinstance(v, str) and '${' in v:
                        var_name = v.strip('${}')
                        if var_name in self.task_variables and self.task_variables[var_name] is not None:
                            params[k] = self.task_variables[var_name]
                            print(f"   [Var Injected] {k} = {params[k]}")
                        else:
                            dependency_missing = True
                            obs = f"Execution Error: Missing dependency variable '{var_name}'."
                            execution_status = "invalid_input"
                            error_code = "missing_dependency"
                            break

                if dependency_missing:
                    print("   [Circuit Break] Dependency missing, abort plan.")
                    circuit_break = True
                else:
                    params = _sanitize_tool_params_for_evidence(params)
                    is_valid, casted_params, err_msg = self._validate_and_cast_params(tool_name, params)
                    if not is_valid:
                        obs = f"Execution Error: {err_msg}"
                        circuit_break = True
                        execution_status = "invalid_input"
                        error_code = "invalid_tool_params"
                    else:
                        if hasattr(self.localization, f"tool_{tool_name}"):
                            producer_module = "localization"
                            casted_params['instruction'] = instruction
                            tool_request = make_tool_request(
                                producer_module, tool_name, casted_params, instruction,
                            )
                            obs, _, _ = self.localization.execute(tool_name, casted_params)
                        elif hasattr(self.perception, f"tool_{tool_name}"):
                            producer_module = "perception"
                            casted_params['instruction'] = instruction
                            tool_request = make_tool_request(
                                producer_module, tool_name, casted_params, instruction,
                            )
                            obs, _, _ = self.perception.execute(tool_name, casted_params)
                        else:
                            obs = f"Execution Error: Tool '{tool_name}' not found."
                            circuit_break = True
                            execution_status = "tool_not_found"
                            error_code = "tool_not_found"

                    # Every tool observation crosses the same envelope.  Keep
                    # the raw provider/module payload in ``evidence`` and do
                    # deterministic field exports only; a second LLM here
                    # used to create lossy, unauditable handoffs.
                    if producer_module:
                        module_result = normalize_tool_result(
                            obs,
                            producer_module=producer_module,
                            tool_name=tool_name,
                            request=tool_request,
                        )
                    if export_vars and not circuit_break and module_result.get("status") == "ok":
                        print(f"   [Extracting] {list(export_vars.keys())}...")
                        extracted = self._extract_variables(module_result, export_vars)
                        valid_extracted = {k: v for k, v in extracted.items() if v is not None}
                        self.task_variables.update(valid_extracted)

                # Update working memory
                # ``obs`` can contain VLM/OCR/ASR evidence verbatim.  A
                # lecture about MATLAB, programming, or debugging may quite
                # legitimately contain the literal text ``Error:``; that is
                # evidence, not a runtime failure.  Only the normalized
                # module envelope below is authoritative for execution state.
                if execution_status == "ok" and module_result.get("status") == "error":
                    execution_status = "module_error"
                    error_code = str(module_result.get("error") or "module_error")[:160]
                output_protocol = _output_contract_for_module(producer_module)
                output_contract_valid = bool(
                    execution_status == "ok" and module_result and result_is_usable(module_result)
                )
                observation_for_memory = (
                    render_result_for_context(module_result)
                    if module_result else str(obs)
                )
                try:
                    evidence_event_ids = [
                        str(item.get("evidence_event_id") or "")
                        for item in runtime_evidence.get_evidence_events()[evidence_start:]
                        if item.get("evidence_event_id")
                    ]
                except Exception:
                    evidence_event_ids = []
                self.work_memory.update(
                    thought=instruction,
                    action=tool_name,
                    observation=observation_for_memory,
                    execution_status=execution_status,
                    error_code=error_code,
                    producer_module=producer_module,
                    output_protocol=output_protocol,
                    output_contract_valid=output_contract_valid,
                    evidence_event_ids=evidence_event_ids,
                    active_time_windows=active_time_windows,
                    tool_params_summary={
                        str(key): str(value)[:160]
                        for key, value in (casted_params or params).items()
                    },
                    tool_request=tool_request,
                    module_result=module_result,
                )
                # A tool result is consumed by the work-memory before the
                # next thinking turn.  Record that concrete handoff so the
                # execution trace distinguishes an implemented memory path
                # from a merely declared bundle contract.
                if producer_module and module_result and output_contract_valid:
                    try:
                        import runtime_evidence
                        memory_fields = set(module_result.keys())
                        evidence_payload = module_result.get("evidence")
                        if isinstance(evidence_payload, dict):
                            memory_fields.update(evidence_payload.keys())
                        runtime_evidence.mark_output_consumed(
                            consumer="memory",
                            tool_name=tool_name,
                            producer_module=producer_module,
                            output_protocol=output_protocol,
                            contract_valid=output_contract_valid,
                            source_event_ids=evidence_event_ids,
                            output_fields=sorted(memory_fields),
                        )
                    except Exception:
                        pass
                if module_result:
                    recent_module_results.append(module_result)
                    try:
                        import runtime_evidence
                        runtime_evidence.record_runtime_trace(
                            "module.result", status=str(module_result.get("status") or ""),
                            payload={"producer_module": producer_module, "tool_name": tool_name,
                                     "request": tool_request}, result=module_result,
                        )
                    except Exception:
                        pass
                if not circuit_break and output_contract_valid:
                    pending_output_tools.append({
                        "tool_name": str(tool_name),
                        "producer_module": producer_module,
                        "output_protocol": output_protocol,
                        "output_contract_valid": output_contract_valid,
                        "evidence_event_ids": evidence_event_ids,
                    })

                # Fuse check
                if circuit_break or ("Error:" in str(obs) and "Not Found" not in str(obs) and "Cannot find" not in str(obs)):
                    print("   [Plan Aborted] Circuit break triggered.")
                    break

        # The mission is over and the experience is accumulated
        self.work_memory.save_session(final_answer, step)
        trajectory = self.work_memory.export_trajectory()

        # End of cost statistics
        latency = utils.cost_tracker.stop_timer()
        answering_cost = utils.cost_tracker.summary()
        answering_cost["latency_sec"] = round(latency, 1)
        structure_cost_getter = getattr(self.structuring, "structure_cost_summary", None)
        structure_build_cost = structure_cost_getter() if callable(structure_cost_getter) else {}
        cost_summary = _merge_cost_ledgers(structure_build_cost, answering_cost)
        # Sandbox evaluation consumes this exact per-task ledger, so later
        # audits never need to infer provider usage from trajectory text.
        self.last_cost_summary = cost_summary
        utils.cost_tracker.print_summary()

        return final_answer, trajectory
