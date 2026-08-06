# MetaVideoAgent Five-Module Protocol

This document defines the responsibilities and data-flow contract of the five
generated modules. It does not prescribe an algorithm or default policy. Every
bundle must provide `video_structuring`, `localization`, `perception`, `memory`,
and `thinking` implementations.

The exact Python signatures, default arguments, return shapes, and minimal call
examples in `generated_runtime_abi_witness.json` are authoritative. If this
document and that witness differ, follow the witness.

## Data flow

```text
video or bounded media window
  -> video_structuring -> time-bounded records and media references
  -> thinking -> localization request -> localization result
  -> direct structure evidence or perception request -> perception result
  -> memory records complete events -> thinking -> next request or final answer
```

Thinking may request localization and perception over multiple rounds. Memory
manages the context of one question; it is not a substitute for the structure
store. Persist the request, normalized result, capability events, consumption
events, and final decision in the trajectory. Smoke repair, probe feedback, and
review all depend on this complete trajectory.

## Shared time, status, and provider rules

- Every media or structure interval uses numeric `start_sec` and `end_sec`, with
  `end_sec > start_sec`.
- During a probe, every interval must intersect the active media window. Full
  evaluation exposes the complete video window.
- A tool result has status `ok`, `unavailable`, or `error`. Do not represent an
  empty result as `ok`. Provider and asset failures must use `error` with a
  useful description.
- Model calls must use registered runtime adapters and capability profiles.
  Generated modules must not create provider clients or raw HTTP requests.
- A media reference is not observation evidence by itself. Preserve its time
  interval, provenance, and availability status.

## 1. video_structuring

Responsibility: persist execution-layer-prepared frames, audio, subtitles,
model output, or media references as retrievable time-bounded records. This
module does not answer questions and must not call a model again from
`addStructure`.

Required ABI:

```python
__init__(workspace_dir, video_id, struct_type="", custom_config=None,
         requires_vector_db=False)
addStructure(**record)
retrieveStructure(**kwargs) -> str
_get_api_params(...)
```

These methods may be inherited from `VideoStructuringBase`; an implementation
only needs to override the behavior required by its generated strategy.

`retrieveStructure` accepts `query`, `top_k`, `start_sec`, and `end_sec` as
keyword arguments and returns search evidence as text. An override must retain
that call shape and return type.

The generic builder supplies records with `metavideoagent_structure_build_request`
semantics:

```json
{
  "start_sec": 12.0,
  "end_sec": 20.0,
  "multimodal_narration": "raw model text or JSON",
  "asr_text": "observed transcript text only",
  "asr_status": "ok | no_speech | not_requested | asset_unavailable | api_error",
  "media_assets": [
    {
      "kind": "frame | audio_segment | subtitle_span",
      "start_sec": 12.0,
      "end_sec": 20.0,
      "path_or_ref": "...",
      "status": "available"
    }
  ],
  "builder_request": {
    "window": {"start_sec": 12.0, "end_sec": 20.0},
    "requested_modalities": ["frames"]
  }
}
```

Keep lists and objects as native JSON values. Preserve the original time window
and `multimodal_narration` when overriding `addStructure` so later retrieval can
consume the same evidence.

## 2. localization

Responsibility: use the structure store and the question context to return
candidate time windows. Return direct structure evidence when sufficient;
otherwise emit explicit perception requests. Localization never returns the
final answer.

Construction ABI: `__init__(env, struct_db=None, custom_config=None)`. Public
tool methods use the `tool_<name>` convention.

Tool output must use `self.make_localization_result(...)` or an equivalent
envelope:

```json
{
  "kind": "localization_result",
  "status": "ok",
  "producer_module": "localization",
  "tool_name": "...",
  "time_ranges": [{"start_sec": 12.0, "end_sec": 20.0}],
  "structure_records": [],
  "perception_requests": [
    {
      "time_range": {"start_sec": 12.0, "end_sec": 20.0},
      "modality": "visual | audio | ocr",
      "purpose": "..."
    }
  ],
  "evidence": {},
  "error": ""
}
```

An `ok` localization result must contain at least one valid time range. Limit
structure reads to the intersection of requested and active windows. Treat
`asr_status=not_supplied` as missing audio evidence, not as an audio conclusion.

## 3. perception

Responsibility: observe only the windows requested by localization. Perception
may use bounded frame, audio, OCR, VLM, or registered external-capability
adapters and must preserve raw responses and parsed claims.

Construction ABI: `__init__(env, struct_db=None, custom_config=None)`. Public
tool methods use `tool_<name>`. Parse requested ranges with
`self._parse_time_ranges` so they remain inside the active media window.

Return `self.make_perception_result(...)` or an equivalent envelope:

```json
{
  "kind": "perception_result",
  "status": "ok",
  "producer_module": "perception",
  "tool_name": "...",
  "time_ranges": [{"start_sec": 12.0, "end_sec": 20.0}],
  "evidence": {"raw_output": "...", "claims": [], "channels": []},
  "error": ""
}
```

An `ok` result must include non-empty raw output, claims, or channels. Otherwise
return `unavailable`.

## 4. memory

Responsibility: retain reasoning, plans, tool requests, normalized results,
capability event IDs, consumption events, and the final answer. Compression is
allowed, but the original trajectory must remain available.

Required ABI:

```text
set_combo, _log_planning, _log_reasoning, update, get_context,
get_context_packet, export_trajectory, save_session
```

`get_context_packet()` returns a structured object containing
`protocol_version`, `kind="memory_context"`, `recent_events`, and
`latest_module_results`. New thinking modules should consume the structured
memory available through `get_runtime_context()["memory"]`.

## 5. thinking

Responsibility: decide whether to request more evidence or return an answer.
Thinking receives the question, video metadata, memory context, and prior module
results. It must not read the workspace, JSONL files, or raw video directly.

Stable call ABI:

```python
__call__(question, video_context, work_context, force_finish=False)
# returns exactly (thought: str, action: "act" | "finish", payload: list | dict)
```

The runtime calls `set_runtime_context(thinking_context)` first. Use
`get_runtime_context()` for lossless structured state. Construct `act` payloads
with `localization_request(...)` or `perception_request(...)`. Construct a
`finish` payload with `finish_payload(...)` or an equivalent object containing
at least `answer`; it may also include `answer_type`, `answer_confidence`, and
`evidence_summary`.

## Candidate verification

Dynamic preflight compiles, injects, and instantiates all five modules through
the real module map. It does not judge answer correctness. Real smoke then
verifies producer-to-consumer handoffs on one execution trajectory. The ABI
witness remains the sole authority for exact signatures and minimal calls.
