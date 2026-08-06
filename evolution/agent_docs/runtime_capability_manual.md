# Runtime Capability Manual

This is the prompt-safe operational manual for generated MetaVideoAgent
modules. The source of truth is
`evolution/agent_docs/capabilities/runtime_capability_profiles.json`.
Each profile binds one capability to a provider/model/transport contract. It
contains no API keys.

## Profile Rule

Diagnosis, Deep Research, and Codex may select only a documented `profile_id`.
They must never write provider SDK calls, raw HTTP, base URLs, API keys,
provider clients, subprocess commands, bare `ffmpeg`, or model
names not represented by a profile. Runtime availability is verified by actual
structured capability events, not by this manual.

The apparent OpenAI-compatible surface is not one universal payload. Text chat,
vision chat, audio chat/transcription, and multimodal embeddings have different
input/output formats. `runtime_evidence` selects the registered adapter from
the profile and enforces those formats.

## Active Media Window Contract

Normal smoke/full evaluation exposes the complete raw video. Targeted probe
exposes only the exact `time_reference` interval(s). All raw-media adapters
must obey this boundary.

- Read `env.get_active_time_windows()` only through a runtime adapter.
- VLM/OCR image tools receive assets derived from `active_time_window_frame`.
- ASR receives one bounded extracted segment per
  `active_time_window_segment`.
- Never bypass the boundary by reading the raw video, JSONL, Chroma, OpenCV,
  subprocess, or `utils` helpers directly.

## Registered Runtime Adapters

Generated code may import only:

```python
import runtime_evidence
```

Use these adapters with a documented `profile_id` when a different profile is
actually needed. Omitting `profile_id` preserves the current default profile.

- LLM: `runtime_evidence.call_text_llm(messages, profile_id=..., temperature=..., max_tokens=...)`
- VLM: `runtime_evidence.inspect_active_frames(env, capability="vlm", prompt=..., max_windows=..., frames_per_window=..., profile_id=...)`
- ASR: `runtime_evidence.transcribe_active_windows(env, max_windows=..., max_seconds_per_window=..., profile_id=...)`
- OCR: `runtime_evidence.inspect_active_frames(env, capability="ocr", prompt=..., max_windows=..., frames_per_window=..., profile_id=...)`
- Embedding: `runtime_evidence.embed_text(input_text, profile_id=...)` or
  `runtime_evidence.embed_image(image_path, profile_id=...)`

Every adapter records profile ID, resolved provider/model, transport, input
provenance, and status. If an adapter is unavailable or errors, return the
structured fallback; never fabricate transcripts or visual observations.

### Localized media calls

`time_ranges` or `candidate_windows` returned by localization are executable
media selectors, not descriptive metadata. A perception tool that receives
such ranges must use the range-scoped adapters below rather than calling an
active-window adapter with the unmodified environment:

- Localized VLM/OCR: `runtime_evidence.inspect_time_ranges(env, time_ranges, capability="vlm"|"ocr", prompt=..., max_windows=..., frames_per_window=..., profile_id=...)`
- Localized ASR: `runtime_evidence.transcribe_time_ranges(env, time_ranges, max_windows=..., max_seconds_per_window=..., profile_id=...)`

They normalize and intersect requested ranges with the active-media contract,
call only the resulting intervals, return `requested_time_ranges` and
`actual_time_ranges`, and preserve normal capability events. If there is no
intersection, they return structured `unavailable`; do not silently fall back
to sampling the full active video.

## Paper-Reproduction Profile Families

- `paper.llm.evolution`: bounded text reasoning with `enable_thinking=false`.
- `paper.vlm.default`: default visual-evidence route in the reproduction profile.
- `paper.asr.default`: bounded segment ASR.
- `paper.ocr.default`: bounded frame OCR.
- `paper.embedding.default`: shared text/image embedding space.

The paper-reproduction review configuration uses the same
`paper.vlm.default` visual-evidence profile plus
`paper.asr.default`. Teacher tools sample frames and transcribe
audio separately for every selected `time_reference` window; they may repeat
the tool call on narrower subranges but never inspect an out-of-contract
range.

The specific model/transport/input/output/cost/source fields are in the JSON
manual. A profile's existence does not make it mandatory or configured.

## Output and Fallback Rules

- Preserve raw model output, parsed evidence, profile provenance, and explicit
  no-speech/no-text/unavailable/error states.
- ASR segment bounds are the default temporal provenance. Do not claim word
  timestamps unless a registered profile explicitly returns and validates them.
- Keep all frame/window/call counts bounded.
- A module that emits confidence or conflict must derive them from channel
  agreement and failures; never use constants.
- A selected capability must be observed through its runtime event before smoke
  or probe can pass.
