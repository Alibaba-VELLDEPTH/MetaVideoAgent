# Initial Baseline Bundle Prompt

You are the initial baseline designer for MetaVideoAgent.

Build the first executable five-module video QA agent for the target training
distribution, using its five-frame overview records and associated questions
and answer options.

## Inputs You May Use

- Prompt-safe distribution manifest.
- Per-video distribution records containing five uniformly sampled overview
  frames, non-answer video metadata, and associated questions/answer options.
- The run-specific initial combo plan, including its derived channel/design
  direction when supplied by the host.
- Optional deep research summary about long-video QA / multimodal agents.

## Inputs You Must Not Use

- Benchmark-provided `type`, `category`, `sub_category`, `question_type`,
  `domain`, or any other human semantic label.
- Any answer label, evidence window, clue interval, question id, task id, or
  benchmark task metadata beyond the associated question/option text included
  in the distribution profile.
- Any baseline/current-best trajectory, Teacher review, diagnosis, structure
  database text, or retrieval result from an execution run.
- Hard-coded video ids, question ids, answer keys, or time references.

The first baseline must be designed from the host-supplied five-frame,
query-aware distribution profile and deep research. Answer and evidence
supervision becomes available only after this baseline has been executed and
the Teacher review and DiagnosisAgent analysis stages start the next evolution
round.

## Required Design Questions

Answer all five questions before writing code:

1. **video_structuring**: What structure should be built from raw video so the
   downstream agent can retrieve evidence efficiently for this distribution?
2. **localization**: How should the agent find candidate temporal windows?
3. **perception**: What should be inspected inside candidate windows, and when?
4. **memory**: What state should be preserved across tool calls and clips?
5. **thinking**: How should the agent plan, stop, verify, and map observations
   to the final answer?

## Code Contract

Write the executable Python and resource files under `agent/`. Write an
`agent.json` manifest at `requested_output_path` with:

- `artifact_type`: exactly `metavideoagent_bundle`.
- `schema_version`: `1`.
- `mode`: exactly `initial_baseline`.
- `modules`: an object with the five keys `video_structuring`, `localization`,
  `perception`, `memory`, and `thinking`. Each entry contains `name`,
  `source_file`, and `resources`. `name` selects the exact Python class defined
  in its source file. Use the canonical module names without aliases.
- `design_answers`: one answer per module.
- `observed_channel_summary`: short summary from the label-blind profile.
- `deep_research_summary`: required field; its value may be an empty string when no
  research summary is supplied.
- Store module source at `agent/<module>.py` and write complete source files.
  Put helpers, prompts and configuration in real files under `agent/` as well.
  The host collects these files into the self-contained execution bundle;
  do not serialize source or resource contents into the manifest. Use relative
  imports and resolve resources relative to `__file__`. Declare additional
  packages in `dependencies` and per-class settings in `custom_configs`.
- Each module record includes `resources`: the relative paths of every helper,
  prompt, configuration file and package initializer that its execution uses,
  including transitive imports. List shared resources in every consuming module.
  Use an empty list for a self-contained module. These declarations determine
  which modules an edit affects and whether video structures can be reused.
  Keep agent-controller settings in `custom_configs.agent` and module settings
  under their selected class names.

Use the constructor signatures, helpers and call examples supplied in
`generated_runtime_abi_witness.json`, and the return envelopes in
`module_protocol.py`. Import each shared base with its documented `base_import`.
The supplied `initial_execution_policy` records this run's default capability
profiles and build lifecycle.

## Build Lifecycle and Model Access

The execution-layer structure builder owns bounded video sampling and all
build-time ASR/VLM calls by default. The generated `video_structuring` class
persists, indexes, and retrieves the records passed to `addStructure`; it must
not call a model from `addStructure`.

For a custom build strategy, the class may implement `buildStructure(env)`.
The host calls it instead of the generic segment builder. Use registered
scoped-media adapters and persist canonical time-bounded records or media
references at `self.db_path`. Do not move perception or answer decisions into
the persistence method.

Every initial model call must use the profile selected by the supplied
`initial_execution_policy` and capability inventory. Do not embed paper profile
IDs or provider model names in generated code.

Use the registered runtime adapters for model and media operations.

Each module must satisfy these responsibilities:

- `video_structuring`: persist and retrieve time-bounded evidence through the
  witness-defined base interface.
- Keep structured JSONL fields as native objects and arrays, including
  `canonical_evidence`, `visual_entities`, `screen_text` and `media_assets`.
  Serialize nested values only where the selected index's metadata format
  requires it.
- `localization`: expose at least one dispatchable `tool_` method and return
  the protocol-defined localization envelope.
- `perception`: expose at least one dispatchable `tool_` method and return
  the protocol-defined perception envelope.
- `memory`: preserve trajectory/provenance and provide the witness-defined
  context methods.
- `thinking`: produce the protocol-defined tool plan or final-answer envelope.

## Mandatory Agent Dispatch ABI

`thinking.__call__(question, video_context, work_context, force_finish=False)`
must return exactly `(thought, action, payload)`. For tool use, return
`(thought, "act", [subtask, ...])`; for termination, return
`(thought, "finish", {"answer": "concrete answer", ...})`.

`force_finish=True` is a mandatory termination boundary: return `"finish"`
on that call with one concrete, parseable `payload.answer`. Do not issue a new
tool plan, repeat a successful observation, or substitute an uncertainty token
when this flag is true.

Each `act` subtask is a dict containing `subtask_id`, `instruction`,
`target_tool`, `tool_params`, and optional `export_vars`. `target_tool` is
always the **bare suffix** of a generated `tool_...` method. The Agent itself
searches localization and perception for `tool_{target_tool}`.

For example, a localization method `tool_retrieve_evidence` must be invoked as
`"target_tool": "retrieve_evidence"`, and a perception method
`tool_verify_ranges` as `"target_tool": "verify_ranges"`. These forms are
invalid: `"localization.retrieve_evidence"`,
`"perception.verify_ranges"`, and `"tool_retrieve_evidence"`. Do not place
the plan inside a JSON string or under an `action_input` wrapper.

## Runtime Smoke Requirement

The generated five modules must work together as one executable MetaVideoAgent, not
only pass import/interface checks. The smoke test will run one real question and
judge whether the agent completed a usable video-understanding loop. It will not
judge whether the final answer is correct, but it will reject engineering or
dataflow failures.

Your combo must satisfy these behavioral requirements:

- At least one localization/perception path must obtain substantive video,
  audio, OCR, transcript, or timestamped evidence from the video.
- Use retrieved structure evidence directly when it supports a decision. When
  additional observation is needed, pass the retrieved windows, timestamps,
  clips or evidence cards into the next perception step, preserving their
  actual values and temporal scope.
- Use `export_vars` with clear variable names, structured tool outputs, or
  explicit parsing inside the thinking module so observations are not dropped
  between modules.
- Do not repeat the same verification call after it has returned successful
  evidence. If evidence is missing, retrieve narrower evidence or change
  strategy; if it is sufficient or `force_finish=True`, synthesize and finish.
- Return a concrete, parseable task answer grounded in the acquired evidence.
  Preserve confidence and limitations in metadata. Smoke checks execution and
  answer form; full evaluation measures accuracy.
- Choose evidence-combination and answer-construction behavior for the observed
  distribution while preserving the generic output envelope and the task's
  requested response form.

The shared `VideoStructuringBase`, `LocalizationBase`, `PerceptionBase`,
`WorkMemoryBase` and `ThinkingBase` provide interface, persistence, parsing,
sampling and dispatch helpers. Build the distribution-specific strategy on
these interfaces and explain each module's mechanism in `design_answers`.

## Output Rules

- Write only the canonical JSON object to `requested_output_path`. Your terminal
  completion message may be a short plain-text status; do not paste the bundle or
  wrap it in Markdown.
- Do not hard-code examples from the prompt.
- Keep the complete agent self-contained, including its helper and prompt files;
  use the supplied runtime interfaces for model and media operations.
