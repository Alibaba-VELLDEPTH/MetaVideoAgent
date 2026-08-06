# Initial Baseline Bundle Prompt

You are the initial baseline designer for MetaVideoAgent.

Your task is not to improve an existing baseline. Your task is to build the
first executable five-module video QA combo for a hidden MetaVideoAgent training
distribution.

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
- A fixed historical baseline combo as the design target.

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

Return an `initial_baseline_bundle` JSON object with:

- `artifact_type`: exactly `metavideoagent_bundle`.
- `schema_version`: `1`.
- `mode`: exactly `initial_baseline`.
- `combo`: mapping the five canonical execution keys to generated class names:
  `video_structuring`, `localization`, `perception`, `memory`, and `thinking`.
  Do not emit aliases such as `struct` or `work_mem`.
- `modules`: exactly five module records. Every record must contain
  `module_type`, `name`, `source_file`, and `code`. `name` must be the exact
  generated Python class name and must equal the corresponding value in
  `combo`; do not emit alternate class-name fields or schema aliases.
- `design_answers`: one answer per module.
- `observed_channel_summary`: short summary from the label-blind profile.
- `deep_research_summary`: required field; its value may be an empty string when no
  research summary is supplied.

Exact constructor signatures, inherited helpers, return shapes, and minimal
runtime call witnesses are defined only by the generated
`generated_runtime_abi_witness.json` and `module_protocol.py` supplied to
Codex. Do not copy or infer an alternate ABI from this design document.
For each generated module, use the witness's exact `base_import` statement;
do not infer a `*_base` module name from the base class name.

The host supplies an `initial_execution_policy` artifact. It is binding and,
together with the ABI witness, overrides research suggestions, the plan, and
optional capability inventories whenever they disagree on runtime behavior.

## Execution-Owned Build Lifecycle and Capability Pins

The execution-layer structure builder owns bounded video sampling and all
build-time ASR/VLM calls. The generated `video_structuring` class only
persists, indexes, and retrieves the records passed to `addStructure`; it must
not call a model from `addStructure`.

Every initial model call must use the profile selected by the supplied
`initial_execution_policy` and capability inventory. Do not embed paper profile
IDs or provider model names in generated code.

Unregistered external/local visual routes are disabled. Use only the registered
runtime adapters; do not add an external wrapper.

Each module must satisfy these responsibilities:

- `video_structuring`: persist and retrieve time-bounded evidence through the
  witness-defined base interface.
- `video_structuring` JSONL artifacts must keep structured fields native:
  `subject_registry` as a JSON object/dict and list-like fields such as
  `visual_entities`, `screen_text`, `spatial_relations`,
  `discriminative_keywords`, and `uncertainties` as JSON arrays/lists. Only
  serialize these fields with `json.dumps` when writing Chroma metadata; do not
  store stringified JSON in JSONL rows.
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
- If one tool retrieves candidate windows, timestamps, clips, evidence cards,
  or other grounding data, the thinking module must pass that data to the next
  verification/perception step. Do not hard-code empty values such as
  `candidate_windows: []`.
- Use `export_vars` with clear variable names, structured tool outputs, or
  explicit parsing inside the thinking module so observations are not dropped
  between modules.
- Do not repeat the same verification call after it has returned successful
  evidence. If evidence is missing, retrieve narrower evidence or change
  strategy; if it is sufficient or `force_finish=True`, synthesize and finish.
- The final answer must be grounded in a previous observation and must not say
  it is an unsupported guess, generic fallback, or failure message.
- The final answer must be a concrete task answer, not a description of the
  retrieval process, a provider error, or an uncertainty/failure sentinel. This
  is an observable output contract, not a prescription for the thinking
  algorithm: the bundle may choose its own bounded control flow and evidence
  combination approach. Preserve limitations in structured metadata rather than
  replacing the answer with a fallback token.
- A successful smoke run is one where the agent returns a specific parseable
  answer string, even if answer accuracy is not
  evaluated at smoke time.
- The thinking module may choose its own evidence-combination and answer
  construction mechanism. It must only respect the runtime's generic output
  envelope and requested response form; do not implement a fixed task taxonomy,
  a fixed option parser, or prompt-example-specific answer logic.

Do not submit a baseline that merely instantiates shell classes. The runtime
only provides neutral infrastructure bases such as `VideoStructuringBase`,
`LocalizationBase`, `PerceptionBase`, `WorkMemoryBase`, and `ThinkingBase`;
these bases contain interface plumbing, persistence, parsing, sampling, and
dispatch helpers, but no project-approved strategy. You may inherit these bases
and import existing runtime utilities, API clients, parsers, or low-level
helpers for interface compatibility, but each generated module must implement a
distribution-specific behavior derived from the observed profile and research.
Explain the new mechanism in `thought`/`design_answers`.

## Output Rules

- Write only the canonical JSON object to `requested_output_path`. Your terminal
  completion message may be a short plain-text status; do not paste the bundle or
  wrap it in Markdown.
- Do not hard-code examples from the prompt.
- Keep code self-contained and import only existing execution runtime modules.
