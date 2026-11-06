# Map an existing agent into MetaVideoAgent

Produce an executable initial five-module MetaVideoAgent bundle by adapting the
agent in `original_agent/`. Preserve the source agent's algorithm and observable
behavior. This initialization maps capabilities; it does not introduce a new
evolution strategy. The resulting Agent0 enters the same smoke, training
evaluation, Teacher, Diagnosis, Research and automated evolution loop.

## Inspect the source execution first

Read the real entrypoint and follow the complete question-to-answer path through
helper files, loaded prompts, model bindings, tool schemas, memory and stopping
conditions. Record the original call sequence and explain where each operation
belongs in the five MVA responsibilities. Preserve original prompts and template
rendering, sampling parameters, tool names, native tool-call IDs, message order,
batch boundaries, budgets, state transitions and recovery behavior.

Map video preparation/storage to video_structuring, evidence location to
localization, evidence extraction to perception, retained state to memory, and
decision/control flow to thinking. A source agent without indexing does not
need a fabricated index, embedding request or additional VLM call. Its custom
`buildStructure(env)` can persist time-bounded media references. A module with no
corresponding source capability can be a small ABI adapter; explain its role
honestly rather than inventing capabilities to fill five slots.

## Adapt the integration, not the algorithm

Use the supplied MVA ABI witness, module protocol and runtime capability manual.
Keep original agent-owned helpers and prompts in the bundle. Rewrite import and
resource locations to the self-contained `agent/` package. Route model requests
through registered profiles with equivalent models and parameters; keys and
endpoints remain environment configuration, never source literals. If a required
transport or dependency cannot be represented, report that specific incompatibility
instead of silently changing the method or claiming equivalence.

For native function calling use `runtime_evidence.call_chat_llm`, preserving
the returned assistant message and native tool calls. One returned batch maps to
one thinking turn's act plan; each item may carry `native_tool_call` and
`native_assistant_message`. Read results from the next runtime context and rebuild
the original tool messages using their original IDs. Do not replace native calls
with an invented JSON-generation prompt. Use `custom_configs.agent.max_steps` to
retain the original round budget. Honor MVA's terminal `force_finish` boundary.

## Check and explain the mapping

With fixed or stubbed provider responses, compare the original and adapted call
order, prompts, parameter values, state transitions, tool arguments/results and
final answer. Keep the checks and any unresolved differences in
`implementation_notes`. Do not describe a runtime smoke pass as proof of semantic
equivalence. The host subsequently runs real compile/injection, smoke and train
evaluation through the common initialization pipeline.

## Output

Write the adapted executable files under `agent/` and their `agent.json`
manifest at `requested_output_path`: `artifact_type` is
`metavideoagent_bundle`, `schema_version` is 1, `mode` is `initial_baseline`,
and `modules` maps the five canonical module names to their exact class `name`,
`source_file` and `resources`. Use source locations `agent/<module>.py`. Write
all required helpers, prompts, configurations and binary resources as real files
under `agent/`. The host collects them into the self-contained execution bundle.
Do not put source strings in the manifest or leave runtime references to
`original_agent/` or the user's checkout.
Each module record also lists `resources`: the full relative helper/resource
closure used by that module, including transitive imports and package
initializers. Shared files belong in every consuming module's list; use `[]`
for modules with no additional files. Preserve original resource contents and
loading behavior while recording this ownership.

Include `design_answers` with the per-module mapping rationale,
`observed_channel_summary` (empty when not observed), `deep_research_summary`
(empty for a mapping-only initialization), `implementation_notes`, declared
`dependencies`, and `custom_configs` for original module/agent settings. Return
only a short completion status after writing the bundle.
