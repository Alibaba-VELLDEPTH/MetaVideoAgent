"""Dynamic registries populated by generated MetaVideoAgent bundles.

The public runtime deliberately contains no hand-written strategy pool. The
initial bundle and each evolved bundle register their five module classes here
before execution.
"""


AGENT_CONFIG = {
    "max_steps": 10,
    "llm_temperature": 0.1,
    "vlm_temperature": 0.2,
    "max_output_tokens": 2048,
}


STRUCTURING_MAP = {}
WORK_MEMORY_MAP = {}
THINKING_MAP = {}
LOCALIZATION_MAP = {}
PERCEPTION_MAP = {}
