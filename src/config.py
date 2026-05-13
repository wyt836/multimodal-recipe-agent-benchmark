"""Central config: paths, model names, env-var lookups, LLM factory.

Single LLM (DeepSeek V4 Flash) is used across V0/V1/V2/V3 so the only
variable between systems is architecture, not the underlying language model.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR    = PROJECT_ROOT / "data"
EVAL_DIR    = PROJECT_ROOT / "eval"
IMAGES_DIR  = PROJECT_ROOT / "images"
CHROMA_DIR  = PROJECT_ROOT / "chroma_db"

RECIPES_JSON = DATA_DIR / "recipes.json"
QUERIES_JSON = EVAL_DIR / "queries.json"
MEMORY_JSON  = DATA_DIR / "memory.json"

RESULTS_DIR        = EVAL_DIR / "results"
CACHE_DIR          = RESULTS_DIR / "cache"
RAW_OUTPUTS_JSONL  = RESULTS_DIR / "raw_outputs.jsonl"
METRICS_CSV        = RESULTS_DIR / "metrics_summary.csv"
PER_QUERY_CSV      = RESULTS_DIR / "per_query_results.csv"
FAILURES_CSV       = RESULTS_DIR / "failure_analysis.csv"

# ---------------------------------------------------------------- Model config
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

LLM_MODEL    = "deepseek-v4-flash"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"

# ChromaDB collections
TEXT_COLLECTION  = "recipes_text"
IMAGE_COLLECTION = "recipes_image"

# ---------------------------------------------------------------- System list
# (system_name, kwargs passed into v3_agent.run for ablations; {} for non-V3)
SYSTEMS = [
    ("V0",             {}),
    ("V1",             {}),
    ("V2",             {}),
    ("V3",             {}),
    ("V3-no-memory",   {"no_memory":   True}),
    ("V3-no-planner",  {"no_planner":  True}),
    ("V3-no-verifier", {"no_verifier": True}),
]

# ---------------------------------------------------------------- LLM factory
def get_llm(temperature: float = 0.0, **kwargs):
    """Return a configured ChatOpenAI client pointed at DeepSeek.

    All systems should call this so the LLM is held constant across V0..V3.
    max_retries=3 lets the underlying OpenAI SDK retry transient 429/5xx errors
    with backoff before raising. Non-transient errors and exhausted retries
    bubble up; the outer benchmark loop catches per-record and skips without
    writing to cache.
    """
    from langchain_openai import ChatOpenAI
    api_key = os.environ.get(LLM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {LLM_API_KEY_ENV} is not set. "
            f"Get a key at https://platform.deepseek.com and `setx {LLM_API_KEY_ENV} <key>`."
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=api_key,
        temperature=temperature,
        max_retries=3,
        timeout=120,
        **kwargs,
    )


def ensure_dirs():
    """Make sure all writable output dirs exist before the benchmark starts."""
    for d in (DATA_DIR, RESULTS_DIR, CACHE_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)
