"""V2 — multimodal RAG with a FIXED pipeline.

Per turn: query BOTH the text index and the image index in parallel, fuse
the candidate sets (dedupe, keep ordering preference by score), then ask
the LLM to answer from the fused context. No routing — both indices are
always queried. The contrast with V3 is exactly the routing decision.
"""
import time
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm
from retrieval import build_indices, search_image, search_text
from tools import recipes_for_context


SYSTEM_PROMPT = (
    "You are a recipe assistant. Answer the user's question using ONLY the "
    "recipe context below. The context was retrieved from two sources: the "
    "recipe text index and the recipe image index (capturing visual properties "
    "like colors and presentation). Cite specific recipes by name where helpful. "
    "If the answer is not in the context, say so plainly.\n\n"
    "Recipe context:\n{context}"
)


def _fuse(text_hits, image_hits, k_total: int = 5) -> list[str]:
    """Reciprocal-rank fusion-ish: alternate text and image, dedupe, cap at k_total."""
    out: list[str] = []
    seen = set()
    for i in range(max(len(text_hits), len(image_hits))):
        if i < len(text_hits):
            rid = text_hits[i][0]
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
        if i < len(image_hits):
            rid = image_hits[i][0]
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
        if len(out) >= k_total:
            break
    return out[:k_total]


def run(query_turns: list[dict], k_text: int = 3, k_image: int = 3) -> dict:
    build_indices()
    llm = get_llm(temperature=0.0)

    per_turn_responses = []
    per_turn_retrieved = []
    total_in = total_out = 0
    t0 = time.time()

    for turn in query_turns:
        if turn["role"] != "user":
            continue
        text_hits  = search_text(turn["content"],  k=k_text)
        image_hits = search_image(turn["content"], k=k_image)
        fused_ids = _fuse(text_hits, image_hits, k_total=5)

        ctx = recipes_for_context(fused_ids)
        msg = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=ctx)),
            HumanMessage(content=turn["content"]),
        ]
        resp = llm.invoke(msg)
        per_turn_responses.append(resp.content)
        per_turn_retrieved.append(fused_ids)
        usage = getattr(resp, "usage_metadata", None) or {}
        total_in  += usage.get("input_tokens", 0)
        total_out += usage.get("output_tokens", 0)

    return {
        "answer": per_turn_responses[-1] if per_turn_responses else "",
        "per_turn_responses": per_turn_responses,
        "retrieved_recipe_ids": per_turn_retrieved[-1] if per_turn_retrieved else [],
        "per_turn_retrieved": per_turn_retrieved,
        "tool_calls": [],
        "latency_seconds": time.time() - t0,
        "token_usage": {"input": total_in, "output": total_out},
    }
