"""V1 — text-only RAG. Single text index, top-k=3, no memory, no routing.

Per turn: encode the user query, retrieve top-3 recipes from the text index,
generate an answer conditioned on the retrieved context. Turns are independent.
"""
import time
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm
from retrieval import build_indices, search_text
from tools import recipes_for_context


SYSTEM_PROMPT = (
    "You are a recipe assistant. Answer the user's question using ONLY the recipe "
    "context provided below. If the answer is not in the context, say so plainly.\n\n"
    "Recipe context:\n{context}"
)


def run(query_turns: list[dict], k: int = 3) -> dict:
    build_indices()  # no-op if already built
    llm = get_llm(temperature=0.0)

    per_turn_responses = []
    per_turn_retrieved = []
    total_in = total_out = 0
    t0 = time.time()

    for turn in query_turns:
        if turn["role"] != "user":
            continue
        hits = search_text(turn["content"], k=k)
        retrieved_ids = [rid for rid, _ in hits]
        ctx = recipes_for_context(retrieved_ids)
        msg = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=ctx)),
            HumanMessage(content=turn["content"]),
        ]
        resp = llm.invoke(msg)
        per_turn_responses.append(resp.content)
        per_turn_retrieved.append(retrieved_ids)
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
