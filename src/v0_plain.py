"""V0 — plain LLM baseline. No retrieval, no memory, no tools.

Each turn is processed independently with no prior-turn context — exactly the
condition we want to compare against V3's memory layer for the QF4 family.
"""
import time
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm


SYSTEM_PROMPT = (
    "You are a helpful recipe assistant. Answer the user's question concisely. "
    "If you don't know a recipe from your own knowledge, say so rather than inventing details."
)


def run(query_turns: list[dict]) -> dict:
    llm = get_llm(temperature=0.0)
    per_turn_responses = []
    total_in = total_out = 0
    t0 = time.time()

    for turn in query_turns:
        if turn["role"] != "user":
            continue
        msg = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=turn["content"])]
        resp = llm.invoke(msg)
        per_turn_responses.append(resp.content)
        usage = getattr(resp, "usage_metadata", None) or {}
        total_in  += usage.get("input_tokens", 0)
        total_out += usage.get("output_tokens", 0)

    return {
        "answer": per_turn_responses[-1] if per_turn_responses else "",
        "per_turn_responses": per_turn_responses,
        "retrieved_recipe_ids": [],  # V0 retrieves nothing
        "tool_calls": [],
        "latency_seconds": time.time() - t0,
        "token_usage": {"input": total_in, "output": total_out},
    }
