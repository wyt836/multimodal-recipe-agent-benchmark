"""V3 — LangGraph agent with 6 nodes, 5 tools, 3-layer memory.

Graph:
    START → memory_load → planner → executor → verifier → generator → memory_update → END

Ablation flags accepted by run():
    no_memory=True     → skip memory_load + memory_update (memory stays empty)
    no_planner=True    → replace structured planner with reactive bind_tools loop
    no_verifier=True   → skip the verifier (executor results go straight to generator)

Single-LLM design: every LLM call (planner, verifier, generator, memory_update,
reactive-loop) uses the same DeepSeek V4 Flash instance from config.get_llm().
"""
import json
import os
import re
import time
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import memory as mem
import tools
from config import get_llm

DEBUG = bool(os.environ.get("V3_DEBUG"))


# ============================================================== Agent state
class AgentState(TypedDict, total=False):
    user_query:            str
    memory:                dict
    plan:                  list           # [{tool, args}]
    step_results:          list           # one entry per plan step
    candidate_recipe_ids:  list[str]
    verified_recipe_ids:   list[str]
    answer:                str
    tool_calls:            list           # full log for the record
    token_usage:           dict           # {"input": int, "output": int}
    flags:                 dict           # ablation flags


# ============================================================== Helpers
def _add_tokens(state: AgentState, resp) -> None:
    usage = getattr(resp, "usage_metadata", None) or {}
    tu = state.setdefault("token_usage", {"input": 0, "output": 0})
    tu["input"]  += usage.get("input_tokens",  0)
    tu["output"] += usage.get("output_tokens", 0)


def _parse_json_block(text: str) -> dict | None:
    """Extract the first JSON object from an LLM response."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = fence.group(1) if fence else None
    if raw is None:
        m = re.search(r"\{.*\}", text, re.S)
        raw = m.group(0) if m else None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _resolve_refs(args: Any, step_results: list) -> Any:
    """Replace '$step_N' tokens with the Nth step's result (1-indexed)."""
    if isinstance(args, str):
        m = re.fullmatch(r"\$step_(\d+)", args)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < len(step_results):
                return step_results[i]
            return args
        return args
    if isinstance(args, list):
        return [_resolve_refs(x, step_results) for x in args]
    if isinstance(args, dict):
        return {k: _resolve_refs(v, step_results) for k, v in args.items()}
    return args


TOOL_REGISTRY = {
    "search_text_content":  tools.search_text_content,
    "search_image_visual":  tools.search_image_visual,
    "filter_recipes":       tools.filter_recipes,
    "compute_aggregate":    tools.compute_aggregate,
    "get_recipe_by_id":     tools.get_recipe_by_id,
}


# ============================================================== Nodes
def node_memory_load(state: AgentState) -> AgentState:
    if state.get("flags", {}).get("no_memory"):
        state["memory"] = mem._fresh()
        return state
    state["memory"] = mem.load_memory()
    if DEBUG: print("[memory_load]", state["memory"])
    return state


PLANNER_SYS = """You are the planner of a recipe-QA agent.
Given the user query and the user's memory, output a JSON plan: a list of tool steps that will collect evidence to answer the query. Each step has a 'tool' name and 'args' dict.

Available tools:
- search_text_content(query: str, k: int=3) → list of {recipe_id, score, name}. Use for conceptual / ingredient / cuisine queries.
- search_image_visual(query: str, k: int=3) → same shape, retrieves by visual properties (colors, presentation, layout).
- filter_recipes(criteria: dict) → list[recipe_id]. Apply HARD structured filters. Supported criteria: cuisine, cuisines, cuisines_excluded, main_protein, spicy_level, dietary_tags (must have ALL), allergens_excluded (must have NONE), ingredients_required (must have ALL), ingredients_excluded, max_total_time_min, max_cooking_time_min, max_prep_time_min, min_* equivalents.
- compute_aggregate(recipe_ids: list, op: str, field: str) → numeric or recipe_id. op ∈ {min, max, mean, sum, count, argmin, argmax}. field is a numeric recipe field like total_time_min, cooking_time_min, prep_time_min, servings.
- get_recipe_by_id(recipe_id: str) → full recipe dict.

You can refer to a previous step's result by passing the string "$step_N" (1-indexed) as an argument value.

Rules:
- Use search_image_visual ONLY when the query references visual properties (color, layout, "looks like", presentation).
- Use filter_recipes whenever the query has structured constraints (allergen exclusion, dietary, time bound, cuisine, must-have ingredients).
- For "fastest" / "shortest" / "quickest" / "average" / "how many", chain filter_recipes → compute_aggregate.
- If the user's query references "average cooking time" or similar phrasing about totals, use field=total_time_min (not cooking_time_min) — total time captures prep+cook.
- ALWAYS incorporate the user's allergies and dietary preferences from memory as constraints when relevant.
- If the user asks for a recipe that is clearly not in the knowledge base (e.g., a cuisine no recipe represents), produce an empty plan; the generator will refuse.

Respond with ONLY a JSON object: {"plan": [ {"tool": "...", "args": {...}}, ... ], "reasoning": "<one short sentence>"}."""


def node_planner(state: AgentState) -> AgentState:
    if state.get("flags", {}).get("no_planner"):
        # reactive mode handles plan/executor in one shot; leave state empty.
        return state
    llm = get_llm(temperature=0.0)
    prompt = (
        f"User query: {state['user_query']}\n\n"
        f"User memory:\n{mem.render_for_prompt(state['memory'])}\n\n"
        f"Produce the plan JSON now."
    )
    resp = llm.invoke([SystemMessage(content=PLANNER_SYS), HumanMessage(content=prompt)])
    _add_tokens(state, resp)
    parsed = _parse_json_block(resp.content) or {}
    state["plan"] = parsed.get("plan", []) or []
    if DEBUG: print("[planner] plan =", json.dumps(state["plan"], indent=2))
    return state


def node_executor(state: AgentState) -> AgentState:
    if state.get("flags", {}).get("no_planner"):
        return _reactive_executor(state)

    step_results = []
    call_log = []
    for i, step in enumerate(state.get("plan", []), start=1):
        tool_name = step.get("tool")
        raw_args = step.get("args", {})
        args = _resolve_refs(raw_args, step_results)
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            step_results.append(None)
            call_log.append({"step": i, "tool": tool_name, "args": args, "error": "unknown_tool"})
            continue
        try:
            out = fn(**args) if isinstance(args, dict) else fn(args)
        except Exception as e:
            out = None
            call_log.append({"step": i, "tool": tool_name, "args": args, "error": str(e)})
            step_results.append(out)
            continue
        step_results.append(out)
        call_log.append({"step": i, "tool": tool_name, "args": args, "result_summary": _summarise(out)})

    state["step_results"] = step_results
    state.setdefault("tool_calls", []).extend(call_log)
    state["candidate_recipe_ids"] = _collect_candidates(step_results)
    if DEBUG: print("[executor] candidates =", state["candidate_recipe_ids"])
    return state


def _summarise(out: Any) -> Any:
    if isinstance(out, list):
        return out[:5] + (["..."] if len(out) > 5 else [])
    if isinstance(out, dict):
        return {k: out[k] for k in list(out)[:5]}
    return out


def _collect_candidates(step_results: list) -> list[str]:
    """Pull every recipe_id mentioned by any tool output into a single list (dedupe, keep order)."""
    seen, out = set(), []
    for r in step_results:
        for rid in _ids_from_result(r):
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
    return out


def _ids_from_result(r: Any) -> list[str]:
    if isinstance(r, str):
        return [r]
    if isinstance(r, dict):
        if "id" in r:
            return [r["id"]]
        if "recipe_id" in r:
            return [r["recipe_id"]]
        return []
    if isinstance(r, list):
        out = []
        for x in r:
            out.extend(_ids_from_result(x))
        return out
    return []


# ----------- reactive (no-planner) executor: ReAct-style bind_tools loop
def _reactive_executor(state: AgentState) -> AgentState:
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def search_text_content(query: str, k: int = 3):
        """Semantic search over recipe text. Returns top-k recipes."""
        return tools.search_text_content(query, k)

    @lc_tool
    def search_image_visual(query: str, k: int = 3):
        """Search recipes by visual properties (colors, layout, presentation)."""
        return tools.search_image_visual(query, k)

    @lc_tool
    def filter_recipes(criteria: dict):
        """Apply hard structured filters; returns list of recipe ids."""
        return tools.filter_recipes(criteria)

    @lc_tool
    def compute_aggregate(recipe_ids: list, op: str, field: str):
        """Run min/max/mean/sum/count/argmin/argmax on a numeric recipe field."""
        return tools.compute_aggregate(recipe_ids, op, field)

    @lc_tool
    def get_recipe_by_id(recipe_id: str):
        """Look up a full recipe by id."""
        return tools.get_recipe_by_id(recipe_id)

    # DeepSeek's thinking mode breaks bind_tools multi-turn loops: the API requires
    # `reasoning_content` to be round-tripped in echoed assistant messages, but
    # langchain-openai doesn't preserve it. Disable thinking via DeepSeek's
    # Anthropic-style schema `thinking: {"type": "disabled"}`. Only this path needs
    # it — the structured planner uses single-shot calls and isn't affected.
    bound = get_llm(
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    ).bind_tools([
        search_text_content, search_image_visual,
        filter_recipes, compute_aggregate, get_recipe_by_id,
    ])
    fn_map = {
        "search_text_content":  lambda **kw: tools.search_text_content(**kw),
        "search_image_visual":  lambda **kw: tools.search_image_visual(**kw),
        "filter_recipes":       lambda **kw: tools.filter_recipes(**kw),
        "compute_aggregate":    lambda **kw: tools.compute_aggregate(**kw),
        "get_recipe_by_id":     lambda **kw: tools.get_recipe_by_id(**kw),
    }

    sys = SystemMessage(content=(
        "You are a recipe-QA agent. Use the available tools to gather evidence, then stop "
        "calling tools and let the answer-generation step run. Respect any user allergies / "
        "dietary preferences from the memory context."
    ))
    msgs = [sys, HumanMessage(content=(
        f"User memory:\n{mem.render_for_prompt(state['memory'])}\n\n"
        f"User query: {state['user_query']}"
    ))]
    step_results, call_log = [], []
    for _ in range(6):  # cap rounds of reactive tool calling
        resp = bound.invoke(msgs)
        _add_tokens(state, resp)
        msgs.append(resp)
        tc = getattr(resp, "tool_calls", None) or []
        if not tc:
            break
        from langchain_core.messages import ToolMessage
        for call in tc:
            name = call["name"]
            args = call.get("args", {}) or {}
            fn = fn_map.get(name)
            try:
                out = fn(**args) if fn else None
            except Exception as e:
                out = f"error: {e}"
            step_results.append(out)
            call_log.append({"tool": name, "args": args, "result_summary": _summarise(out)})
            msgs.append(ToolMessage(content=json.dumps(out, default=str)[:2000], tool_call_id=call["id"]))

    state["step_results"] = step_results
    state["tool_calls"] = call_log
    state["candidate_recipe_ids"] = _collect_candidates(step_results)
    return state


VERIFIER_SYS = """Extract HARD constraints from the user query and memory into a JSON criteria dict for filter_recipes.
Only include keys you are confident about — they will be applied strictly. Skip keys that don't apply.
Supported keys: cuisines_excluded (list), allergens_excluded (list), dietary_tags (list — must have ALL), ingredients_excluded (list), max_total_time_min (number), max_cooking_time_min (number), max_prep_time_min (number), spicy_level (string), main_protein (string).
Respond with ONLY: {"criteria": { ... }}"""


def node_verifier(state: AgentState) -> AgentState:
    if state.get("flags", {}).get("no_verifier"):
        state["verified_recipe_ids"] = state.get("candidate_recipe_ids", [])
        return state

    candidates = state.get("candidate_recipe_ids", [])
    if not candidates:
        state["verified_recipe_ids"] = []
        return state

    llm = get_llm(temperature=0.0)
    resp = llm.invoke([
        SystemMessage(content=VERIFIER_SYS),
        HumanMessage(content=(
            f"User query: {state['user_query']}\n"
            f"User memory:\n{mem.render_for_prompt(state['memory'])}"
        )),
    ])
    _add_tokens(state, resp)
    parsed = _parse_json_block(resp.content) or {}
    criteria = parsed.get("criteria", {}) or {}

    # Memory-based safety net: ALWAYS enforce known allergies, regardless of LLM extraction.
    extra_allergens = list(state["memory"].get("user_profile", {}).get("allergies", []))
    if extra_allergens:
        criteria.setdefault("allergens_excluded", [])
        for a in extra_allergens:
            if a not in criteria["allergens_excluded"]:
                criteria["allergens_excluded"].append(a)
        # Also block by ingredient name to catch allergens not declared in `allergens` field.
        criteria.setdefault("ingredients_excluded", [])
        for a in extra_allergens:
            if a not in criteria["ingredients_excluded"]:
                criteria["ingredients_excluded"].append(a)

    if not criteria:
        state["verified_recipe_ids"] = candidates
        return state

    allowed = set(tools.filter_recipes(criteria))
    verified = [rid for rid in candidates if rid in allowed]

    # Fallback: if verifier wiped everything but candidates existed, keep originals
    # so the generator can explain rather than answer from nothing.
    state["verified_recipe_ids"] = verified if verified else candidates
    state.setdefault("tool_calls", []).append({"tool": "verifier_filter", "args": criteria,
                                                "result_summary": verified[:5]})
    if DEBUG: print("[verifier] criteria=", criteria, "verified=", verified)
    return state


GENERATOR_SYS = """You are the final answer generator of a recipe-QA agent.
Answer the user's question concisely and directly. Use ONLY the verified recipe context and the user memory below.
- If the verified context is empty, say plainly that you don't have a matching recipe in your knowledge base.
- When recommending recipes, mention them by name and (where helpful) by id.
- Respect the user's allergies and dietary preferences.
- For numerical questions, give the exact number from the recipe data."""


def node_generator(state: AgentState) -> AgentState:
    verified = state.get("verified_recipe_ids", [])
    ctx = tools.recipes_for_context(verified) if verified else "(no recipes available)"
    llm = get_llm(temperature=0.0)
    resp = llm.invoke([
        SystemMessage(content=GENERATOR_SYS),
        HumanMessage(content=(
            f"User memory:\n{mem.render_for_prompt(state['memory'])}\n\n"
            f"Verified recipes:\n{ctx}\n\n"
            f"User query: {state['user_query']}"
        )),
    ])
    _add_tokens(state, resp)
    state["answer"] = resp.content
    return state


MEMORY_UPDATE_SYS = """Extract durable facts about the user from their most recent message.
Output ONLY this JSON (skip keys with nothing to add):
{
  "allergies":    [strings],
  "dietary":      [strings, e.g. "vegetarian", "vegan"],
  "dislikes":     [strings],
  "preferences":  [strings],
  "cooked":       [recipe name strings the user said they cooked, ate, or made],
  "constraints":  {key: value, key: value}   // session-scoped numeric/categorical constraints
}
If the user said something like "I cooked X yesterday", put X in "cooked".
If the user expressed a temporary constraint like "I want something under 30 minutes", put {"max_total_time_min": 30} in "constraints".
If the user said "I don't like spicy", put "spicy" in "dislikes"."""


def node_memory_update(state: AgentState) -> AgentState:
    if state.get("flags", {}).get("no_memory"):
        return state
    llm = get_llm(temperature=0.0)
    resp = llm.invoke([
        SystemMessage(content=MEMORY_UPDATE_SYS),
        HumanMessage(content=f"User message: {state['user_query']}"),
    ])
    _add_tokens(state, resp)
    parsed = _parse_json_block(resp.content) or {}
    mem.apply_extracted_facts(state["memory"], parsed)

    # Always record the last recommendation made (whether verified or not).
    if state.get("verified_recipe_ids"):
        mem.set_last_recommendation(state["memory"], state["verified_recipe_ids"][0])

    mem.save_memory(state["memory"])
    if DEBUG: print("[memory_update] facts=", parsed, "memory=", state["memory"])
    return state


# ============================================================== Graph wiring
def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("memory_load",   node_memory_load)
    g.add_node("planner",       node_planner)
    g.add_node("executor",      node_executor)
    g.add_node("verifier",      node_verifier)
    g.add_node("generator",     node_generator)
    g.add_node("memory_update", node_memory_update)

    g.add_edge(START, "memory_load")
    g.add_edge("memory_load",   "planner")
    g.add_edge("planner",       "executor")
    g.add_edge("executor",      "verifier")
    g.add_edge("verifier",      "generator")
    g.add_edge("generator",     "memory_update")
    g.add_edge("memory_update", END)
    return g.compile()


_GRAPH = None
def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ============================================================== Entry point
def run(query_turns: list[dict],
        no_memory: bool = False,
        no_planner: bool = False,
        no_verifier: bool = False,
        chat_continue: bool = False) -> dict:
    """Run the V3 agent over a list of conversational turns.

    chat_continue=False (default, benchmark mode): reset memory before this call
        so each benchmark query is independent.
    chat_continue=True (REPL mode): preserve memory across calls so the chat
        agent accumulates user facts (allergies, preferences) over the session.
        The caller is responsible for the initial mem.reset_memory().
    """
    from retrieval import build_indices
    build_indices()

    flags = {"no_memory": no_memory, "no_planner": no_planner, "no_verifier": no_verifier}

    per_turn_responses = []
    per_turn_retrieved = []
    all_tool_calls = []
    total_in = total_out = 0
    t0 = time.time()

    # Memory reset contract: benchmark resets per-call to keep queries
    # independent; chat mode skips the reset so prior turns' facts persist.
    if not chat_continue:
        mem.reset_memory()

    graph = _graph()
    for turn in query_turns:
        if turn["role"] != "user":
            continue
        init: AgentState = {
            "user_query": turn["content"],
            "memory": {},
            "tool_calls": [],
            "token_usage": {"input": 0, "output": 0},
            "flags": flags,
        }
        final = graph.invoke(init)
        per_turn_responses.append(final.get("answer", ""))
        per_turn_retrieved.append(final.get("verified_recipe_ids", []) or
                                   final.get("candidate_recipe_ids", []))
        all_tool_calls.append(final.get("tool_calls", []))
        tu = final.get("token_usage", {})
        total_in  += tu.get("input", 0)
        total_out += tu.get("output", 0)

    return {
        "answer": per_turn_responses[-1] if per_turn_responses else "",
        "per_turn_responses": per_turn_responses,
        "retrieved_recipe_ids": per_turn_retrieved[-1] if per_turn_retrieved else [],
        "per_turn_retrieved": per_turn_retrieved,
        "tool_calls": all_tool_calls,
        "latency_seconds": time.time() - t0,
        "token_usage": {"input": total_in, "output": total_out},
    }
