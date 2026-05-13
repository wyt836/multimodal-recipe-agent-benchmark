"""3-layer memory store backed by data/memory.json.

Layers:
  - user_profile:   stable per-user facts (allergies, dietary, dislikes, preferences)
  - session_state:  current conversation's compound constraints + last recommendation
  - cooking_history: list of dishes the user said they cooked

The benchmark resets memory between test cases (not between turns within a case).
"""
import json
from typing import Iterable

from config import MEMORY_JSON


EMPTY_STATE: dict = {
    "user_profile": {
        "allergies":   [],
        "dietary":     [],
        "dislikes":    [],
        "preferences": [],
    },
    "session_state": {
        "current_constraints":  {},
        "last_recommendation":  None,
    },
    "cooking_history": [],
}


def load_memory() -> dict:
    """Read memory.json. If the file is missing, return a fresh empty state."""
    if not MEMORY_JSON.exists():
        return _fresh()
    with open(MEMORY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(state: dict) -> None:
    MEMORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def reset_memory() -> dict:
    """Wipe memory.json back to an empty state and return it."""
    state = _fresh()
    save_memory(state)
    return state


def _fresh() -> dict:
    # deep-copy via json round-trip; cheap for this tiny structure.
    return json.loads(json.dumps(EMPTY_STATE))


# ---------------------------------------------------------------- Mutators
def add_allergens(state: dict, items: Iterable[str]) -> None:
    _extend_unique(state["user_profile"]["allergies"], items)


def add_dislikes(state: dict, items: Iterable[str]) -> None:
    _extend_unique(state["user_profile"]["dislikes"], items)


def add_dietary(state: dict, items: Iterable[str]) -> None:
    _extend_unique(state["user_profile"]["dietary"], items)


def add_preferences(state: dict, items: Iterable[str]) -> None:
    _extend_unique(state["user_profile"]["preferences"], items)


def add_to_history(state: dict, recipe_id_or_name: str) -> None:
    if recipe_id_or_name and recipe_id_or_name not in state["cooking_history"]:
        state["cooking_history"].append(recipe_id_or_name)


def set_constraint(state: dict, key: str, value) -> None:
    state["session_state"]["current_constraints"][key] = value


def set_last_recommendation(state: dict, recipe_id: str | None) -> None:
    state["session_state"]["last_recommendation"] = recipe_id


def apply_extracted_facts(state: dict, facts: dict) -> None:
    """Merge an LLM-extracted facts dict (see prompt in v3_agent.memory_update)."""
    if not isinstance(facts, dict):
        return
    add_allergens(state,    facts.get("allergies",   []))
    add_dietary(state,      facts.get("dietary",     []))
    add_dislikes(state,     facts.get("dislikes",    []))
    add_preferences(state,  facts.get("preferences", []))
    for cooked in facts.get("cooked", []) or []:
        add_to_history(state, cooked)
    for k, v in (facts.get("constraints", {}) or {}).items():
        set_constraint(state, k, v)


def _extend_unique(target: list, items: Iterable[str]) -> None:
    for s in items or []:
        s_norm = s.strip().lower()
        if s_norm and s_norm not in target:
            target.append(s_norm)


def render_for_prompt(state: dict) -> str:
    """Compact human-readable rendering used inside V3 prompts."""
    up = state["user_profile"]
    ss = state["session_state"]
    parts = []
    if up["allergies"]:
        parts.append(f"User allergies: {', '.join(up['allergies'])}")
    if up["dietary"]:
        parts.append(f"User dietary: {', '.join(up['dietary'])}")
    if up["dislikes"]:
        parts.append(f"User dislikes: {', '.join(up['dislikes'])}")
    if up["preferences"]:
        parts.append(f"User preferences: {', '.join(up['preferences'])}")
    if state["cooking_history"]:
        parts.append(f"Recently cooked: {', '.join(state['cooking_history'])}")
    if ss["current_constraints"]:
        cc = "; ".join(f"{k}={v}" for k, v in ss["current_constraints"].items())
        parts.append(f"Active constraints: {cc}")
    if ss["last_recommendation"]:
        parts.append(f"Last recommended: {ss['last_recommendation']}")
    return "\n".join(parts) if parts else "(no prior user info)"
