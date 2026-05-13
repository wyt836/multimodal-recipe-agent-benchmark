"""Rule-based scoring: 11 constraint checks + retrieval/faithfulness/visual-grounding
metrics + 5-class failure classifier.

All evaluation is local: no LLM-as-judge. The structured metadata in recipes.json
and the constraint params in queries.json make this purely deterministic.
"""
import re
from typing import Iterable

# ---------------------------------------------------------------- Helpers
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _lower(s: str) -> str:
    return (s or "").lower()


def _recipe_aliases(r: dict) -> list[str]:
    """Strings that count as 'mentioning' this recipe."""
    aliases = [r["id"].lower(), r["name"].lower()]
    # id with underscores → space (e.g. tomato_egg_stirfry → tomato egg stirfry)
    aliases.append(r["id"].replace("_", " ").lower())
    # id with underscores → hyphen
    aliases.append(r["id"].replace("_", "-").lower())
    return list(set(aliases))


def _mentions_recipe(text: str, recipe: dict) -> bool:
    t = _lower(text)
    return any(alias in t for alias in _recipe_aliases(recipe))


def _mentioned_recipe_ids(text: str, recipes_by_id: dict) -> set[str]:
    t = _lower(text)
    found = set()
    for rid, r in recipes_by_id.items():
        if any(alias in t for alias in _recipe_aliases(r)):
            found.add(rid)
    return found


def _extract_numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM_RE.findall(text or "")]


def _select_turn_answer(per_turn_responses: list[str], turn_index) -> str:
    if not per_turn_responses:
        return ""
    if turn_index is None:
        return per_turn_responses[-1]
    if 0 <= turn_index < len(per_turn_responses):
        return per_turn_responses[turn_index]
    return per_turn_responses[-1]


# ---------------------------------------------------------------- 11 checks
def check_answer_contains_number(answer: str, params: dict, recipes_by_id: dict) -> bool:
    nums = _extract_numbers(answer)
    if "accepted_values" in params:
        wanted = set(float(v) for v in params["accepted_values"])
        return any(n in wanted for n in nums)
    if "value_range" in params:
        lo, hi = params["value_range"]
        return any(float(lo) <= n <= float(hi) for n in nums)
    return False


def check_answer_contains_text(answer: str, params: dict, recipes_by_id: dict) -> bool:
    t = _lower(answer)
    return any(_lower(s) in t for s in params.get("any_of", []))


def check_answer_does_not_contain_text(answer: str, params: dict, recipes_by_id: dict) -> bool:
    t = _lower(answer)
    return all(_lower(s) not in t for s in params.get("none_of", []))


def check_answer_mentions_recipe(answer: str, params: dict, recipes_by_id: dict) -> bool:
    rid = params["recipe_id"]
    r = recipes_by_id.get(rid)
    return bool(r) and _mentions_recipe(answer, r)


def check_answer_mentions_all_recipes(answer: str, params: dict, recipes_by_id: dict) -> bool:
    rids = params["recipe_ids"]
    min_required = params.get("min_required", len(rids))
    hits = sum(1 for rid in rids if rid in recipes_by_id and _mentions_recipe(answer, recipes_by_id[rid]))
    return hits >= min_required


def check_answer_does_not_mention(answer: str, params: dict, recipes_by_id: dict) -> bool:
    for rid in params["forbidden_recipe_ids"]:
        r = recipes_by_id.get(rid)
        if r and _mentions_recipe(answer, r):
            return False
    return True


def check_answer_recipes_subset_of(answer: str, params: dict, recipes_by_id: dict) -> bool:
    allowed = set(params["allowed_recipe_ids"])
    mentioned = _mentioned_recipe_ids(answer, recipes_by_id)
    return mentioned.issubset(allowed) and len(mentioned) > 0


def check_answer_recipe_count(answer: str, params: dict, recipes_by_id: dict) -> bool:
    n = len(_mentioned_recipe_ids(answer, recipes_by_id))
    if "exact" in params and n != params["exact"]:
        return False
    if "min_count" in params and n < params["min_count"]:
        return False
    if "max_count" in params and n > params["max_count"]:
        return False
    return True


def check_answer_indicates_no_recipe(answer: str, params: dict, recipes_by_id: dict) -> bool:
    t = _lower(answer)
    return any(_lower(p) in t for p in params.get("refusal_phrases", []))


def check_answer_identifies_as_superlative(answer: str, params: dict, recipes_by_id: dict) -> bool:
    rid = params["recipe_id"]
    r = recipes_by_id.get(rid)
    if not r:
        return False
    t = _lower(answer)
    if not _mentions_recipe(answer, r):
        return False
    return any(_lower(s) in t for s in params.get("superlative_terms", []))


def check_answer_acknowledges_conflict(answer: str, params: dict, recipes_by_id: dict) -> bool:
    t = _lower(answer)
    return any(_lower(s) in t for s in params.get("hedge_terms", []))


CONSTRAINT_DISPATCH = {
    "answer_contains_number":           check_answer_contains_number,
    "answer_contains_text":             check_answer_contains_text,
    "answer_does_not_contain_text":     check_answer_does_not_contain_text,
    "answer_mentions_recipe":           check_answer_mentions_recipe,
    "answer_mentions_all_recipes":      check_answer_mentions_all_recipes,
    "answer_does_not_mention":          check_answer_does_not_mention,
    "answer_recipes_subset_of":         check_answer_recipes_subset_of,
    "answer_recipe_count":              check_answer_recipe_count,
    "answer_indicates_no_recipe":       check_answer_indicates_no_recipe,
    "answer_identifies_as_superlative": check_answer_identifies_as_superlative,
    "answer_acknowledges_conflict":     check_answer_acknowledges_conflict,
}


def check_constraints(query: dict, per_turn_responses: list[str], recipes_by_id: dict
                      ) -> tuple[bool, list[dict]]:
    """Return (all_passed, [{type, turn_index, passed} per constraint])."""
    detail = []
    all_passed = True
    for c in query["verifiable_constraints"]:
        fn = CONSTRAINT_DISPATCH.get(c["type"])
        if fn is None:
            detail.append({"type": c["type"], "turn_index": c.get("turn_index"),
                           "passed": False, "error": "unknown constraint type"})
            all_passed = False
            continue
        ans = _select_turn_answer(per_turn_responses, c.get("turn_index"))
        try:
            passed = bool(fn(ans, c.get("params", {}), recipes_by_id))
        except Exception as e:
            passed = False
            detail.append({"type": c["type"], "turn_index": c.get("turn_index"),
                           "passed": False, "error": str(e)})
            all_passed = False
            continue
        detail.append({"type": c["type"], "turn_index": c.get("turn_index"), "passed": passed})
        if not passed:
            all_passed = False
    return all_passed, detail


# ---------------------------------------------------------------- Metric calcs
def recall_at_k(retrieved: Iterable[str], gold: Iterable[str], k: int = 3) -> float | None:
    gold = list(gold)
    if not gold:
        return None  # not applicable (e.g. out-of-KB probe)
    top = list(retrieved)[:k]
    hits = len(set(top) & set(gold))
    return hits / len(set(gold))


def faithfulness(answer: str, retrieved_recipes: list[dict], recipes_by_id: dict) -> float | None:
    """Numeric claims and recipe-name mentions must be supported by retrieved context.
    Score = supported_claims / total_claims. None if there are no checkable claims.
    """
    if not answer:
        return None
    retrieved_ids = {r["id"] for r in retrieved_recipes}

    # Numeric claims --------------------------------------------------------
    numeric_pool: set[float] = set()
    for r in retrieved_recipes:
        for field in ("prep_time_min", "cooking_time_min", "total_time_min", "servings"):
            if isinstance(r.get(field), (int, float)):
                numeric_pool.add(float(r[field]))
        for ing in r.get("ingredients", []):
            amt = ing.get("amount")
            if isinstance(amt, (int, float)):
                numeric_pool.add(float(amt))

    nums_in_answer = _extract_numbers(answer)
    # If the answer is a long enumeration, ignore the list-bullet small integers.
    skip_enum = len(nums_in_answer) > 5

    claims = 0
    supported = 0
    for n in nums_in_answer:
        if skip_enum and n in {1.0, 2.0, 3.0, 4.0, 5.0}:
            continue
        claims += 1
        if n in numeric_pool:
            supported += 1

    # Recipe-name claims ----------------------------------------------------
    mentioned_ids = _mentioned_recipe_ids(answer, recipes_by_id)
    for rid in mentioned_ids:
        claims += 1
        if rid in retrieved_ids:
            supported += 1

    if claims == 0:
        return None
    return supported / claims


def visual_grounding(answer: str, retrieved_recipes: list[dict]) -> float | None:
    """For QF2 only: extract visual descriptors from answer; check support in
    retrieved recipes' visual_attributes."""
    if not retrieved_recipes:
        return None
    color_words = {"red", "green", "yellow", "white", "black", "brown",
                   "orange", "pink", "purple", "golden", "beige", "gray", "grey"}
    presentation_words = {"layered", "soup", "bowl", "plate", "garnished",
                          "fried", "crispy", "creamy", "stir", "noodle"}
    t = _lower(answer)
    found = set()
    for w in color_words | presentation_words:
        if w in t:
            found.add(w)

    if not found:
        return None

    pool = set()
    for r in retrieved_recipes:
        va = r.get("visual_attributes", {})
        for c in va.get("dominant_colors", []) or []:
            pool.add(_lower(c))
        pres = _lower(va.get("presentation", ""))
        for w in presentation_words:
            if w in pres:
                pool.add(w)
        for vi in va.get("visible_ingredients", []) or []:
            pool.add(_lower(vi))

    matched = sum(1 for w in found if w in pool or any(w in p for p in pool))
    return matched / len(found)


# ---------------------------------------------------------------- Failure taxonomy
def classify_failure(query: dict, retrieved_ids: list[str], answer: str,
                     recipes_by_id: dict) -> str:
    """Map a failed (system, query) record to one of 5 Lecture-9 categories."""
    family = query.get("family")
    gold = set(query.get("gold_recipe_ids", []) or [])

    # query_grounding: query asserts an exclusion that was violated.
    for c in query["verifiable_constraints"]:
        if c["type"] == "answer_does_not_mention":
            for rid in c["params"]["forbidden_recipe_ids"]:
                r = recipes_by_id.get(rid)
                if r and _mentions_recipe(answer, r):
                    return "query_grounding"
        if c["type"] == "answer_does_not_contain_text":
            t = _lower(answer)
            for s in c["params"].get("none_of", []):
                if _lower(s) in t:
                    return "query_grounding"

    # retrieval: gold not in retrieved and answer is non-empty (and gold exists).
    if gold and retrieved_ids and not (gold & set(retrieved_ids)) and answer.strip():
        return "retrieval"
    if gold and not retrieved_ids and answer.strip():
        return "retrieval"

    # reasoning: multi-hop family with wrong superlative/count.
    if family == "multi_hop":
        return "reasoning"

    # citation: multi-turn answer cites a recipe outside any allowed set.
    if family == "conversational":
        for c in query["verifiable_constraints"]:
            if c["type"] == "answer_recipes_subset_of":
                allowed = set(c["params"]["allowed_recipe_ids"])
                mentioned = _mentioned_recipe_ids(answer, recipes_by_id)
                if mentioned and not mentioned.issubset(allowed):
                    return "citation"

    # grounding: retrieval was right but answer is wrong on a checked value.
    return "grounding"
