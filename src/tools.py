"""V3's 5 callable tools.

Two semantic tools (text / image search) and three structured tools
(filter, aggregate, lookup). The split is deliberate: it makes the
planner's routing decisions interpretable and gives the report clean
ablation handles ("what if we drop structured tools?").
"""
import json
from functools import lru_cache
from typing import Any

from config import RECIPES_JSON


# ---------------------------------------------------------------- Recipe cache
@lru_cache(maxsize=1)
def _recipes_by_id() -> dict[str, dict]:
    with open(RECIPES_JSON, "r", encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)}


# ---------------------------------------------------------------- Semantic tools
# (Lazy import of `retrieval` keeps the structured tools usable without chromadb installed.)
def search_text_content(query: str, k: int = 3) -> list[dict]:
    """Search the text index. Returns [{recipe_id, score, name}]."""
    import retrieval
    by_id = _recipes_by_id()
    return [
        {"recipe_id": rid, "score": score, "name": by_id[rid]["name"]}
        for rid, score in retrieval.search_text(query, k=k)
        if rid in by_id
    ]


def search_image_visual(query: str, k: int = 3) -> list[dict]:
    """Search the image index using a text query against CLIP-encoded images."""
    import retrieval
    by_id = _recipes_by_id()
    return [
        {"recipe_id": rid, "score": score, "name": by_id[rid]["name"]}
        for rid, score in retrieval.search_image(query, k=k)
        if rid in by_id
    ]


# ---------------------------------------------------------------- Structured tools
def filter_recipes(criteria: dict) -> list[str]:
    """Apply hard structured filters over the full recipe set.

    Supported criteria keys (all optional, ANDed together):
      - cuisine, cuisines:          single string or list
      - main_protein:               single string
      - spicy_level:                single string
      - dietary_tags:               list — recipe must include ALL
      - allergens_excluded:         list — recipe must include NONE
      - ingredients_required:       list — recipe's ingredient names must include ALL
      - ingredients_excluded:       list — recipe's ingredient names must include NONE
      - max_total_time_min, max_cooking_time_min, max_prep_time_min: numeric
      - min_total_time_min, min_cooking_time_min, min_prep_time_min: numeric
      - cuisines_excluded:          list — recipe.cuisine must not be any
    """
    by_id = _recipes_by_id()
    results = []
    for rid, r in by_id.items():
        if _passes(r, criteria):
            results.append(rid)
    return results


def _norm_allergen(s: str) -> str:
    """Strip a trailing plural 's' so 'peanuts' matches 'peanut' across data sources."""
    s = (s or "").strip().lower()
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s


def _passes(r: dict, c: dict) -> bool:
    def as_list(v):
        return v if isinstance(v, list) else [v]

    if "cuisine" in c and r["cuisine"].lower() != str(c["cuisine"]).lower():
        return False
    if "cuisines" in c and r["cuisine"].lower() not in [s.lower() for s in c["cuisines"]]:
        return False
    if "cuisines_excluded" in c and r["cuisine"].lower() in [s.lower() for s in c["cuisines_excluded"]]:
        return False
    if "main_protein" in c and r["main_protein"].lower() != str(c["main_protein"]).lower():
        return False
    if "spicy_level" in c and r["spicy_level"].lower() != str(c["spicy_level"]).lower():
        return False

    if "dietary_tags" in c:
        rt = {t.lower() for t in r.get("dietary_tags", [])}
        if not {t.lower() for t in c["dietary_tags"]}.issubset(rt):
            return False
    if "allergens_excluded" in c:
        ra = {_norm_allergen(t) for t in r.get("allergens", [])}
        if ra & {_norm_allergen(t) for t in c["allergens_excluded"]}:
            return False

    ing_names = {i["name"].lower() for i in r.get("ingredients", [])}
    if "ingredients_required" in c:
        for need in c["ingredients_required"]:
            need_l = need.lower()
            if not any(need_l in n for n in ing_names):
                return False
    if "ingredients_excluded" in c:
        for forbid in c["ingredients_excluded"]:
            forbid_l = forbid.lower()
            if any(forbid_l in n for n in ing_names):
                return False

    for field, key in (("total_time_min", "max_total_time_min"),
                       ("cooking_time_min", "max_cooking_time_min"),
                       ("prep_time_min", "max_prep_time_min")):
        if key in c and r.get(field, 0) > c[key]:
            return False
    for field, key in (("total_time_min", "min_total_time_min"),
                       ("cooking_time_min", "min_cooking_time_min"),
                       ("prep_time_min", "min_prep_time_min")):
        if key in c and r.get(field, 0) < c[key]:
            return False
    return True


def compute_aggregate(recipe_ids: list[str], op: str, field: str) -> Any:
    """Run an aggregation across a set of recipes.

    op ∈ {"min", "max", "mean", "sum", "count", "argmin", "argmax"}.
    For argmin/argmax, returns the recipe_id of the winning recipe.
    For numeric ops, returns the numeric value.
    """
    by_id = _recipes_by_id()
    rows = [(rid, by_id[rid][field]) for rid in recipe_ids
            if rid in by_id and isinstance(by_id[rid].get(field), (int, float))]
    if not rows and op != "count":
        return None
    if op == "count":
        return len(recipe_ids)
    values = [v for _, v in rows]
    if op == "min":     return min(values)
    if op == "max":     return max(values)
    if op == "sum":     return sum(values)
    if op == "mean":    return sum(values) / len(values)
    if op == "argmin":  return min(rows, key=lambda kv: kv[1])[0]
    if op == "argmax":  return max(rows, key=lambda kv: kv[1])[0]
    raise ValueError(f"unknown op: {op}")


def get_recipe_by_id(recipe_id: str) -> dict | None:
    return _recipes_by_id().get(recipe_id)


# ---------------------------------------------------------------- For prompts
def recipes_for_context(recipe_ids: list[str], compact: bool = True) -> str:
    """Render a list of recipes into a string for inclusion in an LLM prompt."""
    by_id = _recipes_by_id()
    parts = []
    for rid in recipe_ids:
        r = by_id.get(rid)
        if not r:
            continue
        if compact:
            ings = ", ".join(i["name"] for i in r["ingredients"])
            parts.append(
                f"- {r['name']} (id={r['id']}, cuisine={r['cuisine']}, "
                f"prep={r['prep_time_min']}min, cook={r['cooking_time_min']}min, "
                f"total={r['total_time_min']}min, servings={r['servings']}, "
                f"protein={r['main_protein']}, spicy={r['spicy_level']}, "
                f"dietary={r.get('dietary_tags', [])}, allergens={r.get('allergens', [])}, "
                f"ingredients=[{ings}]). "
                f"Caption: {r['image_caption']}"
            )
        else:
            parts.append(str(r))
    return "\n".join(parts)
