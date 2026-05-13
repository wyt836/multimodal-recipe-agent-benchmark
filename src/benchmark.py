"""Benchmark entry point.

Two stages:
  A. Execute each (system, query) → JSONL + per-record cache
  B. Score everything from the JSONL → three CSVs

Cache invariant: a cache file is written ONLY on a fully successful record.
Failed (system, query) records are logged and skipped — never written to cache.
This guarantees that re-running with cache will retry just the failed pairs.

Usage:
    python src/benchmark.py
    python src/benchmark.py --no-cache
    python src/benchmark.py --systems V0,V3 --queries QF1.1,QF3.2
    python src/benchmark.py --score-only
"""
import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

import eval_utils as eu
from config import (
    CACHE_DIR, ensure_dirs, FAILURES_CSV, METRICS_CSV,
    PER_QUERY_CSV, QUERIES_JSON, RAW_OUTPUTS_JSONL, SYSTEMS,
)
import memory as mem
import retrieval


# ============================================================== Loaders
def load_queries() -> list[dict]:
    with open(QUERIES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def load_recipes_by_id() -> dict[str, dict]:
    return {r["id"]: r for r in retrieval.load_recipes()}


SYSTEM_MODULES = {
    "V0":             "v0_plain",
    "V1":             "v1_text_rag",
    "V2":             "v2_multimodal_rag",
    "V3":             "v3_agent",
    "V3-no-memory":   "v3_agent",
    "V3-no-planner":  "v3_agent",
    "V3-no-verifier": "v3_agent",
}


def system_run(system_name: str, kwargs: dict, query: dict) -> dict:
    """Dispatch to the right system module's run() function."""
    module_name = SYSTEM_MODULES[system_name]
    module = importlib.import_module(module_name)
    return module.run(query["turns"], **kwargs)


# ============================================================== Stage A
def _cache_path(system: str, qid: str) -> Path:
    safe_sys = system.replace("/", "_")
    return CACHE_DIR / f"{safe_sys}__{qid}.json"


def stage_a(systems: list[tuple[str, dict]], queries: list[dict],
            use_cache: bool = True) -> tuple[list[dict], list[dict]]:
    """Run every (system, query). Returns (records, failures)."""
    ensure_dirs()
    records, failures = [], []

    for system_name, kwargs in systems:
        for q in queries:
            qid = q["query_id"]
            cache_file = _cache_path(system_name, qid)

            if use_cache and cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
                print(f"[cache] {system_name:<16} {qid}")
                continue

            print(f"[run]   {system_name:<16} {qid}  ", end="", flush=True)
            t0 = time.time()
            try:
                # Memory contract: reset between test cases. v3_agent.run() does this
                # internally; for V0/V1/V2 it's a no-op since they don't touch memory.
                mem.reset_memory()
                output = system_run(system_name, kwargs, q)
                rec = {
                    "system":  system_name,
                    "query_id": qid,
                    "family":  q["family"],
                    "subtype": q.get("subtype"),
                    "gold_recipe_ids": q.get("gold_recipe_ids", []),
                    **output,
                }
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
                records.append(rec)
                print(f"ok  ({time.time() - t0:.1f}s)")
            except Exception as e:
                # CRITICAL: do not write cache on failure.
                failures.append({"system": system_name, "query_id": qid, "error": str(e)})
                print(f"FAIL ({time.time() - t0:.1f}s): {e}")
                if "--trace" in sys.argv:
                    traceback.print_exc()

    # Write fresh raw_outputs.jsonl
    with open(RAW_OUTPUTS_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    return records, failures


def load_records_from_jsonl() -> list[dict]:
    if not RAW_OUTPUTS_JSONL.exists():
        return []
    out = []
    with open(RAW_OUTPUTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ============================================================== Stage B
def stage_b(records: list[dict], queries: list[dict]) -> None:
    """Score records → 3 CSVs."""
    ensure_dirs()
    recipes_by_id = load_recipes_by_id()
    queries_by_id = {q["query_id"]: q for q in queries}

    per_query_rows = []
    failure_rows = []

    for rec in records:
        q = queries_by_id.get(rec["query_id"])
        if q is None:
            continue
        per_turn = rec.get("per_turn_responses", []) or [rec.get("answer", "")]
        retrieved = rec.get("retrieved_recipe_ids", []) or []
        applicable = set(q.get("applicable_metrics", []))

        passed, detail = eu.check_constraints(q, per_turn, recipes_by_id)
        recall = eu.recall_at_k(retrieved, q.get("gold_recipe_ids", []), k=3) \
                 if "recall@3" in applicable else None
        retrieved_recipes = [recipes_by_id[r] for r in retrieved if r in recipes_by_id]
        faith = eu.faithfulness(rec.get("answer", ""), retrieved_recipes, recipes_by_id) \
                if "faithfulness" in applicable else None
        visual = eu.visual_grounding(rec.get("answer", ""), retrieved_recipes) \
                 if q.get("family") == "cross_modal" else None

        tool_calls = rec.get("tool_calls", [])
        tool_call_count = _count_tool_calls(tool_calls)

        row = {
            "system":           rec["system"],
            "query_id":         rec["query_id"],
            "family":           rec["family"],
            "subtype":          rec.get("subtype"),
            "task_success":     int(passed),
            "recall@3":         recall,
            "faithfulness":     faith,
            "visual_grounding": visual,
            "latency_seconds":  rec.get("latency_seconds"),
            "tool_calls":       tool_call_count,
            "tokens_input":     rec.get("token_usage", {}).get("input"),
            "tokens_output":    rec.get("token_usage", {}).get("output"),
            "constraint_detail": json.dumps(detail, ensure_ascii=False),
        }
        per_query_rows.append(row)

        if not passed:
            failure_rows.append({
                "system":      rec["system"],
                "query_id":    rec["query_id"],
                "family":      rec["family"],
                "failure_type": eu.classify_failure(q, retrieved, rec.get("answer", ""), recipes_by_id),
                "retrieved_recipe_ids": ",".join(retrieved),
                "gold_recipe_ids":      ",".join(q.get("gold_recipe_ids", [])),
                "answer_excerpt":       (rec.get("answer", "") or "")[:240].replace("\n", " "),
            })

    per_q_df = pd.DataFrame(per_query_rows)
    fail_df  = pd.DataFrame(failure_rows)
    summary_df = _summarise(per_q_df)

    per_q_df.to_csv(PER_QUERY_CSV, index=False, encoding="utf-8")
    fail_df.to_csv(FAILURES_CSV, index=False, encoding="utf-8")
    summary_df.to_csv(METRICS_CSV, index=False, encoding="utf-8")
    print(f"[score] wrote {PER_QUERY_CSV.name}, {FAILURES_CSV.name}, {METRICS_CSV.name}")


def _count_tool_calls(tool_calls) -> int:
    """tool_calls can be a flat list (V3 single-turn) or list-of-lists (V3 multi-turn)."""
    if not tool_calls:
        return 0
    if tool_calls and isinstance(tool_calls[0], list):
        return sum(len(x) for x in tool_calls)
    return len(tool_calls)


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(system, family) means of all metric columns."""
    if df.empty:
        return df
    numeric_cols = ["task_success", "recall@3", "faithfulness", "visual_grounding",
                    "latency_seconds", "tool_calls", "tokens_input", "tokens_output"]
    grouped = df.groupby(["system", "family"], dropna=False)[numeric_cols].mean(numeric_only=True)
    grouped["n"] = df.groupby(["system", "family"]).size()
    return grouped.reset_index()


# ============================================================== CLI
def main():
    parser = argparse.ArgumentParser(description="Run the V0..V3 recipe-QA benchmark.")
    parser.add_argument("--no-cache",   action="store_true", help="Force rerun, ignore cached records.")
    parser.add_argument("--systems",    type=str, default=None,
                        help="Comma-separated subset, e.g. V0,V3 or V3-no-memory.")
    parser.add_argument("--queries",    type=str, default=None,
                        help="Comma-separated query ids, e.g. QF1.1,QF3.2.")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip stage A; rescore from existing raw_outputs.jsonl.")
    parser.add_argument("--trace",      action="store_true", help="Print full tracebacks on failure.")
    args = parser.parse_args()

    queries = load_queries()
    if args.queries:
        wanted = set(args.queries.split(","))
        queries = [q for q in queries if q["query_id"] in wanted]

    systems = SYSTEMS
    if args.systems:
        wanted = set(args.systems.split(","))
        systems = [(n, kw) for (n, kw) in SYSTEMS if n in wanted]

    if args.score_only:
        records = load_records_from_jsonl()
        if not records:
            print("[error] no raw_outputs.jsonl to score. Run without --score-only first.")
            return
        stage_b(records, load_queries())
        return

    print(f"[plan] {len(systems)} systems × {len(queries)} queries = {len(systems)*len(queries)} records")
    records, failures = stage_a(systems, queries, use_cache=not args.no_cache)
    print(f"[stage A] done: {len(records)} records ok, {len(failures)} failed")
    if failures:
        for f in failures:
            print(f"  - FAIL {f['system']} {f['query_id']}: {f['error']}")
    stage_b(records, load_queries())
    print("[done]")


if __name__ == "__main__":
    main()
