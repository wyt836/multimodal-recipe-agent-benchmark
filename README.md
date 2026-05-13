# INFS4205/7205 Assignment 3 — Personalised Multimodal Recipe Agent

A comparative study of 4 recipe-QA architectures (V0 plain LLM → V1 text RAG → V2 multimodal RAG → V3 LangGraph agent) plus 3 V3 ablations, evaluated over 20 fixed queries with rule-based metrics.

## Install

Activate your conda env first, then:

```
pip install -r src/requirements.txt
```

All commands below assume the `recipe_agent` conda env is active — use
`python`, not the Windows `py` launcher (`py` bypasses conda and goes to
the system Python, splitting packages across two interpreters).

Then set the DeepSeek API key. Pick one:

```powershell
# A. Current PowerShell session only (takes effect immediately)
$env:DEEPSEEK_API_KEY = "<your-key>"

# B. Persistent across new shells (reopen PowerShell to pick it up)
setx DEEPSEEK_API_KEY "<your-key>"
```

Verify with `echo $env:DEEPSEEK_API_KEY` — should print your key.

## Run

```
python src/benchmark.py                          # full run: 7 systems × 20 queries = 140 records
python src/benchmark.py --systems V0,V3          # restrict systems
python src/benchmark.py --queries QF1.1,QF3.5    # restrict queries
python src/benchmark.py --no-cache               # force rerun
python src/benchmark.py --score-only             # rescore existing raw_outputs.jsonl
py src/visualize.py                          # render 4 result figures to eval/results/figures/
```

## Outputs (in `eval/results/`)

- `raw_outputs.jsonl` — per-record system output (one line per system×query)
- `metrics_summary.csv` — means grouped by (system, family)
- `per_query_results.csv` — full per-record table with all metrics
- `failure_analysis.csv` — 5-class taxonomy for every failed record

Per-record cache lives in `eval/results/cache/`. ChromaDB index lives in `chroma_db/` (built on first run).
