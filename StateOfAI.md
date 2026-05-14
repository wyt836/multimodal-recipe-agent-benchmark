# State of AI Use

This project was completed with assistance from AI tools, including Claude Code
and Codex. AI was used as a development and writing aid throughout the
assignment, including but not limited to understanding the assignment
requirements, designing the recipe knowledge-base schema, implementing and
debugging the code, planning the evaluation pipeline, drafting and refining the
report structure, and producing tables and visualisations for the final report.

The author remained actively involved in the project. I reviewed and directed
the AI-generated suggestions, made design decisions about the personalised
recipe domain and evaluation setup, inspected the code and results, interpreted
the experimental findings, and ensured that the submitted work reflects my own
understanding of the system. The final code, report, evaluation design, and
conclusions were checked and accepted by me.

## AI-Assisted Contributions

AI assistance was particularly useful for the following parts of the project:

- **Knowledge-base and data-structure design:** AI helped design the structured
  recipe schema, including recipe metadata, ingredients, dietary tags,
  allergens, cooking times, image captions, and visual attributes. These fields
  made it possible to support both retrieval and deterministic evaluation.

- **Retrieval and multimodal indexing:** AI assisted with implementing the
  OpenCLIP and ChromaDB retrieval layer, including separate text and image
  collections, query functions, and recipe-context formatting for downstream
  language-model calls.

- **Agent architecture:** AI helped implement the LangGraph-based V3 agent,
  including memory loading and updating, planning, tool execution, verification,
  and final response generation. It also assisted in designing the tool set for
  text search, image search, structured filtering, aggregate computation, and
  recipe lookup.

- **Ablation and benchmarking pipeline:** AI assisted with building the
  benchmark runner, cache handling, metric calculation, and comparison across
  V0 plain LLM, V1 text RAG, V2 multimodal RAG, V3 full agent, and V3 ablation
  variants.

- **Evaluation queries and rule-based scoring:** AI helped formulate the query
  families and deterministic constraints used to score factual, cross-modal,
  multi-hop, and conversational tasks without relying on an LLM judge.

- **Report planning and presentation:** AI supported the organisation of the
  report narrative, tables, figures, and discussion of results, including the
  interpretation that the no-planner ablation outperformed the full V3 agent in
  some settings.
