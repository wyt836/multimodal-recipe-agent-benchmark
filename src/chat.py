"""Interactive REPL over the V3 recipe agent.

Memory persists across user turns within a single REPL session, so the agent
accumulates allergies / dietary preferences / cooking history as you talk.

Run:
    python src/chat.py

Commands:
    quit / exit / Ctrl-C    leave the session
    /reset                  wipe accumulated memory mid-conversation
    /memory                 print the current memory state
    /help                   show this help
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import memory as mem
import v3_agent
from retrieval import build_indices


HELP = """\
Talk to the recipe agent — try things like:
  "How long does miso soup take to cook?"
  "I'm allergic to peanuts. Any quick Thai dishes I can make?"
  "What recipe has a layered presentation?"
  "I cooked miso soup yesterday. Suggest something different."
Commands: /reset, /memory, /help, quit
"""


def _count_tools(tool_calls) -> int:
    if not tool_calls:
        return 0
    if tool_calls and isinstance(tool_calls[0], list):
        return sum(len(x) for x in tool_calls)
    return len(tool_calls)


def main():
    print("Recipe agent (V3) — interactive mode")
    print("Building / loading vector indices...")
    build_indices()
    mem.reset_memory()  # start each REPL session clean
    print(HELP)

    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", ":q"}:
            break
        if user == "/help":
            print(HELP)
            continue
        if user == "/reset":
            mem.reset_memory()
            print("[memory cleared]")
            continue
        if user == "/memory":
            print(json.dumps(mem.load_memory(), indent=2, ensure_ascii=False))
            continue

        try:
            result = v3_agent.run(
                [{"role": "user", "content": user}],
                chat_continue=True,
            )
        except Exception as e:
            print(f"[error] {e}")
            continue

        print(f"\n{result['answer']}")
        toks = result["token_usage"]
        print(
            f"\n[{result['latency_seconds']:.1f}s | "
            f"{toks['input'] + toks['output']} tokens | "
            f"{_count_tools(result['tool_calls'])} tool calls | "
            f"retrieved: {', '.join(result.get('retrieved_recipe_ids', [])) or '-'}]"
        )


if __name__ == "__main__":
    main()
