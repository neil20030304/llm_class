"""
Exercise C — Asta-powered research chatbot with GPT-4o-mini.

At startup, this script:

  1. Sends `tools/list` to the Asta MCP server and pulls every tool schema.
  2. Maps each schema to OpenAI function-calling format (MCP `inputSchema` is
     already valid JSON Schema, so the mapping is direct).
  3. Runs a tool-call loop: GPT-4o-mini decides which tool to call, we POST
     a `tools/call` to Asta, append the result as a `tool` message, and
     hand control back to the model until it produces a text answer.

Run modes:

    python exercise_c.py                          # interactive REPL
    python exercise_c.py --scripted               # run the 4 demo queries
    python exercise_c.py --ask "your question"    # single one-shot query

Truncation: tool results from Asta can be huge (e.g. a paper with full
abstracts for every citation). We cap each result at MAX_TOOL_BYTES before
appending to the conversation so the context budget doesn't blow up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI

from mcp_client import call_tool, list_tools, tool_error, tool_text

MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
MAX_TOOL_BYTES = 6000     # truncate Asta payloads before sending to the model
MAX_TURNS = 8             # safety bound on the tool-call loop

SYSTEM_PROMPT = (
    "You are a research assistant with live access to the Semantic Scholar corpus "
    "(225M+ academic papers) through the Ai2 Asta MCP toolkit. "
    "When a user asks about papers, authors, citations, or references, call the "
    "appropriate Asta tool. Prefer narrow `fields` lists to keep responses small. "
    "Cite paper titles and years in your final answer; do not invent IDs.\n\n"
    "Important tool semantics:\n"
    "  * `get_citations(paper_id)` returns papers that CITE the given paper "
    "(downstream impact).\n"
    "  * To get a paper's REFERENCES (its bibliography / what it builds on), "
    "call `get_paper(paper_id, fields='title,references.title,references.year')` "
    "— `references` is a FIELD on get_paper, not a separate tool.\n"
    "  * `search_paper_by_title` is exact-match; if it errors, fall back to "
    "`search_papers_by_relevance` with the title as a keyword."
)


# ---------------------------------------------------------------------------
# Schema translation: MCP → OpenAI
# ---------------------------------------------------------------------------
def mcp_to_openai_tool(mcp_tool: dict) -> dict:
    """MCP tool dict → OpenAI function-calling tool dict.

    `inputSchema` from MCP is already valid JSON Schema, which is exactly
    what OpenAI's `function.parameters` field expects.
    """
    description = (mcp_tool.get("description") or "").strip()
    return {
        "type": "function",
        "function": {
            "name": mcp_tool["name"],
            "description": description[:1024],
            "parameters": mcp_tool.get("inputSchema") or {"type": "object", "properties": {}},
        },
    }


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run an Asta tool and return a string suitable for a `tool` message.

    On error, returns the server-side error text so the model can recover
    (e.g. by retrying with a different paper ID).
    """
    try:
        result = call_tool(name, arguments)
    except Exception as exc:  # network/auth error — surface to the model
        return f"[MCP transport error] {exc}"

    err = tool_error(result)
    if err:
        return f"[Asta error] {err}"

    text = tool_text(result) or "(empty result)"
    if len(text) > MAX_TOOL_BYTES:
        text = text[:MAX_TOOL_BYTES] + f"\n…[truncated {len(text) - MAX_TOOL_BYTES} bytes]"
    return text


# ---------------------------------------------------------------------------
# One conversation turn (user message → final assistant text)
# ---------------------------------------------------------------------------
def run_turn(client: OpenAI, messages: list[dict], openai_tools: list[dict]) -> str:
    """Run the tool-call loop until the model emits a plain text answer."""
    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""
            messages.append({"role": "assistant", "content": content})
            return content

        # Persist the assistant turn so subsequent tool messages have a parent.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  ↳ tool call: {tc.function.name}({json.dumps(args)})")
            result_text = execute_tool(tc.function.name, args)
            print(f"    result: {len(result_text)} chars")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

    return "(stopped: hit MAX_TURNS without a final answer)"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def make_chatbot() -> tuple[OpenAI, list[dict], list[dict]]:
    """Connect to Asta, fetch tools, build the OpenAI client and message log."""
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY is not set. Add it to topic7MCP/.env.")

    print(f"[startup] fetching tool schemas from Asta MCP …")
    mcp_tools = list_tools()
    openai_tools = [mcp_to_openai_tool(t) for t in mcp_tools]
    print(f"[startup] {len(openai_tools)} tools registered: "
          f"{', '.join(t['function']['name'] for t in openai_tools)}")
    print(f"[startup] LLM: {MODEL}\n")

    client = OpenAI()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    return client, messages, openai_tools


def scripted_demo() -> None:
    """Run the 4 task-brief queries against a fresh chatbot."""
    queries = [
        "Find 3 recent papers about large language model agents and give me their titles.",
        "Who wrote 'Attention Is All You Need'? Just give me the author names.",
        "List 3 recent papers (2024 onward) that cite the original BERT paper "
        "(ARXIV:1810.04805).",
        "Summarize what the 'Attention Is All You Need' paper builds on by looking "
        "at 5 of its references.",
    ]
    client, messages, tools = make_chatbot()

    for i, q in enumerate(queries, 1):
        print("=" * 70)
        print(f"Q{i}: {q}")
        print("=" * 70)
        messages.append({"role": "user", "content": q})
        answer = run_turn(client, messages, tools)
        print(f"\nA{i}: {answer}\n")


def interactive() -> None:
    client, messages, tools = make_chatbot()
    print("Type 'quit' or Ctrl-D to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user.lower() in {"quit", "exit", "q"}:
            return
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        answer = run_turn(client, messages, tools)
        print(f"\nAssistant: {answer}\n")


def one_shot(question: str) -> None:
    client, messages, tools = make_chatbot()
    print(f"Q: {question}")
    messages.append({"role": "user", "content": question})
    answer = run_turn(client, messages, tools)
    print(f"\nA: {answer}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--scripted", action="store_true",
                     help="run the 4 demo queries from the task brief")
    grp.add_argument("--ask", metavar="QUESTION",
                     help="run a single one-shot query and exit")
    args = ap.parse_args()

    if args.scripted:
        scripted_demo()
    elif args.ask:
        one_shot(args.ask)
    else:
        interactive()


if __name__ == "__main__":
    main()
