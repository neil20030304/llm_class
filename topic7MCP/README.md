# Topic 7 — MCP Tool Integration with Ai2 Asta

Four exercises that wire an LLM agent to the **Ai2 Asta Scientific Corpus Tool** — an MCP server exposing search and navigation over 225M+ academic papers (Semantic Scholar) — and use **GPT-4o-mini** as the brain. The arc goes from raw tool discovery (Ex A) through hand-rolled tool calls (Ex B) to a dynamically-configured chatbot (Ex C) and a fully autonomous research agent (Ex D).

## Table of contents

| File | What it is |
| ---- | ---------- |
| [`mcp_client.py`](mcp_client.py) | Tiny SSE-aware MCP client (`list_tools` / `call_tool` / `tool_result` / `tool_error`) used by every exercise |
| [`exercise_a.py`](exercise_a.py) | **Exercise A** — `tools/list` discovery; prints every Asta tool, params, defaults |
| [`exercise_b.py`](exercise_b.py) | **Exercise B** — three direct drills: search, citations, references |
| [`exercise_c.py`](exercise_c.py) | **Exercise C** — chatbot that fetches tool schemas at startup and runs a GPT-4o-mini tool-call loop |
| [`exercise_d.py`](exercise_d.py) | **Exercise D** — autonomous "citation neighbourhood" agent (no LLM in the planning loop) |
| [`exercise_a_output.txt`](exercise_a_output.txt) | Captured `tools/list` output (8 tools) |
| [`exercise_b_output.txt`](exercise_b_output.txt) | Captured output of the three drills |
| [`exercise_c_output.txt`](exercise_c_output.txt) | Captured output of the scripted 4-query demo |
| [`exercise_d_output.txt`](exercise_d_output.txt) | Captured stderr+stdout from running the agent on *Attention Is All You Need* |
| [`exercise_d_output.md`](exercise_d_output.md) | The rendered markdown report from that same run |

## Prerequisites

```bash
pip install openai requests python-dotenv

# topic7MCP/.env  (gitignored)
OPENAI_API_KEY=sk-...
ASTA_API_KEY=...
LLM_MODEL=gpt-4o-mini
```

Then before running anything, export the keys:

```bash
export ASTA_API_KEY=$(grep ^ASTA_API_KEY .env | cut -d= -f2-)
# OPENAI_API_KEY is assumed to already live in your shell env
```

---

## Transport note: Asta is SSE-flavoured JSON-RPC

The Asta endpoint (`https://asta-tools.allen.ai/mcp/v1`) is a **streamable HTTP** MCP server. Two non-obvious details surfaced during this assignment:

1. **`Accept` header must include `text/event-stream`.** A plain `application/json` Accept header returns `HTTP 406 Not Acceptable`.
2. **Every response is one SSE frame**, formatted as
   ```
   event: message
   data: {"jsonrpc":"2.0","id":1,"result":{...}}
   ```
   so `response.json()` fails — you have to grab the `data:` line and `json.loads` it yourself. That's exactly what [`mcp_client._parse_sse`](mcp_client.py) does.

Tool payloads also come in two shapes, redundantly:

- `result["content"][i]["text"]` — one stringified JSON object per content item.
- `result["structuredContent"]["result"]` — the same data already parsed as a Python list/dict.

`mcp_client.tool_result()` always prefers the structured form. Errors arrive with `isError: True` and the error message in `content[0].text`; `mcp_client.tool_error()` surfaces that.

---

## The eight tools Asta actually exposes

The lecture brief used short names (`search_papers`, `get_references`). The live server names are slightly different — Exercise A pins them down:

| Tool | Purpose |
| ---- | ------- |
| `get_paper` | Full metadata for one paper by SS ID / DOI / ArXiv / Corpus / etc. `references` and `citations` are *fields*, not separate tools. |
| `get_paper_batch` | Same as above, but for a list of IDs in one call. |
| `get_citations` | Papers that cite a given paper (downstream impact). Supports `publication_date_range`. |
| `search_authors_by_name` | Free-text author lookup → returns author IDs + paper/citation counts. |
| `get_author_papers` | Papers by a specific `authorId`. |
| `search_papers_by_relevance` | Keyword / semantic search over the full corpus. |
| `search_paper_by_title` | Exact-ish title match (errors with "Title match not found" if it can't pin one down). |
| `snippet_search` | Returns matching text snippets, optionally restricted to a paper or venue list. |

---

## Exercise A — Discover the Asta tools

> *Which tool would you use to find all papers about "transformer attention mechanisms"?*
> `search_papers_by_relevance` — keyword/semantic over 225M+ papers.

> *Which tool would you use to find who else published in the same area as a specific author?*
> `search_authors_by_name` → `get_author_papers` (get their papers), then either `get_citations` on a representative paper or `search_papers_by_relevance` on a derived topic keyword to surface peers.

Run:

```bash
python exercise_a.py
```

Sends a `tools/list` JSON-RPC message and prints each tool's name, one-line description, and required/optional parameters with defaults. Full output in [`exercise_a_output.txt`](exercise_a_output.txt) — 8 tools registered.

---

## Exercise B — Three direct tool drills

Run:

```bash
python exercise_b.py
```

| Drill | Tool | What it does |
| ----- | ---- | ------------ |
| 1 | `search_papers_by_relevance` | Top-5 recent papers for `"large language model agents"` |
| 2 | `get_citations` | First 10 papers citing BERT (`ARXIV:1810.04805`) from `2023-01-01:` onward |
| 3 | `get_paper` (`fields=references.title,references.year`) | All 41 references of *Attention Is All You Need* (`ARXIV:1706.03762`), sorted by year |

**Two adjustments from the brief**

- The actual server tool is `search_papers_by_relevance`, not `search_papers`.
- There is no standalone `get_references` — you ask for `references` as a *field* on `get_paper`. This is the same field-projection pattern Semantic Scholar uses; MCP just passes the field string through.

**Original Drill 3 seed (ReAct, `ARXIV:2210.03629`) was swapped for *Attention Is All You Need*.** The Semantic Scholar record for ReAct is malformed — the title is stored as `"LANGUAGE MODELS"` and `references` is `null`, which makes the server return `isError: True` with `"'NoneType' object is not iterable"` whenever you ask for `references.title`. Attention Is All You Need is the same shape of paper (foundational, dense bibliography) and resolves cleanly.

**Across the three drills, the result shapes differed in one main way:** `get_citations` wraps every entry as `{"citingPaper": {...}}` (because the same tool can also return *cited-by-source* relations); the other two tools return paper dicts directly. We unwrap that explicitly in the script.

---

## Exercise C — Asta-powered research chatbot

Run:

```bash
python exercise_c.py --scripted              # runs the 4 demo queries
python exercise_c.py --ask "your question"   # one-shot
python exercise_c.py                         # interactive REPL
```

Architecture (visible in [`exercise_c.py`](exercise_c.py)):

```
startup:
  ── tools/list ──▶ Asta MCP
  ◀── 8 tool schemas (inputSchema = valid JSON Schema)
  convert each via mcp_to_openai_tool() → OpenAI tools[]

each user turn:
  loop ≤ MAX_TURNS:
    GPT-4o-mini ◀── messages, tools[], tool_choice="auto"
    if reply.tool_calls:
      for each call:
        POST tools/call to Asta, truncate to 6000 bytes
        append {role:"tool", tool_call_id, content} to messages
    else:
      reply is the final answer → return
```

**Schema translation is one-liner.** Because MCP's `inputSchema` is already valid JSON Schema, the conversion is mechanical:

```python
def mcp_to_openai_tool(mcp_tool):
    return {
        "type": "function",
        "function": {
            "name": mcp_tool["name"],
            "description": mcp_tool["description"][:1024],
            "parameters": mcp_tool["inputSchema"],
        },
    }
```

**Calling the tool back** is just a `tools/call` POST and a write into the message log:

```python
result_text = execute_tool(name, args)   # → string, truncated to 6KB
messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
```

**Demo output** (full transcript in [`exercise_c_output.txt`](exercise_c_output.txt)):

| Query | Tool chosen | Outcome |
| ----- | ----------- | ------- |
| "Find 3 recent papers about LLM agents" | `search_papers_by_relevance` | 3 titles, one tool call |
| "Who wrote *Attention Is All You Need*?" | `search_paper_by_title(fields=authors)` | 8 authors, one tool call |
| "3 recent (2024+) papers citing BERT" | `get_citations(publication_date_range="2024-01-01:")` | 3 titles, one tool call |
| "Summarize what *Attention Is All You Need* builds on by looking at 5 references" | `get_paper(fields="references.title,references.year")` | 6KB payload → 5-paper synthesis |

**One model-side gotcha discovered while running this.** With only the raw tool descriptions, GPT-4o-mini conflated `get_citations` with `references` on Q4 and got stuck in a loop calling `get_citations` repeatedly. Adding a short clarification to the system prompt fixed it in one shot:

```
* `get_citations(paper_id)` returns papers that CITE the given paper.
* To get a paper's REFERENCES, call get_paper(paper_id,
  fields='title,references.title,references.year') — references is a FIELD.
```

That's the most interesting practical lesson here: schemas alone don't always disambiguate semantics. A few sentences of *human-curated* tool guidance, placed in the system prompt, can save the model from a high-cost reasoning loop.

---

## Exercise D — Citation Network Explorer Agent

Run:

```bash
python exercise_d.py ARXIV:1706.03762 --out report.md
```

This is **not** a chatbot. The LLM only writes the one-paragraph summary at the end — every tool call is orchestrated in Python so the pipeline is reproducible:

```
seed (get_paper, fields=…,references.title,…,references.citationCount,…,references.authors)
   │
   ├─▶ pick_foundational()    sort refs by citationCount desc → top 5
   ├─▶ get_citations(publication_date_range=f"{currentYear-3}-01-01:")
   ├─▶ for each seed author: get_author_papers → most-cited other paper
   ├─▶ cross_pollination()    set intersection (ref-authors ∩ citer-authors)
   └─▶ llm_summarize(abstract)   ← only LLM call in the whole pipeline
                                  
   ────▶ render_report() → markdown
```

**Ordering matters and is explicit.** The recent-citations call needs the seed's `paperId` (the resolved Semantic Scholar ID, not whatever the user passed), so it has to come after `get_seed`. Author-profile calls likewise need `authorId`s out of the seed payload. The dependency chain is:

```
get_seed ──▶ {paperId, authors[].authorId, references[]} ──┬──▶ get_citations
                                                            ├──▶ get_author_papers ×N
                                                            └──▶ pick_foundational (pure)
```

**Sample run on `ARXIV:1706.03762`** (full report in [`exercise_d_output.md`](exercise_d_output.md)):

- Foundational works (by elsewhere-citations): ResNet (227,782×), Adam (166,060×), LSTM (104,402×), Dropout (43,097×), Inception v3 (30,846×).
- Recent developments (2026 citing papers): medical image segmentation, multivariate time-series water-quality, traffic-signal RL, automated-driving safety metric, atrophic-gastritis segmentation — i.e. Transformers showing up in domains far from NMT.
- Author profiles: Vaswani → GNN survey, Shazeer → T5, Parmar → Conformer, Uszkoreit → ViT, Jones / Polosukhin → Natural Questions, Gomez → evolutionary biology, Kaiser → TensorFlow.

---

## Closing discussion

**What did writing tool schemas dynamically buy us?** Exercise A is what would normally be a manual translation step (read the docs → write an OpenAI function-tool dict → keep it in sync as the API evolves). With MCP it's a 6-line function. Concretely, when Asta added a new tool tomorrow, Exercise C would pick it up at next startup with zero code change — that's the durable value.

**What did it cost?** Two things. First, **transport quirks**: a server that speaks JSON-RPC-over-SSE rather than plain JSON broke `requests.json()` and would have broken any naive OpenAPI-style client. Second, **semantic ambiguity**: the auto-discovered tool descriptions don't always disambiguate close concepts (Ex C's `get_citations` vs. `references` confusion). A self-describing protocol gets you wire-compatible, not semantically clear — humans still need to write a few lines of tool guidance for non-obvious cases.

**What did we choose to include in context, and what did response quality look like when we passed everything?** In Exercise C every tool result was hard-capped at 6KB. In Exercise D the full `references` payload of *Attention Is All You Need* was ~30KB; that's fine for parsing in Python but too much for the LLM context budget if we summarized every reference. Truncation strategy depends on who the consumer is: Python wants the whole array, the LLM wants the top-K already sorted.

**What would it take to let the LLM decide Exercise D's order?** Make it a proper chatbot — register all five tools, hand it a multi-step prompt, and let it plan. It would probably do better than I expected on the easy steps and worse on the cross-pollination bookkeeping (set intersection is much cleaner in code than in a tool-call loop). The honest answer is that Ex D is a good benchmark for *which steps actually need the LLM* — here, only the natural-language summary did.

**What's missing from the MCP ecosystem today?** Mostly two things. (1) A discovery layer that's actually usable — Asta's URL is in a class hand-out, not a registry I could `pip install`. (2) A standard way to encode *semantic relationships between tools* (e.g. "this field is the same entity as that tool's input") so clients don't have to learn ad-hoc rules like "references is a field on get_paper, not its own tool".
