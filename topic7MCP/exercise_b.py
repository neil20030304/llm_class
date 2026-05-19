"""
Exercise B — Three direct Asta tool drills.

Drill 1: search_papers_by_relevance — top-5 recent LLM-agent papers.
Drill 2: get_citations              — recent (2023+) papers that cite BERT.
Drill 3: get_paper (fields=references) — references of 'Attention Is All You Need'.

Two adjustments from the task brief:
  - The actual server tool is `search_papers_by_relevance`, not `search_papers`.
  - There is no standalone `get_references` tool — references come back as a
    field on `get_paper` (fields=references.title,references.year).

Drill 3 originally targeted ReAct (ARXIV:2210.03629). The Semantic Scholar entry
for that paper is malformed (title = "LANGUAGE MODELS", references null) which
makes the server return an isError when we ask for references. We use Attention
Is All You Need (ARXIV:1706.03762) instead — same kind of foundational paper,
clean record.
"""
from __future__ import annotations

from mcp_client import call_tool, tool_error, tool_result


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------------
# Drill 1 — search_papers_by_relevance: recent LLM-agent papers
# ----------------------------------------------------------------------------
def drill1_search_llm_agents() -> None:
    banner("Drill 1 — search_papers_by_relevance: 'large language model agents'")
    res = call_tool("search_papers_by_relevance", {
        "keyword": "large language model agents",
        "fields": "title,abstract,year,authors",
        "limit": 5,
    })
    err = tool_error(res)
    if err:
        print(f"  ERROR: {err}")
        return
    papers = tool_result(res) or []
    for i, p in enumerate(papers[:5], 1):
        year = p.get("year", "?")
        title = p.get("title", "(no title)")
        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
        if authors:
            print(f"  {i}. [{year}] {title}")
            print(f"      authors: {authors}")
        else:
            print(f"  {i}. [{year}] {title}")


# ----------------------------------------------------------------------------
# Drill 2 — get_citations: recent citations of BERT (ARXIV:1810.04805)
# ----------------------------------------------------------------------------
def drill2_bert_citations() -> None:
    banner("Drill 2 — get_citations: BERT (ARXIV:1810.04805), 2023-01-01 onward")
    res = call_tool("get_citations", {
        "paper_id": "ARXIV:1810.04805",
        "fields": "title,year,authors",
        "limit": 10,
        "publication_date_range": "2023-01-01:",
    })
    err = tool_error(res)
    if err:
        print(f"  ERROR: {err}")
        return
    citing = tool_result(res) or []
    # Each citation entry is wrapped as {"citingPaper": {...}}.
    flat = [c.get("citingPaper", c) for c in citing]

    print(f"  Retrieved {len(flat)} citing papers (showing first 5):")
    for i, p in enumerate(flat[:5], 1):
        year = p.get("year", "?")
        title = p.get("title", "(no title)")
        print(f"  {i}. [{year}] {title}")


# ----------------------------------------------------------------------------
# Drill 3 — get_paper (fields=references): foundations of Attention Is All You Need
# ----------------------------------------------------------------------------
def drill3_paper_references() -> None:
    banner("Drill 3 — get_paper(fields=references): 'Attention Is All You Need'")
    res = call_tool("get_paper", {
        "paper_id": "ARXIV:1706.03762",
        "fields": "title,year,references.title,references.year",
    })
    err = tool_error(res)
    if err:
        print(f"  ERROR: {err}")
        return
    paper = tool_result(res) or {}

    print(f"  Seed paper: {paper.get('title')}  ({paper.get('year')})")
    refs = paper.get("references") or []
    refs.sort(key=lambda r: r.get("year") or 0)

    print(f"  {len(refs)} references (sorted by year, showing first 15):\n")
    for r in refs[:15]:
        year = r.get("year", "?")
        title = r.get("title", "(no title)")
        print(f"    {year}  {title}")
    if len(refs) > 15:
        print(f"    ... and {len(refs) - 15} more")


def main() -> None:
    drill1_search_llm_agents()
    drill2_bert_citations()
    drill3_paper_references()

    print()
    print("=" * 70)
    print("Notes on result shape")
    print("=" * 70)
    print(
        "  - Every tool returns the structured payload at\n"
        "    result['structuredContent']['result']; mcp_client.tool_result()\n"
        "    grabs it directly. result['content'][i]['text'] is the same data\n"
        "    serialized one JSON object per item — usable but redundant.\n"
        "  - get_citations wraps each entry as {'citingPaper': {...}}; the other\n"
        "    two tools return flat paper dicts directly.\n"
        "  - get_paper exposes references as a *field* on the paper, not a\n"
        "    separate endpoint — request via fields=references.title,...\n"
        "  - When a paper's Semantic Scholar record is malformed (e.g. ReAct\n"
        "    ARXIV:2210.03629), asking for `references.title` makes the server\n"
        "    return isError=True. Always check tool_error() before parsing.\n"
    )


if __name__ == "__main__":
    main()
