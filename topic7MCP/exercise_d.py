"""
Exercise D — Citation Network Explorer Agent.

Takes a paper id (any form Asta accepts: ARXIV:..., CorpusId:..., DOI:..., sha)
and autonomously assembles a markdown "citation neighbourhood" report:

  1. Seed paper metadata (title, year, authors, fields of study, abstract).
  2. Foundational works — 5 references from the bibliography, sorted by their
     citation count (most-cited first).
  3. Recent developments — up to 5 papers from the last 3 years that cite
     the seed paper.
  4. Author profiles — for each seed author, their most-cited other paper.
  5. Cross-pollination — does any citing paper share an author with the
     reference list? (bonus)

The LLM is used ONLY for generation: it never decides which tool to call.
Tool orchestration is hard-coded in Python so the pipeline is reproducible.

Usage:

    python exercise_d.py ARXIV:1706.03762                 # default seed
    python exercise_d.py CorpusId:13756489
    python exercise_d.py ARXIV:1810.04805 --out report.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections import defaultdict
from typing import Any

from openai import OpenAI

from mcp_client import call_tool, tool_error, tool_result

MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
RECENT_YEARS = 3
N_REFERENCES = 5
N_CITERS = 5


# ---------------------------------------------------------------------------
# Asta wrappers — every call returns either the parsed payload or raises.
# ---------------------------------------------------------------------------
def fetch(name: str, **arguments) -> Any:
    res = call_tool(name, arguments)
    err = tool_error(res)
    if err:
        raise RuntimeError(f"{name}{arguments} failed: {err}")
    return tool_result(res)


def get_seed(paper_id: str) -> dict:
    fields = (
        "title,year,abstract,authors,fieldsOfStudy,venue,"
        "references.paperId,references.title,references.year,references.citationCount,"
        "references.authors"
    )
    paper = fetch("get_paper", paper_id=paper_id, fields=fields)
    if not isinstance(paper, dict):
        raise RuntimeError(f"Unexpected get_paper payload: {paper!r}")
    return paper


def get_recent_citers(paper_id: str, n: int) -> list[dict]:
    cutoff = dt.date.today().year - RECENT_YEARS
    raw = fetch(
        "get_citations",
        paper_id=paper_id,
        fields="title,year,authors,abstract",
        limit=n * 4,                      # over-fetch — many will be older or partial
        publication_date_range=f"{cutoff}-01-01:",
    ) or []
    flat = [c.get("citingPaper", c) for c in raw]
    flat.sort(key=lambda p: (p.get("year") or 0), reverse=True)
    return flat[:n]


def get_author_top_paper(author_id: str, exclude_paper_id: str) -> dict | None:
    """Most-cited other paper by this author (excluding the seed)."""
    papers = fetch(
        "get_author_papers",
        author_id=author_id,
        paper_fields="title,year,citationCount",
        limit=100,
    ) or []
    candidates = [p for p in papers if p.get("paperId") != exclude_paper_id]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.get("citationCount") or 0), reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Reduction / cross-pollination
# ---------------------------------------------------------------------------
def pick_foundational(refs: list[dict], n: int) -> list[dict]:
    """References sorted by citationCount (None treated as 0), most-cited first."""
    refs = [r for r in refs if r.get("title")]
    refs.sort(key=lambda r: (r.get("citationCount") or 0), reverse=True)
    return refs[:n]


def cross_pollination(refs: list[dict], citers: list[dict]) -> list[str]:
    """Author names appearing both in the reference list and in citing papers."""
    ref_authors: dict[str, set[str]] = defaultdict(set)
    for r in refs:
        for a in r.get("authors") or []:
            ref_authors[a.get("name", "")].add(r.get("title", ""))

    overlap = []
    for c in citers:
        for a in c.get("authors") or []:
            name = a.get("name", "")
            if name and name in ref_authors:
                overlap.append(
                    f"**{name}** — cited in {next(iter(ref_authors[name]))!r}, "
                    f"author of citing paper {c.get('title','')!r}"
                )
    return overlap


# ---------------------------------------------------------------------------
# LLM-generated narrative (one paragraph summary of the seed abstract)
# ---------------------------------------------------------------------------
def llm_summarize(seed: dict) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        return f"_(LLM summary skipped: OPENAI_API_KEY not set)_\n\n{seed.get('abstract','')}"

    client = OpenAI()
    abstract = seed.get("abstract") or "(no abstract available)"
    title = seed.get("title", "Untitled")
    year = seed.get("year", "?")
    prompt = (
        f"Summarize the following paper in one tight paragraph (~80 words). "
        f"Focus on the contribution and why it mattered. Do not invent anything "
        f"the abstract doesn't say.\n\n"
        f"Title: {title} ({year})\n"
        f"Abstract: {abstract}"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_report(seed: dict, summary: str, refs: list[dict],
                  citers: list[dict], author_picks: list[tuple[dict, dict | None]],
                  overlap: list[str]) -> str:
    lines = []
    title = seed.get("title", "Untitled")
    year = seed.get("year", "?")
    venue = seed.get("venue") or "—"
    fields = ", ".join(seed.get("fieldsOfStudy") or []) or "—"
    authors = ", ".join(a.get("name", "") for a in seed.get("authors") or []) or "—"

    lines.append(f"# Citation Neighbourhood: *{title}* ({year})\n")
    lines.append(f"**Authors:** {authors}  ")
    lines.append(f"**Venue:** {venue}  ")
    lines.append(f"**Fields:** {fields}\n")
    lines.append("## Summary\n")
    lines.append(summary + "\n")

    # Foundational works
    lines.append(f"## Foundational Works — Top {len(refs)} references by citation count\n")
    if not refs:
        lines.append("_No references retrievable._\n")
    for r in refs:
        ry = r.get("year", "?")
        rt = r.get("title", "(untitled)")
        rc = r.get("citationCount", 0) or 0
        lines.append(f"- [{ry}] **{rt}** — cited {rc:,}× elsewhere")
    lines.append("")

    # Recent developments
    lines.append(f"## Recent Developments — citing papers from the last {RECENT_YEARS} years\n")
    if not citers:
        lines.append("_No recent citing papers found in the requested window._\n")
    for c in citers:
        cy = c.get("year", "?")
        ct = c.get("title", "(untitled)")
        ca = ", ".join(a.get("name", "") for a in (c.get("authors") or [])[:3])
        lines.append(f"- [{cy}] **{ct}**" + (f" — {ca}" if ca else ""))
    lines.append("")

    # Author profiles
    lines.append("## Author Profiles — most-cited other paper per seed author\n")
    for author, top in author_picks:
        name = author.get("name", "(unknown)")
        if top is None:
            lines.append(f"- **{name}** — no other papers found in the corpus")
        else:
            ty = top.get("year", "?")
            tt = top.get("title", "(untitled)")
            tc = top.get("citationCount", 0) or 0
            lines.append(f"- **{name}** — *{tt}* ({ty}, cited {tc:,}×)")
    lines.append("")

    # Cross-pollination
    lines.append("## Cross-Pollination — authors appearing in both directions\n")
    if not overlap:
        lines.append("_No authors found in both the reference list and the citing-paper list._\n")
    else:
        for line in overlap:
            lines.append(f"- {line}")
    lines.append("")

    lines.append("---\n")
    lines.append("_Generated by `exercise_d.py` against the Ai2 Asta MCP server._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(paper_id: str) -> str:
    print(f"[1/5] fetching seed paper: {paper_id}", file=sys.stderr)
    seed = get_seed(paper_id)
    seed_paper_id = seed.get("paperId", paper_id)

    print(f"[2/5] picking {N_REFERENCES} foundational references", file=sys.stderr)
    refs = pick_foundational(seed.get("references") or [], N_REFERENCES)

    print(f"[3/5] fetching recent citing papers (last {RECENT_YEARS}y)",
          file=sys.stderr)
    citers = get_recent_citers(seed_paper_id, N_CITERS)

    print(f"[4/5] resolving author profiles", file=sys.stderr)
    author_picks: list[tuple[dict, dict | None]] = []
    for a in seed.get("authors") or []:
        aid = a.get("authorId")
        if not aid:
            author_picks.append((a, None))
            continue
        try:
            top = get_author_top_paper(aid, seed_paper_id)
        except Exception as exc:
            print(f"    skipped author {a.get('name')}: {exc}", file=sys.stderr)
            top = None
        author_picks.append((a, top))

    print(f"[5/5] LLM summary of the abstract", file=sys.stderr)
    summary = llm_summarize(seed)

    overlap = cross_pollination(refs, citers)
    return render_report(seed, summary, refs, citers, author_picks, overlap)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper_id", nargs="?", default="ARXIV:1706.03762",
                    help="paper id (default: ARXIV:1706.03762 — Attention Is All You Need)")
    ap.add_argument("--out", help="write the markdown report to this file as well")
    args = ap.parse_args()

    report = run(args.paper_id)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\n[done] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
