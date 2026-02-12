"""Parse a LangGraph agent's markdown response into structured sections.

The travel agent produces markdown with ### headers like:
  ### 🌤️ Weather Summary
  ### 🎒 Packing List
  ### 🎯 Activity Ideas

This module extracts those sections into a dict the frontend can render.
"""

import re

_HEADER_RE = re.compile(r"^#{1,4}\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+)$")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+(.+)$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

_WEATHER_KEYWORDS = ("weather", "summary", "overview", "forecast", "conditions")
_PACKING_KEYWORDS = ("pack", "bring", "clothing", "essentials", "accessories")
_ACTIVITY_KEYWORDS = ("activit", "things to do", "suggestion", "ideas")


def _matches_any(text: str, keywords: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _extract_list_items(text: str) -> list[str]:
    """Pull bullet-point or numbered list items from a block of text."""
    items: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        match = _BULLET_RE.match(line) or _NUMBERED_RE.match(line)
        if match:
            item = _BOLD_RE.sub(r"\1", match.group(1)).strip()
            if item:
                items.append(item)
    return items


def _extract_prose(text: str) -> str:
    """Return the first non-empty paragraph (ignoring bullets/headers)."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:  # end of first paragraph
                break
            continue
        if _BULLET_RE.match(stripped) or _NUMBERED_RE.match(stripped):
            continue
        if stripped.startswith("#"):
            continue
        lines.append(stripped)
    return " ".join(lines)


def parse_agent_response(markdown: str) -> dict:
    """
    Split a markdown response into structured sections.

    Returns:
        {
            "summary": str,      # prose weather/trip overview
            "packing": [str],    # packing list items
            "activities": [str], # activity suggestions
            "raw": str,          # the original markdown
        }
    """
    result: dict = {
        "summary": "",
        "packing": [],
        "activities": [],
        "raw": markdown,
    }

    # Split on ### (or ## / #) headers, keeping the header text
    parts = re.split(r"(?=^#{1,4}\s)", markdown, flags=re.MULTILINE)

    for part in parts:
        header_match = re.match(r"^#{1,4}\s*(.*)", part)
        header = header_match.group(1) if header_match else ""
        body = part[header_match.end():] if header_match else part

        if _matches_any(header, _WEATHER_KEYWORDS):
            result["summary"] = _extract_prose(body) or result["summary"]

        elif _matches_any(header, _PACKING_KEYWORDS):
            result["packing"].extend(_extract_list_items(body))

        elif _matches_any(header, _ACTIVITY_KEYWORDS):
            result["activities"].extend(_extract_list_items(body))

    # If no sections were parsed (agent returned unstructured text),
    # use the whole thing as the summary.
    if not result["summary"] and not result["packing"]:
        result["summary"] = markdown.strip()

    return result
