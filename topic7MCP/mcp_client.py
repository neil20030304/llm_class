"""
mcp_client.py — tiny streamable-HTTP MCP client for Ai2's Asta Scientific Corpus.

Asta's MCP endpoint speaks JSON-RPC 2.0 wrapped in Server-Sent Events: every
response comes back as `event: message\\ndata: {...}\\n\\n`. The `requests`
library's `.json()` chokes on that, so we parse the SSE frame ourselves.

Two helpers everyone needs:

    list_tools()                 -> list of MCP tool dicts (tools/list)
    call_tool(name, arguments)   -> raw JSON-RPC `result` dict (tools/call)

Plus a `tool_text(result)` convenience that pulls the JSON-encoded body out of
`result["content"][0]["text"]` (Asta wraps every payload that way).
"""
from __future__ import annotations

import itertools
import json
import os
from typing import Any

import requests

ASTA_URL = "https://asta-tools.allen.ai/mcp/v1"

_id_counter = itertools.count(1)


def _headers() -> dict:
    api_key = os.environ.get("ASTA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ASTA_API_KEY is not set. Add it to topic7MCP/.env and "
            "`set -a && . ./.env && set +a` before running."
        )
    return {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-api-key": api_key,
    }


def _parse_sse(text: str) -> dict:
    """Extract the JSON payload from a single `event: message\\ndata: {...}` frame."""
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:"):].strip())
    raise ValueError(f"No SSE `data:` line in response:\n{text[:500]}")


def _rpc(method: str, params: dict | None = None) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": next(_id_counter),
        "method": method,
        "params": params or {},
    }
    resp = requests.post(ASTA_URL, headers=_headers(), json=payload, timeout=60)
    resp.raise_for_status()
    msg = _parse_sse(resp.text)
    if "error" in msg:
        raise RuntimeError(f"MCP error from {method}: {msg['error']}")
    return msg["result"]


def list_tools() -> list[dict]:
    """Return the list of tool definitions exposed by the Asta MCP server."""
    return _rpc("tools/list")["tools"]


def call_tool(name: str, arguments: dict[str, Any]) -> dict:
    """Invoke `name` with `arguments`; return the JSON-RPC `result` dict.

    The dict contains:
        content:           list[{type:"text", text:"<JSON string>"}]
        structuredContent: {"result": <parsed payload>}   (only on success)
        isError:           bool
    On `isError`, the first content item's text is the server-side error
    message. Callers should check via tool_error()/tool_result().
    """
    return _rpc("tools/call", {"name": name, "arguments": arguments})


def tool_error(result: dict) -> str | None:
    """Return the error message if the call failed, else None."""
    if result.get("isError"):
        content = result.get("content") or []
        return content[0].get("text", "unknown error") if content else "unknown error"
    return None


def tool_text(result: dict) -> str:
    """Concatenate every content[i]['text'] (one JSON object per item on success)."""
    content = result.get("content") or []
    return "\n".join(item.get("text", "") for item in content)


def tool_result(result: dict):
    """Return the structured payload, preferring `structuredContent.result`.

    Asta puts the parsed value at `structuredContent.result`. We fall back
    to JSON-parsing the concatenated content items only if that's missing.
    Returns None if the call errored.
    """
    if tool_error(result):
        return None
    structured = result.get("structuredContent") or {}
    if "result" in structured:
        return structured["result"]
    # Fallback: parse each content item as JSON and return as a list
    items = []
    for c in result.get("content") or []:
        text = c.get("text", "")
        if not text:
            continue
        try:
            items.append(json.loads(text))
        except json.JSONDecodeError:
            items.append(text)
    if len(items) == 1:
        return items[0]
    return items or None
