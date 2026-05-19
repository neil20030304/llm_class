# Exercise A — Discover the Asta Tools.
#
# Q: Which tool would you use to find all papers about "transformer attention mechanisms"?
# A: search_papers_by_relevance (keyword/semantic search over 225M+ papers).
#
# Q: Which tool would you use to find who else published in the same area as a specific author?
# A: search_authors_by_name → get_author_papers to list their work, then
#    search_papers_by_relevance on those topics (or get_citations on a representative
#    paper) to surface co-authors and other researchers in the same area.
"""
Send a `tools/list` JSON-RPC message to the Asta MCP endpoint and print every
tool's name, description, and parameter list.

This is exactly the discovery handshake an MCP-aware client performs at
startup — doing it by hand makes the automation in exercise_c / exercise_d
legible.
"""
from mcp_client import list_tools


def main() -> None:
    tools = list_tools()

    print(f"Discovered {len(tools)} tools from Asta MCP server\n")
    print("=" * 70)

    for tool in tools:
        name = tool["name"]
        description = (tool.get("description") or "").strip().splitlines()[0]
        schema = tool.get("inputSchema", {}) or {}
        properties = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        print(f"\nTool: {name}")
        print(f"  Description: {description}")

        for param_name, info in properties.items():
            ptype = info.get("type", "?")
            tag = "REQUIRED" if param_name in required else "optional"
            default = info.get("default")
            default_str = f"  default={default!r}" if default is not None else ""
            print(f"  [{tag:8s}] {param_name} ({ptype}){default_str}")

        print("-" * 70)


if __name__ == "__main__":
    main()
