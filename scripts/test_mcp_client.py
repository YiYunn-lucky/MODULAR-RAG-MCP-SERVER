"""Temporary MCP client smoke test: connects to the stdio server and calls tools.

Usage:
    python scripts/test_mcp_client.py
"""

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> int:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.server"],
        cwd=str(REPO_ROOT),
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Tools available:", [t.name for t in tools.tools])

            result = await session.call_tool(
                "query_knowledge_hub",
                arguments={"query": "DeepSeek 多模态模型", "top_k": 2, "collection": "knowledge_hub"},
            )
            text = result.content[0].text if result.content else str(result)
            out = f"query_knowledge_hub result:\n{text[:1200]}"
            open(REPO_ROOT / "_mcp_result.txt", "w", encoding="utf-8").write(out)
            print("query_knowledge_hub: OK, response saved")
            return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
