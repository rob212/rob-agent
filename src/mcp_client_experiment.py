import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()


async def main():
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp@latest"],
        env={
            "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        },
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            result = await session.call_tool(
                "tavily_search",
                arguments={
                    "query": "Who won the womens curling final at the 2026 winter Olympics?"
                },
            )
            print("Search Result:")
            print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
