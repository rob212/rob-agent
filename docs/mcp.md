# MCP - Standardising tools

In November 2024, Anthropic introduces MCP (Model Context Protocol) as an attempt to standardise the world of LLM tool integration. Where APIs established a common langage for communication between services, MCP establishes a common languge for LLM agents to communicate with tools. Just as you call any REST API using the same HTTP patters, an MCP-compatible agent can use any MCP-compatible tool without custom integration code. 

With MCP, agents can more easily integrate tools from services likes Google Drive, Slack, Tavily and an ever growing number of more providers.

MCP consists of a three-component architecture: Host, Client and Server. 

## MCP's Three Components

The *Host* is the primary environment or application where the AI "lives." In our context of building an AI agent, our orchestration code is the Host. It is responsible for managing the user's session, holding the conversation history, and deciding when it needs to reach out to a Client to get more information.

The *MCP Client* is the bridge inside your agent's architecture. When the Host decides it needs a tool, the Client handles the communication. It's job is to maintain a 1-to-1 connection with a specific Server. From an implementation standpoint, the Client doesn't actually "know" how to read a database or search the web; it simply knows how to ask the MCP Server what it’s capable of and pass the AI’s requests along.

THe *MCP Server* hosts the actual tools and executes them on demand. It responds to two types of requests: "what tools do you have?" (returning tool definitions) and "execute this tool with these arguements" (running the tool and returning results). MCP Servers can connect to extermal services like databases, APIs etc. You effectively "plug in" an MCP Server to your AI agent via your Client. 


> ℹ️ MCP actually defines _three_ types of capabilites that servers can provide: tools (functions the LLM can call), prompts (resuable prompt templates), and resources (data the LLM can access). For now we will focus on the tools, but it is important to be aware of the other capabilities for future learning.

## MCP Tool Interfaces

*Tool Discovery*: we previously [defined our own functionality to convert a Python function to a tool definition](/docs/implementing-a-web-search-tool.md#better-defining-our-tool-definitions) that an LLM can consume. With MCP, the client can request tool definitions from an MCP Server, which returns them in a standardised format. Meaning the tool developer of the MCP Server implements the schema once and any MCP compatible client can retrieve and use it. 

*Tool Execution*: we also built our own `tool_execution` function as part of our `simple_agent_loop` whilst [implementing our web search tool](/docs/implementing-a-web-search-tool.md#tool-execution-utilities). This bridged the LLMs output and the actual function calls. MCP standardises this too. The client sends a tool execution request to the server, which handles execution and return the results in a consistent format. 

## Transport Mechanisms

MCP supports three transport mechanisms for communicatin between MCP clients and MCP servers: 

*stdio (Standard I/O)*: this is the simplest approach where the client launches the seerver as a subprocess and communicates through standard input/output streams. Since everything runs locally on the same machine, there's no network overhead. This is sufficient for local development and when tools don't need to be shared across machines. 

*HTTP*: enables remote communication over the network. The client sends requests via HTTP, and the server streams responses back. This works well when servers need to run on separate machines or be shared across multiple clients.

*WebSocket*: provides full bidirectional communication, allowing both client and server to initiate messages. This is useful when servers need to push updates to clients proactively.

For our learning we will first use stdio transport since it's the simplest to setup and suffiecient to learn the MCP concepts we are interested in. You can setich to HTTP or WebSockets if you need remote or shared servers without changing your tool implementation. 

## Running our first MCP server 

Let's experiment with MCP but running and existing server locally. We'll use the [official Tavily MCP server](https://docs.tavily.com/documentation/mcp#tavily-mcp-server), which provides the same web search capability we built manually previously. This time it is packages as a ready-to-use MCP server. 

MCP servers in the ecosystem are typically distributed as npm packages and run using npx. If you don't have Node.js installed, you'll need to install it first.

Verify your node version: 
```bash
node --version
```

With Node.js installed, we can launch the Tavily MCP server using a single command, but first we need to set our Tavily API ket as an environment variable via our terminal: 
```bash
export TAVILY_API_KEY=<your tavily api key>
```

Now launch the server with the [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector). This is an interactive browser based developer tool for testing  and debugging MCP servers:

```bash
npx @modelcontextprotocol/inspector npx -y tavily-mcp@latest
```

This command runs the @modelcontextprotocol/inspector package without instlling it globally on your machine, pointing it at the latest tavily mcp server package. Open the URL shown in the terminal to begin exploring the server via the Inspector. 

Click on the *Connect* button in order to connect the inspector to the Tavily MCP Server. If you receive a 'Connection Error' you may need to copy the 'Session token' that was displayed in the terminal when you ran the inspector and paste it into the _Configuration_ > _'Proxy Session Token'_ field. 

You will see several tabs relating to the MCP capabilities we discussed above, we are interested in the *Tools* tab. Select *List Tools* and you should see the available tools that the Tavily MCP Server provides, which mirrors those available to us via the client we used previously. 

Select the 'tavily_search' to expand it's details. You'll see the tools description and parameter schema, formatted in the standard MCP format. Experiment with a question like 'who won the womens 2026 winter olympic curling final' and click the *Run Tool*. 

So what have we just done? We have downloaded and ran a production-ready MCP server, connected to it using a standard client interface. Discovered available tools and executed a tool and recieved results. Compare this to our previously implementation of basic search functionality vie defining our own functino, tool definition and execution. In addition the Tavily MCP server includes other tools and more robust error handling, the benefits of MCP become clearer. 

## The MCP Client 

Now let's explore building an MCP Client that can discover and use these tools programmatically within our agent. Whilst interacting with the Inspector to examine the Tavily MCP server was helpful for testing, our agent needs to interact with MCP programmaticaly. 

The MCP Client is what our agent will use to discover and invoke tools from MCP servers. It handles the protocol details: establishing connections, requesting tool definitions, executing tools andparsing results. 

We start by installing the MCP Python SDK for hands-on practice:

```bash
uv add mcp
```

Let's write a very simple client that connects to the Tavily MCP server, list it's available tools and perform a web serach. This will involve the following occuring in our code:
1. *StdioServerParameters* specify how to launch the server. This is the `npx` command to launch the tavily server, pass our API Key via _Environment_ variables
2. *stdio_client* launches the server as a subprocess. We will use `async with` to ensure proper cleanup when we are done. It returns read and write streams that the session uses for communication.
3. *ClientSession* provides the high-level API. After calling `initialize()`, we can interact with the server using standard methods like `list_tools()` and `call_tool()`.
4. *session_list_tools()* requests *all* available tools from the server. The server returns tools definitions including names, descriptions and parameters schemas.
5. *session.call_tool()* executes a specified tool. We pass the tool name and a dictionary or arguements. The server runs the tool and returns the result. 

In our `src` directory create a `mcp_client_experiment.py` with the following:


```python
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
                "tavily-search",
                arguments={
                    "query": "Who won the womens curling final at the 2026 winter Olympics?"
                },
            )
            print("Search Result:")
            print(result.content)


if __name__ == "__main__":
    asyncio.run(main())

```