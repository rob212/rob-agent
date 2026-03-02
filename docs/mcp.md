# MCP - Standardising tools

In November 2024, Anthropic introduces MCP (Model Context Protocol) as an attempt to standardise the world of LLM tool integration. Where APIs established a common langage for communication between services, MCP establishes a common languge for LLM agents to communicate with tools. Just as you call any REST API using the same HTTP patterns, an MCP-compatible agent can use any MCP-compatible tool without custom integration code.

With MCP, agents can more easily integrate tools from services likes Google Drive, Slack, Tavily and an ever growing number of providers.

MCP consists of a three-component architecture: **Host**, **Client** and **Server**.

## MCP's Three Components

The **_Host_** is the primary environment or application where the AI "lives." In our context of building an AI agent, our orchestration code is the Host. It is responsible for managing the user's session, holding the conversation history, and deciding when it needs to reach out to a Client to get more information.

The **_MCP Client_** is the bridge **inside** your agent's architecture. When the Host decides it needs a tool, the Client handles the communication. It's job is to maintain a 1-to-1 connection with a specific MCP Server. From an implementation standpoint, the Client doesn't actually "know" how to read a database or search the web; it simply knows how to ask the MCP Server what it’s capable of and pass the AI’s requests along.

The **_MCP Server_** hosts the actual tools and executes them on demand. It responds to two types of requests:

- "what tools do you have?" (returning tool definitions)
- "execute this tool with these arguements" (running the tool and returning results)

MCP Servers can connect to external services like databases, APIs, messaging systems etc. You effectively "plug in" an MCP Server to your AI agent via your Client.

> ℹ️ MCP actually defines _three_ types of capabilites that servers can provide: tools (functions the LLM can call), prompts (resuable prompt templates), and resources (data the LLM can access). For now we will focus on the tools, but it is important to be aware of the other capabilities for future learning.

## 🪛 MCP Tool Interfaces

_Tool Discovery_: we previously [defined our own functionality to convert a Python function to a tool definition](implementing-a-web-search-tool.md#better-defining-our-tool-definitions) that an LLM can consume. With MCP, the client can request tool definitions from an MCP Server, which returns them in a standardised format. Meaning the tool developer of the MCP Server implements the schema once and any MCP compatible client can retrieve and use it.

_Tool Execution_: we also built our own `tool_execution` function as part of our `simple_agent_loop` whilst [implementing our web search tool](implementing-a-web-search-tool.md#tool-execution-utilities). This bridged the LLMs output and the actual function calls. MCP standardises this too. The client sends a tool execution request to the server, which handles execution and return the results in a consistent format.

## 🚛 Transport Mechanisms

MCP supports three transport mechanisms for communication between MCP clients and MCP servers:

**_stdio (Standard I/O)_**: this is the simplest approach where the client launches the server as a subprocess and communicates through standard input/output streams. Since everything runs locally on the same machine, there's no network overhead. This is sufficient for local development and when tools don't need to be shared across machines.

**_HTTP_**: enables remote communication over the network. The client sends requests via HTTP, and the server streams responses back. This works well when servers need to run on separate machines or be shared across multiple clients.

**_WebSocket_**: provides full bidirectional communication, allowing both client and server to initiate messages. This is useful when servers need to push updates to clients proactively.

For our learning we will first use stdio transport since it's the simplest to setup and suffiecient to learn the MCP concepts we are interested in. You can switch to HTTP or WebSockets if you need remote or shared servers without changing your tool implementation.

## 🏃‍♀️ Running our first MCP server

Let's experiment with MCP by running an existing server locally. We'll use the [official Tavily MCP server](https://docs.tavily.com/documentation/mcp#tavily-mcp-server), which provides the same web search capability we built manually previously. This time it is packages as a ready-to-use MCP server.

MCP servers in the ecosystem are typically distributed as npm packages and run using npx. If you don't have Node.js installed, you'll need to install it first.

Verify your node version:

```bash
node --version
```

With Node.js installed, we can launch the Tavily MCP server using a single command, but first we need to set our Tavily API ket as an environment variable via our terminal:

```bash
export TAVILY_API_KEY=<your tavily api key>
```

Now launch the server with the [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector). This is an interactive browser based developer tool for testing and debugging MCP servers:

```bash
npx @modelcontextprotocol/inspector npx -y tavily-mcp@latest
```

This command runs the @modelcontextprotocol/inspector package without instlling it globally on your machine, pointing it at the latest tavily mcp server package. Open the URL shown in the terminal to begin exploring the server via the Inspector.

> Once running you should see a link to the locally running inspector in your terminal, e,g: "MCP Inspector is up an running at http://127.0.0.1:6274 🚀"

Click on the _Connect_ button in order to connect the inspector to the Tavily MCP Server. If you receive a 'Connection Error' you may need to copy the 'Session token' that was displayed in the terminal when you ran the inspector and paste it into the _Configuration_ > _'Proxy Session Token'_ field.

You will see several tabs relating to the MCP capabilities we discussed above, we are interested in the _Tools_ tab. Select _List Tools_ and you should see the available tools that the Tavily MCP Server provides, which mirrors those available to us via the client we used previously.

Select the '**tavily-search**' to expand it's details. You'll see the tools description and parameter schema, formatted in the standard MCP format. Experiment with a question like 'who won the womens 2026 winter olympic curling final' and click the "Run Tool".

So what have we just done? We have downloaded and ran a production-ready MCP server, connected to it using a standard client interface. Discovered available tools and executed a tool and recieved results. Compare this to our previously implementation of basic search functionality vie defining our own functino, tool definition and execution. In addition the Tavily MCP server includes other tools and more robust error handling, the benefits of MCP become clearer.

## The MCP Client

Now let's explore the MCP Client more and how it can discover and use tools programmatically within our agent. Whilst interacting with the Inspector to examine the Tavily MCP server was helpful for testing, our agent needs to interact with MCP programmaticaly.

The MCP Client is what our agent will use to discover and invoke tools from MCP servers. It handles the protocol details: establishing connections, requesting tool definitions, executing tools andparsing results.

We start by installing the MCP Python SDK for hands-on practice:

```bash
uv add mcp
```

Let's write a very simple client that connects to the Tavily MCP server, list it's available tools and perform a web serach. This will involve the following occuring in our code:

1. **_StdioServerParameters_** specify how to launch the server. This is the `npx` command to launch the tavily server, pass our API Key via _Environment_ variables
2. **_stdio_client_** launches the server as a subprocess. We will use `async with` to ensure proper cleanup when we are done. It returns read and write streams that the session uses for communication.
3. **_ClientSession_** provides the high-level API. After calling `initialize()`, we can interact with the server using standard methods like `list_tools()` and `call_tool()`.
4. **_session_list_tools()_** requests _all_ available tools from the server. The server returns tools definitions including names, descriptions and parameters schemas.
5. **_session.call_tool()_** executes a specified tool. We pass the tool name and a dictionary or arguments. The server runs the tool and returns the result.

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
                "tavily_search",
                arguments={
                    "query": "Who won the womens curling final at the 2026 winter Olympics?"
                },
            )
            print("Search Result:")
            print(result.content)


if __name__ == "__main__":
    asyncio.run(main())

```

You can run this from in your `src` directory via the following command:

```bash
uv run python mcp_client_experiment.py
```

Doing so results in an output like the following:

```bash
Available tools:
  - tavily_search: Search the web for current information on any topic. Use for...
  - tavily_extract: Extract content from URLs. Returns raw page content in markd...
  - tavily_crawl: Crawl a website starting from a URL. Extracts content from p...
  - tavily_map: Map a websites structure. Returns a list of URLs found star...
  - tavily_research: Perform comprehensive research on a given topic or question....
Search Result:
[
    TextContent(
        type='text',
        text='Detailed Results:\n\nTitle: Sweden beat Switzerland to win gold...',
        annotations=None,
        meta=None
    )
]
```

## Converting MCP Tools for LLM Use

The tools returned by `list_tools` use MCP's schema format. Because MCP is a standardized protocol and the LLM provider (e.g. OpenAI) have their own specific API format, we need to convert the response before sending it back to the LLM.

Fortunately, we already built the foundation for this in our `utilities/tool_definition.py` module. The `format_tool_definition` function we created takes a _name_, _description_, and _parameters_ dictionary, and returns a properly structured OpenAI tool definition.

MCP tool objects provide exactly these three pieces of information through their `name`, `description`, and
`inputSchema` attributes. This makes the conversion straightforward.

We can add the following function to our `utilities/tool_definition.py` file:

```python
def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI tool format."""
    return [
        _format_tool_definition(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
        )
        for tool in mcp_tools.tools
    ]
```

With this conversion function, we can retrieve tools from any MCP server and immediately use them with OpenAI's API. Here's how it would work in practice if we were to adapt our `mcp_client_experiment.py` implemetation:

```python
async with stdio_client(server_params) as (read_stream, write_stream):
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()

        # List available tools
        tools_result = await session.list_tools()
        openai_format_tools = mcp_tools_to_openai_format(tools_result)
        for tool in openai_format_tools:
            print(tool)
```

## Building our own MCP Server to further our learning.

So far, we've used external MCP Servers that have been built by others. While this is a common way to build effective AI agents using community-provided servers, understanding how to create our own MCP Server is a good practice to learn how they work under the hood.

There also may be instances that a service we wish to add as a tool to our agent does not yet have an MCP Server implemented, and we are the ones that build it.

Building an MCP Server is surprisingly simple. We will use the [FastMCP library](https://gofastmcp.com/getting-started/welcome) that will allow us to convert an existing Python function into an MCP-compatible tool by adding just a decorator.

Let's transform our `search_web` function [we wrote](implementing-a-web-search-tool.md#adding-search-options) into a fully functional MCP Server.

Let’s start by installing the required libraries.

```bash
uv add fastmcp
```

Now, create a new file called `tavily_mcp_server.py`. We'll take our existing `search_web` function and wrap it with the MCP infrastructure. The FastMCP instance creates our server with a name that identifies it to clients. The `@mcp.tool()` decorator registers our function as an MCP tool.

FastMCP automatically extracts the function name, parameters, and type hints to generate the tool schema. The docstring becomes the tool's description, which helps LLMs understand when and how to use the tool. Finally, `mcp.run(transport='stdio')` starts the server using standard input/output for communication.

```python
import os
from tavily import TavilyClient
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

mcp = FastMCP("custom-tavily-search")

@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Search results as formatted string
    """
    try:
        response = tavily_client.search(
            query,
            max_results=max_results,
        )
        results = response.get("results", [])
        return "\n\n".join(
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
            for r in results
        )
    except Exception as e:
        return f"Error searching web: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='stdio')
```

Notice how little code we added to our original function. The core logic remains unchanged from our [original implementation](implementing-a-web-search-tool.md#adding-search-options); we simply wrapped it with the MCP infrastructure.

We can test this working using the MCP Inspector, ensure you are in the `src` directory and run:

```python
 uv run npx @modelcontextprotocol/inspector uv run python tavily_mcp_server.py
```

In the inspector, navigate to 'Tools' and you should see our `search_web` function, which you can run. (Make sure you have exported your TAVILY_API_KEY and have added the session token in to the 'Proxy Session Token' in the Inspector Configuration as we did previously.)

## Connecting our MCP Server to the Client

Now let's verify our custom MCP Server works by connecting to it with the [client code we wrote](mcp.md#the-mcp-client).

The only change we need is updating `StdioServerParameters` to point to our custom server instead of the official Tavily package.

We make this change via the `mcp_client_experiment.py`:

```python
server_params = StdioServerParameters(
    command="uv",
    args=["run", "tavily_mcp_server.py"],
    env={
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
    }
)

async with stdio_client(server_params) as (read_stream, write_stream):
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()

        # List available tools
        tools_result = await session.list_tools()
        print("Available tools:")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")
```

The only difference being the fact we now reference our bespoke 'tavily_mcp_server' rather than the official tavily mcp server.

If we were to run this we would see that where the official Tavily server exposed four tools (tavily-search, tavily-extract, tavily-crawl, tavily-map). Our custom server exposes just one tool: 'search_web'. Yet the client code is nearly identical. This is the power of standardization: whether you're connecting to a sophisticated official server or a simple custom one, the client interface remains the same.

## Summary

Let's summarise our learnings. In this section we converted our custom `search_web` function into an MCP server with minimum changes. We connected to our MCP server using the same MCP Client patters we used for the official Tavily server and we verifiyed that tool discovery and execution work the same.

The core value proposition of MCP is that tool implementation and tool usage are decoupled. MCP server developers can focus on building reliable tools and AI Agent developers can focus on their loop via reasoning and orchestration. The protocol handles everything in between.

## Recap

- Tools extend LLM capabilities, enabling them to access APIs, custom functionality, external information, databases etc.
- **Tool calling** is the mechanism which LLMs generate **structured outputs** specifying which tools to use and their associated parameters. The LLM acts as a mediator between the user requests and the available tools, but **does not** execute the tools itself.
- Tool execution is handled by our code, which feeds the results back into the LLms context for further reasoning.
- **Tool definitions** are structured schemas that describe available tools, their parameters and expected outputs. Clear and explicit definitions are essential for reliable tool selection by the LLM.
- Building custom tools involves implementing the tool function, creating tool definitions, and building execution infrastructure. While straightforward for simple cases, custom tools create a maintenance burden and can introduce inconsistencies as projects scale.
- MCP (Model Context Protocol) standardizes tool development through a client-server architecture. MCP servers host and execute tools, while MCP clients discover and invoke them. This seperation of concerns enables tool reuse across the ecosystem.

## 🪜 Next steps

Now we have a better understanding of tools and MCP, we have the building blocks needed for a complete agent.

Next we will combine these capabilities into a more robust agent framework, implementing propert tool abstraction, error handling and the iterative reasoning loop that transforms an LLM with tools into a true agent.

We'll start this by learning about Reasoning + Acting via [ReAct](react.md)
