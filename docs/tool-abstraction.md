# Tool Abstraction

Previously, we [build tools as Python functions](tool-definitions#custom-tools) and created a utility to convert them into a tool definition for the LLM. This simple approach was useful for our learning but as we evolve, we will build a more unified tool interface that will enable us to add tools more easily.

## Why do we need a unified tool interface?

As a reminder here is the `search_web` function we wrote, then used `function_to_tool_definition` to generate the schema that the LLM needs.

```python
def search_web(query: str, max_results: int = 3) -> str:
    """Search the web for information."""
    return tavily_client.search(query, max_results=max_results)

search_tool_definition = function_to_tool_definition(search_web)
search_web(query="Python tutorials")
```

This works, but the function and its metadata are seperate entities. When building an agent, we need to pass the tool definitions to the LLM and then map names back to the function itself when the LLM wants us to execute it, making keeping things in sync a manual overhead.

If we had a tool abstraction via a decorator, the function and it's metadata could be bundled together. Let's implement this with two new classes: `BaseTool` which defines the interface and `FunctionTool` that will wrap existing functions.

## BaseTool

Every tool in our framework will inherit from `BaseTool`, an abstract base class that defines the interface for a tool.

A tool must have three properties:

- _name_: for identification
- _description_: for explaining it's purpose to the LLM
- _tool_definition_: a schema that tells the LLM how to use it

A tools must also be executable, which we'll represent with an async execute method that receives `ExecutionContext` along with any arguments.

We will provide some sensible defaults when initialised, if no name is givenm it uses the class name. If no description is given, it uses the docstring. This will hopefully reduce boilerplate when creating simple tools. The `tool_definition` propoert returns the schema that tells the LLM what parameters the tool accepts. We will store it as `_tool_definition` to allow subclasses to generate it lazily if needed.

```python
class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(
        self,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition

    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)
```

The execute method is abstract because every tool must implement its own logic. Notice that it always receives context as its first parameter. This is the key design decision that enables context propagation: the Agent passes `ExecutionContext` to every tool, and tools can use it if they need access to execution state. Tools that don't need context simply ignore it.

The `__call__` function provides syntactic convenience. Instead of having to write `tool.execute(context, query="test")`, we can write `tool(context, query="test")`.

Now we have a `BaseTool` defined that all of our tools will extend assuring that they must follow the contract. Next we'll implement `FunctionTool` to wrap the simple functions we created as tools.

## FunctionTool

`FunctionTool` acts as an adapter that wraps existing functions with the BaseTool interface. The AI Agent will pass `ExecutionContext` to every tool, but some of our simple tools may not need the context. For example our `search_web` tool has a signature of `search_web(query: str)` not `search_web(context: ExecutionContext, query: str)`.

We will inspect each tool function's signature at initialisation. If the function has a `context` parameter, we pass it. If not, we omit it. This will allow simple functions to work unchanged, while context-aware function can access execution state when needed.

Our `execute` method uses the information from our initialisation to determine whether to forward the context or not.

```python
class FunctionTool(BaseTool):
    """Wraps a Python function as a BaseTool."""

    def __init__(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None
    ):
        self.func = func
        self.needs_context = 'context' in inspect.signature(func).parameters

        name = name or func.__name__
        description = description or (func.__doc__ or "").strip()
        tool_definition = tool_definition or self._generate_definition()

        super().__init__(
            name=name,
            description=description,
            tool_definition=tool_definition
        )

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """Execute the wrapped function."""
        if self.needs_context:
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        # Handle both sync and async functions
        if inspect.iscoroutine(result):
            return await result
        return result

    def _generate_definition(self) -> Dict[str, Any]:
        """Generate tool definition from function signature."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)

```

Here is `FunctionTool` in action. The function's name, docstring and parameters are automatically extracted to a create a fully functional tool that our LLM understands how to interact with and what it's for.

> Python's `eval` function takes a string containing python expressions and executes them, i.e. it will perform numerical calculations returning a float.

```python
def calculator(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)

calc_tool = FunctionTool(calculator)

print(calc_tool.name)          # "calculator"
print(calc_tool.description)   # "Calculate mathematical expressions."
await calc_tool(context, expression="1234 * 5678")  # 7006652
```

## The @Tool Decorator

For convenience, we provide a decorator that creates FunctionTool instances. This is the equivalent of writing `calculator = FunctionTool(calculator), but with much cleaner syntax. This would enable us to use the @tool decorator as follows:

```python
@tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)
```

We can also implement our decorator to accept optional parameters to override the tool's metadata for added flexibility. Instead of using the function's name and docstring, you can specify custom values that will appear in the tool definition sent to the LLM. For example:

```python
@tool(name="web_search", description="Search the internet for information")
def search_web(query: str) -> str:
    """Search the web."""
    return tavily_client.search(query)
```

We define this `tool` decorator as follows. (Be aware that this is all pure Python code and not AI Agent specific. Just exploring how we might begin to define useful utilities if we were to build up our own Agentic Framework from scratch to demystify the "magic" that 3rd party AI Agent frameworks may offer)

```python
# decorator.py

from typing import Callable, Union, Dict, Any
from .function_tool import FunctionTool

def tool(
    func: Callable = None,
    *,
    name: str = None,
    description: str = None,
    tool_definition: Union[Dict[str, Any], str] = None
) -> Union[Callable, FunctionTool]:

    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(
            func=f,
            name=name,
            description=description,
            tool_definition=tool_definition
        )

    if func is not None:
        return decorator(func)
    return decorator
```

This supports both the simple @tool decorator pattern with no arguments **and** the decorator pattern with arguments.

## Integrating MCP Tools

Previously, [we explored MCP](mcp.md) and connected our AI agent to MCP servers to discover and execute tools. Now we need to use those tools in our agent framework. The challenge is that MCP tools live on external servers and are executes via `session.call_tool()`, while our agent expects BaseTool instances with an `execute` method.

We will therefore create a wrapper function for each MCP tool that calls the server, then wrap that function with `FunctionTool`. This approach allows us to reuse what we've built.

The `load_mcp_tools` function connects to an MCP server, retrieves all available tools and converts each one into a `FunctionTool`. It takes connection parameters (like the command to start the server) and returns a list of tools ready to use with our agent.

```python
async def load_mcp_tools(connection: dict) -> list[BaseTool]:
    """Load tools from an MCP server and convert to FunctionTools."""
    tools = []

    async with stdio_client(StdioServerParameters(**connection)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            for mcp_tool in mcp_tools.tools:
                func_tool = _create_mcp_tool(mcp_tool, connection)
                tools.append(func_tool)

    return tools
```

The core logic lives will then live in `_create_mcp_tool`. For each MCP tool, it creates a wrapper function called `call_mcp` that establishes a connection to the server and executes the tool with whatever arguments are passed. This wrapper function is then wrapped with FunctionTool. Notice that we pass `tool_definition` explicitly rather than letting FunctionTool generate it, because the MCP server already provides a complete schema via `mcp_tool.inputSchema`.

```python
def _create_mcp_tool(mcp_tool, connection: dict) -> FunctionTool:
    """Create a FunctionTool that wraps an MCP tool."""

    async def call_mcp(**kwargs):
        async with stdio_client(StdioServerParameters(**connection)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(mcp_tool.name, kwargs)
                return _extract_text_content(result)

    tool_definition = {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        }
    }

    return FunctionTool(
        func=call_mcp,
        name=mcp_tool.name,
        description=mcp_tool.description,
        tool_definition=tool_definition
    )
```

Let's see an example of this in use to ensure that this adapter logic allows us to intergrate MCP tools seamlessly with our agent. The following example loads tools from the Tavily MCP server and combines them with the local calculator tool. Our agent doesn't know or care that some tools are local functions and others call remote MCP servers. They all implement the same BaseTool interface and their origins are abstracted away.

```python
connection = {
    "command": "npx",
    "args": ["-y", "tavily-mcp@latest"],
    "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
}
mcp_tools = await load_mcp_tools(connection)

agent = Agent(
    model=LlmClient(model="gpt-5"),
    tools=[calculator, *mcp_tools],
    instructions="You are a helpful assistant."
)
```

> Note: This implementation is purely for learning how we _could_ manage our tools if we were to build our own ai agent framework. Caveat being it is not production ready and creates a new connection for each tool call which is **not** performant due to this additional overhead. We could maintain a persistent session if MCP tool usage is frequent but again, this is all just for learning.

## Summary

We built a unified tool interface to simplify tool management in our agent. Previously, functions and their metadata existed as separate entities requiring manual synchronization. We introduced BaseTool, an abstract base class defining the tool contract (name, description, tool_definition, and execute method), and FunctionTool, which wraps regular Python functions as tools while automatically extracting metadata from function signatures and docstrings.

We also implemented a @tool decorator for cleaner syntax. Critically, FunctionTool inspects whether functions need ExecutionContext and conditionally passes it, allowing simple tools to work unchanged while context-aware tools can access execution state.

With our tool abstraction exploration complete, we now turn to the [LLM Communication Layer](llm-communication-layer.md), which manages how information flows between our agent and language models through three components: **LlmRequest** (selects and formats information for the LLM), **LlmClient** (handles API calls), and **LlmResponse** (standardizes responses across different LLM providers).
