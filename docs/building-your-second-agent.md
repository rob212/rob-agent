# Building your second agent

We will define a new `Agent` class that comprises of all the components we explored in our [Loop learnings](react.md). It will coordinate information flow between the LLM and tools to solve problems step by step.

Our Agent initializes with `_setup_tools` then `run` creates an `ExecutionContext` and repeatedly calls `step` until completion. Each `step` performs one think-act cycle; `think` calls the LLM to decide what to do next and `act` executes any requested tools, updating our `ExecutionContext` with each iteration.

Let's implement these five methods in a logical order:

1. `_setup_tools`: Registers the tools the agent will use, converting the list into a dictionary for efficient lookup
2. `run`: The entry point that receives user requests, manages the execution loop and returns the final result
3. `step`: Performs one complete think-act cycle, updating ExecutionContext with each iteration
4. `think`: Calls the LLM to analyze the situation and decide the next action
5. `act`: Executes the tools selected by the LLM and captures their results

Our goal at the end of this section is to have a working ai agent that can solve multi-step problems. We'll also add structured output capabilities, allowing the agent to return Pydantic models instead of free-form text when your application requires predictable data formats.

## Agent class

We'll begin with the `Agent` class constructor and the `_setup_tools` method. The constructor receives all the components we've built previously and stores them for use during execution.

```python
# src/agents/agent_2.py

class Agent:
    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] = None,
        instructions: str = "",
        max_steps: int = 10,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.tools = self._setup_tools(tools or [])
```

The parameters are as follows:

- model: The LlmClient instance that handles LLM communication
- tools: List of BaseTool instances the agent can use
- instructions: System prompot that defines the agent's behaviour
- max_steps: Safetly limit to prevent infinite loops when the agent keeps calling tools without reaching a conclusion

Our `_setup_tools` method prepares the tools list for use. For now, it simply returns the list unchanged, but this method serves as an extension point for future enhancements like adding default tools.

```python
# src/agents/agent_2.py

def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
    return tools
```

With our basic structure in place, let's implement the `run` method that orchestrates the entire execution.

## The run method

The `run` method is the entry point for agent execution. It creates the execution environment, manages the think-act loop, and returns the result.

To give callers access to the full execution trace, we define AgentResult that bundles the final output with its ExecutionContext. This allows inspection of every step the agent took, which proves invaluable for debugging and analysis.

```python
@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: str | BaseModel
    context: ExecutionContext
```

The method first checks if an ExecutionContext was provided, creating one if not. It then wraps the user's input in an Event and adds it to the context. The main loop repeatedly calls step() until either a final result is obtained or the maximum step limit is reached. After each step, we check if the last event represents a final response and extract the result if so.

```python
# src/agents/agent_2.py

async def run(
    self,
    user_input: str,
    context: ExecutionContext = None
) -> str:
    # Create or reuse context
    if context is None:
        context = ExecutionContext()

    # Add user input as the first event
    user_event = Event(
        execution_id=context.execution_id,
        author="user",
        content=[Message(role="user", content=user_input)]
    )
    context.add_event(user_event)

    # Execute steps until completion or max steps reached
    while not context.final_result and context.current_step < self.max_steps:
        await self.step(context)

        # Check if the last event is a final response
        last_event = context.events[-1]
        if self._is_final_response(last_event):
            context.final_result = self._extract_final_result(last_event)

    return AgentResult(output=context.final_result, context=context)
```

The helper methods handle completion detection and result extraction. The `_is_final_response` method checks whether an event represents a final answer by examining its contents. An event is final when it contains neither tool calls nor tool results, meaning the LLM provided a direct answer. The `_extract_final_result` method iterates through the event's content to find the assistant's message.

```python
# src/agents/agent_2.py

def _is_final_response(self, event: Event) -> bool:
    """Check if this event contains a final response."""
    has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
    has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
    return not has_tool_calls and not has_tool_results

def _extract_final_result(self, event: Event) -> str:
    for item in event.content:
        if isinstance(item, Message) and item.role == "assistant":
            return item.content
    return None
```

Usage is straightforward. Access the answer via `result.output`, and examine execution details through `result.context` when needed.

```python
result = await agent.run("What is 1234 * 5678?")
print(result.output)                      # "7006652"
print(result.context.current_step)

```

## The step method

The `step` method performs one complete think-act cycle. It prepares a request for the LLM, gets the LLM's decision, and executes any tools the LLM requests.

The method starts by calling `_prepare_llm_request` to package the current context into a format suitable for the LLM. It then calls `think` to get the LLM's response and wraps that response in an Event, recording it in the context. If the response contains tool calls, the method calls act() to execute them and record the results as another Event. Finally, it increments the step counter.

```python
# src/agents/agent_2.py

async def step(self, context: ExecutionContext):
    # for visibility as we experiment and learn
     print(f"[Step {context.current_step + 1}]")
    # Prepare what to send to the LLM
    llm_request = self._prepare_llm_request(context)

    # Get LLM's decision
    llm_response = await self.think(llm_request)

    # Record LLM response as an event
    response_event = Event(
        execution_id=context.execution_id,
        author=self.name,
        content=llm_response.content,
    )
    context.add_event(response_event)

    # Execute tools if the LLM requested any
    tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
    if tool_calls:
        tool_results = await self.act(context, tool_calls)
        tool_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=tool_results,
        )
        context.add_event(tool_event)

    context.increment_step()
```

The `_prepare_llm_request` method extracts information from ExecutionContext and packages it into an LlmRequest. It flattens all events into a list of content items, combines them with the agent's instructions and available tools, and sets the tool choice to "auto" so the LLM can decide whether to use tools.

```python
# src/agents/agent_2.py

def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
    # Flatten events into content items
    flat_contents = []
    for event in context.events:
        flat_contents.extend(event.content)

    return LlmRequest(
        instructions=[self.instructions] if self.instructions else [],
        contents=flat_contents,
        tools=self.tools,
        tool_choice="auto" if self.tools else None,
    )
```

Notice the separation between ExecutionContext and LlmRequest. ExecutionContext holds the complete execution history, while LlmRequest contains only what we choose to send to the LLM. Currently, we send everything, but this separation is where context engineering happens. You could summarize old messages, omit irrelevant tool results, or inject additional information, all by modifying how you build the LlmRequest without touching the original context.

## Think and act methods

The `think` and `act` methods handle the two core operations in each step: getting the LLM's decision and executing the requested tools.

The `think` method is intentionally simple. It takes the prepared LlmRequest and passes it to the LlmClient, returning whatever response comes back. All the complexity of building the request happens `in _prepare_llm_request`, and all the complexity of parsing the response happens in LlmClient. This keeps `think` focused on a single responsibility.

```python
# src/agents/agent_2.py

async def think(self, llm_request: LlmRequest) -> LlmResponse:
    return await self.model.generate(llm_request)
```

The `act` method executes the tools that the LLM requested. It first builds a dictionary mapping tool names to tool instances for efficient lookup. Then it iterates through each ToolCall, retrieves the corresponding tool, executes it with the provided arguments, and collects the results. The method passes ExecutionContext to each tool execution, allowing tools to access execution state if they need it. Each execution is wrapped in a try-except block, capturing failures as error results rather than crashing the entire agent.

```python
# src/agents/agent_2.py

async def act(
    self,
    context: ExecutionContext,
    tool_calls: List[ToolCall]
) -> List[ToolResult]:
    tools_dict = {tool.name: tool for tool in self.tools}
    results = []

    for tool_call in tool_calls:
        if tool_call.name not in tools_dict:
            raise ValueError(f"Tool '{tool_call.name}' not found")

        tool = tools_dict[tool_call.name]

        try:
            output = await tool(context, **tool_call.arguments)
            results.append(ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="success",
                content=[output],
            ))
        except Exception as e:
            results.append(ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="error",
                content=[str(e)],
            ))

    return results
```

When a tool execution fails, the error message becomes part of the conversation history. The LLM sees this failure in the next step and can adapt its approach, perhaps trying a different tool or rephrasing its query. This graceful error handling is one of the advantages of the ReAct pattern: the agent can recover from failures rather than stopping entirely.

#### ... More coming soon, this is still very much a work in progress and will likely contain errors as I experiment.
