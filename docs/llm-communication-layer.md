# LLM Communication Layer

With `ExecutionContext` storing our execution state and tools unified under `BaseTool`, we need one more piece: a way to communicate with the LLM. This communication layer bridges the gap between our internal data structures and the LLM's API requirements.

## Why a communication layer?

Consider what happens when our agent needs to call the LLM. ExecutionContext contains a list of Events, each holding ContentItems like Messages, ToolCalls, and ToolResults. But LLM APIs expect a specific message format: a list of dictionaries with roles and content. Someone needs to translate between these representations.

We also face the challenge of provider diversity. While we use LiteLLM to abstract away most provider differences, we still need to structure our requests consistently and parse responses into a standard format. Without this layer, translation logic would scatter throughout the Agent class, making it harder to maintain and extend.

Our solution will be to implement three components, working together:

**LlmRequest** packages what we want to send: instructions, conversation contents and available tools. It serves as a staging area where we select and organise information before the API call to our LLM.

**LlmClient** handles the actual API communication. It transforms LlmRequest into the format our LLM APIs expect, makes the actual call and converts the response back.

**LlmResponse** standardises what we receive. Regardless of which provider we use, responses come back in the same format that our Agent can process.

This separation keeps each component focused. LlmRequest knows nothing about API formats. LlmClient knows nothing about ExecutionContext. Each piece does one job well.

## **LlmRequest**: Selecting what to send

`LlmRequest` is the outbound gate from our agent to the LLM. It will hold everything we need for a single LLM call, organised into four components.

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class LlmRequest(BaseModel):
    """Request object for LLM calls."""
    instructions: List[str] = Field(default_factory=list)
    contents: List[ContentItem] = Field(default_factory=list)
    tools: List[BaseTool] = Field(default_factory=list)
    tool_choice: Optional[str] = None
```

The **instructions** field holds system prompt fragments. Rather than a single monolithic prompt, we allow multiple instruction strings that get combined. This flexibility proves useful when instructions come from different sources: base agent instructions, task-specific guidance, or dynamically generated context.

The **contents** field contains the conversation history as `ContentItem` objects: Messages from users and assistants, ToolCalls the LLM requested, and ToolResults from executions. This is the core context the LLM uses to understand the current situation.

The **tools** field lists available tools as BaseTool instances. LlmClient will extract its definitions when building the API request.

The **tool_choice** field controls how the LLM selects tools. Setting it to "auto" lets the LLM decide freely. Setting it to "required" forces tool usage, which becomes important for our structured output later.

Notice what `LlmRequest` **does not** contain: it has no reference to ExecutionContext or Events. Our Agent is responsible for extracting relevant information from ExecutionContext and packaging it into LlmRequest. This separation is intentional. It makes LlmRequest a simple data container while keeping context selection logic in the Agent where it belongs.

This is where we will do our _context engineering_. We will implement a `_prepare_llm_request` in our Agent, which decides what information from ExecutionContext goes into each LlmRequest. For now, we will flatten all events into contents.

## **LlmResponse**: Standardising what we receive

`LlmResponse` is the inbound gate, standardizing LLM responses regardless of which provider generated them.

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class LlmResponse(BaseModel):
    """Response object from LLM calls."""
    content: List[ContentItem] = Field(default_factory=list)
    error_message: Optional[str] = None
    usage_metadata: Dict[str, Any] = Field(default_factory=dict)
```

The **content** field contains what the LLM produced, represented as `ContentItem` objects. A response might include a Message with text, one or more ToolCalls requesting tool execution, or both. Using the same ContentItem types [we defined earlier](execution-context.md#what-happends-during-agent-execution) keeps our data model consistent throughout the system.

The **error_message** field captures failures. When an API call fails due to network issues, rate limits, or invalid requests, we store the error here rather than raising an exception. This allows the Agent to handle failures gracefully, perhaps by retrying or informing the user.

The **usage_metadata** field tracks token consumption. Knowing how many input and output tokens each call uses helps with cost management and debugging. For now, we simply store this information. We will explore monitoring and cost tracking later.

## **LlmClient**: The LLM provider adapter

LlmClient bridges LlmRequest and LlmResponse with actual LLM APIs. Thanks to LiteLLM, which we introduced earlier, we can support multiple LLM providers through a single implementation.

```python
class LlmClient:
    """Client for LLM API calls using LiteLLM."""

    def __init__(self, model: str, **config):
        self.model = model
        self.config = config

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Generate a response from the LLM."""
        try:
            messages = self._build_messages(request) # 1
            tools = [t.tool_definition for t in request.tools] if request.tools else None # 2

            # 3
            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                **({"tool_choice": request.tool_choice}
                   if request.tool_choice else {}),
                **self.config
            )

            return self._parse_response(response)
        except Exception as e:
            return LlmResponse(error_message=str(e))
```

Our constructor takes a model identifier (like "gpt-5-mini") and optional configuration parameters such as temperature or max_tokens.

The `generate` method orchestrates the API call in three steps. First, it builds the messages list from the request. Second, it extracts tool definitions with a simple list comprehension. Third, it calls LiteLLM's acompletion and parses the response. If anything fails, it returns an LlmResponse with the error captured rather than crashing.

TOur `_build_messages` method transforms LlmRequest contents into the message format that LLM APIs expect. The main complexity here is that OpenAI's API requires assistant messages and their tool calls to appear together in a single message object, while we store them as separate ContentItems 🤦‍♂️.

```python
def _build_messages(self, request: LlmRequest) -> List[dict]:
    """Convert LlmRequest to API message format."""
    messages = []

    for instruction in request.instructions:
        messages.append({"role": "system", "content": instruction})

    for item in request.contents:
        if isinstance(item, Message):
            messages.append({"role": item.role, "content": item.content})

        elif isinstance(item, ToolCall):
            tool_call_dict = {
                "id": item.tool_call_id,
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": json.dumps(item.arguments)
                }
            }
            # Append to previous assistant message if exists
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(tool_call_dict)
            else:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call_dict]
                })

        elif isinstance(item, ToolResult):
            messages.append({
                "role": "tool",
                "tool_call_id": item.tool_call_id,
                "content": str(item.content[0]) if item.content else ""
            })

    return messages
```

The logic handles each ContentItem type differently. Messages become standard message objects. ToolCalls get appended to the preceding assistant message's tool_calls array, since our agent stores them consecutively from the same LLM response. If a ToolCall appears without a preceding assistant message, we create one with null content. ToolResults become tool-role messages linked back to their originating call via tool_call_id.

## Parsing responses

... coming soon
