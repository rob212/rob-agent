# ExecutionContext

We need ExecutionContext to avoid scattering state across multiple variables and to simplify how components share information. Now, let's determine what we need to store by examining what actually happens when an agent runs.

## What happends during agent execution?

Recalling our Kipchoge problem, when an agent solves this problem, the following of series of events is likely:

1. User asks the question: "How long would it take Kipchoge to run to the Moon?"
2. LLM decides it needs information → requests `search_web` tool.
3. Tool executes and returns: "Kipchoge's record is 2:01:09 for 42.195km."
4. LLM needs more information → requests `search_wikipedia` tool.
5. Tool executes and returns: "Moon's perigee is 356,500 km'.
6. LLM can now calculate → requests calculator tool.
7. Tool executes and returns: "17034"
8. LLM provides the final answer: "Approximately 17,000 hours."

Examing this sequence, we can identify three distinct types of occurences:

**Messages** are text exchanges in the conversation. The user's initial question and the LLM's final answer are both examples of messages. Each message has a clear role (user, assistant, or system) and text content.

**Tool Calls** occur when the LLM decides to use a tool. The LLM specifies which tool to call and with what arguments. In step 2 above, the LLM requests `search_web` with the query "Kipchoge marathon world record pace".

**Tool Results** capture what happens when tools execute. They include the tool's outpuy and whether execution succeeded or failed. Step 3 would be a successful result containing Kipchoges' marathon time.

Let's define these as data types in Python using Pydantic.

```python
# react_agents/types/contents.py

from typing import Literal, Union
from pydantic import BaseModel

class Message(BaseModel):
    """A text message in the conversation."""
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str

class ToolCall(BaseModel):
    """LLM's request to execute a tool."""
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict

class ToolResult(BaseModel):
    """Result from tool execution."""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list

ContentItem = Union[Message, ToolCall, ToolResult]

```

The `type` field in each class serves as a discriminator, making it easy to identify what kind of content we're dealing with. The `tool_call_id` links a _ToolResult_ bac to it's originating _ToolCall_ which is necessary when multiple tools execute in parallel.

These content tpyes capture _what_ happened, but for debugging and analysis, we also need to know _who_ produced each piece of content and _when_. We wrap content items with metadata using an `Event` class.

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Event(BaseModel):
    """A recorded occurrence during agent execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str  # "user" or agent name
    content: List[ContentItem] = Field(default_factory=list)

```

The `execution_id` groups all events from a single agent run, enabling us to trace an entire problem-solving session. The `author` field distinguishes between user input and agent actions. Here's an illustation of how one step from Kipchoge problem could look as an Event.

```python
Event(
    execution_id="abc-123",
    author="research_agent",
    content=[ToolCall(
        tool_call_id="call-1",
        name="search_web",
        arguments={"query": "Kipchoge marathon world record"}
    )]
)
```

Each step in the our agent's execution becomes an Event, creating a complete audit trail that proves invaluable when debugging.

## Implementing ExecutionContext

Now we know what to store, implementing our `ExecutionContext` is fairly straightforward.

```python
# react_agents/models/execution_context.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ..types.events import Event.  # location of our previously defined Event
from ..types.contents import Message # location of our previously defined Message
from pydantic import BaseModel
import uuid

@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None

    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)

    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1

```

Each field serves a specific purpose:

- `execution_id`: Unique identifier for this execution session, automatically generated
- `events`: Chronological list of all Events that occur during execution
- `current_step`: Counter to prevent infinite loops (agent stops after max_steps)
- `state`: Flexible key-value store for custom data that tools might need, such as API configurations or intermediate results
- `final_result`: Holds the agent's answer when execution completes

> What are the `dataclass` and `field` imports? These are nothing specific to AI Agents and are Python features.
> `dataclass` is a decorator to indicate a class primarily exists to store data. It automatically generates boilerplate methods that you would otherwise have to write manually such as **init**, **repr** and **eq**.
> `field` is a function used when a simple assigment isn't enough when working with mutable types.

`ExecutionContext` can also be passed to tools that need access to the execution state. A tool might read from state or examine previous events in the execution history. For this to work, we need a consistent way to define tools that can optionally receive context. To unify the function-based tools under a common interface that supports this capability, let's start our implementation of [Tool Abstraction](tool-abstraction.md).
