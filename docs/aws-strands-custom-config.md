# Agent with Custom Configuration

Now let's create a production-ready agent with custom configuration. This example shows how to build a specialized research assistant with a clear identity and capabilities.

The `name` parameter helps with debugging and logging in multi-agent systems. The `system_prompt` defines the agent's role, expertise, and behavioral guidelines—think of it as the agent's job description and operating instructions.

The `tools` parameter equips the agent with specific capabilities: calculator for mathematical operations, time for temporal queries, and python_code for executing code.

Finally, `model_id` explicitly specifies which AI model to use, giving you control over performance, cost, and capabilities. This pattern—clear identity, focused role, appropriate tools, and explicit model selection—is the foundation for building reliable, specialized agents.

```python
from strands import Agent
from strands_tools import calculator
from dotenv import load_dotenv

# laod credentials from our .env file
load_dotenv()

# Create a specialized agent with tools and custom prompt
research_agent = Agent(
    name="research_assistant",
    system_prompt="""You are a research specialist who provides
    factual, well-sourced information. Always cite your sources.""",
    tools=[calculator],
    model="us.anthropic.claude-sonnet-4-20250514-v1:0"
)

# Use the agent
result = research_agent("Calculate the compound interest on £10,000 at 5% for 10 years")
print(result)
```

Notice that we utilise the `strands_agents_tools` package to easily import from a community-supported tools. See the [Tools documention](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/tools/community-tools-package/#available-tools) for further details.

Previosuly when we implemented our our AI Agent Framework from scratch we had to build the [logic to define our own tool decorator](tool-definitions.md). Strands has already done this for us and also allows for MCP tools which we can conver later.
