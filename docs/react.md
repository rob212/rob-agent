# ReAct (Reasoning + Acting)

So far we have been focssing on two of the core components to AI Agents, the brain 🧠 (the LLM), and the tools (⚒️). We will now focus more on the third component, the loop 🔁.

We introduce the ReAct pattern by building an AI Agent by chaining multi-step reasoning and actions to solve problems end-to-end and adding robust error handling.

ReAct (Reasoning + Acting) is not a framework but a way of designing agents that mirrors how humans solve problems:

**Assess the situation** -> **decide what information or action is needed** -> **execute that action** -> **observe the results** -> **repeat**

The goals of this section include:

- Building a ToolCallingAgent using ReAct
- Chain multi-step reasoning and actions to solve problems end-to-end
- Add more robust error handling
- Implement a callback system

We will build an Agent that can achieve the above and we'll evaluate it on the Gaia benchmark.

## How ReAct agents work

Let's go back to our original question we explored when [implementing our first agent](building-your-first-agent.md).

"If marathon runner Eliud Kipchoge could maintain his world record pace indefinitely, how long would it take him to reach the Moon?"

Let's deconstruct how we as a human, might answer this question. It could look something like this, "I need to find Kipchoge's marathon pace and the Earth to Moon distance". You would then search for this information, calculate the results and formualte your answer. This cycle of thinking and acting is the basis for how ReAct agents operate.

ReAct stands for Reasoning plus Acting, is the foundational pattern behind most modern AI Agents. The core idea is simple: instead of trying to answer everything at once, the AI agent alternates between reasoning about **what** it needs and **taking action** to get it.

Let's attempt to breakdown how a ReAct agent would tackle the Kipchoge problem:

```
Thought: I need to find Kipchoge's marathon world record pace.
Action: search_web("Kipchoge marathon world record")
Observation: Kipchoge's record is 2:01:09 for 42.195 km (Berlin 2022)

Thought: Now I need the Earth-Moon distance at closest approach.
Action: search_wikipedia("Moon perigee distance")
Observation: The Moon's perigee (closest approach) is 356,500 km

Thought: I can now calculate the time needed.
Action: calculator("356500 / (42.195 / 2.0186) * 1")
Observation: 17,034 hours

Thought: I have enough information to answer.
Final Answer: Approximately 17,000 hours
```

Notice the pattern. At each step, the agent explicitly states what is's thinking, takes concrete action, and observes the result. This cycle repeats until the AI agent has gathered enough information to provide the final answer.

This approach is highly adaptable, if the first search returned unhelpful results, the agent could try a different query. If the calculation seemed wrong, it could verify by searching for additional information. Unlike a traditional software engineering implementation via a fixed script that breaks when something unexpected happens, a ReAct agent adjusts its approach based on what it discovers. (This also brings with it new consequences we need to consider as engineers such as costs spiralling or the concept of unknown paths our system can take and what checks/constraints do we need to apply to priovide safeguards).

We don't execute a predetermined sequence of steps and instead mirror how a human might naturally solve problems. We think, act, observe the results, and decide what to do next. The LLM (the brain 🧠) handles the reasoning, while tools handle the actions. Together, they form a system that can tackle problems neither could solve independently.

## The Completed ReAct AI Agent

Let's work backwards and take a sneak preview about what we are working towards building in this section:

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()

from react_agent.tools import calculator, search_web, search_wikipedia, get_wikipedia_page
from react_agent.models.openai import OpenAILlm
from react_agent.agents.tool_calling_agent_base import ToolCallingAgent

gaia_system_prompt = """
You are a general AI assistant.
I will ask you a question.
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
"""

kipchoge_problem = """
If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon at its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest integer.
"""

async def main():
    tools = [search_web, calculator, search_wikipedia, get_wikipedia_page]
    model = OpenAILlm(model="gpt-5")
    agent = ToolCallingAgent(model=model, tools=tools, instruction=gaia_system_prompt, max_steps=20)
    result, context = await agent.run(kipchoge_problem, return_context=True)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

This implementation includes a number of learnings and iterations of our code, let's start smaller and build up to this.

## A Simpler ReAct Agent to get us started

Let's take a look at a smaller implementation we can start to work on to understand the fundamentals of ReAct.

```python
from react_agent import Agent, LlmClient
from react_agent.tools import calculator, search_web

agent = Agent(
    model=LlmClient(model="gpt-5-mini"),
    tools=[calculator, search_web],
    instructions="You are a helpful assistant"
)

result = await agent.run("What is 1234 * 5678?")
```

The agent receives a user question, decides whther to use tools, executes them if needed and returns a final answer. For the multiplication question above, it would likely call the calculator tool and return "7,006,652". This isn't anything new to our previous AI Agent, but the implementation behind it will allow us to build towards more involved and complex problems.

Let's explor ethe information flow and components that will make this possible.

## Information Flow

Up until now our AI agents job is straightforward: receive the user input, ask the LLM what to do next, execute tools if requested, feed the results back to the LLM and repeat until done. The LLM decides when it has enough information to provide the final answer.

### **ExecutionContext** - Why Messages Alone Aren't Enough

Previously we maintained a list of 'messages' in the form of a Python list. This was the 'conversation history' with the LLM and seemed to work well in our previous agents. However, running an effective agent actually requires tracking much more information:

- **event**: The history of all interactions, including user messages, LLM responses, and tool execution results
- **current_step**: A counter to prevent infinite loops (what if the agent keeps calling tools forever?)
- **execution_id**: A unique identifier for debugging and logging (which execution produced this error?)
- **state**: A scratchpad for dynamic data during execution, such as task progress, intermediate results, or user preferences that tools might need
- **final_result**: A flag to know when execution is complete

If we were to scatter this information across multiple variables and pass them indivually to each method, the core principles of software engineering teaches us the code quickly becomes unwieldy and diffucult to maintain. Every method would rrequire a long parameter list and future refactors would require modifying every method signature.

Our solution will be to consolidate everything into a single container. All methods receive this one container, and when new information is needed, we only modify the container definition. This container will be the **ExecutionContext**, the central storage for all execution states.

Our `ExecutionContext` will sit at the centre of our agent architecture, where all information converges and manages all execution state. When our agent needs to call the LLM, it extracts relevant information from `ExecutionContext` into an `LLMRequest` object. When the LLM responds, the response flows directly back to the `ExecutionContext`. When tools execute, the agent passes `ExecutionContext` to them so they can access any state they need and their results are (you guessed it) recorded back into `ExecutionContext`.

This architecture will simplify our method signatures since every method just receives the context. Secondly, it enables **context propagation**, where the agent can pass `ExecutionContext` to tools that need access to execution state (for example, a tool that checks permissions before acting).

We maintain a distinct seperation between `LlmRequest` and `ExecutionContext`. While the ExecutionContext stores everything, LlmRequest is a _curated subset_ containing only what the LLM needs for a specific call. This allows for effective context engineering: deciding what information to include, what to omit, and how to format it for optimal LLM performance.

## Implementation Roadmap

Here are the components we are going to build in order to implement the information flow we just outlined.

1. ExecutionContext - the central storage that manages all information during execution
2. Tool Abstraction - Improvements on our tool definitions to unify tools under a consistent interace that can receive context
3. LLM Communication Layer - _LlmRequest_: selects information for the LLM. _LlmClient_: handles API calls. _LlmResponse_: standardizes responses
4. Agent - the orchestrator that creates context, coordinates information flow, and implements the _think-act_ loop

Let's start with buidling our [ExecutionContext](execution-context.md).
