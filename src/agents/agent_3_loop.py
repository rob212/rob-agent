import os
import json
from tavily import TavilyClient
from dotenv import load_dotenv, find_dotenv

from utilities.tool_definition import simple_agent_loop
from tools.calculator import calculator

load_dotenv(find_dotenv())

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

SYSTEM_PROMPT = """ You are a helpful assistant. 
Use the search tool when you need current information."""


## Search Web Tool function definition
def search_web(
    query: str,
    max_results: int = 2,
) -> list | str:
    """Search the web for the given query."""
    try:
        response = tavily_client.search(
            query,
            max_results=max_results,
        )
        return response.get("results")
    except Exception as e:
        return f"Error: Search failed - {e}"


## Define our Tools via our utility functions
tools = [search_web, calculator]

QUESTION = "Who won gold medal in the womans curling at 2026 winter olympics?"

# start our agent loop passing the system prompt, user question and list of
# availbale tools to our LLM, in this instace gpt-5-mini
result, context = simple_agent_loop(SYSTEM_PROMPT, QUESTION, tools, "gpt-5-mini")

print(result)
# print out the final context that took place in the agent loop to learn about how the LLM interacted with
# the tools we provided to it.
print(" ------------ context --------------------")
print(json.dumps(context, indent=2, default=str))
