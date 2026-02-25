# %%

## Simple call through to OpenAI via the openai python library


import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "My name is Rob"}]
)
print(response.choices[0].message.content)

## --------------------------------------------------------------

## Introducing liteLLM as a wrapper around or LLM provider call 

from litellm import completion
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

## Basic call to LLM via LiteLLM wrapper

response1 = completion(
    model="gpt-5-mini", messages=[{"role": "user", "content": "My name is Rob"}]
)
print(response1.choices[0].message.content)

## ---------------------------------------------------------------

## LLMs are stateless, therefore you need tp accumulate conversation content an provide it
## to subsequent LLM prompts

messages = []

# First Exchange
messages.append({"role": "user", "content": "My name is Rob."})
response2 = completion(model="gpt-5-mini", messages=messages)
assistant_message1 = response2.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_message1})
print(assistant_message1)

# Second Exchange - includes previous conversation history
messages.append({"role": "user", "content": "What is my name?"})
response3 = completion(model="gpt-5-mini", messages=messages)
assistant_message2 = response3.choices[0].message.content
print(assistant_message2)


## ---------------------------------------------------------------

## Structured Output - modern LLM support Structured Output which allows you to
## specifiy the LLM's response format, e.g. JSON.
## Here we use Pydantic (Python library for data validation) to define our desired data structure
## that we want the LLM to adhere to in it's response. In this case the LLM should respond
## in a JSON format.

from pydantic import BaseModel
from litellm import completion


class ExtractedInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None


response4 = completion(
    model="gpt-5-mini",
    messages=[
        {
            "role": "user",
            "content": "My name is John Smith, my email is john@example.com, and my phone number is 07712345678",
        }
    ],
    response_format=ExtractedInfo,
)

result = response4.choices[0].message.content
print(result)


## ---------------------------------------------------------------

## Asynchronous calls - You may which to process multiple LLM requests simultaneously.
## Sending multiple requests sequentially will increase the total execution time. We can handle
## asynchronous programming with async/await syntax and te asyncio library. LiteLLM supports
## this through the `acompletion` function

## When sending many requests simultaneously, two issues can arise: rate limits from the API provider
## and transient failures from network issues or server overload. LiteLLM's `num_retries` parameter
## handles transient failures with automatice exponential backoff. For rate limiting we can use Python's
## asyncio.Semaphore to limit how many requests run concurrently. By wrapping the `acompletion` call
## we can ensure that no matter how many tasks are runningm only a limited number of actual API calls happen
## at once.

import asyncio
from litellm import acompletion

# Limit to 10 concurrent requests
semaphore = asyncio.Semaphore(10)


async def call_llm(prompt: str) -> str:
    """LLM call with rate limiting and automatic retry."""
    async with semaphore:
        response = await acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            num_retries=3,  # Automatic rety with exponential backoff
        )
        return response.choices[0].message.content


# Even if we had 100 prompts, only 10 API calls run at a time
prompts = [
    "What is 2 + 2?",
    "What is the capital of Japan?",
    "Who wrote Romeo and Juliet?",
]

# Execute all requests concurrently
## `return_exceptions=True` argument prevents a single failure from cancelling all other tasks,
## Instead, exceptions are returned as values in the results list, allowing us to handle failures gracefully
## while still getting results from successful calls.
tasks = [call_llm(p) for p in prompts]
results = await asyncio.gather(*tasks, return_exceptions=True)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")


## -----------------------------------------------------

## Importing the GAIA dataset from HuggingFace to use to evaulate our agent.
## It consists of a collection of level 1, 2 and 3 tasks, each requiring more complexity in an agent
## and it's available tolling to solve. The dataset contains questions with expected answers with to assess our agent's
## response against.

from datasets import load_dataset

level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
print(f"Number of Level 1 problems: {len(level1_problems)}")
print(f" problems: {level1_problems.features}")


## -----------------------------------------------------

# Implementing a web search tool.
# An essential capability for a research agent is the ability to access the web for up to date information that may not
# be part of our LLM's training data.

# we will achieve this via [Tavily](https://www.tavily.com/). Tavily allows for up to 1,000 free API calls per month.

import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))


def search_web(query: str, max_results: int = 2) -> list:
    response = tavily_client.search(query, max_results=max_results)
    return response.get("results")


search_web("Kipchoge's marathon world record")


## -------------------------------------------------------

# Expanding a web search with additional search options and basic error handling


def search_web(
    query: str,
    max_results: int = 2,
    topic: str = "general",
    time_range: str | None = None,
    country: str | None = None,
) -> list | str:
    """Search the web for the given query."""
    try:
        response = tavily_client.search(
            query,
            max_results=max_results,
            topic=topic,
            time_range=time_range,
            country=country,
        )
        return response.get("results")
    except Exception as e:
        return f"Error: Search failed - {e}"


results = search_web(
    query="Kipchoge's marathon world record",
    topic="news",
    time_range="year",
    country="united kingdom",
)

print(results)

## -------------------------------------------------------------

# Python `inspect` in order to extract a functions details as a building block for us
# creating a utility function that automatically converts a Python function into a tool
# definition

import inspect


def example_tool(input_1: str, input_2: int = 1):
    """docstring for example_tool"""
    return


print(f"function name: {example_tool.__name__}")
print(f"function docstring: {example_tool.__doc__}")
print(f"function signature: {inspect.signature(example_tool)}")


## -------------------------------------------------------------

# Our utility function to automatically define out Structured Output tool definition for a given
# function is as follows.


def function_to_input_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [param.name for param in signature.parameters.values()]

    return {
        "type": "object",
        "properties": parameters,
        "required": required,
    }


def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def function_to_tool_definition(func) -> dict:
    return format_tool_definition(
        func.__name__, func.__doc__ or "", function_to_input_schema(func)
    )


# We test our new `function_to_tool_definition()` with our `search_web` function.

search_tool_definition = function_to_tool_definition(search_web)
print(search_tool_definition)


# This results in the following Python dictionary
# {'type': 'function', 'function': {'name': 'search_web', 'description': 'Search the web for the given query.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}, 'max_results': {'type': 'integer'}, 'topic': {'type': 'string'}, 'time_range': {'type': 'string'}, 'country': {'type': 'string'}}, 'required': ['query', 'max_results', 'topic', 'time_range', 'country']}}}
