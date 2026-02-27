# 📜 Basic Concepts

We will start by covering some basic concepts that we will utilise in our AI agents. To start we will simply be calling through to an LLM (OpenAI) via code.

We willthen introduce [liteLLM](https://www.litellm.ai/) as a wrapper to access LLMs from our code. We will also do some basic evaluation of our results via the [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA).

Let's get started.

## 🔗 Project Resources

All code examples, documentation, and tool definitions discussed in this guide can be found in the repository:
👉 [GitHub: rob-agent](https://github.com/rob212/rob-agent)

### If you wish to follow along, the details of my setup can be found in [prerequisites](/docs/prerequisites.md)

## Concepts

I created ['basic-concepts.py'](https://github.com/rob212/rob-agent/blob/main/src/basic-concepts.py) as a scratch-pad for experimenting with some of the basic concepts we will be using as we develop AI agents. This is the scratchpad I use to try out code and run quikly with `Shift + Enter` to understand how it works. Once satisfied, I build out AI agents using these concepts under the `/src/agents` directory.

## 🏗️ Step 1: The Basic Connection

The foundation of any agent is the ability to communicate with a model. We start by creating a simple interface to send a prompt and receive a string.

To do this I will use the **gpt-5-mini** model by OpenAI, in order to call it via the OpenAI python sdk you will need to [sign-up and create an API Key](https://platform.openai.com/api-keys). Note you will need to put some credit in your account ($5 is the minimum but should be sufficient for this learning).

Ensure you have copied the `.env.example` file to a new `.env` file where you can add your API Keys.

```bash
cp .env.example .env
```

> NOTE: ensure that your `.env` file is referenced in your `.gitignore` file if you are planning to push your code to a git repository, to ensure your keys are not exposed.

## A simple call through to OpenAI

The foundation of any agent is the ability to communicate with a model. We start by creating a simple interface to send a prompt and receive a response.

```python
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
```

Note the use of the 'role' to distinguish the actor that was responsible for the content. This becomes relevant was our context evolves and we need to keep track of the originator of each interaction.

The actual response text is found in `response.choices[0].message.content`. The reason `choices` is a list is that you can request multiple responses using the `n` parameter, though typically you only use the first one.

The OpenAI client supports both the **chat.completions** syntax and the newer **client.responses**. Which was introduced by OpenAI to better reflect the multimodal nature of it's gpt modals. Anthropic, Gemini and other LLMs will all provide a slightly different syntax, increasing the need for boilerplate code in our agent.

Thankfully to decouple this tight dependency there are numerous tools that abstract the underyling calls to major LLMs via a wrapper. [LiteLLM](https://www.litellm.ai/) is one we will use.

## LiteLLM - Wrapper to our model call

You can add the litellm dependency to your project as follows:

```bash
uv add litellm
```

By importing the `completion` function from liteLLM our same call can be made like so:

```python
from litellm import completion
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

## Basic call to LLM via LiteLLM wrapper

response1 = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "My name is Rob"}]
)
print(response1.choices[0].message.content)
```

At the time of writing (Feb 2025) litellm let's you call over 100 LLMs through this unified completion interface, allowing us to easily swap out LLMs easily. Provided you also have an Anthropic API Key configured in your `.env` file you can test this out by simply upding the model to a claude model:

```python
response1 = completion(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "My name is Rob"}]
)
```

## LLM APIs are stateless

When you use ChatGPT or Claude via their web interface, they appear to remember previous conversations. However, LLM APIs are stateless, so each API call is indepenent and has no memory of the previous call. Let's test this out for ourselves:

```python
from litellm import completion
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

response1 = completion(
    model="gpt-5-mini", messages=[{"role": "user", "content": "Hi, my name is Rob"}]
)
print(response1.choices[0].message.content)

response2 = completion(
    model="gpt-5-mini", messages=[{"role": "user", "content": "What is my name?"}]
)
print(response2.choices[0].message.content)
```

Even though we introduced ourselves in the first call, the subsequent second call has no memory of it. Therefore to maintain conversation history we must manage it ourselves.

```python
messages = []

# First Exchange
messages.append({"role": "user", "content": "My name is Rob."})
response = completion(model="gpt-5-mini", messages=messages)
assistant_message1 = response.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_message1})
print(assistant_message1)

# Second Exchange - includes previous conversation history
messages.append({"role": "user", "content": "What is my name?"})
response2 = completion(model="gpt-5-mini", messages=messages)
assistant_message2 = response2.choices[0].message.content
print(assistant_message2)
```

We accumulate all conversation content in the `messages` list and pass the entire history with each call. We ensure that the messages have a corresponding role, the `user` being the human asking the question, and `assistant` being the LLM.

## Structured Output

We are familiar with the natural languages text that LLMs generate and it is great for humans to read, but incovenient for programs to process. Most modern LLM providers (OpenAI, Anthropic, Gemini etc) support 'Structured Output', a feature that allows us to instruct LLMs to generate responses in a defiend format, like JSON.

A common approach to this is by defining your desired output format using Python's [Pydantic](https://docs.pydantic.dev/latest/) library. Pydantic is a library for data validation that lets you define data structures as classes. By inheriting from `BaseModel` you can create a schema that specifies field names and their types.

Let's play with an example `ExtractedInfo` that defines three fields: `name` and `email` are required strings, while `phone` is an optional string that defaults to `None` if not provided.

```python
from pydantic import BaseModel


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
```

We pass our Pydantic model to `response_format`, a parameter [supported by litellm](https://docs.litellm.ai/docs/completion/json_mode). If you inspect the result from our gpt-5-mini model, you should observe the LLM adhering to our defined ExtractedInfo schema:

```python
{"name":"John Smith","email":"john@example.com","phone":"07712345678"}
```

Try experimenting by altering the initial user question, to remove the phone number and running the prompt again. This time the LLM has no possible way of knowing the phone number so excludes it from it's structured response. How does this differ if you remove one of the none optional fields from the content?

Structured output plays a crucual role in agent development. In tool calling, which we'll cover later, the LLM must output which tool to call with which arguments in a structured format. One of the core capabilites of structured output is the ability to convert user intent into appropriate actions.

## Asynchronous calls

It is possible that you may need to process multiple LLM requests simultaneously when developing your agents. This could be due to comparing responses from multiple models, running a multi-agent system or evaludating dozens of problems in a benchmark test.

This is handled as it would be in any Python program via `async/await` and the `asyncio` library. Litellm supports asynchronous calls through the `acompletion` function.

```python
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

```

This example includes measures to prevent two common issues when sending many requests simultaneously. Rate limits from the API provider and transient failures from network issues or server overload. LiteLLM's `num_retries` parameter handles transient failures with automatice exponential backoff.

For rate limiting we can use Python's `asyncio.Semaphore` to limit how many requests run concurrently. By wrapping the `acompletion` call we can ensure that no matter how many tasks are running only a limited number of actual API calls happen at once.

## System Prompt

System prompts are particularly important for agents. Agents act autonomously across multiple steps. System prompts define the behavioral rules that guide all of these decisions. From a context engineering perspective, the system prompt is information that is always included in the context.

Every time the agent calls a tool, analyzes results, or decides on its next action, the system prompt sits at the front of the context, guiding the agent's judgment. This is why the quality of the system prompt determines the quality of the agent's overall behavior.

When wirting a system prompt, you typically provide this context to the LLM with the role of 'system'. For example using the `acompletion` call via liteLLM:

```python
response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
```

You can see some examples of a production agents system prompt as [Anthropic publishes Claude's system promtps](https://docs.anthropic.com/en/release-notes/system-prompts). Note this Claude system prompt is applied when interacting through the claude.ai website or mobile chat interface. When calling Claude through the API, developers write their own system prompts. Claude's system prompt serves 4 main roles: defining the products identity, specifying output format and style, setting boundaries on prohibited behaviour and clarifying the limits of it's knowledge.

## 📏 Measuring our Agents capabilities with Gaia

When working with the non-determinstic nature of LLMs (the same prompt will return different responses each time, due to the probabilstic nature of LLMs) traditional means of software testing are not appropriate. How do we know if our agent is actually getting better? We need a benchmark in place we can measure our agent against to allow us to assess whether each tweak we make to our prompts and logic we are actually making progress.

I intened to make a 'Research Agent', a system that can gather information from multiple sources, analyse findings ans produce comprehensive answers. Therefore we need test cases that have clear, verifiable answers, represent realistic user requests and cover a range of difficulty levels.

Gaia (General AI Assistants) is a dataset that includes a range of a questions that ranges in complexity based on it's level (1, 2, or 3). Level 1 questions generally require no tools, or at most one tool, with no more that n5 steps. Level 2 questions involve more steps, roughly between 5 and 10 and the combinaition of different tools. Level 3 questions are designed for a near-perfect general assistant, requiring arbitrarily long sequences of actions. An example of a potential Level 1 question is this:

> "If Eliud Kipchoge could maintain his marathon pace indefinitely, how long would it take to run to the Moon?"

As a user, for us to answer this question might comprise of the following steps:

1. Google search to determine who 'Eliud Kipchoge' is.
2. Once we have determined Eluid is a Olympian long-distance runner, we may look up Wikipedia to find his marathon pace.
3. We then perform a web search to find the distance from the earth to the moon.
4. We then discover due to the elliptical path of the moon, the distance varies and we need to decide on which measurement(s) to use.
5. We perform a calculation using these numbers to determine the time it would take Eliud to run to the moon.

To solve this our agent would need the ability to call tools to perform calculations, potentially the ability to access the web if it does not possess this information in its initial training data. Our agent also needs to be able to reason over a number of steps.

> The answer by the way, rounded to the nearest 1000 hours is 17,000 hours.

By using a subset of these Gaia questions we can assess the capability of our agent and measure it's accuracy as we iterate our implementation.

### Loading the Gaia dataset

We can obtain the Gaia dataset from [HuggingFace](https://huggingface.co/) (_oversimplification_: think of HuggingFace as the GitHub of AI models). You will need a HuggingFace account and must accept the dataset's terms of use before accessing it.

Visit [https://huggingface.co/datasets/gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) and click the "Agree and access repository" button to accept the dataset's terms of use. You will then need to create an Access Token in your HuggingFace account and add this in your `.env` file under `HF_TOKEN` so that your code can retrieve the dataset.

You will need the `datasets` library from HuggingFace in your dependencies:

```bash
uv add datasets
```

The following code will load the Level 1 problems from the Gaia dataset in Python code which we can use in our agent for evaluating their ability to solve them:

```python
from datasets import load_dataset

level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
print(f"Number of Level 1 problems: {len(level1_problems)}")
print(f" problems: {level1_problems.features}")
```

Examing the output of these, allows us to find the "Eluid Kipchoge" marathon problem:

```python
{
    "task_id": "e1fc63a2-da7a-432f-be78-7c4a95598703",
    "question": (
        "If Eliud Kipchoge could maintain his record-making marathon pace "
        "indefinitely, how many thousand hours would it take him to run the "
        "distance between the Earth and the Moon at its closest approach? "
        "Please use the minimum perigee value on the Wikipedia page for the Moon "
        "when carrying out your calculation. Round your result to the nearest "
        "1000 hours and do not use any comma separators if necessary."
    ),
    "level": 1,
    "final_answer": "17",
    "file_name": "",
    "annotator_metadata": {
        "steps": (
            "1. Googled Eliud Kipchoge marathon pace.\n"
            "2. Found the minimum perigee distance to the Moon.\n"
            "3. Calculated total hours at constant pace.\n"
            "4. Rounded to the nearest 1000 hours."
        ),
        "tools": [
            "Web browser",
            "Search engine",
            "Calculator",
        ],
        "num_tools": 3,
    },
}
```

The 'question' field contains what a user would ask our agent. 'final_answer' is what we will assert to be the correct answer, in this case simply "17." The 'file_name' field indicates whether the problem includes an attached file (empty here means no attachment). The 'annotator_metadata' reveals what the problem creators needed to solve it, which includes a web browser, a search engine, and a calculator.

This metadata is particularly revealing. Even though this is a Level 1 problem, the "easiest" category, the annotators needed three different tools to solve it. They had to search for Kipchoge's marathon pace, look up the Earth-Moon perigee distance on Wikipedia, and perform calculations.

## Building our first agent

Now we have explored a number of concepts, we can create our first agent utilising these. We then test our agent using some Level 1 questions from the Gaia dataset.

I document this in [Building Your First Agent](building-your-first-agent.md).
