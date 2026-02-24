# 🤖 Building Your First Agent

We will start by covering some basic concepts that we will utilise in our AI agents. To start we will simply be calling through to an LLM (OpenAI and Anthropic) via code.

We will introduce [liteLLM](https://www.litellm.ai/) as a wrapper to access LLMs from our code. We will also do some basic evaluation of our results via the [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA). 

Let's get started. 

## 🔗 Project Resources
All code examples, benchmark scripts, and tool definitions discussed in this guide can be found in the repository:
👉 [GitHub: rob-agent](https://github.com/rob212/rob-agent)

## 📋 Prerequisites
Before we start building, ensure your development environment is configured for modern Python performance and security.

## ⚡ The uv Package Manager
For this project, we use uv, an extremely fast Python package installer and resolver written in Rust. It replaces pip, pip-tools, and virtualenv while being significantly faster.

To install uv on macOS via Homebrew:

```bash
brew install uv
```

If you are following along with the accompanying [code repo](https://github.com/rob212/rob-agent) you can sync from the root of the project locally to pull all dependencies down: 

```bash
uv sync
```

If starting from scratch you will need to add dependencies individually to your own `pyproject.toml` via `uv add <dependency>` in your terminal. 

## 🛠️ Core Requirements
* Python 3.12+
* OpenAI account (optional to also have an Anthropic account if you wish to use other claude models)
* Hugging face account

---

## Concepts

I created ['concepts.py'](https://github.com/rob212/rob-agent/blob/main/src/concepts.py) as a scratch-pad for experimenting with some of the basic concepts we will be using as we develop AI agents. 




## 🏗️ Step 1: The Basic Connection
The foundation of any agent is the ability to communicate with a model. We start by creating a simple interface to send a prompt and receive a string.

To do this I will use the **gpt-5-mini** model by OpenAI, in order to call it via the OpenAI python sdk you will need to [sign-up and create an API Key](https://platform.openai.com/api-keys). Note you will need to put some credit in your account ($5 is the minimum but should be suffiecient for this learning). 

Ensure you have copied the `.env.example` file to a new `.env` file where you can add your API Keys.

```bash
cp .env.example .env
```

> NOTE: ensure that your `.env` file is referenced in your `.gitignore` file if you are planning to push your code to a git repository, to ensure your keys are not exposed. 

### A simple call through to OpenAI

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

The OpenAI client supports both the **chat.completions** syntax and the newer **client.responses**. Which was introduced by OpenAI to better reflect the multimodal nature of it's gpt modals. Anthropic, Gemeni and other LLMs will all provide a slighly different syntax, increasing the need for boilerplate code in our agent. 

Thankfully to decouple this tight dependency there are numerous tools that abstract the underyling calls to major LLMs via a wrapper. [LiteLLM](https://www.litellm.ai/) is one we will use. 

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

