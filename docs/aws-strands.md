# AWS Strands & AgentCore

So far we have learned about AI agents by manually building them from scratch. The 'brain' of our agent is an LLM that we access via the wrapper LiteLLM in order to decouple our implementation and allow us to easily swap out or use LLMs from multiple providers.

The 'tools' and 'loop' proportion has involved us building our own framework in Python. Most of our implementation is the generic infrastructure that is common to all modern agentic AI. Building this has been useful for our understanding, and allows for full controlled of our bespoke agentic system, however, maintaining this in a professional real-world environment is cumbersome.

It is akin to building a modern web application using html, css and vanilla javascript, rolling our own authentication and routing etc. Modern frameworks such as Next.js or Angular exist to allow us to quickly built robust application but focussing on business value and features.

This is no different in the world of AI Agents. Numerous platforms exist that offer opiniated solutions to the common boilerplate such as tooling and the agentic loop. Some of these include [CrewAi](https://crewai.com/), [Ag2](https://www.ag2.ai/), [Google Agent Development Kit](https://google.github.io/adk-docs/) and [AWS Strands](https://strandsagents.com/latest/).

I will be exploring with AWS Strands and the [AWS Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/).

**AWS Strands** and **AWS Bedrock AgentCore** are two complementary parts of the AWS generative AI ecosystem designed to move AI agents from simple experiments into secure, enterprise-grade production environments.

To understand how they work together, it helps to use an analogy: Strands is the "Brain" (the logic), while AgentCore is the "Operating System" (the infrastructure).

## What is AWS Strands?

AWS Strands (specifically the Strands Agents SDK) is an open-source, model-driven framework for building AI agents. Unlike traditional frameworks that require you to hardcode complex "if-then" workflows, Strands relies on the reasoning capabilities of Large Language Models (LLMs) to plan and execute tasks.

Key Features:

- The Agentic Loop: It manages the cycle where an agent receives a goal, reasons about it, selects a tool, observes the result, and repeats until the task is done.
- Model Agnostic: While it works natively with Amazon Bedrock (e.g., Claude, Amazon Nova), it also supports external models like OpenAI, Gemini, or even local models via Ollama.
- Tool-First Design: It makes it trivial to turn any Python function into an agent tool using a simple @tool decorator. It also supports the Model Context Protocol (MCP), allowing agents to use thousands of pre-made tools.
- Minimal Boilerplate: You can define a fully functional agent in just a few lines of code by providing a model, a prompt, and a list of tools.

## What is AWS Bedrock AgentCore?

Amazon Bedrock AgentCore is a fully managed platform for deploying and operating these agents at scale. This includes guardrails, monitoring, authentication and memory.

Key Components:

- Runtime: A serverless environment (similar to Lambda but built for agents) that can run tasks for up to 8 hours with complete session isolation.
- Memory: Managed short-term and long-term memory so agents can remember users across different sessions.
- Gateway: A secure "front door" that connects agents to company APIs and databases without manual glue code.
- Identity & Policy: Manages what an agent is allowed to do (e.g., "Agent X can read S3 but cannot delete files") using AWS IAM and Cedar policies.

## Getting started with Strands

I will be using Python and `uv` to experiment with Strands.

First we initiate a new project with 'uv' to hold our code:

```bash
uv init strands-play-1
cd strands-play-1
```

Now we add some of the core dependencies to get started:

```bash
uv add strands-agents strands-agents-tools strands-agents-builder boto3
```

### Adding AWS Credentials

In order to access an LLM via AWS we need an AWS Account and validAIM Credentials.

In order to create an Access key, log into your AWS Account via the Management Console:

1. Navigate to **IAM** -> **Users** -> Select your user
2. Go to **Security credentials** tab
3. Click **Create access key**
4. Choose **Command Line Interface (CLI)** as use case
5. Copy both the Access Key ID and Secret Accese Key

#### Store your credentials in you AWS Profile

> Note there are numerous ways you can configure your credentials, see the [official strands documentation](https://strandsagents.com/latest/documentation/docs/user-guide/quickstart/python/#configuring-credentials) for details. I am taking an opininated approch of an IAM Access Key.

Create a new named profile:

```bash
aws configure --profile strands-demo
```

This creates/updates ~/.aws/credentials:

```bash
[default]
aws_access_key_id = YOUR_OLD_KEY
aws_secret_access_key = YOUR_OLD_SECRET

[strands-demo]
aws_access_key_id = YOUR_NEW_KEY
aws_secret_access_key = YOUR_NEW_SECRET
```

And ~/.aws/config:

```bash
[default]
region = us-east-1

[profile strands-demo]
region = us-east-1
```

#### Create a `.env` file for your project

In the root of your 'strands-play-1' directory:

```bash
touch .env
```

Add the following to your `.env` file:

```bash
AWS_ACCESS_KEY_ID=your-new-access-key
AWS_SECRET_ACCESS_KEY=your-new-secret-key
AWS_DEFAULT_REGION=us-east-1
```

**Ensure you add `.env` to your `.gitignore`**:

```bash
echo ".env" >> .gitignore
```

Now we add the `dotenv` dependency to our project so we can easily access our credentials in code:

```bash
uv add dotenv
```

## Creating our first Simple Agent

Now we can create our first agent using Strands. Create a new `agent_1.py` file with the following:

```python
from strands import Agent
from dotenv import load_dotenv

# laod credentials from our .env file
load_dotenv()

# Create a simple agent with default settings, e.g. Claude 3 Sonnet as the model
agent = Agent()

# Ask the agent a simple question
response = agent("Tell me about agentic AI")
print(response)
```

You can run this via 'uv':

```bash
uv run agent_1.py
```

This should print the output of our simple agent to your terminal, giving you some kind of response liek this (remember models are non-deterministic so every response will differ):

```md
Agentic AI refers to artificial intelligence systems designed to act autonomously as independent agents, capable of making decisions, taking actions, and pursuing goals with minimal human oversight. Here are the key aspects:

## Core Characteristics

**Autonomy**: These systems can operate independently, making decisions without constant human input or supervision.

**Goal-directed behavior**: They work toward specific objectives, adapting their strategies based on changing conditions.

**Environmental interaction**: Agentic AI can perceive, analyze, and respond to its environment in real-time.

**Proactive decision-making**: Rather than just responding to prompts, they can initiate actions and plan ahead.

## Current Applications

- **AI assistants** that can manage complex, multi-step tasks
- **Autonomous vehicles** navigating traffic and making driving decisions
- **Trading algorithms** that buy and sell securities based on market conditions
- **Game-playing AI** like those that master chess or Go
- **Robotic systems** in manufacturing and logistics

## Technical Approaches

- **Reinforcement learning** for goal optimization
- **Multi-agent systems** where multiple AI agents interact
- **Large language models** enhanced with planning and tool-use capabilities
- **Hybrid architectures** combining different AI techniques

## Benefits and Concerns

**Potential benefits**: Increased efficiency, 24/7 operation, handling complex tasks at scale

**Key concerns**: Alignment with human values, accountability for decisions, potential for unintended consequences, and ensuring appropriate human oversight

Agentic AI represents a significant step toward more capable and independent AI systems, though it also raises important questions about control, safety, and responsibility.Agentic AI refers to artificial intelligence systems designed to act autonomously as independent agents, capable of making decisions, taking actions, and pursuing goals with minimal human oversight. Here are the key aspects:

## Core Characteristics

**Autonomy**: These systems can operate independently, making decisions without constant human input or supervision.

**Goal-directed behavior**: They work toward specific objectives, adapting their strategies based on changing conditions.

**Environmental interaction**: Agentic AI can perceive, analyze, and respond to its environment in real-time.

**Proactive decision-making**: Rather than just responding to prompts, they can initiate actions and plan ahead.

## Current Applications

- **AI assistants** that can manage complex, multi-step tasks
- **Autonomous vehicles** navigating traffic and making driving decisions
- **Trading algorithms** that buy and sell securities based on market conditions
- **Game-playing AI** like those that master chess or Go
- **Robotic systems** in manufacturing and logistics

## Technical Approaches

- **Reinforcement learning** for goal optimization
- **Multi-agent systems** where multiple AI agents interact
- **Large language models** enhanced with planning and tool-use capabilities
- **Hybrid architectures** combining different AI techniques

## Benefits and Concerns

**Potential benefits**: Increased efficiency, 24/7 operation, handling complex tasks at scale

**Key concerns**: Alignment with human values, accountability for decisions, potential for unintended consequences, and ensuring appropriate human oversight

Agentic AI represents a significant step toward more capable and independent AI systems, though it also raises important questions about control, safety, and responsibility.
```

## Summary

Congratulations, you have just setup and run your first simple AI Agent via Strands.

As you can see it has extremely little boiler code and obfuscates almost all of the details via a simple interface.
