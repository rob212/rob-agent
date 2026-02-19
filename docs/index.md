# 🤖 Welcome to Building an AI Agent from scratch

> Building AI agents deliberately.  
> Measuring everything.  
> Improving incrementally.

---

## What This Project Is

This is a research-driven journey into building an AI agent **from first principles**.

No frameworks.  
No abstractions hiding the mechanics.  
No magic.

Each capability is added intentionally and evaluated against the **GAIA benchmark** to measure real performance improvements.

This documentation captures lessons learned and hopefully acts as a useful tutorial for others. 

---

## Why Build From Scratch?

Modern agent frameworks are powerful — but they can abstract away the core elements of all AI Agents have in common. By building and Agent from scratch we will develop an understanding of the fundamentals of AI Agents as a solid foundational learning.


By building each layer manually, we can:

- Understand the mechanics
- Measure the impact
- Avoid accidental complexity
- Create reproducible experiments

---

## What is an AI Agent? 

An AI agent is a program that autonomously decides what actions to take and when to stop based on its current context and goals. At its core, it consists of three elements. 

* 🧠 **Brain** 
* 🔧 **Tools**
* 🔁 **Loop**

   
The LLM serves as the agent's brain. It understands the current situation and decides what to do next. 

Tools are the means of interacting with the external world—web search, code execution, and database access. 

The Loop is the structure that repeats this process until the goal is achieved. 
   
   
The LLM being the "brain" means it doesn't just generate text—it decides which tool to use and when to stop. This is what distinguishes an agent from a plain LLM or traditional software.

---

## Workflows vs Agents

In workflows, developers predefine the execution flow and use LLMs to perform specific steps within that structure. In agents, LLMs dynamically determine their own processes, deciding which actions to take and when to stop.

A workflow is a system where developers explicitly design the sequence of operations, with LLMs executing specific steps within that predefined structure. The key characteristic is predictability: given the same input, the system follows the same path through the workflow.

There are numerous well documented workflow techniques including:

* chaining - connect multiple LLM calls together in a predfined sequence
* router - introduce conditional logic where LLM decides predefined path to take next.

AI Workflows have their place, and like all good software engineering, you should pick the right tool for the job. For this series, I will purposefully focus on AI Agents for learning purposes.

In a professional setting, a hybrid approach is often the best solution. Combining determistic code, workflows and agents enables you to control the more predictable parts of the system, with agentic AI tackling well scoped tasks. This can be explanded upon to have multple sub-agents working with more focussed SLM (Small Language Models) to reduce costs and latency. This is out of scope, but is an interested topic to tackle after your explorations in with AI agents. 

## Evaluation Philosophy

In 2023, Meta and HuggingFace released GAIA (General AI Assistants), a dataset that systematically collects tasks requiring agents. We will use it to track changes in our agent's performance as we add new techniques and capabilities. We'll see firsthand how each component, such as tool use, memory, and planning, actually improves the agent's problem-solving capabilities.

GAIA consists of question-answer pairs. Each question requires multi-step reasoning, web searches, calculations, and more. They're difficult to solve with a single LLM call, and it's not easy to predefine a clear workflow either. These are problems that naturally require an agentic approach.

> Since 2023, model capabilities have improved significantly, with more challenging benchmarks emerging. However, GAIA problems range from straightforward to genuinely difficult even for current models, making it well-suited for learning agent development.

We measure:

- **Accuracy**
- **Solvability rate**
- **Unsolvable failure patterns**
- **Capability gaps**

This project treats agent improvements as experiments — not assumptions.

---

## Roadmap

We progress in stages:

1. [Our 1st agent - Closed LLM baseline](agent-1.md)
2. [Tool integration](tools.md)

Each stage introduces one new capability and measures its effect.

