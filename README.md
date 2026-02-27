# AI Agents From Scratch

This repository documents my journey building AI agents from first principles.

The goal is not to use a framework — but to understand and implement the core building blocks that make modern AI agents work.

## 📚 Start Here – Learning Notes & Guide

This repository includes a structured MkDocs guide that documents my learning journey, experiments, and explanations.

If you're following along or want the step-by-step breakdown behind the code, read the full documentation here:

👉 **[Read the Learning Guide](https://rob212.github.io/rob-agent/)**

The guide acts as:

- Structured learning notes
- Explanations of AI agent concepts
- Setup walkthroughs
- Experiments and findings

The code in this repo supports the material in the docs.

## 🚀 Project Goals

We will progressively build:

- Closed LLM baseline
- Agentic reasoning loop
- Tool usage
- Knowledge base (RAG)
- Web access
- Context engineering strategies
- Memory
- MCP tooling
- Multi-agent systems
- Evaluation via Gaia dataset

## 🧪 Why GAIA?

GAIA contains:

- Multi-step reasoning problems
- Knowledge-dependent questions
- Tool-requiring tasks
- Web-requiring tasks
- Intentionally unsolvable tasks

It allows us to measure:

- Baseline performance
- Impact of tool access
- Impact of web access
- Reduction in "unsolvable" responses
- Structural agent improvements

What cannot be measured cannot be improved, this is particularly important when dealing with the non-determistic nature of LLMs.

## 🏗 Project Structure

```
Rob-agent/
│
├── src/
|   |-- concepts.py
│   ├── agents/
│       ├── agent-1.py
│
├── docs/        # MkDocs documentation site
│
├── mkdocs.yml
├── pyproject.toml
|-- .env.example
|-- LICENSE
└── .gitignore
```

## 🛠 Setup

This project uses uv for dependency management. (docs: https://docs.astral.sh/uv/getting-started/installation/)

#### Install uv (docs: https://docs.astral.sh/uv/getting-started/installation/)

- Homebrew (macOS):

```bash
brew install uv
```

- Verify installation:

```bash
uv --version
```

### 1️⃣ Clone the repository

```bash
git clone https://github.com/rob212/rob-agent.git
cd rob-agent
```

### 2️⃣ Install dependencies

```bash
uv sync
```

### 3️⃣ Add Environment variables

Copy the example env file and set your API keys:

```bash
cp .env.example .env
```

- Open `.env` and provide the necessary keys (e.g., `OPENAI`, `ANTHROPIC`, `HugggingFace` etc).

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HF_KEY=your_huggingface_key
```

## 🤖 Running the agents

In order to run the agents to see the results of how they respond to Gaia dataset questions. Run from the `src` directory in the project, specifying your desired agent. For example to run `agent_1`:

```bash
uv run python -m agents.agent_1
```

This will result in the first 20, Level 1 questions in the Gaia dataset being sent to all of the LLM models in the `MODELS` list in the agent. In the case of agent_1 this includes "gpt-5","gpt-5-mini", "anthropic/claude-sonnet-4-5" and "anthropic/claude-haiku-4-5".

Once completed an Accuracy table will be displayed outlining the model, the "Judged Accuracy" of it's response compared with the accepted answer in the Gaia dataset. Finally the "Judged Solvable" outlines how many of the 20 tasks the LLM thought it could solve without any further information or tools.

| Model                       | Judged Accuracy | Judged Solvable |
| --------------------------- | --------------- | --------------- |
| gpt-5-mini                  | 6/20 (30%)      | 8/20 (40%)      |
| gpt-5                       | 10/20 (50%)     | 12/20 (60%)     |
| anthropic/claude-sonnet-4-5 | 2/20 (10%)      | 9/20 (45%)      |
| anthropic/claude-haiku-4-5  | 2/20 (10%)      | 6/20 (30%)      |

## 🤝 Contributions

This repository is primarily an educational and research exercise.

If you’d like to discuss design decisions or experiments, feel free to open an issue.

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
