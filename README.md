# AI Agents From Scratch

This repository documents my journey building AI agents from first principles.

The goal is not to use a framework — but to understand and implement the core building blocks that make modern AI agents work.

Each stage is evaluated against the GAIA benchmark dataset to measure improvements and validate architectural changes.

This project evolves incrementally, with measurable progress at each step.



## 🚀 Project Goals

We will progressively build:
* Closed LLM baseline
* Agentic reasoning loop
* Tool usage
* Knowledge base (RAG)
* Web access
* Context engineering strategies
* Memory
* MCP tooling
* Multi-agent systems

Each phase is evaluated using GAIA to measure:
* Accuracy
* Solvability
* Capability gaps
* Failure modes

## 🧪 Why GAIA?

GAIA contains:
* Multi-step reasoning problems
* Knowledge-dependent questions
* Tool-requiring tasks
* Web-requiring tasks
* Intentionally unsolvable tasks

It allows us to measure:
* Baseline performance
* Impact of tool access
* Impact of web access
* Reduction in "unsolvable" responses
* Structural agent improvements

What cannot be measured cannot be improved, this is particularly improtant when dealing with the non-determistic nature of LLMs.

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
│   |-- index.md
|
├── mkdocs.yml
├── pyproject.toml
|-- .env.example
|-- LICENSE
└── .gitignore
```

## 🛠 Setup

This project uses uv for dependency management. (docs: https://docs.astral.sh/uv/getting-started/installation/)

####  Install uv (docs: https://docs.astral.sh/uv/getting-started/installation/)
- Homebrew (macOS):
```bash
brew install uv
```
- Verify installation:
```bash
uv --version
```

###  1️⃣ Clone the repository

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

## 📚 Documentation (MkDocs)

This repository includes a full documentation site powered by MkDocs with the Material theme. https://www.mkdocs.org/

The documentation expands on lessons learned, decisions and observations

## ▶️ Running the Docs Locally
```bash
uv run mkdocs serve
```
You should see:
```bash
Serving on http://127.0.0.1:8000/
```
Open that URL in your browser.

The docs live inside:
```bash
/docs
```

Navigation is configured in:
```bash
mkdocs.yml
```

## 📦 Building Static Docs

To build the static site:
```bash 
uv run mkdocs build
```

This generates a site/ folder that can be deployed to:

* GitHub Pages
* Netlify
* Vercel
* Any static host

## 🤝 Contributions

This repository is primarily an educational and research exercise.

If you’d like to discuss design decisions or experiments, feel free to open an issue.

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.