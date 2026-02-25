# Prerequisties

Outlining my setup for this project, should you wish to replicate it.

- Code Editor: [VSCode](https://code.visualstudio.com/)
- Python version: 3.12.10
- Node version: 20.19.1

## ⚡ The uv Package Manager

For this project, I am using [uv](https://docs.astral.sh/uv/), an extremely fast Python package installer and resolver written in Rust. It replaces pip, pip-tools, and virtualenv while being significantly faster.

To install uv on macOS via Homebrew:

```bash
brew install uv
```

If you are following along with the accompanying [code repo](https://github.com/rob212/rob-agent) you can sync from the root of the project locally to pull all dependencies down:

```bash
uv sync
```

If starting from scratch you will need to add dependencies individually to your own `pyproject.toml` via `uv add <dependency>` in your terminal.

## 🛠️ Accounts you will require

- OpenAI account (optional to also have an Anthropic account if you wish to use other claude models)
- Hugging face account

## 🗒️ Running Python samples as an interactive notebook inside VSCode - _Optional_

This project uses a regular `.py` files (e.g. concepts.py) to experiment with code as we learn. In order to speed up feedback I use an interactive notebook inside VSCode. By adding special cell markers (# %%), the file behaves like a Jupyter notebook while remaining a standard Python script.

This allows you to run small sections of code independently and see the output immediately in a note view.

If you wish to do the same, you will need to install the following VSCode extensions:

- [Python (Microsoft)](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Jupyter (Microsoft)](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

These enable notebook-style execution inside .py files.

Ensure you have `ipykernel` installed as a dev dependency:

```bash
uv add --dev ipykernel
```

Why? `ipykernel` allows your virtual environment to act as a Jupyter kernel so VSCode can execute code interactively.

For any file that you wish to interact with as a notebook, ensure you have added `# %%` to the top of the file. (See `concepts.py` as an example). This is a _cell marker_ which instructs the VSCode Jupyter extension to treat any code below this as an executable notebook cell.

Now you can highight a portion of code in this `concepts.py` file and press `Shift + Enter` on your keyboard. Doing so will open an interactive notes panel for you to see the results of the code block running quickly.

### 🏁 Getting started with our first agent

Now let's move onto experimenting with our [first simple agent](/docs/agent-1.md)
