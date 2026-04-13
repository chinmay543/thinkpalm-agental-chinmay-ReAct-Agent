# ReActPro — Multi-Tool Smart AI Assistant

A Python **ReAct-style agent** that answers user queries by orchestrating **OpenAI tool calls** across three capabilities: a **safe calculator**, **web search** (DuckDuckGo), and **persistent key–value memory**.

**Repository:** [chinmay543/thinkpalm-agental-chinmay-ReAct-Agent](https://github.com/chinmay543/thinkpalm-agental-chinmay-ReAct-Agent)

## How it works

1. The user sends a natural-language message.
2. `run_agent` in `src/react_agent/agent.py` calls the OpenAI **Chat Completions** API with `tools=TOOL_SCHEMAS` and `tool_choice="auto"`.
3. The model may emit **function calls** (`calculator`, `web_search`, `memory_read`, `memory_write`).
4. Each call is executed via `dispatch_tool` in `src/react_agent/tools.py`; results are returned as **tool** messages and the model continues until it produces a final assistant message without further tool calls (or until a max round limit).

This follows the **Reason + Act** pattern: the model reasons about which tool to use, acts (calls tools), observes results, and repeats as needed.

## Project layout

| Path | Purpose |
|------|--------|
| `src/react_agent/agent.py` | OpenAI loop, system prompt, tool-call handling |
| `src/react_agent/tools.py` | Calculator (AST-safe math), `web_search` (DDGS), `MemoryStore` + tool schemas |
| `src/react_agent/main.py` | CLI: loads `.env`, one-shot or interactive REPL |
| `pyproject.toml` | Package metadata, dependencies, `react-agent` console script |
| `requirements.txt` | Same dependencies for `pip install -r` workflows |

## Requirements

- Python **3.10+**
- An **OpenAI API key** with available quota ([OpenAI API keys](https://platform.openai.com/api-keys))

## Setup

```bash
cd ReActPro
python -m pip install -e .
```

Create a `.env` file in the project root (do **not** commit it; it is listed in `.gitignore`):

```env
OPENAI_API_KEY=sk-your-key-here
# Optional:
# OPENAI_MODEL=gpt-4o-mini
```

## Usage

One-shot:

```bash
python -m react_agent.main "What is (21 + 9) * 2?"
```

Interactive:

```bash
python -m react_agent.main
```

Or use the installed script (if your `Scripts` directory is on `PATH`):

```bash
react-agent "Hello"
```

## Tools (summary)

- **calculator** — Evaluates numeric expressions with `+ - * / ** % //`, parentheses, and unary `+`/`-` (implemented with a restricted AST evaluator, not `eval()` on arbitrary code).
- **web_search** — Text search via DuckDuckGo (`duckduckgo-search`).
- **memory_read** / **memory_write** — JSON file-backed store at `memory.json` in the project root (also gitignored by default).

## License

Add a license file if you intend to open-source under specific terms.
