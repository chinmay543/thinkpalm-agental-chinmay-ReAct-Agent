"""Tool implementations: calculator, web search, persistent memory."""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
from typing import Any

from duckduckgo_search import DDGS

# Allowed AST nodes for safe arithmetic
_BIN_OPS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported binary operator")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported unary operator")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate a numeric expression with + - * / ** % // and parentheses."""
    expression = expression.strip()
    if not expression:
        return "Error: empty expression"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        if result == int(result):
            return str(int(result))
        return str(round(result, 12))
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return concise snippets."""
    query = query.strip()
    if not query:
        return "Error: empty query"
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=max_results))
        if not rows:
            return "No results found."
        lines = []
        for i, r in enumerate(rows, 1):
            title = r.get("title", "") or ""
            body = r.get("body", "") or ""
            href = r.get("href", "") or ""
            lines.append(f"{i}. {title}\n   {body}\n   {href}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


class MemoryStore:
    """Simple JSON file-backed key-value memory."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(__file__).resolve().parent.parent.parent / "memory.json"
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            try:
                raw = self.path.read_text(encoding="utf-8")
                self._data = json.loads(raw) if raw.strip() else {}
                if not isinstance(self._data, dict):
                    self._data = {}
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def read(self, key: str) -> str:
        key = key.strip()
        if not key:
            return "Error: empty key"
        if key not in self._data:
            return f"(no value stored for key '{key}')"
        return self._data[key]

    def write(self, key: str, value: str) -> str:
        key = key.strip()
        if not key:
            return "Error: empty key"
        self._data[key] = value
        self._save()
        return f"Stored key '{key}' ({len(value)} chars)."

    def delete(self, key: str) -> str:
        key = key.strip()
        if key in self._data:
            del self._data[key]
            self._save()
            return f"Deleted key '{key}'."
        return f"No key '{key}' to delete."


# Shared default store for the process
_default_memory: MemoryStore | None = None


def get_memory_store(path: Path | None = None) -> MemoryStore:
    global _default_memory
    if path is not None:
        return MemoryStore(path)
    if _default_memory is None:
        _default_memory = MemoryStore()
    return _default_memory


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression with numbers, + - * / ** % //, parentheses, and unary +/-.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '(2 + 3) * 7.5'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for current facts, news, definitions, or anything not in memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read a previously stored value by key from long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key",
                    }
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Save a fact or note to long-term memory for later turns (persists on disk).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short key name"},
                    "value": {"type": "string", "description": "Text to remember"},
                },
                "required": ["key", "value"],
            },
        },
    },
]


def dispatch_tool(name: str, arguments: str, memory: MemoryStore | None = None) -> str:
    """Parse JSON arguments and run the named tool."""
    store = memory or get_memory_store()
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return f"Invalid tool arguments JSON: {e}"

    if name == "calculator":
        return calculate(args.get("expression", ""))
    if name == "web_search":
        return web_search(args.get("query", ""))
    if name == "memory_read":
        return store.read(args.get("key", ""))
    if name == "memory_write":
        return store.write(args.get("key", ""), args.get("value", ""))
    return f"Unknown tool: {name}"
