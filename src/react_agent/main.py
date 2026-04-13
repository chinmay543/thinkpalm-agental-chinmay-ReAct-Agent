"""CLI entry: load .env, run one query or interactive loop."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from react_agent.agent import run_agent


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Multi-tool ReAct agent (calculator + search + memory)")
    parser.add_argument("query", nargs="*", help="User message (omit for interactive mode)")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL"),
        help="OpenAI model name (default: OPENAI_MODEL or gpt-4o-mini)",
    )
    args = parser.parse_args()

    if args.query:
        text = " ".join(args.query)
        try:
            out = run_agent(text, model=args.model)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        print(out)
        return

    print("ReAct agent — type a question, or 'quit' to exit.", flush=True)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        try:
            out = run_agent(line, model=args.model)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            continue
        print(out, flush=True)


if __name__ == "__main__":
    main()
