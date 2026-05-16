"""
CLI entry point.

Week 1: profiles the CSV and exits — LLM codegen is wired in Week 2.
Week 2: runs the full agent loop and prints the answer + chart path.

Usage:
    python main.py <csv_path> "<question>"
"""

import sys
from pathlib import Path

from agent.profiler import profile_csv


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python main.py <csv_path> \"<question>\"")
        sys.exit(1)

    csv_path = sys.argv[1]
    question = sys.argv[2]

    if not Path(csv_path).exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    print(f"Question : {question}")
    print()
    print(profile_csv(csv_path))
    print("[Week 1] LLM codegen not yet wired.")
    print("         Run `python eval/run_eval.py` to exercise the execution harness.")


if __name__ == "__main__":
    main()
