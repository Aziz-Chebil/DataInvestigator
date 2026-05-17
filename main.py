"""
Data-Analysis Agent — interactive REPL.

Usage:
    python main.py <csv_path>

Loads the CSV once, cleans it, generates EDA charts, then enters a
question loop. Type 'exit' or press Ctrl+C to quit.

Requires OPENAI_API_KEY in .env or environment.
"""

import os
import sys
from pathlib import Path


def _separator(char: str = "-", width: int = 60) -> None:
    print(char * width)


def main() -> None:
    from agent._env import load_dotenv
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python main.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    from agent.cleaner import clean_csv
    from agent.eda import generate_eda
    from agent.agent import run

    # ------------------------------------------------------------------ #
    # Step 0a — Profile + clean
    # ------------------------------------------------------------------ #
    print(f"\nLoading {Path(csv_path).name} ...")
    _separator("=")

    cleaned_path, cleaning_report = clean_csv(csv_path)

    print("Cleaning report:")
    print(cleaning_report)
    print(f"\nCleaned CSV saved to: {cleaned_path}")

    # ------------------------------------------------------------------ #
    # Step 0b — Auto EDA charts
    # ------------------------------------------------------------------ #
    _separator()
    print("Generating EDA charts ...")
    eda_paths = generate_eda(cleaned_path)
    print(f"  {len(eda_paths)} charts saved to: output/eda/")
    for p in eda_paths:
        print(f"    {Path(p).name}")

    # ------------------------------------------------------------------ #
    # REPL
    # ------------------------------------------------------------------ #
    _separator("=")
    print("Ready. Ask a question about the data, or type 'exit' to quit.\n")

    while True:
        try:
            question = input("Question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        print()
        result = run(cleaned_path, question)

        _separator()
        print(f"Answer      : {result.answer}")
        print(f"Verification: {result.verification}")
        print(f"Attempts    : {result.attempt_count}  ({result.outcome.value})")
        if result.chart_path:
            print(f"Chart       : {result.chart_path}")
        _separator()
        print()


if __name__ == "__main__":
    main()
