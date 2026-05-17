import os
import re

import openai

from .executor import run_code
from .types import OutcomeKind

_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

_SYSTEM = """\
You are a data analysis auditor. Given a question, the answer that was computed, and the
code that produced it, write a short Python script that performs an INDEPENDENT sanity check.

The check should verify a property that MUST hold if the answer is correct. Good examples:
  - "sum of all group counts equals total row count"
  - "filtered subset row count is <= total row count"
  - "the stated maximum value is >= the stated minimum value"
  - "a stated percentage is between 0 and 100"

The script must:
  1. Load the CSV from the exact path provided.
  2. Perform the check using only pandas and the Python standard library.
  3. Print exactly one of these two lines as the final output:
       PASS
       FAIL: <short reason>

Return ONLY raw Python code. No explanation. No markdown fences."""


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def verify(
    question: str,
    answer: str,
    code: str,
    csv_path: str,
    chart_path: str,
    timeout: int = 15,
) -> str:
    """
    Ask the LLM to propose a sanity-check script, run it, and return the result string.

    Returns one of:
      "PASS"
      "FAIL: <reason>"
      "VERIFY_ERROR: <what went wrong>"
      "VERIFY_TIMEOUT"
    """
    client = openai.OpenAI()

    user_content = (
        f"CSV path: {csv_path!r}\n\n"
        f"Question: {question}\n\n"
        f"Answer given: {answer}\n\n"
        f"Code that produced the answer:\n{code}"
    )

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    verify_code = _extract_code(response.choices[0].message.content)

    result = run_code(verify_code, chart_path, timeout=timeout)

    if result.outcome == OutcomeKind.TIMEOUT:
        return "VERIFY_TIMEOUT"
    if result.outcome == OutcomeKind.EXCEPTION:
        last_err = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown error"
        return f"VERIFY_ERROR: {last_err}"

    # Pull the last non-empty line — that's the PASS/FAIL verdict
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    return lines[-1] if lines else "VERIFY_ERROR: no output"
