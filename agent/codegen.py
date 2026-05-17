import os
import re
from typing import Callable, Optional

import openai

_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

_SYSTEM = """\
You are a Python code generator for data analysis tasks.

Given a CSV data profile and a user question, write a complete runnable Python script that:
1. Loads the CSV using pandas from the exact path string provided — do not change the path.
2. Answers the question using ONLY pandas, matplotlib, and the Python standard library.
3. Prints the final answer on the line immediately after this exact marker (the marker must be
   alone on its own line):
       ===ANSWER===
4. If a chart is appropriate, saves it with matplotlib to the exact chart path string provided.
   - Call  matplotlib.use('Agg')  before importing pyplot.
   - Never call plt.show().
5. Handles NaN values explicitly when they could affect the result.

Hard rules:
- Use ONLY pandas, matplotlib, and the Python standard library. Nothing else.
- No machine learning, prediction, or forecasting of any kind.
- The script must be completely self-contained and runnable as-is.
- Always print ===ANSWER=== followed by the answer — even for a single number.
- Return ONLY the raw Python code. No explanation. No markdown fences."""

_RETRY_SUFFIX = """\

--- RETRY INSTRUCTIONS ---
Your previous attempt failed. Fix the error and return corrected code only.

Error output from the previous attempt:
{feedback}

Previous (broken) code:
{prev_code}
---"""


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def make_codegen_fn(
    question: str,
    profile: str,
    csv_path: str,
    chart_path: str,
) -> Callable[[Optional[str], Optional[str]], str]:
    """
    Returns a codegen callable that fits the execute_with_retry interface:
        fn(prev_code, error_feedback) -> python_code_string
    All context (question, profile, paths) is captured in the closure.
    """
    client = openai.OpenAI()

    base_user_msg = (
        f"CSV profile:\n{profile}\n\n"
        f"CSV path (use this exact string in your code): {csv_path!r}\n"
        f"Chart path (save here if producing a chart): {chart_path!r}\n\n"
        f"Question: {question}"
    )

    def fn(prev_code: Optional[str], error_feedback: Optional[str]) -> str:
        if prev_code is None:
            user_content = base_user_msg
        else:
            user_content = base_user_msg + _RETRY_SUFFIX.format(
                feedback=error_feedback, prev_code=prev_code
            )

        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        return _extract_code(response.choices[0].message.content)

    return fn
