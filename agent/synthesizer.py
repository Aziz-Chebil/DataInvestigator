import os

import openai

_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

_SYSTEM = """\
You are a concise data analyst reporting a finding to a non-technical stakeholder.
Given the user's question and the raw computed result, write one or two plain sentences
that directly answer the question. State numbers clearly. Do not mention code, DataFrames,
pandas, or any technical implementation detail."""


def synthesize(question: str, raw_answer: str, chart_written: bool) -> str:
    client = openai.OpenAI()

    user_content = f"Question: {question}\n\nComputed result:\n{raw_answer}"

    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    if chart_written:
        answer += " (A chart has been saved.)"
    return answer
