"""
Week 1 eval harness — uses hardcoded Python code strings instead of the LLM.

Proves:
  - subprocess execution and stdout/stderr capture
  - outcome classification (clean_success / exception / timeout / ran_but_no_answer)
  - retry loop: planted error in tips_04_retry recovers on attempt 2
  - chart file detection (tips_02 saves a chart)

Run from the project root:
    python eval/run_eval.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.executor import execute_with_retry
from agent.profiler import profile_csv
from agent.types import OutcomeKind, RunResult

OUTPUT_DIR = Path(__file__).parent.parent / "output"
CHART_PATH = str(OUTPUT_DIR / "chart.png")
CASES_FILE = Path(__file__).parent / "cases.json"

# ---------------------------------------------------------------------------
# Hardcoded code templates — one list per case, index = attempt number.
# Format placeholders: {csv} = csv path (repr-quoted), {chart} = chart path.
# Any braces inside the generated Python code must be doubled {{ }}.
# ---------------------------------------------------------------------------
_CODES: dict[str, list[str]] = {
    "tips_01": [
        """\
import pandas as pd
df = pd.read_csv({csv!r})
avg = df['tip'].mean()
print('===ANSWER===')
print(f'The average tip amount is ${{avg:.4f}}')
""",
    ],

    # tips_02 also saves a chart — proves chart_written detection
    "tips_02": [
        """\
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
df = pd.read_csv({csv!r})
avg_by_day = df.groupby('day')['tip'].mean()
best_day = avg_by_day.idxmax()
avg_by_day.plot(kind='bar', title='Average Tip by Day')
plt.tight_layout()
plt.savefig({chart!r})
plt.close()
print('===ANSWER===')
print(f'Day with highest average tip: {{best_day}}')
""",
    ],

    "tips_03": [
        """\
import pandas as pd
df = pd.read_csv({csv!r})
pct = df['smoker'].value_counts(normalize=True)['Yes'] * 100
print('===ANSWER===')
print(f'{{pct:.1f}}% of dining parties included smokers')
""",
    ],

    # Planted error case: attempt 1 uses wrong column name 'tip_amount' -> KeyError.
    # attempt 2 uses the correct name 'tip'.
    "tips_04_retry": [
        """\
import pandas as pd
df = pd.read_csv({csv!r})
corr = df[['total_bill', 'tip_amount']].corr().iloc[0, 1]
print('===ANSWER===')
print(f'Correlation: {{corr:.4f}}')
""",
        """\
import pandas as pd
df = pd.read_csv({csv!r})
corr = df[['total_bill', 'tip']].corr().iloc[0, 1]
print('===ANSWER===')
print(f'Correlation between total_bill and tip: {{corr:.4f}}')
""",
    ],

    "titanic_01": [
        """\
import pandas as pd
df = pd.read_csv({csv!r})
n = int(df['Survived'].sum())
print('===ANSWER===')
print(f'{{n}} passengers survived')
""",
    ],
}


def _make_codegen(case_id: str, csv_path: str, chart_path: str):
    codes = _CODES[case_id]
    attempt_idx = 0

    def fn(prev_code: Optional[str], feedback: Optional[str]) -> str:
        nonlocal attempt_idx
        idx = min(attempt_idx, len(codes) - 1)
        attempt_idx += 1
        return codes[idx].format(csv=csv_path, chart=chart_path)

    return fn


def _check_answer(answer: Optional[str], case: dict) -> bool:
    if answer is None:
        return False
    kind = case["kind"]
    expected = str(case["expected_answer"])
    tolerance = case.get("tolerance")

    if kind == "numeric":
        # Extract all numbers from the answer and accept if any is within tolerance
        nums = re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))
        tol = float(tolerance) if tolerance is not None else 0.01
        for num_str in nums:
            try:
                if abs(float(num_str) - float(expected)) <= tol:
                    return True
            except ValueError:
                continue
        return False

    if kind == "string":
        return expected.lower() in answer.lower()

    return False


def _print_table(rows: list[dict]) -> None:
    col_w = [20, 22, 10, 8, 6]
    header = (
        f"{'ID':<{col_w[0]}}"
        f"{'OUTCOME':<{col_w[1]}}"
        f"{'ATTEMPTS':<{col_w[2]}}"
        f"{'CHART':<{col_w[3]}}"
        f"{'PASS':<{col_w[4]}}"
    )
    sep = "-" * sum(col_w)
    print("\n" + "=" * sum(col_w))
    print("EVAL RESULTS - Week 1 (hardcoded harness)")
    print("=" * sum(col_w))
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['id']:<{col_w[0]}}"
            f"{r['outcome'].value:<{col_w[1]}}"
            f"{r['attempts']:<{col_w[2]}}"
            f"{'yes' if r['chart'] else 'no':<{col_w[3]}}"
            f"{'PASS' if r['passed'] else 'FAIL':<{col_w[4]}}"
        )
    print(sep)
    passed = sum(r["passed"] for r in rows)
    print(f"  {passed}/{len(rows)} passed\n")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    cases: list[dict] = json.loads(CASES_FILE.read_text())
    rows: list[dict] = []

    for case in cases:
        csv_path = str(Path(case["dataset"]).resolve())
        case_id = case["id"]

        print(f"\n{'='*60}")
        print(f"Case: {case_id}")
        print(f"Q:    {case['question']}")
        print(f"{'='*60}")

        # Always profile first — profile goes into codegen context in Week 2
        profile = profile_csv(csv_path)
        # Print a compact excerpt so the terminal stays readable
        profile_lines = profile.splitlines()
        print("\n".join(profile_lines[:6]) + "\n  ...")

        codegen_fn = _make_codegen(case_id, csv_path, CHART_PATH)
        final, history = execute_with_retry(codegen_fn, CHART_PATH)

        for r in history:
            status = "OK" if r.outcome == OutcomeKind.CLEAN_SUCCESS else "FAIL"
            print(f"  attempt {r.attempt}: [{status}] {r.outcome}")
            if r.outcome != OutcomeKind.CLEAN_SUCCESS and r.stderr:
                # Print only the last line of stderr (usually the exception type)
                last_err = r.stderr.strip().splitlines()[-1]
                print(f"    -> {last_err}")

        if final.answer:
            print(f"  answer: {final.answer}")

        rows.append(
            {
                "id": case_id,
                "outcome": final.outcome,
                "attempts": final.attempt,
                "chart": final.chart_written,
                "passed": _check_answer(final.answer, case),
            }
        )

    _print_table(rows)


if __name__ == "__main__":
    main()
