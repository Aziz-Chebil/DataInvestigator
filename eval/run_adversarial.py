"""
Adversarial eval harness — robustness testing on messy real-world CSVs.

Usage:
    python eval/run_adversarial.py           # normal mode
    python eval/run_adversarial.py --chaos   # chaos mode: forces KeyError on attempt 1

Instrumentation recorded per case:
  - attempts used
  - runtime (seconds)
  - exception type (first failure)
  - verifier result
  - whether retry recovered (attempt > 1 AND succeeded)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent._env import load_dotenv
from agent.executor import execute_with_retry, run_code
from agent.profiler import profile_csv
from agent.verifier import verify
from agent.types import OutcomeKind

OUTPUT_DIR = Path(__file__).parent.parent / "output"
CHART_PATH  = str(OUTPUT_DIR / "adv_chart.png")
VERIFY_PATH = str(OUTPUT_DIR / "adv_verify.png")
CASES_FILE  = Path(__file__).parent / "adversarial_cases.json"


# ---------------------------------------------------------------------------
# Chaos codegen: attempt 1 injects a guaranteed KeyError;
# attempts 2-3 use the real LLM with the traceback as feedback.
# ---------------------------------------------------------------------------
def _chaos_codegen(question: str, profile: str, csv_path: str) -> Callable:
    from agent.codegen import make_codegen_fn
    real_fn = make_codegen_fn(question, profile, csv_path, CHART_PATH)
    attempt_idx = 0

    def fn(prev_code: Optional[str], feedback: Optional[str]) -> str:
        nonlocal attempt_idx
        attempt_idx += 1
        if attempt_idx == 1:
            # Force a KeyError to trigger the retry path
            return (
                f"import pandas as pd\n"
                f"df = pd.read_csv({csv_path!r})\n"
                f"result = df['CHAOS_COL_DOES_NOT_EXIST'].mean()\n"
                f"print('===ANSWER===')\n"
                f"print(result)\n"
            )
        return real_fn(prev_code, feedback)

    return fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_exception_type(stderr: str) -> str:
    if not stderr:
        return "—"
    for line in reversed(stderr.strip().splitlines()):
        line = line.strip()
        m = re.match(r"^([A-Za-z][A-Za-z0-9]*(?:Error|Exception|Warning|Expired))", line)
        if m:
            return m.group(1)
    return "unknown"


def _check_answer(answer: Optional[str], case: dict) -> bool:
    if answer is None:
        return False
    kind     = case["kind"]
    expected = str(case["expected_answer"])
    tolerance = case.get("tolerance")

    if kind == "numeric":
        nums = re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))
        tol  = float(tolerance) if tolerance is not None else 0.01
        return any(
            abs(float(n) - float(expected)) <= tol
            for n in nums
            if _is_float(n)
        )
    if kind == "string":
        return expected.lower() in answer.lower()
    return False


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _print_table(records: list[dict], chaos: bool) -> None:
    mode = "CHAOS MODE" if chaos else "normal"
    w = [22, 28, 14, 5, 9, 14, 14, 6]
    sep = "-" * sum(w)

    print(f"\n{'='*sum(w)}")
    print(f"ADVERSARIAL EVAL — {mode}")
    print(f"{'='*sum(w)}")
    print(
        f"{'ID':<{w[0]}}{'TRAP':<{w[1]}}{'OUTCOME':<{w[2]}}"
        f"{'ATT':<{w[3]}}{'RETRY?':<{w[4]}}{'EXCEPTION':<{w[5]}}"
        f"{'VERIFY':<{w[6]}}{'PASS':<{w[7]}}"
    )
    print(sep)

    for r in records:
        print(
            f"{r['id']:<{w[0]}}"
            f"{str(r['trap'])[:26]:<{w[1]}}"
            f"{r['outcome'][:13]:<{w[2]}}"
            f"{r['attempts']:<{w[3]}}"
            f"{'yes' if r['retry_recovered'] else 'no':<{w[4]}}"
            f"{r['exception_type'][:13]:<{w[5]}}"
            f"{r['verification'][:13]:<{w[6]}}"
            f"{'PASS' if r['passed'] else 'FAIL':<{w[7]}}"
        )

    print(sep)
    passed    = sum(r["passed"]          for r in records)
    recovered = sum(r["retry_recovered"] for r in records)
    failures  = [r["id"] for r in records if not r["passed"]]
    print(f"  {passed}/{len(records)} passed")
    print(f"  {recovered} case(s) recovered via retry")
    if failures:
        print(f"  Failures: {', '.join(failures)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chaos",
        action="store_true",
        help="Chaos mode: forces KeyError on attempt 1 to stress-test retry recovery",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    from agent.codegen import make_codegen_fn

    OUTPUT_DIR.mkdir(exist_ok=True)
    cases: list[dict] = json.loads(CASES_FILE.read_text())
    records: list[dict] = []

    for case in cases:
        csv_path = str(Path(case["dataset"]).resolve())
        case_id  = case["id"]

        print(f"\n{'='*60}")
        print(f"Case : {case_id}")
        print(f"Trap : {case.get('trap', '—')}")
        print(f"Q    : {case['question'][:70]}")
        print(f"{'='*60}")

        t0 = time.time()

        # Profile
        try:
            profile = profile_csv(csv_path)
        except Exception as exc:
            print(f"  PROFILER FAILED: {exc}")
            records.append({
                "id": case_id, "trap": case.get("trap", "—"),
                "outcome": "profiler_error", "attempts": 0,
                "retry_recovered": False, "exception_type": type(exc).__name__,
                "passed": False, "verification": "SKIPPED",
                "runtime_s": round(time.time() - t0, 1),
            })
            continue

        profile_lines = profile.splitlines()
        print("\n".join(profile_lines[:5]) + "\n  ...")

        # Codegen function
        if args.chaos:
            codegen_fn = _chaos_codegen(case["question"], profile, csv_path)
        else:
            codegen_fn = make_codegen_fn(case["question"], profile, csv_path, CHART_PATH)

        final, history = execute_with_retry(codegen_fn, CHART_PATH)
        runtime = round(time.time() - t0, 1)

        # Instrumentation
        exception_type = "—"
        for r in history:
            if r.outcome == OutcomeKind.EXCEPTION:
                exception_type = _extract_exception_type(r.stderr)
                break

        retry_recovered = (
            final.outcome == OutcomeKind.CLEAN_SUCCESS and final.attempt > 1
        )

        # Print attempt log
        for r in history:
            ok = r.outcome == OutcomeKind.CLEAN_SUCCESS
            print(f"  attempt {r.attempt}: [{'OK' if ok else 'FAIL'}] {r.outcome.value}")
            if not ok and r.stderr:
                print(f"    -> {r.stderr.strip().splitlines()[-1][:100]}")

        # Verifier
        verification = "SKIPPED"
        if final.outcome == OutcomeKind.CLEAN_SUCCESS and final.answer:
            try:
                verification = verify(
                    question=case["question"],
                    answer=final.answer,
                    code=final.code,
                    csv_path=csv_path,
                    chart_path=VERIFY_PATH,
                )
            except Exception as exc:
                verification = f"VERIFY_ERR"

        if final.answer:
            print(f"  answer  : {str(final.answer)[:100]}")
        print(f"  verify  : {verification}")
        print(f"  runtime : {runtime}s")

        records.append({
            "id":              case_id,
            "trap":            case.get("trap", "—"),
            "outcome":         final.outcome.value,
            "attempts":        final.attempt,
            "retry_recovered": retry_recovered,
            "exception_type":  exception_type,
            "passed":          _check_answer(final.answer, case),
            "verification":    verification,
            "runtime_s":       runtime,
        })

    _print_table(records, chaos=args.chaos)

    # Save machine-readable results
    results_path = OUTPUT_DIR / "adversarial_results.json"
    results_path.write_text(json.dumps(records, indent=2))
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
