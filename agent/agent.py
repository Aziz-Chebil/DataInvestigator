from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .codegen import make_codegen_fn
from .executor import execute_with_retry
from .profiler import profile_csv
from .synthesizer import synthesize
from .types import OutcomeKind
from .verifier import verify

_OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEFAULT_CHART_PATH = str(_OUTPUT_DIR / "chart.png")
_VERIFY_CHART_PATH = str(_OUTPUT_DIR / "verify_chart.png")


@dataclass
class AgentResult:
    answer: str                 # synthesized natural-language answer (or failure message)
    raw_answer: Optional[str]   # raw text after ===ANSWER===, None on failure
    chart_path: Optional[str]   # path to saved chart, None if no chart produced
    attempt_count: int
    outcome: OutcomeKind
    final_code: str
    verification: str           # "PASS", "FAIL: reason", "VERIFY_ERROR: ...", or "SKIPPED"


def run(
    csv_path: str,
    question: str,
    chart_path: str = DEFAULT_CHART_PATH,
    timeout: int = 30,
) -> AgentResult:
    _OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 0: profile (deterministic, no LLM)
    profile = profile_csv(csv_path)

    # Steps 1-4: codegen + subprocess + retry loop (max 3 attempts)
    codegen_fn = make_codegen_fn(question, profile, csv_path, chart_path)
    final, history = execute_with_retry(codegen_fn, chart_path, timeout)

    if final.outcome != OutcomeKind.CLEAN_SUCCESS:
        tried = "\n\n".join(
            f"Attempt {r.attempt} ({r.outcome.value}):\n{r.stderr[-400:]}"
            for r in history
        )
        return AgentResult(
            answer=(
                f"I could not answer this reliably after {len(history)} attempt(s).\n\n"
                f"{tried}"
            ),
            raw_answer=None,
            chart_path=None,
            attempt_count=len(history),
            outcome=final.outcome,
            final_code=final.code,
            verification="SKIPPED",
        )

    # Step 5: synthesize raw output into a natural-language sentence
    answer = synthesize(question, final.answer, final.chart_written)

    # Step 6: independent sanity-check verification
    verification = verify(
        question=question,
        answer=final.answer,
        code=final.code,
        csv_path=csv_path,
        chart_path=_VERIFY_CHART_PATH,
    )

    return AgentResult(
        answer=answer,
        raw_answer=final.answer,
        chart_path=chart_path if final.chart_written else None,
        attempt_count=final.attempt,
        outcome=final.outcome,
        final_code=final.code,
        verification=verification,
    )
