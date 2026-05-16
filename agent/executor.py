import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from .types import OutcomeKind, RunResult

ANSWER_MARKER = "===ANSWER==="
MAX_ATTEMPTS = 3
_FEEDBACK_MAX_CHARS = 2000


def run_code(
    code: str,
    chart_path: str,
    timeout: int = 30,
) -> RunResult:
    chart_file = Path(chart_path)
    if chart_file.exists():
        chart_file.unlink()

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        stdout, stderr, rc = "", f"TimeoutExpired: exceeded {timeout}s", -1
        timed_out = True

    if timed_out:
        outcome, answer = OutcomeKind.TIMEOUT, None
    elif rc != 0:
        outcome, answer = OutcomeKind.EXCEPTION, None
    elif ANSWER_MARKER in stdout:
        outcome = OutcomeKind.CLEAN_SUCCESS
        answer = stdout.split(ANSWER_MARKER, 1)[1].strip()
    else:
        outcome, answer = OutcomeKind.RAN_BUT_NO_ANSWER, None

    return RunResult(
        outcome=outcome,
        stdout=stdout,
        stderr=stderr,
        return_code=rc,
        answer=answer,
        chart_written=chart_file.exists(),
        attempt=0,  # caller sets this
        code=code,
    )


def _build_feedback(result: RunResult) -> str:
    parts: list[str] = []
    if result.stderr:
        parts.append("STDERR:\n" + result.stderr[-_FEEDBACK_MAX_CHARS:])
    if result.outcome == OutcomeKind.RAN_BUT_NO_ANSWER and result.stdout:
        parts.append(
            f"STDOUT (no {ANSWER_MARKER!r} marker found):\n"
            + result.stdout[-_FEEDBACK_MAX_CHARS:]
        )
    return "\n\n".join(parts) or "No output produced."


def execute_with_retry(
    codegen_fn: Callable[[Optional[str], Optional[str]], str],
    chart_path: str,
    timeout: int = 30,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[RunResult, list[RunResult]]:
    """
    Run the write→execute→observe→retry loop.

    codegen_fn(prev_code, error_feedback) -> Python code string.
      - prev_code and error_feedback are None on the first attempt.
      - On subsequent attempts they carry the last code and its truncated traceback.

    Returns (final_result, all_attempt_results).
    The caller should check final_result.outcome; if not CLEAN_SUCCESS after
    max_attempts, degrade gracefully — never fabricate an answer.
    """
    history: list[RunResult] = []
    prev_code: Optional[str] = None
    feedback: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        code = codegen_fn(prev_code, feedback)
        result = run_code(code, chart_path, timeout)
        result.attempt = attempt
        history.append(result)

        if result.outcome == OutcomeKind.CLEAN_SUCCESS:
            return result, history

        prev_code = code
        feedback = _build_feedback(result)

    return history[-1], history
