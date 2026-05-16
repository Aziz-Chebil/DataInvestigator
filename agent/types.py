from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OutcomeKind(str, Enum):
    CLEAN_SUCCESS = "clean_success"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    RAN_BUT_NO_ANSWER = "ran_but_no_answer"


@dataclass
class RunResult:
    outcome: OutcomeKind
    stdout: str
    stderr: str
    return_code: int
    answer: Optional[str]   # text after ===ANSWER=== marker, or None
    chart_written: bool
    attempt: int
    code: str
