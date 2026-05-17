# Data-Analysis Agent

A CLI agent that accepts a CSV file and a natural-language question and returns
a natural-language answer plus a chart. It writes Python, runs it in a
subprocess, observes the result or error, and corrects itself up to three times
before giving up honestly.

Built as a portfolio project. The single most important deliverable is the
**Results** section below — the code exists to produce that table.

---

## Architecture

```
Question + CSV path
        │
        ▼
Step 0  Profile CSV (deterministic, no LLM)
        │  shape · dtypes · head · describe · null counts · categorical samples
        ▼
Step 1  LLM call #1 — generate Python code  ◄─── retry with truncated traceback ◄──┐
        │  system prompt: pandas/matplotlib only; print after ===ANSWER===;           │
        │  save chart to fixed path; load CSV from injected path                      │
        ▼                                                                             │
Step 2  Execute in subprocess                                                         │
        │  subprocess.run([sys.executable, "-c", code], timeout=30, capture_output=True)
        ▼                                                                             │
Step 3  Classify outcome                                                              │
        │  clean_success · exception · timeout · ran_but_no_answer                   │
        ├── not clean_success AND attempts < 3 ──────────────────────────────────────┘
        │
        ▼
Step 5  LLM call #2 — synthesize natural-language answer from raw output
        │
        ▼
Step 6  LLM call #3 — verification (independent sanity-check code, run via same harness)
        │  e.g. "sum of group counts == len(df)", "filtered rows <= total rows"
        │  contradiction → retry once or downgrade confidence
        ▼
Answer · chart path · attempt count · verification result
```

**Sandbox:** `subprocess.run` with a 30 s timeout. Isolation is intentionally
weak; this is acceptable because I control the input CSVs. Production hardening
= wrap the subprocess in a Docker container with no network and read-only
filesystem mounts.

---

## Results

> Status: **Week 3 complete — 20/20 pass. Model: `gpt-4o-mini`, `temperature=0`.**

| ID | Question | Dataset | Trap | Expected | Pass | Attempts |
|---|---|---|---|---|---|---|
| tips_01 | Average tip amount | tips.csv | — | 2.9983 | PASS | 1 |
| tips_02 | Day with highest avg tip | tips.csv | — | Sun | PASS | 1 |
| tips_03 | % smokers | tips.csv | — | 38.1% | PASS | 1 |
| tips_04 | Correlation bill↔tip | tips.csv | — | 0.6757 | PASS | 1 |
| tips_05 | Avg tip percentage | tips.csv | computed column | 16.08% | PASS | 1 |
| tips_06 | Smokers vs non-smokers tip | tips.csv | ambiguous framing | smoker | PASS | 1 |
| tips_07 | Lunch vs Dinner avg bill | tips.csv | — | Dinner | PASS | 1 |
| tips_08 | Tip > $5 count | tips.csv | — | 18 | PASS | 1 |
| tips_09 | Avg bill for party ≥ 4 | tips.csv | — | $29.31 | PASS | 1 |
| titanic_01 | Survivors count | titanic.csv | — | 342 | PASS | 1 |
| titanic_02 | Avg age of survivors | titanic.csv | NaN (177 nulls) | 28.34 | PASS | 1 |
| titanic_03 | Highest survival rate by class | titanic.csv | rate vs count | 1st class | PASS | 1 |
| titanic_04 | Female survival % | titanic.csv | — | 74.2% | PASS | 1 |
| titanic_05 | Most common embarkation port | titanic.csv | NaN (2 nulls) | S | PASS | 1 |
| titanic_06 | Avg fare by class | titanic.csv | — | ~84 (1st class) | PASS | 1 |
| titanic_07 | Passengers traveling alone | titanic.csv | — | 537 | PASS | 1 |
| titanic_08 | Highest fare paid | titanic.csv | — | 512.33 | PASS | 1 |
| titanic_09 | Women survived more than men by how many? | titanic.csv | rate vs count | 124 | PASS | 1 |
| iris_01 | Flowers per species | iris.csv | — | 50 each | PASS | 1 |
| iris_02 | Species with largest avg petal length | iris.csv | — | virginica | PASS | 1 |

**20 / 20 correct. All on first attempt.**

The retry loop was not triggered by the LLM (the profile gives it correct column names and types, so it writes correct code). Retry recovery was verified in Week 1 via a deliberately planted `KeyError`.

---

## Failure taxonomy

### Clean benchmark (tips / Titanic / Iris)

20/20 correct. All on first attempt. The profile-first approach (dtype, null count, sample values) eliminates the most common failure categories on well-structured datasets. This benchmark is **not hard enough to break the agent**.

### Adversarial benchmark

5 synthetic datasets with deliberate corruption: currency-formatted numbers, mixed boolean representations, categorical abbreviation variants, mixed datetime formats, free-text contamination in numeric columns, NaN-heavy columns.

**Normal mode: 15/20 (75%). Chaos mode: 15/20 (75%). Retry recovery (chaos): 20/20 (100%).**

| ID | Trap | Pass | Attempts | Retry recovered | Notes |
|---|---|---|---|---|---|
| adv_sales_01 | currency + boolean | PASS | 2 | yes | ValueError on attempt 1; LLM fixed currency parsing on attempt 2 |
| adv_sales_02 | categorical normalization | **FAIL** | 1 | no | Got 120, expected 161 — matched "Male/male/MALE" but missed "M" |
| adv_sales_03 | currency + outliers | PASS | 2 | yes | ValueError on attempt 1; retry recovered |
| adv_sales_04 | currency + outliers | PASS | 2 | yes | ValueError on attempt 1; retry recovered |
| adv_sales_05 | boolean normalization | PASS | 1 | no | Handled True/yes/1/Y correctly |
| adv_emp_01 | free-text age | PASS | 1 | no | Extracted numeric from "23 years", "N/A" etc. |
| adv_emp_02 | currency salary | PASS | 1 | no | Stripped $ and , correctly |
| adv_emp_03 | categorical normalization | **FAIL** | 1 | no | Got 34, expected 43 — matched "Engineering/ENGINEERING" but missed "Eng" |
| adv_emp_04 | NaN-heavy (40%) | PASS | 1 | no | dropna handled correctly |
| adv_trans_01 | row-level vs aggregate | PASS | 1 | no | Chose row-level (correct); output decimal not % (defensible) |
| adv_trans_02 | percentage vs count | PASS | 1 | no | |
| adv_trans_03 | control | PASS | 1 | no | |
| adv_trans_04 | count vs percentage | PASS | 1 | no | |
| adv_events_01 | mixed datetime | **FAIL** | 1 | no | Got 2, expected 5 — `pd.to_datetime` misparse on 4-format column |
| adv_events_02 | mixed datetime | **FAIL** | 1 | no | Got 2, expected 30 — same datetime parsing failure |
| adv_events_03 | control | PASS | 1 | no | |
| adv_events_04 | control | PASS | 1 | no | |
| adv_survey_01 | boolean normalization | **FAIL** | 1 | no | Got 55.5%, expected 66% — some boolean variants not covered |
| adv_survey_02 | NaN-heavy (70%) | PASS | 1 | no | Correctly used 61/200 non-null values |
| adv_survey_03 | NaN-heavy (30%) | PASS | 1 | no | |

### Observed failure patterns

| Pattern | Cases | Root cause | Retry helps? |
|---|---|---|---|
| **Mixed datetime parsing** | adv_events_01, adv_events_02 | `pd.to_datetime` on 4-format column silently misparses some dates; no exception raised so retry never triggers | No — silent wrong answer |
| **Abbreviated categorical variants** | adv_sales_02, adv_emp_03 | LLM normalizes case ("male"→"MALE") but does not handle abbreviations ("M", "Eng") — they are not in the profile's top values | No — silent wrong answer |
| **Boolean variant coverage** | adv_survey_01 | Covered some variants (Yes/yes) but missed others (TRUE/1/Y) in the same column | No — silent wrong answer |

### Key insight: what retry can and cannot fix

The chaos mode result is the most informative: inject a guaranteed `KeyError` on attempt 1 across all 20 cases — **20/20 cases recovered on attempt 2 or 3** (100%). The retry mechanism is fully functional for recoverable errors.

But the 5 persistent failures have no exception. The code runs to completion and prints a wrong number. Retry never fires because the outcome is `clean_success`. The LLM would need to *know the expected answer* to detect the error — which it cannot. This is the fundamental ceiling: **retry recovers runtime errors; it cannot detect silent logical errors.**

To close this gap: the Step 6 verifier is the right tool, but it needs to be calibrated to write checks that can actually catch off-by-abbreviation errors (e.g., "assert count_with_normalization == count_without"). That is future work.

---

## Setup

```bash
# 1. Create virtual environment and install dependencies (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt

# 2. Add your OpenAI API key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 3. Download eval datasets
python eval/download_datasets.py

# 4a. Run eval harness — hardcoded, no API calls (Week 1 sanity check)
python eval/run_eval.py

# 4b. Run eval harness — real LLM codegen
python eval/run_eval.py --llm

# 4c. Run the interactive agent (cleans data, EDA charts, then Q&A loop)
python main.py path/to/data.csv

# 5. Generate adversarial datasets (one-time)
python eval/generate_adversarial.py

# 6a. Run adversarial eval — normal mode
python eval/run_adversarial.py

# 6b. Run adversarial eval — chaos mode (forces KeyError on attempt 1)
python eval/run_adversarial.py --chaos
```

---

## Eval datasets

| File | Source | Rows | Columns |
|---|---|---|---|
| tips.csv | [mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data) | 244 | 7 |
| titanic.csv | [datasciencedojo/datasets](https://github.com/datasciencedojo/datasets) | 891 | 12 |
| iris.csv | [mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data) | 150 | 5 |

Ground-truth answers are hand-computed from the known dataset statistics and
verified on first run. See `eval/cases.json` for expected values and tolerances.

---

## Project status

- [x] Week 1 — execution harness, profiler, eval infrastructure (Checkpoint 1)
- [x] Week 2 — LLM codegen (#1) + synthesis (#2), retry loop live (Checkpoint 2)
- [x] Week 3 — cleaner, auto EDA, REPL loop, verifier (#3), 20-case eval, failure analysis (Checkpoint 3)
- [ ] Stretch — Streamlit UI (if approved)
