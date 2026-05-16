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

> Status: **Week 1 complete — hardcoded harness, 5/5 pass. LLM not yet wired.**

| ID | Question | Dataset | Expected | Got | Pass | Attempts | Chart |
|---|---|---|---|---|---|---|---|
| tips_01 | Average tip amount | tips.csv | 2.9983 | 2.9983 | PASS | 1 | no |
| tips_02 | Day with highest avg tip | tips.csv | Sun | Sun | PASS | 1 | yes |
| tips_03 | % smokers | tips.csv | 38.1% | 38.1% | PASS | 1 | no |
| tips_04_retry | Correlation bill↔tip (planted error) | tips.csv | 0.6757 | 0.6757 | PASS | 2 | no |
| titanic_01 | Passengers survived | titanic.csv | 342 | 342 | PASS | 1 | no |
| | | | | | | | |
| *Week 2 rows — LLM-generated answers* | | | | | | | |

---

## Failure taxonomy

Expected failure categories (to be filled with real data at Checkpoint 3):

| Category | Description |
|---|---|
| Wrong aggregation level | Groups at the wrong granularity (sum where mean was asked, etc.) |
| Silent NaN | Computation drops nulls without flagging it; answer looks clean but is misleading |
| Ambiguous interpretation | Question has >1 valid reading; agent picks one without flagging uncertainty |
| dtype error | Numeric column read as string, or date column parsed incorrectly |
| No answer marker | Code runs cleanly but omits `===ANSWER===`; classified as `ran_but_no_answer` |

---

## Setup

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Add your Anthropic API key (needed from Week 2 onward)
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

# 3. Download eval datasets
python eval/download_datasets.py

# 4a. Week 1 — run the hardcoded harness
python eval/run_eval.py

# 4b. Week 2+ — run the full agent
python main.py path/to/data.csv "your question here"
```

---

## Eval datasets

| File | Source | Rows | Columns |
|---|---|---|---|
| tips.csv | [mwaskom/seaborn-data](https://github.com/mwaskom/seaborn-data) | 244 | 7 |
| titanic.csv | [datasciencedojo/datasets](https://github.com/datasciencedojo/datasets) | 891 | 12 |

Ground-truth answers are hand-computed from the known dataset statistics and
verified on first run. See `eval/cases.json` for expected values and tolerances.

---

## Project status

- [x] Week 1 — execution harness, profiler, eval infrastructure (Checkpoint 1)
- [ ] Week 2 — LLM codegen (#1) + synthesis (#2), retry loop live (Checkpoint 2)
- [ ] Week 3 — verification (#3), expanded eval (~20 Qs), failure analysis (Checkpoint 3)
- [ ] Stretch — Streamlit UI (after Checkpoint 3, if approved)
