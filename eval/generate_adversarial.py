"""
Generates synthetic messy CSV datasets for adversarial evaluation.

Run once before using the adversarial harness:
    python eval/generate_adversarial.py

Prints ground-truth values for each dataset so cases.json can be verified.
All generation is deterministic (seed=42).
"""

import random
import re
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent / "datasets"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def gen_sales() -> None:
    """
    adversarial_sales.csv
    Traps: currency-formatted amount, boolean inconsistency (8 variants),
           gender categorical normalization (4 variants each), outliers (2%).
    """
    n = 300
    male_variants   = ["Male", "male", "M", "MALE"]
    female_variants = ["Female", "female", "F", "FEMALE"]
    true_variants   = ["True", "yes", "1", "Y"]
    false_variants  = ["False", "no", "0", "N"]
    regions = ["North", "South", "East", "West"]

    rows = []
    for i in range(n):
        is_male = random.random() < 0.55
        gender = random.choice(male_variants if is_male else female_variants)
        is_completed = random.random() < 0.70
        completed = random.choice(true_variants if is_completed else false_variants)
        region = random.choice(regions)
        amount = random.uniform(50000, 100000) if random.random() < 0.02 \
                 else random.uniform(10, 800)
        rows.append({
            "transaction_id": i + 1,
            "customer_gender": gender,
            "amount": f"${amount:,.2f}",
            "completed": completed,
            "region": region,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "adversarial_sales.csv", index=False)

    amounts     = df["amount"].str.replace(r"[$,]", "", regex=True).astype(float)
    completed_b = df["completed"].str.lower().isin(["true", "yes", "1", "y"])
    male_b      = df["customer_gender"].str.upper().isin(["MALE", "M"])

    print("=== adversarial_sales.csv ===")
    print(f"  rows             : {len(df)}")
    print(f"  completed count  : {completed_b.sum()}")
    print(f"  completion rate  : {completed_b.mean()*100:.2f}%")
    print(f"  male count       : {male_b.sum()}")
    print(f"  revenue (all)    : {amounts.sum():.2f}")
    print(f"  revenue (done)   : {amounts[completed_b].sum():.2f}")
    print(f"  mean amount      : {amounts.mean():.2f}")
    print(f"  median amount    : {amounts.median():.2f}")
    print()


def gen_employees() -> None:
    """
    adversarial_employees.csv
    Traps: free-text age ("23 years", "N/A", "unknown"), salary as currency,
           department name with case/abbreviation variants, NaN-heavy performance_score (40%).
    """
    n = 150
    dept_map = {
        "Engineering": ["Engineering", "engineering", "ENGINEERING", "Eng"],
        "Marketing":   ["Marketing", "marketing", "MARKETING", "Mktg"],
        "Sales":       ["Sales", "sales", "SALES"],
        "HR":          ["HR", "hr", "Human Resources"],
    }
    age_junk = ["N/A", "unknown", "n/a", "?", "not provided"]

    rows = []
    for i in range(n):
        r = random.random()
        if r < 0.18:
            age = random.choice(age_junk)
        elif r < 0.28:
            age = f"{random.randint(22, 60)} years"
        else:
            age = str(random.randint(22, 60))

        salary = random.randint(40_000, 120_000)
        dept_key = random.choice(list(dept_map))
        dept = random.choice(dept_map[dept_key])
        perf = round(random.uniform(1, 10), 1) if random.random() > 0.40 else None

        rows.append({
            "employee_id": i + 1,
            "age": age,
            "salary": f"${salary:,}",
            "department": dept,
            "performance_score": perf,
            "years_at_company": round(random.uniform(0.5, 15), 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "adversarial_employees.csv", index=False)

    def _parse_age(v):
        s = str(v).strip().lower()
        if s in {"n/a", "unknown", "?", "not provided", "nan"}:
            return None
        m = re.search(r"\d+", s)
        return int(m.group()) if m else None

    valid_ages = pd.Series([_parse_age(a) for a in df["age"]]).dropna()
    salaries   = df["salary"].str.replace(r"[$,]", "", regex=True).astype(float)
    eng_b      = df["department"].str.upper().isin(["ENGINEERING", "ENG"])

    print("=== adversarial_employees.csv ===")
    print(f"  rows             : {len(df)}")
    print(f"  valid age count  : {len(valid_ages)}")
    print(f"  avg age          : {valid_ages.mean():.2f}")
    print(f"  avg salary       : {salaries.mean():.2f}")
    print(f"  engineering count: {eng_b.sum()}")
    print(f"  perf non-null    : {df['performance_score'].notna().sum()}")
    print(f"  avg perf score   : {df['performance_score'].mean():.2f}")
    print()


def gen_transactions() -> None:
    """
    adversarial_transactions.csv
    Traps: row-level vs aggregate profit margin, return % vs return count.
    """
    n = 500
    categories = ["Electronics", "Clothing", "Food", "Books", "Sports"]

    rows = []
    for i in range(n):
        cat     = random.choice(categories)
        revenue = round(random.uniform(10, 1000), 2)
        cost    = round(revenue * random.uniform(0.40, 0.90), 2)
        profit  = round(revenue - cost, 2)
        is_ret  = random.random() < 0.12

        rows.append({
            "transaction_id": i + 1,
            "category": cat,
            "revenue": revenue,
            "cost": cost,
            "profit": profit,
            "is_return": is_ret,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "adversarial_transactions.csv", index=False)

    row_margins  = df["profit"] / df["revenue"]
    agg_margin   = df["profit"].sum() / df["revenue"].sum()
    top_cat      = df.groupby("category")["revenue"].sum().idxmax()

    print("=== adversarial_transactions.csv ===")
    print(f"  rows                  : {len(df)}")
    print(f"  return count          : {df['is_return'].sum()}")
    print(f"  return rate           : {df['is_return'].mean()*100:.2f}%")
    print(f"  avg row-level margin  : {row_margins.mean()*100:.2f}%")
    print(f"  aggregate margin      : {agg_margin*100:.2f}%")
    print(f"  top revenue category  : {top_cat}")
    print(f"  revenue by category:")
    print(df.groupby("category")["revenue"].sum().round(2).to_string())
    print()


def gen_events() -> None:
    """
    adversarial_events.csv
    Traps: mixed datetime formats (4 different formats in same column),
           weekend counting requires parsing those formats first.
    """
    n = 100
    fmt_fns = [
        lambda d: d.strftime("%Y-%m-%d"),
        lambda d: d.strftime("%m/%d/%Y"),
        lambda d: d.strftime("%d-%b-%Y"),
        lambda d: d.strftime("%B %d, %Y"),
    ]
    base = pd.Timestamp("2023-01-01")

    rows, actual_dates = [], []
    for i in range(n):
        dt  = base + pd.Timedelta(days=random.randint(0, 364))
        fmt = random.choice(fmt_fns)
        dur = round(random.uniform(0.5, 8.0), 2)
        actual_dates.append(dt)
        rows.append({
            "event_id": i + 1,
            "event_date": fmt(dt),
            "duration_hours": dur,
            "city": random.choice(["Paris", "London", "New York", "Tokyo", "Sao Paulo"]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "adversarial_events.csv", index=False)

    dates_s    = pd.Series(actual_dates)
    jan_count  = int((dates_s.dt.month == 1).sum())
    wknd_count = int((dates_s.dt.dayofweek >= 5).sum())

    print("=== adversarial_events.csv ===")
    print(f"  rows              : {n}")
    print(f"  january events    : {jan_count}")
    print(f"  weekend events    : {wknd_count}")
    print(f"  avg duration (h)  : {df['duration_hours'].mean():.2f}")
    print()


def gen_survey() -> None:
    """
    adversarial_survey.csv
    Traps: boolean normalization (6 variants), NaN-heavy nps_score (70%) and rating (30%).
    """
    n = 200
    yes_v = ["Yes", "yes", "Y", "y", "TRUE", "1"]
    no_v  = ["No",  "no",  "N", "n", "FALSE", "0"]

    rows = []
    for i in range(n):
        satisfied = random.random() < 0.65
        response  = random.choice(yes_v if satisfied else no_v)
        nps       = random.randint(0, 10) if random.random() > 0.70 else None
        rating    = random.randint(1, 5)  if random.random() > 0.30 else None

        rows.append({
            "respondent_id": i + 1,
            "satisfied": response,
            "nps_score": nps,
            "rating": rating,
            "age_group": random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "adversarial_survey.csv", index=False)

    sat_b = df["satisfied"].str.lower().isin(["yes", "y", "true", "1"])

    print("=== adversarial_survey.csv ===")
    print(f"  rows              : {len(df)}")
    print(f"  satisfied count   : {sat_b.sum()}")
    print(f"  satisfaction rate : {sat_b.mean()*100:.2f}%")
    print(f"  nps non-null      : {df['nps_score'].notna().sum()}")
    print(f"  avg nps           : {df['nps_score'].mean():.2f}")
    print(f"  rating non-null   : {df['rating'].notna().sum()}")
    print(f"  avg rating        : {df['rating'].mean():.2f}")
    print()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    gen_sales()
    gen_employees()
    gen_transactions()
    gen_events()
    gen_survey()
    print("All adversarial datasets generated in", OUT_DIR)


if __name__ == "__main__":
    main()
