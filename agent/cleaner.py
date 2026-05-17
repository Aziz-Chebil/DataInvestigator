import pandas as pd
from pathlib import Path

_NULL_DROP_THRESHOLD = 0.70  # drop columns with more than this fraction of nulls


def clean_csv(path: str | Path) -> tuple[str, str]:
    """
    Clean a CSV and save the result to output/cleaned.csv.

    Steps (all deterministic, no LLM):
      1. Drop columns that are >70% null.
      2. Fill remaining numeric nulls with the column median.
      3. Fill remaining categorical nulls with 'Unknown'.

    Returns (cleaned_csv_path, human-readable cleaning report).
    """
    df = pd.read_csv(path)
    report: list[str] = []

    # Step 1 — drop mostly-null columns
    null_frac = df.isna().mean()
    to_drop = null_frac[null_frac > _NULL_DROP_THRESHOLD].index.tolist()
    if to_drop:
        df = df.drop(columns=to_drop)
        report.append(f"Dropped columns (>{_NULL_DROP_THRESHOLD:.0%} null): {to_drop}")

    # Step 2 — numeric nulls → median
    for col in df.select_dtypes(include="number").columns:
        n = int(df[col].isna().sum())
        if n > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            report.append(f"  '{col}': filled {n} nulls with median ({median:.4g})")

    # Step 3 — categorical nulls → 'Unknown'
    for col in df.select_dtypes(include="object").columns:
        n = int(df[col].isna().sum())
        if n > 0:
            df[col] = df[col].fillna("Unknown")
            report.append(f"  '{col}': filled {n} nulls with 'Unknown'")

    out_dir = Path(__file__).parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "cleaned.csv"
    df.to_csv(out_path, index=False)

    summary = "\n".join(report) if report else "No cleaning needed — no nulls found."
    return str(out_path), summary
