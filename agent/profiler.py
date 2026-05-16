import pandas as pd
from pathlib import Path

_MAX_UNIQUE_SHOW = 20  # show all values for categoricals with <= this many distinct values


def profile_csv(path: str | Path) -> str:
    df = pd.read_csv(path)
    lines: list[str] = []

    lines.append(f"=== CSV PROFILE: {Path(path).name} ===")
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")

    lines.append("Columns (name: dtype  [nulls]):")
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        pct = nulls / len(df) * 100
        null_note = f"  [{nulls} nulls, {pct:.1f}%]" if nulls > 0 else ""
        lines.append(f"  {col!r}: {df[col].dtype}{null_note}")
    lines.append("")

    lines.append("First 5 rows:")
    lines.append(df.head().to_string(index=True))
    lines.append("")

    lines.append("describe(include='all'):")
    lines.append(df.describe(include="all").to_string())
    lines.append("")

    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        lines.append("Categorical columns — unique counts and top values:")
        for col in obj_cols:
            n = df[col].nunique()
            lines.append(f"  {col!r}: {n} unique values")
            if n <= _MAX_UNIQUE_SHOW:
                for val, cnt in df[col].value_counts().head(10).items():
                    lines.append(f"    {val!r}: {cnt}")
        lines.append("")

    return "\n".join(lines)
