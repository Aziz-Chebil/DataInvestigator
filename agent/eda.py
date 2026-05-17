import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

_MAX_CATEGORIES = 20  # skip bar charts for columns with more unique values than this


def generate_eda(path: str | Path) -> list[str]:
    """
    Generate exploratory charts for every column and save to output/eda/.

    Charts produced (all deterministic, no LLM):
      - Histogram for each numeric column
      - Correlation heatmap if there are >= 2 numeric columns
      - Bar chart (value counts) for each low-cardinality categorical column

    Returns a list of saved file paths.
    """
    df = pd.read_csv(path)
    eda_dir = Path(__file__).parent.parent / "output" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [
        c for c in df.select_dtypes(include="object").columns
        if df[c].nunique() <= _MAX_CATEGORIES
    ]

    # --- Histograms ---
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[col].dropna(), bins=20, edgecolor="white")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        out = eda_dir / f"dist_{col}.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        saved.append(str(out))

    # --- Correlation heatmap ---
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        n = len(num_cols)
        fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(num_cols, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation heatmap")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
        fig.tight_layout()
        out = eda_dir / "correlation_heatmap.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        saved.append(str(out))

    # --- Categorical bar charts ---
    for col in cat_cols:
        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(max(5, len(counts) * 0.6 + 1), 4))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"Value counts: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        out = eda_dir / f"counts_{col}.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        saved.append(str(out))

    return saved
