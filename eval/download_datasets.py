"""
Downloads eval datasets from public sources.
Run once before using the eval harness:

    python eval/download_datasets.py
"""

import urllib.request
from pathlib import Path

DATASETS = {
    "tips.csv": (
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    ),
    "titanic.csv": (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    ),
    "iris.csv": (
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    ),
}

OUT_DIR = Path(__file__).parent / "datasets"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, url in DATASETS.items():
        dest = OUT_DIR / name
        if dest.exists():
            print(f"  {name}: already present, skipping.")
            continue
        print(f"  {name}: downloading ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"done ({size_kb} KB).")
    print("All datasets ready in", OUT_DIR)


if __name__ == "__main__":
    main()
