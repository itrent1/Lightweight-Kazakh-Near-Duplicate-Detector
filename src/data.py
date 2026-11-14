"""Chunked CSV conversion, split loading and exploratory data analysis."""
import json
from collections import Counter

import pandas as pd

from . import config


def base_id(pair_id: str) -> str:
    """Original document id: '14366_noisy_hard_aug' -> '14366'."""
    return str(pair_id).split("_")[0]


def convert_csv_to_parquet(
    csv_path=config.RAW_CSV,
    out_dir=config.PROCESSED_DIR,
    chunksize: int = config.CSV_CHUNKSIZE,
) -> dict:
    """Stream the 1.77 GB CSV in chunks and store every split as Parquet parts."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    parts: Counter = Counter()
    reader = pd.read_csv(csv_path, chunksize=chunksize, dtype={"id": str})
    for chunk in reader:
        for split, sub in chunk.groupby("split"):
            split_dir = out_dir / str(split)
            split_dir.mkdir(exist_ok=True)
            sub.to_parquet(split_dir / f"part_{parts[split]:04d}.parquet", index=False)
            parts[split] += 1
            counts[str(split)] += len(sub)

    summary = dict(counts)
    (out_dir / "split_counts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_split(split: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load one split ('train', 'validation' or 'test') from the Parquet parts."""
    split_dir = config.PROCESSED_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split '{split}' is missing. Run the CSV -> Parquet conversion first.")
    if columns is not None:
        # base_id is derived from id, so it is never stored in the Parquet files
        columns = [c for c in columns if c != "base_id"]
        if "id" not in columns:
            columns = ["id"] + columns
    df = pd.read_parquet(split_dir, columns=columns)
    df["base_id"] = df["id"].map(base_id)
    return df


def run_eda(csv_path=config.RAW_CSV, chunksize: int = config.CSV_CHUNKSIZE) -> dict:
    """One streaming pass over the raw CSV: class balance, lengths, missing values, duplicates."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    total = 0
    missing: Counter = Counter()
    split_counts: Counter = Counter()
    type_counts: Counter = Counter()
    type_by_split: Counter = Counter()
    len_content: list[int] = []
    len_modified: list[int] = []
    content_hashes: set[int] = set()
    repeated_content_rows = 0
    identical_pairs = 0
    base_ids: set[str] = set()

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype={"id": str}):
        total += len(chunk)
        missing.update({k: int(v) for k, v in chunk.isna().sum().items() if v})
        split_counts.update(chunk["split"].value_counts().to_dict())
        type_counts.update(chunk["type_duplicate"].value_counts().to_dict())
        type_by_split.update(zip(chunk["split"], chunk["type_duplicate"]))
        len_content.extend(chunk["content_len_tokens"].dropna().astype(int).tolist())
        len_modified.extend(chunk["modified_content_len_tokens"].dropna().astype(int).tolist())
        identical_pairs += int((chunk["content"] == chunk["modified_content"]).sum())
        base_ids.update(chunk["id"].map(base_id).tolist())

        for text in chunk["content"].fillna("").tolist():
            h = hash(text)
            if h in content_hashes:
                repeated_content_rows += 1
            else:
                content_hashes.add(h)

    summary = {
        "total_rows": total,
        "split_counts": dict(split_counts),
        "type_counts": dict(type_counts),
        "type_by_split": {f"{s}|{t}": n for (s, t), n in type_by_split.items()},
        "missing_values": dict(missing),
        "content_len_tokens": pd.Series(len_content).describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_dict(),
        "modified_len_tokens": pd.Series(len_modified).describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_dict(),
        "rows_with_repeated_content": repeated_content_rows,
        "identical_content_pairs": identical_pairs,
        "unique_original_documents": len(base_ids),
    }

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (config.PROCESSED_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_type_plot(summary["type_counts"])
    return summary


def save_type_plot(type_counts: dict) -> None:
    """Bar chart of duplicate types, saved to results/."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [t for t in config.TYPE_LABELS if t in type_counts]
    values = [type_counts[t] for t in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_title("KazakhTextDuplicates v2.0: duplicate types")
    ax.set_ylabel("pairs")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "duplicate_types.png", dpi=120)
    plt.close(fig)
