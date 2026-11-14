"""Group-aware split by original document (base_id).

The official split of KazakhTextDuplicates v2.0 is not group-aware: 99.85% of its test rows have
the same original text in train. To evaluate honestly, all rows are pooled and re-split so that
every original document (and all its augmented variants) lands in exactly one split.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config
from .data import load_split

SPLIT_NAMES = ("train", "validation", "test")


def build_grouped_split(test_size: float = 0.2, val_size: float = 0.1,
                        seed: int = config.RANDOM_STATE) -> dict:
    pooled = pd.concat([load_split(s) for s in SPLIT_NAMES], ignore_index=True)

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trval_idx, test_idx = next(outer.split(pooled, groups=pooled["base_id"]))
    trval = pooled.iloc[trval_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(inner.split(trval, groups=trval["base_id"]))

    parts = {
        "train": trval.iloc[train_idx],
        "validation": trval.iloc[val_idx],
        "test": pooled.iloc[test_idx],
    }

    summary = {"seed": seed, "test_size": test_size, "val_size": val_size, "splits": {}}
    base_sets = {}
    for name, df in parts.items():
        out_dir = config.PROCESSED_DIR / f"group_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.parquet"):
            old.unlink()
        df.drop(columns=["base_id"]).to_parquet(out_dir / "part_0000.parquet", index=False)
        base_sets[name] = set(df["base_id"])
        summary["splits"][name] = {"rows": int(len(df)), "original_documents": len(base_sets[name])}

    summary["shared_original_documents"] = {
        "train_test": len(base_sets["train"] & base_sets["test"]),
        "train_validation": len(base_sets["train"] & base_sets["validation"]),
        "validation_test": len(base_sets["validation"] & base_sets["test"]),
    }
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (config.PROCESSED_DIR / "group_split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
