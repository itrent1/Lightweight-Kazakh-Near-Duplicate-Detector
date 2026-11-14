"""Train-test leakage check: shared original documents, identical texts and near-duplicate texts."""
from __future__ import annotations

import json

import pandas as pd
from datasketch import MinHashLSH

from . import config
from .data import load_split
from .text_utils import minhash_signature, normalize


def leakage_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_norm = train_df["content"].map(normalize)
    test_norm = test_df["content"].map(normalize)
    train_texts = set(train_norm)
    train_bases = set(train_df["base_id"])

    exact_mask = test_norm.isin(train_texts)

    # Near-duplicates: MinHash LSH over unique train texts, queried with unique non-exact test texts.
    lsh = MinHashLSH(threshold=config.MINHASH_LSH_THRESHOLD, num_perm=config.MINHASH_PERMS)
    for i, text in enumerate(pd.unique(train_norm)):
        lsh.insert(str(i), minhash_signature(text))

    candidate_texts = pd.unique(test_norm[~exact_mask])
    near_texts = {t for t in candidate_texts if lsh.query(minhash_signature(t))}
    near_mask = test_norm.isin(near_texts)

    report = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_rows_identical_text_in_train": int(exact_mask.sum()),
        "test_rows_near_duplicate_in_train": int((near_mask & ~exact_mask).sum()),
        "test_rows_with_any_text_overlap": int((exact_mask | near_mask).sum()),
        "shared_original_documents": len(train_bases & set(test_df["base_id"])),
        "test_rows_whose_original_is_in_train": int(test_df["base_id"].isin(train_bases).sum()),
        "near_duplicate_threshold": config.MINHASH_LSH_THRESHOLD,
    }
    for key in ("test_rows", "test_rows_identical_text_in_train", "test_rows_near_duplicate_in_train"):
        report[f"{key}_pct"] = round(100 * report[key] / max(report["test_rows"], 1), 2)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "leakage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def run_leakage_check() -> dict:
    cols = ["id", "base_id", "content"]
    return leakage_report(load_split(config.TRAIN_SPLIT, cols), load_split(config.TEST_SPLIT, cols))
