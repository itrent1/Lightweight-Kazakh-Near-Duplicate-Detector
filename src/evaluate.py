"""Benchmark: TF-IDF, MinHash, zero-shot e5-small and fine-tuned e5-small on the test split."""
from __future__ import annotations

import json
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .baseline import (
    TfidfBaseline,
    minhash_pair_scores,
    minhash_signatures,
    minhash_topk,
    rowwise_dot,
    sparse_topk,
)
from .data import load_split
from .model import embed, faiss_topk, load_encoder, model_size_mb, pair_features

FINETUNED = config.FINETUNED_METHOD
COLUMNS = ["id", "base_id", "category", "content", "modified_content", "type_duplicate",
           "similarity_score", "content_len_tokens", "modified_content_len_tokens"]


@dataclass
class MethodOutput:
    name: str
    sims: np.ndarray       # similarity score for each test pair
    X_clf_fit: np.ndarray  # classifier features on validation pairs
    X_clf_test: np.ndarray # classifier features on test pairs
    topk: np.ndarray       # retrieval ranking of the sampled test queries
    fit_seconds: float
    infer_seconds: float
    model_size_mb: float | None


def _timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def run_tfidf(fit_df: pd.DataFrame, clf_df: pd.DataFrame, test: pd.DataFrame, query_idx: np.ndarray,
              corpus_mask: np.ndarray) -> MethodOutput:
    vec, fit_s = _timed(
        lambda: TfidfBaseline().fit(pd.concat([fit_df["content"], fit_df["modified_content"]]).tolist())
    )

    def _clf_features():
        A = vec.transform(clf_df["content"].tolist())
        B = vec.transform(clf_df["modified_content"].tolist())
        return rowwise_dot(A, B)[:, None]

    X_clf_fit, t_clf = _timed(_clf_features)

    def _infer():
        A = vec.transform(test["content"].tolist())
        B = vec.transform(test["modified_content"].tolist())
        sims = rowwise_dot(A, B)
        topk = sparse_topk(A[query_idx], B[corpus_mask].T.tocsc(), max(config.TOP_K))
        return sims, topk

    (sims, topk), infer_s = _timed(_infer)
    size = len(pickle.dumps(vec)) / 1e6
    return MethodOutput("tfidf_cosine", sims, X_clf_fit, sims[:, None], topk,
                        fit_s + t_clf, infer_s, size)


def run_minhash(fit_df: pd.DataFrame, clf_df: pd.DataFrame, test: pd.DataFrame, query_idx: np.ndarray,
                corpus_mask: np.ndarray) -> MethodOutput:
    k_max = max(config.TOP_K)

    def _clf_features():
        a = minhash_signatures(clf_df["content"].tolist())
        b = minhash_signatures(clf_df["modified_content"].tolist())
        return minhash_pair_scores(a, b)[:, None]

    X_clf_fit, fit_s = _timed(_clf_features)

    def _infer():
        a = minhash_signatures(test["content"].tolist())
        b = minhash_signatures(test["modified_content"].tolist())
        sims = minhash_pair_scores(a, b)
        corpus = [b[i] for i in np.flatnonzero(corpus_mask)]
        topk = minhash_topk([a[i] for i in query_idx], corpus, k_max)
        return sims, topk

    (sims, topk), infer_s = _timed(_infer)
    return MethodOutput("minhash_lsh", sims, X_clf_fit, sims[:, None], topk, fit_s, infer_s, None)


def run_encoder(name: str, model_path, clf_df: pd.DataFrame, test: pd.DataFrame, query_idx: np.ndarray,
                corpus_mask: np.ndarray) -> MethodOutput:
    model, load_s = _timed(lambda: load_encoder(model_path))

    def _clf_features():
        ea = embed(model, clf_df["content"].tolist())
        eb = embed(model, clf_df["modified_content"].tolist())
        return pair_features(ea, eb)

    X_clf_fit, fit_s = _timed(_clf_features)

    def _infer():
        ea = embed(model, test["content"].tolist())
        eb = embed(model, test["modified_content"].tolist())
        sims = (ea * eb).sum(axis=1)
        topk = faiss_topk(ea[query_idx], eb[corpus_mask], max(config.TOP_K))
        return ea, eb, sims, topk

    (ea, eb, sims, topk), infer_s = _timed(_infer)
    X_clf_test = pair_features(ea, eb)
    return MethodOutput(name, sims, X_clf_fit, X_clf_test, topk, fit_s + load_s, infer_s, model_size_mb(model))


def relevance_sets(query_bases: np.ndarray, corpus_bases: np.ndarray) -> list[set[int]]:
    """A corpus item is relevant to a query if it comes from the same original document."""
    index: dict[str, list[int]] = defaultdict(list)
    for j, b in enumerate(corpus_bases):
        index[b].append(j)
    return [set(index[b]) for b in query_bases]


def recall_at_k(topk: np.ndarray, relevant: list[set[int]], ks=config.TOP_K) -> dict:
    """recall_at_k: share of relevant items found in the top-k (capped by 1/|relevant|, ~1/8 here).
    hit_at_k: share of queries with at least one relevant item in the top-k (the useful signal here)."""
    metrics = {}
    for k in ks:
        recalls, hits = [], []
        for row, rel in zip(topk, relevant):
            if not rel:
                continue
            found = len(set(row[:k].tolist()) & rel)
            recalls.append(found / len(rel))
            hits.append(1.0 if found else 0.0)
        metrics[f"recall_at_{k}"] = float(np.mean(recalls)) if recalls else None
        metrics[f"hit_at_{k}"] = float(np.mean(hits)) if hits else None
    return metrics


def score_output(out: MethodOutput, clf_labels: np.ndarray, test: pd.DataFrame, relevant: list[set[int]]):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    _, clf_s = _timed(lambda: clf.fit(out.X_clf_fit, clf_labels))
    types = clf.predict(out.X_clf_test)
    y = test["type_duplicate"].values
    metrics = {
        "method": out.name,
        "n_test_pairs": int(len(test)),
        "spearman": float(spearmanr(out.sims, test["similarity_score"]).correlation),
        "f1_macro": float(f1_score(y, types, average="macro")),
        "accuracy": float(accuracy_score(y, types)),
        **recall_at_k(out.topk, relevant),
        "fit_seconds": round(out.fit_seconds + clf_s, 2),
        "infer_seconds": round(out.infer_seconds, 2),
        "infer_seconds_per_1000": round(out.infer_seconds / len(test) * 1000, 3),
        "model_size_mb": None if out.model_size_mb is None else round(out.model_size_mb, 2),
        "rss_mb": round(psutil.Process().memory_info().rss / 1e6, 1),
    }
    preds = pd.DataFrame({"pair_id": test["id"].values, "pred_score": out.sims, "pred_type": types})
    return metrics, preds, clf


def majority_baseline(clf_labels: np.ndarray, test: pd.DataFrame):
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(clf_labels), 1)), clf_labels)
    types = dummy.predict(np.zeros((len(test), 1)))
    y = test["type_duplicate"].values
    metrics = {
        "method": "majority_baseline",
        "n_test_pairs": int(len(test)),
        "spearman": None,
        "f1_macro": float(f1_score(y, types, average="macro")),
        "accuracy": float(accuracy_score(y, types)),
        "recall_at_1": None, "recall_at_5": None, "recall_at_10": None,
        "hit_at_1": None, "hit_at_5": None, "hit_at_10": None,
        "fit_seconds": 0.0, "infer_seconds": 0.0, "infer_seconds_per_1000": 0.0,
        "model_size_mb": None, "rss_mb": None,
    }
    preds = pd.DataFrame({"pair_id": test["id"].values, "pred_score": None, "pred_type": types})
    return metrics, preds


def run_benchmark(include_finetuned: bool = True, log=print) -> dict:
    """Run every approach on the test split; save metrics to results/ and the type classifier to models/."""
    train = load_split(config.TRAIN_SPLIT, COLUMNS)
    validation = load_split(config.VAL_SPLIT, COLUMNS)
    test = load_split(config.TEST_SPLIT, COLUMNS)

    fit_df = train.sample(n=min(config.TFIDF_FIT_SIZE, len(train)), random_state=config.RANDOM_STATE)
    rng = np.random.default_rng(config.RANDOM_STATE)
    # Retrieval corpus: all non-exact variants of the test documents (exact copies of the
    # original text would make retrieval trivial). Queries are sampled from all test rows.
    corpus_mask = (test["type_duplicate"] != "exact").values
    corpus_bases = test["base_id"].values[corpus_mask]
    query_idx = rng.choice(len(test), size=min(config.RETRIEVAL_QUERIES, len(test)), replace=False)
    relevant = relevance_sets(test["base_id"].values[query_idx], corpus_bases)
    clf_labels = validation["type_duplicate"].values

    runs = [
        ("tfidf_cosine", lambda: run_tfidf(fit_df, validation, test, query_idx, corpus_mask)),
        ("minhash_lsh", lambda: run_minhash(fit_df, validation, test, query_idx, corpus_mask)),
        ("e5_zero_shot", lambda: run_encoder("e5_zero_shot", config.BASE_MODEL_NAME, validation, test,
                                             query_idx, corpus_mask)),
    ]
    if include_finetuned and config.FINETUNED_MODEL_DIR.exists():
        runs.append((FINETUNED, lambda: run_encoder(FINETUNED, config.FINETUNED_MODEL_DIR, validation, test,
                                                    query_idx, corpus_mask)))

    results, predictions = [], {}
    metrics, preds = majority_baseline(clf_labels, test)
    results.append(metrics)
    predictions[metrics["method"]] = preds

    for name, runner in runs:
        log(f"-> {name}")
        out = runner()
        metrics, preds, clf = score_output(out, clf_labels, test, relevant)
        results.append(metrics)
        predictions[name] = preds
        if name == FINETUNED:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, config.MODELS_DIR / "type_clf_e5_finetuned.joblib")
        log(f"   f1_macro={metrics['f1_macro']:.4f} spearman={metrics['spearman']:.4f} "
            f"recall@10={metrics['recall_at_10']:.4f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "benchmark.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"results": results, "predictions": predictions, "test": test}
