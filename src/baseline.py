"""Baselines: TF-IDF cosine (word + char n-grams) and MinHash Jaccard with LSH candidate retrieval."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from datasketch import MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from . import config
from .text_utils import minhash_signature, normalize


def _dedup_texts(texts: list[str]) -> tuple[list[str], list[int]]:
    """Return unique texts and an index list mapping each original position to its unique entry."""
    unique: list[str] = []
    idx_map: dict[str, int] = {}
    indices: list[int] = []
    for t in texts:
        idx = idx_map.get(t)
        if idx is None:
            idx = len(unique)
            idx_map[t] = idx
            unique.append(t)
        indices.append(idx)
    return unique, indices


class TfidfBaseline:
    """L2-normalised TF-IDF over word 1-2 grams and char_wb 3-5 grams (concatenated)."""

    def __init__(self) -> None:
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000, sublinear_tf=True
        )

    def fit(self, texts: list[str]) -> "TfidfBaseline":
        clean = [normalize(t) for t in texts]
        self.word_vec.fit(clean)
        self.char_vec.fit(clean)
        return self

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        if not texts:
            return sp.csr_matrix((0, 0), dtype=np.float32)
        unique, indices = _dedup_texts(texts)
        clean = [normalize(t) for t in unique]
        X = sp.hstack([self.word_vec.transform(clean), self.char_vec.transform(clean)]).tocsr()
        return l2_normalize(X)[indices]


def rowwise_dot(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    """Cosine of matching rows of two L2-normalised sparse matrices."""
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


def sparse_topk(Q: sp.csr_matrix, C: sp.csc_matrix, k_max: int, chunk: int = 500) -> np.ndarray:
    """Indices of the k_max most cosine-similar corpus rows for every query row (chunked)."""
    out = np.empty((Q.shape[0], k_max), dtype=np.int64)
    for start in range(0, Q.shape[0], chunk):
        S = (Q[start : start + chunk] @ C).toarray()
        idx = np.argpartition(-S, k_max - 1, axis=1)[:, :k_max]
        vals = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        out[start : start + chunk] = np.take_along_axis(idx, order, axis=1)
    return out


def minhash_signatures(texts: list[str]) -> list:
    if not texts:
        return []
    unique, indices = _dedup_texts(texts)
    sigs = [minhash_signature(t) for t in unique]
    return [sigs[i] for i in indices]


def minhash_pair_scores(sigs_a: list, sigs_b: list) -> np.ndarray:
    """Estimated Jaccard similarity of word-shingle sets for aligned pairs."""
    return np.array([a.jaccard(b) for a, b in zip(sigs_a, sigs_b)], dtype=np.float32)


def minhash_topk(query_sigs: list, corpus_sigs: list, k_max: int) -> np.ndarray:
    """LSH candidates (Jaccard >= retrieval threshold), re-ranked by estimated Jaccard."""
    lsh = MinHashLSH(threshold=config.MINHASH_RETRIEVAL_THRESHOLD, num_perm=config.MINHASH_PERMS)
    for i, sig in enumerate(corpus_sigs):
        lsh.insert(str(i), sig)
    out = np.full((len(query_sigs), k_max), -1, dtype=np.int64)
    for q, qsig in enumerate(query_sigs):
        candidates = [int(c) for c in lsh.query(qsig)]
        if not candidates:
            continue
        ranked = sorted(((qsig.jaccard(corpus_sigs[c]), c) for c in candidates), reverse=True)
        top = [c for _, c in ranked[:k_max]]
        out[q, : len(top)] = top
    return out
