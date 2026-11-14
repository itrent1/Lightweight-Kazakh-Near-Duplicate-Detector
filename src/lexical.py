"""Order-, containment- and length-aware lexical features for a pair of documents.

Why they are needed: in KazakhTextDuplicates the duplicate type is defined by *how* the text was changed,
not by meaning. A semantic embedding hides exactly these differences:

* partial_cut  - the modified text is the first ~70% of the original  (B is contained in A, shorter)
* partial_ext  - the modified text is the original plus a short tail   (A is contained in B, ~3% longer)
* noisy_*      - character-level corruption of increasing strength     (neither text contains the other)
* contextual   - the same words in a different order                   (set overlap ~1.0, n-gram overlap lower)
* exact        - identical text

Cosine of e5 embeddings (and truncation at the model's token limit) cannot see these cues, but cheap
set / n-gram / prefix statistics on the full text can.
"""
from __future__ import annotations

import math

import numpy as np

from .text_utils import normalize

FEATURE_NAMES = [
    "len_ratio_words",
    "log_len_ratio_words",
    "len_ratio_chars",
    "word_jaccard",
    "a_in_b_words",
    "b_in_a_words",
    "trigram_jaccard",
    "a_in_b_trigrams",
    "b_in_a_trigrams",
    "prefix_frac_a",
    "prefix_frac_b",
    "suffix_frac_a",
    "suffix_frac_b",
    "shorter_is_prefix",
]


def _ngrams(words: list[str], n: int) -> set[str]:
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _common_prefix(wa: list[str], wb: list[str]) -> int:
    limit = min(len(wa), len(wb))
    i = 0
    while i < limit and wa[i] == wb[i]:
        i += 1
    return i


def pair_lexical_features(text_a: str, text_b: str) -> list[float]:
    """Feature vector (see FEATURE_NAMES) for one pair, computed on the full untruncated texts."""
    wa, wb = normalize(text_a).split(), normalize(text_b).split()
    na, nb = len(wa), len(wb)
    sa, sb = set(wa), set(wb)
    inter = len(sa & sb)
    union = len(sa | sb)

    ga, gb = _ngrams(wa, 3), _ngrams(wb, 3)
    g_inter = len(ga & gb)
    g_union = len(ga | gb)

    prefix = _common_prefix(wa, wb)
    suffix = _common_prefix(wa[::-1], wb[::-1])
    shorter = min(na, nb)

    return [
        nb / max(na, 1),
        math.log((nb + 1) / (na + 1)),
        len(text_b) / max(len(text_a), 1),
        inter / max(union, 1),
        inter / max(len(sa), 1),
        inter / max(len(sb), 1),
        g_inter / max(g_union, 1),
        g_inter / max(len(ga), 1),
        g_inter / max(len(gb), 1),
        prefix / max(na, 1),
        prefix / max(nb, 1),
        suffix / max(na, 1),
        suffix / max(nb, 1),
        1.0 if shorter and prefix == shorter else 0.0,
    ]


def lexical_features(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Feature matrix (n_pairs x len(FEATURE_NAMES)) for aligned lists of texts."""
    if len(texts_a) != len(texts_b):
        raise ValueError("texts_a and texts_b must have the same length")
    return np.asarray(
        [pair_lexical_features(a, b) for a, b in zip(texts_a, texts_b)], dtype=np.float32
    )
