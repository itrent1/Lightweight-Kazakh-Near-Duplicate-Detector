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
    "len_ratio_norm_chars",
    "word_jaccard",
    "a_in_b_words",
    "b_in_a_words",
    "bigram_jaccard",
    "a_in_b_bigrams",
    "b_in_a_bigrams",
    "trigram_jaccard",
    "a_in_b_trigrams",
    "b_in_a_trigrams",
    "char4_jaccard",
    "a_in_b_char4",
    "b_in_a_char4",
    "char4_minus_word_jaccard",
    "prefix_frac_a",
    "prefix_frac_b",
    "suffix_frac_a",
    "suffix_frac_b",
    "shorter_is_prefix",
    "same_pos_frac_a",
    "same_pos_frac_b",
    "exact_raw",
    "exact_norm",
    "log_words_a",
    "mean_wlen_ratio",
    "punct_ratio",
    "sent_ratio",
]


def _word_ngrams(words: list[str], norm_str: str, n: int) -> set[str]:
    if len(words) < n:
        return {norm_str} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _common_prefix(wa: list[str], wb: list[str]) -> int:
    limit = min(len(wa), len(wb))
    i = 0
    while i < limit and wa[i] == wb[i]:
        i += 1
    return i


def _common_suffix(wa: list[str], wb: list[str]) -> int:
    na, nb = len(wa), len(wb)
    limit = min(na, nb)
    i = 0
    while i < limit and wa[na - 1 - i] == wb[nb - 1 - i]:
        i += 1
    return i


def _text_stats(text: str) -> tuple:
    raw = str(text)
    norm_str = normalize(raw)
    words = norm_str.split()
    word_set = set(words)
    bigrams = _word_ngrams(words, norm_str, 2)
    trigrams = _word_ngrams(words, norm_str, 3)
    char4 = _char_ngrams(norm_str, 4)
    sent = raw.count(".") + raw.count("!") + raw.count("?")
    punct = sent + raw.count(",")
    return (norm_str, words, word_set, bigrams, trigrams, char4, len(raw), punct, sent)


def _jaccard_and_contain(sa: set[str], sb: set[str]) -> tuple[float, float, float]:
    inter = len(sa & sb)
    union = len(sa) + len(sb) - inter
    return (
        inter / max(union, 1),
        inter / max(len(sa), 1),
        inter / max(len(sb), 1),
    )


def _pair_from_stats(sa_stats: tuple, sb_stats: tuple, exact_raw: float) -> list[float]:
    na_str, wa, sa, g2a, g3a, c4a, raw_len_a, punct_a, sent_a = sa_stats
    nb_str, wb, sb, g2b, g3b, c4b, raw_len_b, punct_b, sent_b = sb_stats
    na, nb = len(wa), len(wb)

    word_j, a_in_b_w, b_in_a_w = _jaccard_and_contain(sa, sb)
    g2_j, a_in_b_g2, b_in_a_g2 = _jaccard_and_contain(g2a, g2b)
    g3_j, a_in_b_g3, b_in_a_g3 = _jaccard_and_contain(g3a, g3b)
    c4_j, a_in_b_c4, b_in_a_c4 = _jaccard_and_contain(c4a, c4b)

    prefix = _common_prefix(wa, wb)
    suffix = _common_suffix(wa, wb)
    shorter = min(na, nb)
    same_pos = sum(1 for i in range(shorter) if wa[i] == wb[i])
    mean_wlen_a = len(na_str) / max(na, 1)
    mean_wlen_b = len(nb_str) / max(nb, 1)

    return [
        nb / max(na, 1),
        math.log((nb + 1) / (na + 1)),
        raw_len_b / max(raw_len_a, 1),
        len(nb_str) / max(len(na_str), 1),
        word_j,
        a_in_b_w,
        b_in_a_w,
        g2_j,
        a_in_b_g2,
        b_in_a_g2,
        g3_j,
        a_in_b_g3,
        b_in_a_g3,
        c4_j,
        a_in_b_c4,
        b_in_a_c4,
        c4_j - word_j,
        prefix / max(na, 1),
        prefix / max(nb, 1),
        suffix / max(na, 1),
        suffix / max(nb, 1),
        1.0 if shorter and prefix == shorter else 0.0,
        same_pos / max(na, 1),
        same_pos / max(nb, 1),
        exact_raw,
        1.0 if na_str == nb_str else 0.0,
        math.log(na + 1),
        mean_wlen_b / max(mean_wlen_a, 1e-6),
        (punct_b + 1) / (punct_a + 1),
        (sent_b + 1) / (sent_a + 1),
    ]


def pair_lexical_features(text_a: str, text_b: str) -> list[float]:
    """Feature vector (see FEATURE_NAMES) for one pair, computed on the full untruncated texts."""
    return _pair_from_stats(_text_stats(text_a), _text_stats(text_b), 1.0 if text_a == text_b else 0.0)


def lexical_features(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Feature matrix (n_pairs x len(FEATURE_NAMES)) for aligned lists of texts."""
    if len(texts_a) != len(texts_b):
        raise ValueError("texts_a and texts_b must have the same length")
    cache: dict[str, tuple] = {}

    def _cached(text: str) -> tuple:
        res = cache.get(text)
        if res is None:
            res = _text_stats(text)
            cache[text] = res
        return res

    return np.asarray(
        [_pair_from_stats(_cached(a), _cached(b), 1.0 if a == b else 0.0) for a, b in zip(texts_a, texts_b)],
        dtype=np.float32,
    )
