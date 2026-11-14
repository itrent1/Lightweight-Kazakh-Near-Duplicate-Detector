"""Text normalisation and MinHash helpers shared by the baseline and the leakage check."""
import re

from datasketch import MinHash

from . import config

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace."""
    text = _NON_WORD.sub(" ", str(text).lower())
    return _WS.sub(" ", text).strip()


def word_shingles(text: str, size: int = config.MINHASH_SHINGLE_SIZE) -> set[str]:
    """Set of word n-grams used as MinHash features."""
    tokens = normalize(text).split()
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def minhash_signature(text: str) -> MinHash:
    """MinHash signature of the document's word shingles."""
    m = MinHash(num_perm=config.MINHASH_PERMS, seed=config.RANDOM_STATE)
    m.update_batch([s.encode("utf-8") for s in word_shingles(text)])
    return m
