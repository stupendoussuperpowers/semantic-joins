"""Sparse string-similarity primitives shared by the matchers.

Values are featurized after ``normalize`` into either character q-grams
(padded with ``#``) or word tokens, weighted by TF-IDF or left binary, and
stored as scipy CSR matrices so whole-column similarity joins are matrix
products instead of Python loops.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Sequence

import numpy as np
from scipy import sparse

from semjoin.normalize import normalize


def char_grams(value: str, q: int = 3) -> Counter[str]:
    v = f"{'#' * (q - 1)}{normalize(value)}{'#' * (q - 1)}"
    return Counter(v[i : i + q] for i in range(len(v) - q + 1))


def word_tokens(value: str) -> Counter[str]:
    return Counter(normalize(value).split())


def featurize(value: str, analyzer: str) -> Counter[str]:
    """``analyzer`` is ``"char2"``, ``"char3"``, ... or ``"word"``."""
    if analyzer == "word":
        return word_tokens(value)
    if analyzer.startswith("char"):
        return char_grams(value, int(analyzer[4:] or 3))
    raise ValueError(f"unknown analyzer {analyzer!r}")


class Vectorizer:
    """Featurizes two columns over a shared vocabulary.

    ``weighting`` is ``"tfidf"`` (L2-normalized, so dot products are cosine
    similarities) or ``"binary"`` (0/1 sets, for Jaccard).
    """

    def __init__(self, left: Sequence[str], right: Sequence[str], analyzer: str = "char3", weighting: str = "tfidf"):
        self.analyzer = analyzer
        self.weighting = weighting
        left_feats = [featurize(v, analyzer) for v in left]
        right_feats = [featurize(v, analyzer) for v in right]
        self.vocab: dict[str, int] = {}
        df: Counter[str] = Counter()
        for feats in left_feats + right_feats:
            df.update(feats.keys())
            for g in feats:
                self.vocab.setdefault(g, len(self.vocab))
        n_docs = max(len(left_feats) + len(right_feats), 1)
        self.idf = {g: math.log(1 + n_docs / c) for g, c in df.items()}
        self.L = self._matrix(left_feats)
        self.R = self._matrix(right_feats)

    def _matrix(self, rows: list[Counter[str]]) -> sparse.csr_matrix:
        indptr, indices, data = [0], [], []
        for feats in rows:
            for g, tf in feats.items():
                indices.append(self.vocab[g])
                data.append(1.0 if self.weighting == "binary" else tf * self.idf[g])
            indptr.append(len(indices))
        m = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int64), np.asarray(indptr)),
            shape=(len(rows), max(len(self.vocab), 1)),
        )
        if self.weighting == "tfidf":
            norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
            norms[norms == 0] = 1.0
            m = sparse.diags(1.0 / norms) @ m
        return m.tocsr()


def similarity(a: sparse.csr_matrix, b: sparse.csr_matrix, measure: str) -> sparse.csr_matrix:
    """Pairwise similarity of the rows of ``a`` and ``b`` (only nonzero entries stored).

    ``measure`` is ``"cosine"`` (expects tfidf rows) or ``"jaccard"`` (expects binary rows).
    """
    inter = (a @ b.T).tocsr()
    if measure == "cosine":
        return inter
    if measure != "jaccard":
        raise ValueError(f"unknown measure {measure!r}")
    size_a = np.asarray(a.sum(axis=1)).ravel()
    size_b = np.asarray(b.sum(axis=1)).ravel()
    coo = inter.tocoo()
    union = size_a[coo.row] + size_b[coo.col] - coo.data
    return sparse.csr_matrix((coo.data / union, (coo.row, coo.col)), shape=inter.shape)


def chunked_rows(m: sparse.csr_matrix, chunk: int = 512) -> Iterator[tuple[int, sparse.csr_matrix]]:
    for start in range(0, m.shape[0], chunk):
        yield start, m[start : start + chunk]


def row_top_k(row_indices: np.ndarray, row_data: np.ndarray, k: int) -> list[tuple[int, float]]:
    """The ``k`` largest (column, value) entries of one sparse row, best first."""
    if len(row_data) == 0:
        return []
    if len(row_data) > k:
        part = np.argpartition(-row_data, k - 1)[:k]
    else:
        part = np.arange(len(row_data))
    order = part[np.argsort(-row_data[part], kind="stable")]
    return [(int(row_indices[i]), float(row_data[i])) for i in order]


def top_k(left: Sequence[str], right: Sequence[str], k: int = 10, analyzers: Sequence[str] = ("char3", "word")) -> list[list[tuple[int, float]]]:
    """For each right value, its ``k`` most similar left values (index, score).

    The score is the maximum TF-IDF cosine over ``analyzers``, so a candidate
    close in either characters or words is retrieved.
    """
    best: list[dict[int, float]] = [{} for _ in right]
    for analyzer in analyzers:
        vec = Vectorizer(left, right, analyzer)
        for start, block in chunked_rows(vec.R):
            sims = similarity(block, vec.L, "cosine")
            for i in range(sims.shape[0]):
                lo, hi = sims.indptr[i], sims.indptr[i + 1]
                for j, s in row_top_k(sims.indices[lo:hi], sims.data[lo:hi], k):
                    cur = best[start + i]
                    if s > cur.get(j, 0.0):
                        cur[j] = s
    return [sorted(b.items(), key=lambda kv: (-kv[1], kv[0]))[:k] for b in best]
