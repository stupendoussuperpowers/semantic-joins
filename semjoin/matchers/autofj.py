"""Unsupervised fuzzy join following Auto-FuzzyJoin (Li et al., SIGMOD 2021), Section 3.

autofj: match an entry on the left with the closest value on the r if distance <= theta.

This is fuzzyjoin, but the threshold is decided automatically in the following way:

For each theta, estimate the precision based on how many left entries exist in a 2*theta sphere:
    precision(r, C) = 1 / | \\sigma_L{ f(l, l') if <= 2 * theta }  |

any theta that gives precision >= 0.9 is chosen.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from semjoin.core import Pair, RunContext
from semjoin.matchers.utils_similarity import Vectorizer, chunked_rows, similarity

# name -> (analyzer, weighting, measure)
FUNCTIONS: dict[str, tuple[str, str, str]] = {
    "char3-cosine": ("char3", "tfidf", "cosine"),
    "char2-cosine": ("char2", "tfidf", "cosine"),
    "word-cosine": ("word", "tfidf", "cosine"),
    "char3-jaccard": ("char3", "binary", "jaccard"),
    "word-jaccard": ("word", "binary", "jaccard"),
}
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.02, 0.5, 0.02))


class _Config:
    """One <f, theta>: the left index each right value joins (-1 if none) and its precision."""

    def __init__(self, name: str, theta: float, join: np.ndarray, precision: np.ndarray):
        self.name: str = name
        self.theta: float  = theta
        self.join: np.ndarray = join
        self.precision: np.ndarray = precision


def _configs_for(name: str, left: Sequence[str], right: Sequence[str]) -> list[_Config]:
    analyzer, weighting, measure = FUNCTIONS[name]
    vec = Vectorizer(left, right, analyzer, weighting)
    best_l = np.full(len(right), -1)
    best_d = np.ones(len(right))
    for start, block in chunked_rows(vec.R):
        sims = similarity(block, vec.L, measure)
        for i in range(sims.shape[0]):
            lo, hi = sims.indptr[i], sims.indptr[i + 1]
            if hi > lo:
                idx, data = sims.indices[lo:hi], sims.data[lo:hi]
                top = data.max()
                best_l[start + i] = idx[data >= top - 1e-12].min()  # ties: lowest index
                best_d[start + i] = 1.0 - top

    has = best_l >= 0
    if not has.any():
        return []
    chosen = np.unique(best_l[has])
    radii = 2 * np.asarray(THRESHOLDS)
    # ball[k, t] = #left values within distance 2*theta_t of chosen[k], itself included.
    ball = np.ones((len(chosen), len(THRESHOLDS)))
    if len(chosen):
        ll = similarity(vec.L[chosen], vec.L, measure).tocsr()
        for k, l in enumerate(chosen):
            lo, hi = ll.indptr[k], ll.indptr[k + 1]
            dists = 1.0 - ll.data[lo:hi][ll.indices[lo:hi] != l]
            dists.sort()
            ball[k] += np.searchsorted(dists, radii + 1e-12, side="right")
    pos = np.searchsorted(chosen, best_l.clip(min=0))

    out = []
    for t, theta in enumerate(THRESHOLDS):
        joins = has & (best_d <= theta + 1e-12)
        join = np.where(joins, best_l, -1)
        precision = np.where(joins, 1.0 / ball[pos, t], 0.0)
        out.append(_Config(name, theta, join, precision))
    return out


def _union(cur_join: np.ndarray, cur_p: np.ndarray, c: _Config) -> tuple[np.ndarray, np.ndarray]:
    take = (c.join >= 0) & ((cur_join < 0) | (c.precision > cur_p))
    return np.where(take, c.join, cur_join), np.where(take, c.precision, cur_p)


def _tp_fp(join: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    m = join >= 0
    return float(p[m].sum()), float((1 - p[m]).sum())


class AutoFuzzyJoinStep:
    def __init__(self, target: float = 0.9, functions: str = "all"):
        self.target = target
        if functions == "all":
            self.functions = list(FUNCTIONS)
        else:
            self.functions = [f for f in str(functions).split("+") if f]
            unknown = set(self.functions) - set(FUNCTIONS)
            if unknown:
                raise ValueError(f"unknown functions {sorted(unknown)}; choose from {sorted(FUNCTIONS)}")

    def propose(self, left: Sequence[str], right: Sequence[str], ctx: RunContext, seeds: set[Pair]) -> set[Pair]:
        candidates = [c for f in self.functions for c in _configs_for(f, left, right) if (c.join >= 0).any()]
        join = np.full(len(right), -1)
        prec = np.zeros(len(right))
        chosen = []
        while candidates:
            best, best_key = None, None
            for i, c in enumerate(candidates):
                tp, fp = _tp_fp(*_union(join, prec, c))
                key = (tp / fp if fp > 1e-12 else float("inf"), tp)
                if best_key is None or key > best_key:
                    best, best_key = i, key
            c = candidates.pop(best)
            new_join, new_prec = _union(join, prec, c)
            tp, fp = _tp_fp(new_join, new_prec)
            if tp <= _tp_fp(join, prec)[0] + 1e-12:
                continue  # adds nothing
            if tp / (tp + fp) < self.target:
                break
            join, prec = new_join, new_prec
            chosen.append(f"{c.name}@{c.theta}")
        tp, fp = _tp_fp(join, prec)
        ctx.annotate(autofj_configs=chosen, autofj_est_precision=round(tp / (tp + fp), 4) if tp + fp else None)
        return {(left[l], right[r]) for r, l in enumerate(join) if l >= 0}
