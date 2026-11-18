"""Baseline matchers used to validate the harness end to end."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from semjoin.core import JoinTask, Pair, RunContext
from semjoin.normalize import normalize


class ExactMatcher:
    """Plain equi-join on the raw strings."""

    def match(self, task: JoinTask, ctx: RunContext) -> set[Pair]:
        right = set(task.right)
        return {(v, v) for v in task.left if v in right}


class NormalizedMatcher:
    """Equi-join after ``normalize`` (case, accents, punctuation, spacing)."""

    def match(self, task: JoinTask, ctx: RunContext) -> set[Pair]:
        with ctx.stage("normalize"):
            by_key: dict[str, list[str]] = defaultdict(list)
            for v in task.left:
                by_key[normalize(v)].append(v)
            return {(l, r) for r in task.right for l in by_key.get(normalize(r), ())}


class QgramFuzzyMatcher:
    """q-gram TF-IDF cosine similarity; each right value takes its best left
    value if the similarity is at least ``threshold``."""

    def __init__(self, threshold: float = 0.6, q: int = 3):
        self.threshold = threshold
        self.q = q

    def _grams(self, value: str) -> Counter[str]:
        v = f"{'#' * (self.q - 1)}{normalize(value)}{'#' * (self.q - 1)}"
        return Counter(v[i : i + self.q] for i in range(len(v) - self.q + 1))

    def match(self, task: JoinTask, ctx: RunContext) -> set[Pair]:
        with ctx.stage("fuzzy"):
            left_grams = [self._grams(v) for v in task.left]
            right_grams = [self._grams(v) for v in task.right]
            df = Counter(g for grams in left_grams + right_grams for g in grams)
            n_docs = len(left_grams) + len(right_grams)

            def vectorize(grams: Counter[str]) -> dict[str, float]:
                vec = {g: tf * math.log(1 + n_docs / df[g]) for g, tf in grams.items()}
                norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
                return {g: w / norm for g, w in vec.items()}

            index: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for i, grams in enumerate(left_grams):
                for g, w in vectorize(grams).items():
                    index[g].append((i, w))

            pairs = set()
            for r_value, grams in zip(task.right, right_grams):
                sims: dict[int, float] = defaultdict(float)
                for g, w in vectorize(grams).items():
                    for i, lw in index.get(g, ()):
                        sims[i] += w * lw
                if sims:
                    best = max(sims, key=sims.__getitem__)
                    if sims[best] >= self.threshold:
                        pairs.add((task.left[best], r_value))
            return pairs
