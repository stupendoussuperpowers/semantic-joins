"""Cheap column-pair statistics for the router (proposal step 2).

``profile(left, right)`` describes how the ``right`` values relate to the
``left`` column; the router passes the right values the normalized equi-join
left unmatched. It never calls an LLM. Cross-column statistics are computed
for a seeded sample of at most ``sample`` right values against the whole left
column, because a sampled left side would usually not contain the true match
of a sampled right value. Each statistic then costs one sparse
``sample x len(left)`` product instead of a comparison per value pair.

Statistics, following the routing table in the project proposal:

* ``left_len`` / ``right_len``: mean normalized length of each side, and
  ``left_uniformity`` / ``right_uniformity``: the share of values with the
  most common character shape ("DE" -> "AA", "111-11-1111" -> "999-99-9999").
  A short, uniform side suggests a code <-> value pair (DE <-> Germany).
* ``qgram_cos``: mean over the sample of the best char 3-gram TF-IDF cosine
  to a left value. High for typos and small edits.
* ``token_jaccard``: mean best word-token Jaccard to a left value, and
  ``token_set_equal``: the share whose token set equals some left value's.
  High for reordered tokens ("Smith, John" <-> "John Smith").
* ``acronym_rate``: the share that is an initialism of a left value, or whose
  initialism is a left value (NY <-> New York).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from semjoin.matchers.acronym import initialisms, short_key
from semjoin.matchers.utils_similarity import Vectorizer, similarity
from semjoin.normalize import normalize


@dataclass(frozen=True)
class Profile:
    n_left: int
    n_right: int
    n_sample: int
    left_len: float
    right_len: float
    left_uniformity: float
    right_uniformity: float
    qgram_cos: float
    token_jaccard: float
    token_set_equal: float
    acronym_rate: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy with floats rounded, for ``ctx.annotate``."""
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def shape(value: str) -> str:
    """Character classes of a value: letters -> "A", digits -> "9", anything else kept."""
    return "".join("A" if c.isalpha() else "9" if c.isdigit() else c for c in value.strip())


def _mean(values: Sequence[float] | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def uniformity(values: Sequence[str]) -> float:
    """Share of values with the most common ``shape``."""
    if not values:
        return 0.0
    return Counter(map(shape, values)).most_common(1)[0][1] / len(values)


def best_similarity(sample: Sequence[str], left: Sequence[str], analyzer: str, weighting: str, measure: str) -> np.ndarray:
    """For each sample value, its highest similarity to any left value (0 when none overlap)."""
    if not sample or not left:
        return np.zeros(len(sample))
    vec = Vectorizer(left, sample, analyzer, weighting)
    return similarity(vec.R, vec.L, measure).max(axis=1).toarray().ravel()


def acronym_rate(sample: Sequence[str], left: Sequence[str], max_len: int = 8) -> float:
    """Share of sample values that join a left value as an initialism, in either direction."""
    if not sample:
        return 0.0
    left_initialisms: set[str] = set()
    left_keys: set[str] = set()
    for v in left:
        left_initialisms |= initialisms(v)
        k = short_key(v, max_len)
        if k is not None:
            left_keys.add(k)
    hits = sum(short_key(s, max_len) in left_initialisms or bool(initialisms(s) & left_keys) for s in sample)
    return hits / len(sample)


def profile(left: Sequence[str], right: Sequence[str], sample: int = 100, seed: int = 0, max_len: int = 8) -> Profile:
    rng = random.Random(seed)
    right_sample = rng.sample(list(right), min(sample, len(right)))
    left_sample = rng.sample(list(left), min(sample, len(left)))
    jaccard = best_similarity(right_sample, left, "word", "binary", "jaccard")
    return Profile(
        n_left=len(left),
        n_right=len(right),
        n_sample=len(right_sample),
        left_len=_mean([len(normalize(v)) for v in left_sample]),
        right_len=_mean([len(normalize(v)) for v in right_sample]),
        left_uniformity=uniformity(left_sample),
        right_uniformity=uniformity(right_sample),
        qgram_cos=_mean(best_similarity(right_sample, left, "char3", "tfidf", "cosine")),
        token_jaccard=_mean(jaccard),
        token_set_equal=_mean(jaccard >= 1.0 - 1e-9),
        acronym_rate=acronym_rate(right_sample, left, max_len),
    )
