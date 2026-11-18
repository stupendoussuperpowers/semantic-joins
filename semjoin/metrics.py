"""Pair-level precision / recall / F1.

Conventions for empty sets: precision is 1.0 when nothing is predicted and
recall is 1.0 when there is nothing to find, so a matcher that predicts
nothing on a task with gold pairs scores P=1, R=0, F1=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from semjoin.core import Pair
from semjoin.normalize import normalize


@dataclass(frozen=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score(pred: set[Pair], gold: Iterable[Pair]) -> Counts:
    gold = set(gold)
    tp = len(pred & gold)
    return Counts(tp=tp, fp=len(pred) - tp, fn=len(gold) - tp)


def is_hard(pair: Pair) -> bool:
    """True when the pair is not found by a normalized equi-join."""
    return normalize(pair[0]) != normalize(pair[1])


def hard_only(pairs: Iterable[Pair]) -> set[Pair]:
    return {p for p in pairs if is_hard(p)}


def micro(counts: Iterable[Counts]) -> Counts:
    total = Counts()
    for c in counts:
        total = total + c
    return total
