"""Heuristic mismatch-type tags for gold pairs, used to label tasks that come
without one (Auto-FuzzyJoin, Freyja).

Per pair, after normalization, the first test that passes wins:

    equal                              -> None (an equi-join finds it)
    same tokens, different order       -> reorder
    one side is the other's initials   -> acronym
    equal or contained once text in
    parentheses is dropped             -> fuzzy   ("Song" vs "Song (Band song)")
    character similarity >= 0.8        -> typo
    otherwise                          -> alias

A task gets the dominant pair type if it covers at least ``dominance`` of the
tagged pairs, else ``mixed``. The per-pair counts are kept as ``type_mix``.
The tags are a heuristic for analysis (RQ3), not ground truth.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable

from semjoin.core import Pair
from semjoin.matchers.acronym import initialisms, short_key
from semjoin.normalize import normalize

TYPO_RATIO = 0.8
_PAREN = re.compile(r"\s*\([^)]*\)")


def _is_acronym(full: str, short: str) -> bool:
    key = short_key(short, max_len=6)
    return key is not None and key in initialisms(full)


def tag_pair(left: str, right: str) -> str | None:
    a, b = normalize(left), normalize(right)
    if a == b:
        return None
    if sorted(a.split()) == sorted(b.split()):
        return "reorder"
    if _is_acronym(left, right) or _is_acronym(right, left):
        return "acronym"
    sa, sb = normalize(_PAREN.sub("", left)), normalize(_PAREN.sub("", right))
    if sa and sb and (sa in sb or sb in sa):
        return "fuzzy"
    if SequenceMatcher(None, a, b).ratio() >= TYPO_RATIO:
        return "typo"
    return "alias"


def tag_task(gold: Iterable[Pair], dominance: float = 0.75) -> tuple[str | None, dict[str, int]]:
    """Return (task mismatch type, per-type pair counts)."""
    counts = Counter(t for t in (tag_pair(l, r) for l, r in gold) if t)
    if not counts:
        return None, {}
    top, n = counts.most_common(1)[0]
    mix = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))  # stable across runs
    return (top if n / sum(counts.values()) >= dominance else "mixed"), mix
