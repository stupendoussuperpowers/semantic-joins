"""Acronym / initialism program.

For every value on one side, the following variants are tried in both directions:
- first letter of every word
- first letter of every word except stopwords ("of", "and", "the", ...)
- the same with a trailing parenthetical removed ("Kosovo (region)")
- numerals are kept whole ("Formula 1" -> "f1")
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from semjoin.core import Pair, RunContext
from semjoin.normalize import normalize

STOPWORDS = frozenset({"of", "and", "the", "for", "in", "on", "at", "to", "a", "an", "da", "de", "du", "del", "la", "le", "von", "van", "y", "et", "&"})
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def initialisms(value: str) -> set[str]:
    out = set()
    for base in {value, _PAREN.sub("", value)}:
        words = normalize(base).split()
        if len(words) < 2:
            continue
        letters = lambda ws: "".join(w if w.isdigit() else w[0] for w in ws)
        out.add(letters(words))
        content = [w for w in words if w not in STOPWORDS]
        if len(content) >= 2:
            out.add(letters(content))
    return out


def short_key(value: str, max_len: int) -> str | None:
    k = normalize(value).replace(" ", "")
    return k if 2 <= len(k) <= max_len else None


class AcronymStep:
    def __init__(self, max_len: int = 8, unique: bool = True):
        self.max_len = max_len
        self.unique = unique

    def _join(self, long_side: Sequence[str], short_side: Sequence[str]) -> list[tuple[str, str]]:
        by_acr: dict[str, set[str]] = defaultdict(set)
        for v in long_side:
            for a in initialisms(v):
                by_acr[a].add(v)
        out = []
        for s in short_side:
            k = short_key(s, self.max_len)
            if k is None or k not in by_acr:
                continue
            hits = by_acr[k]
            if self.unique and len(hits) > 1:
                continue
            out.extend((l, s) for l in sorted(hits))
        return out

    def propose(self, left: Sequence[str], right: Sequence[str], ctx: RunContext, seeds: set[Pair]) -> set[Pair]:
        pairs = set(self._join(left, right))  # right is the acronym
        pairs |= {(l, r) for r, l in self._join(right, left)}  # left is the acronym
        return pairs
