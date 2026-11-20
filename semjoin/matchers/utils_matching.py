"""Helpers shared by the strategy matchers.

A strategy matcher's ``match`` typically looks like:

    pairs, remaining = normalize_first(task, ctx)        # proposal step 1
    with ctx.stage("acronym"):
        pairs |= self.find_pairs(task.left, remaining, ctx, pairs)
    if self.residual:
        pairs = add_residual(task, ctx, pairs)           # proposal step 4, in llm_residual.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from semjoin.core import JoinTask, Pair, RunContext
from semjoin.metrics import is_hard
from semjoin.normalize import normalize


def normalized_join(left: Sequence[str], right: Sequence[str]) -> set[Pair]:
    by_key: dict[str, list[str]] = defaultdict(list)
    for v in left:
        by_key[normalize(v)].append(v)
    return {(l, r) for r in right for l in by_key.get(normalize(r), ())}


def unmatched(right: Sequence[str], pairs: set[Pair]) -> list[str]:
    """Right values that appear in no pair, in their original order."""
    matched = {r for _, r in pairs}
    return [r for r in right if r not in matched]


def normalize_first(task: JoinTask, ctx: RunContext) -> tuple[set[Pair], list[str]]:
    """Normalized equi-join; returns its pairs and the right values it left unmatched."""
    with ctx.stage("normalize"):
        pairs = normalized_join(task.left, task.right)
    return pairs, unmatched(task.right, pairs)


def pick_examples(seeds: set[Pair], n: int) -> list[Pair]:
    """Up to ``n`` few-shot examples, preferring pairs that differ after normalization."""
    hard = sorted(p for p in seeds if is_hard(p))
    easy = sorted(p for p in seeds if not is_hard(p))
    pool = hard if len(hard) >= n else hard + easy
    if len(pool) <= n:
        return pool
    stride = len(pool) / n
    return [pool[int(i * stride)] for i in range(n)]


def parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort parse of an LLM reply that should be one JSON object."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return {str(k): v for k, v in obj.items()}
        except json.JSONDecodeError:
            pass
    out: dict[str, Any] = {}
    for m in re.finditer(r'"?([A-Za-z]?\d+)"?\s*:\s*("(?:[^"\\]|\\.)*"|null|\[[^\]]*\]|-?\d+)', text):
        try:
            out[m.group(1)] = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
    return out
