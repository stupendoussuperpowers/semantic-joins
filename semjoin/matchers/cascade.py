"""``cascade``: every cheap strategy in turn, then the residual LLM pass.

normalize -> autofj -> transform -> acronym -> LLM on whatever is still unmatched.
Each strategy only sees the right values the earlier ones left unmatched.
This is the "run everything" comparison point for the router.
"""

from __future__ import annotations

from semjoin.core import JoinTask, Pair, RunContext
from semjoin.matchers.acronym import AcronymMatcher
from semjoin.matchers.autofj import AutoFuzzyJoinMatcher
from semjoin.matchers.llm_residual import add_residual
from semjoin.matchers.transform import TransformMatcher
from semjoin.matchers.utils_matching import normalize_first, unmatched


class CascadeMatcher:
    def __init__(self, residual: bool = True):
        self.residual = residual
        self.strategies = [
            ("autofj", AutoFuzzyJoinMatcher()),
            ("transform", TransformMatcher()),
            ("acronym", AcronymMatcher()),
        ]

    def match(self, task: JoinTask, ctx: RunContext) -> set[Pair]:
        pairs, remaining = normalize_first(task, ctx)
        added = {"normalize": len(pairs)}
        for name, matcher in self.strategies:
            if not remaining:
                break
            with ctx.stage(name):
                new = matcher.find_pairs(list(task.left), remaining, ctx, pairs)
            added[name] = len(new)
            pairs |= new
            remaining = unmatched(task.right, pairs)
        if self.residual:
            before = len(pairs)
            pairs = add_residual(task, ctx, pairs)
            added["llm_residual"] = len(pairs) - before
        ctx.annotate(strategy="cascade", pairs_added=added)
        return pairs
