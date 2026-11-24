"""``router``: profile a column pair and send it to one strategy (proposal steps 1-4).

1. Normalized equi-join; only the right values it leaves unmatched go on.
2. ``profile`` those values against the left column (no LLM, sampled).
3. The first rule whose test passes picks the strategy:

   - ``acronym``:  acronym_rate >= acronym_min                   -> acronym
   - ``code``:     the shorter side has mean length <= code_len
                   and shape uniformity >= code_uniformity        -> llm_gen
   - ``reorder``:  token_set_equal >= reorder_min                -> transform
   - ``typo``:     qgram_cos >= typo_cos                         -> autofj
   - ``fallback``: none of the above                             -> ``fallback`` (llm)

   Acronyms are tested before codes because initialisms are short too.
4. The strategy's ``find_pairs`` runs on the unmatched values, then (with
   ``residual=true``) the batched LLM pass on whatever is still unmatched.

The router returns exactly the pairs the chosen strategy returns on its own,
so ``semjoin summarize --router router`` can compare it with each strategy.
It records its choice as the spec that strategy runs under on its own
(``autofj:residual=true``, or ``autofj`` with ``residual=false``; ``llm``
takes no arguments), plus the rule that fired and the profile. Pass
``strategy_specs(residual)`` as ``--strategies`` so the specs line up.
"""

from __future__ import annotations

from typing import Any, Callable

from semjoin.core import JoinTask, Pair, RunContext
from semjoin.matchers.acronym import AcronymMatcher
from semjoin.matchers.autofj import AutoFuzzyJoinMatcher
from semjoin.matchers.llm_gen import LLMGenerateMatcher
from semjoin.matchers.llm_residual import LLMMatcher, add_residual
from semjoin.matchers.profiling import Profile, profile
from semjoin.matchers.transform import TransformMatcher
from semjoin.matchers.utils_matching import normalize_first

# Strategies the router chooses between, by registry name.
STRATEGIES: dict[str, Callable[[], Any]] = {
    "acronym": AcronymMatcher,
    "llm_gen": LLMGenerateMatcher,
    "transform": TransformMatcher,
    "autofj": AutoFuzzyJoinMatcher,
    "llm": LLMMatcher,
}


def strategy_spec(name: str, residual: bool = True) -> str:
    """The spec under which strategy ``name`` runs on its own the way the router runs it."""
    return f"{name}:residual=true" if residual and name != "llm" else name


def strategy_specs(residual: bool = True) -> list[str]:
    """Every strategy the router can choose, as specs for ``--strategies``."""
    return [strategy_spec(name, residual) for name in STRATEGIES]


class RouterMatcher:
    def __init__(
        self,
        residual: bool = True,
        fallback: str = "llm",
        sample: int = 100,
        seed: int = 0,
        acronym_min: float = 0.5,
        code_len: float = 4.0,
        code_uniformity: float = 0.8,
        # Defaults sit in gaps seen on the dev split: token_set_equal is <= 0.07 on every task
        # autofj wins and 0.35 where transform wins; qgram_cos is >= 0.54 on fuzzy-join tasks
        # and <= 0.26 on code tasks. Re-check them with `semjoin tune` on new runs.
        reorder_min: float = 0.25,
        typo_cos: float = 0.5,
    ):
        if fallback not in STRATEGIES:
            raise ValueError(f"unknown fallback {fallback!r}; choose from {sorted(STRATEGIES)}")
        self.residual = residual
        self.fallback = fallback
        self.sample = sample
        self.seed = seed
        self.acronym_min = acronym_min
        self.code_len = code_len
        self.code_uniformity = code_uniformity
        self.reorder_min = reorder_min
        self.typo_cos = typo_cos

    def route(self, p: Profile) -> tuple[str, str]:
        """(rule that fired, strategy name) for a profile."""
        if p.left_len <= p.right_len:
            short_len, short_uniformity = p.left_len, p.left_uniformity
        else:
            short_len, short_uniformity = p.right_len, p.right_uniformity
        rules = (
            ("acronym", p.acronym_rate >= self.acronym_min, "acronym"),
            ("code", short_len <= self.code_len and short_uniformity >= self.code_uniformity, "llm_gen"),
            ("reorder", p.token_set_equal >= self.reorder_min, "transform"),
            ("typo", p.qgram_cos >= self.typo_cos, "autofj"),
        )
        for rule, passed, strategy in rules:
            if passed:
                return rule, strategy
        return "fallback", self.fallback

    def decide(self, p: Profile) -> tuple[str, str]:
        """Like ``route``, for a profile of the right values the normalized join left unmatched."""
        if p.n_right == 0:
            # Every strategy returns just the normalized pairs, so any choice ties.
            return "nothing_unmatched", self.fallback
        return self.route(p)

    def match(self, task: JoinTask, ctx: RunContext) -> set[Pair]:
        pairs, remaining = normalize_first(task, ctx)
        with ctx.stage("profile"):
            prof = profile(task.left, remaining, sample=self.sample, seed=self.seed)
        rule, strategy = self.decide(prof)
        if strategy == "llm":
            pairs = add_residual(task, ctx, pairs)
        else:
            if remaining:
                with ctx.stage(strategy):
                    pairs |= STRATEGIES[strategy]().find_pairs(list(task.left), remaining, ctx, pairs)
            if self.residual:
                pairs = add_residual(task, ctx, pairs)
        ctx.annotate(strategy=strategy_spec(strategy, self.residual), route_rule=rule, profile=prof.as_dict())
        return pairs
