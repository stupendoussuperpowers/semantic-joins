"""Matcher registry.

Matchers are selected on the command line by spec strings such as
``fuzzy`` or ``fuzzy:threshold=0.7,q=2``. Register new strategies (and the
router) in ``MATCHERS``.
"""

from __future__ import annotations

from typing import Any, Callable

from semjoin.core import Matcher
from semjoin.matchers.acronym import AcronymMatcher
from semjoin.matchers.autofj import AutoFuzzyJoinMatcher
from semjoin.matchers.baselines import ExactMatcher, NormalizedMatcher, QgramFuzzyMatcher
from semjoin.matchers.cascade import CascadeMatcher
from semjoin.matchers.llm_gen import LLMGenerateMatcher
from semjoin.matchers.llm_residual import LLMMatcher
from semjoin.matchers.transform import TransformMatcher

MATCHERS: dict[str, Callable[..., Matcher]] = {
    # Baselines.
    "exact": ExactMatcher,
    "normalized": NormalizedMatcher,
    "fuzzy": QgramFuzzyMatcher,
    # Routing-table strategies; each takes residual=true to add the LLM pass.
    "autofj": AutoFuzzyJoinMatcher,
    "transform": TransformMatcher,
    "acronym": AcronymMatcher,
    "llm_gen": LLMGenerateMatcher,
    # LLM on everything the normalized join misses.
    "llm": LLMMatcher,
    # autofj -> transform -> acronym -> LLM.
    "cascade": CascadeMatcher,
}

def _parse_value(raw: str) -> Any:
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    name, _, args = spec.partition(":")
    kwargs = {}
    for item in filter(None, args.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"bad matcher argument {item!r} in {spec!r}; expected key=value")
        kwargs[key.strip()] = _parse_value(value.strip())
    return name, kwargs


def build_matcher(spec: str) -> Matcher:
    name, kwargs = parse_spec(spec)
    if name not in MATCHERS:
        raise ValueError(f"unknown matcher {name!r}; choose from {sorted(MATCHERS)}")
    return MATCHERS[name](**kwargs)
