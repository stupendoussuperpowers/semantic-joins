"""Matcher registry.

Matchers are selected on the command line by spec strings such as
``fuzzy`` or ``fuzzy:threshold=0.7,q=2``. Register new strategies (and the
router) in ``MATCHERS``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from semjoin.core import Matcher
from semjoin.matchers.baselines import ExactMatcher, NormalizedMatcher, QgramFuzzyMatcher
from semjoin.matchers.pipeline import Pipeline

MATCHERS: dict[str, Callable[..., Matcher]] = {
    # Baselines.
    "exact": ExactMatcher,
    "normalized": NormalizedMatcher,
    "fuzzy": QgramFuzzyMatcher,
    # Routing-table strategies: normalize -> step [-> residual LLM with residual=true].
    "autofj": partial(Pipeline, steps="autofj"),
    "transform": partial(Pipeline, steps="transform"),
    "acronym": partial(Pipeline, steps="acronym"),
    "llm_gen": partial(Pipeline, steps="gen"),
    # LLM on everything the normalized join misses.
    "llm": partial(Pipeline, steps="", residual=True),
    # Every cheap step in sequence, then the residual LLM pass.
    "cascade": partial(Pipeline, steps="autofj+transform+acronym", residual=True),
    # Any combination, e.g. "pipeline:steps=transform+acronym,residual=true".
    "pipeline": Pipeline,
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
