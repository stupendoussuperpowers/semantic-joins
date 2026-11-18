"""Matcher registry.

Matchers are selected on the command line by spec strings such as
``fuzzy`` or ``fuzzy:threshold=0.7,q=2``. Register new strategies (and the
router) in ``MATCHERS``.
"""

from __future__ import annotations

from typing import Any, Callable

from semjoin.core import Matcher
from semjoin.matchers.baselines import ExactMatcher, NormalizedMatcher, QgramFuzzyMatcher

MATCHERS: dict[str, Callable[..., Matcher]] = {
    "exact": ExactMatcher,
    "normalized": NormalizedMatcher,
    "fuzzy": QgramFuzzyMatcher,
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
