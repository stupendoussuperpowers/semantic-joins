"""The shared matching pipeline:
    (normalize -> strategy step(s) -> optional residual LLM pass)
"""

from __future__ import annotations

import inspect
import json
import re
from collections import defaultdict
from typing import Any, Callable, Protocol
from collections.abc import Sequence

from semjoin.core import JoinTask, Pair, RunContext
from semjoin.metrics import is_hard
from semjoin.normalize import normalize


class Step(Protocol):
    def propose(self, left: Sequence[str], right: Sequence[str], ctx: RunContext, seeds: set[Pair]) -> set[Pair]:
        """Return pairs (left value, right value) for values in ``right``."""
        ...


def normalized_join(left: Sequence[str], right: Sequence[str]) -> set[Pair]:
    by_key: dict[str, list[str]] = defaultdict(list)
    for v in left:
        by_key[normalize(v)].append(v)
    return {(l, r) for r in right for l in by_key.get(normalize(r), ())}


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


def _step_registry() -> dict[str, Callable[..., Step]]:
    from semjoin.matchers.acronym import AcronymStep
    from semjoin.matchers.autofj import AutoFuzzyJoinStep
    from semjoin.matchers.llm_gen import LLMGenerateStep
    from semjoin.matchers.transform import TransformStep

    return {
        "autofj": AutoFuzzyJoinStep,
        "transform": TransformStep,
        "acronym": AcronymStep,
        "gen": LLMGenerateStep,
    }


def _check_kwargs(cls: Callable[..., Any], kwargs: dict[str, Any], owner: str) -> None:
    params = inspect.signature(cls).parameters
    unknown = sorted(set(kwargs) - set(params))
    if unknown:
        raise ValueError(f"unknown argument(s) {unknown} for {owner}; accepted: {sorted(params)}")


class Pipeline:
    def __init__(self, steps: str = "autofj+transform+acronym", residual: bool = False, normalize: bool = True, **kwargs: Any):
        from semjoin.matchers.llm_residual import LLMResidualStep

        registry = _step_registry()
        self.step_names = [s for s in str(steps).split("+") if s]
        for name in self.step_names:
            if name not in registry:
                raise ValueError(f"unknown step {name!r}; choose from {sorted(registry)}")
        self.residual = residual
        self.normalize = normalize

        per_step: dict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in kwargs.items():
            owner, sep, param = key.partition("_")
            if sep and (owner in self.step_names or owner == "llm"):
                per_step[owner][param] = value
            elif len(self.step_names) == 1:
                per_step[self.step_names[0]][key] = value
            elif not self.step_names:
                per_step["llm"][key] = value
            else:
                raise ValueError(f"argument {key!r} must be prefixed with a step name ({'/'.join(self.step_names)}) or llm_")
        self.steps: list[tuple[str, Step]] = []
        for name in self.step_names:
            _check_kwargs(registry[name], per_step[name], name)
            self.steps.append((name, registry[name](**per_step[name])))
        _check_kwargs(LLMResidualStep, per_step["llm"], "llm residual")
        self.residual_step = LLMResidualStep(**per_step["llm"]) if residual else None

    @property
    def strategy_name(self) -> str:
        name = "+".join(self.step_names) or "none"
        return f"{name}+llm" if self.residual else name

    def match(self, task: JoinTask, ctx: RunContext) -> set[Pair]:
        left = list(task.left)
        pairs: set[Pair] = set()
        step_pairs: dict[str, int] = {}
        if self.normalize:
            with ctx.stage("normalize"):
                pairs = normalized_join(left, task.right)
            step_pairs["normalize"] = len(pairs)

        stages = [(name, step) for name, step in self.steps]
        if self.residual_step is not None:
            stages.append(("llm_residual", self.residual_step))
        for name, step in stages:
            matched = {r for _, r in pairs}
            remaining = [r for r in task.right if r not in matched]
            if not remaining:
                step_pairs[name] = 0
                continue
            with ctx.stage(name):
                new = step.propose(left, remaining, ctx, set(pairs))
            remaining_set = set(remaining)
            new = {p for p in new if p[1] in remaining_set}
            step_pairs[name] = len(new)
            pairs |= new

        ctx.annotate(strategy=self.strategy_name, step_pairs=step_pairs)
        return pairs
