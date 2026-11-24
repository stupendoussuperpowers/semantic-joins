"""Re-route a finished run under other router thresholds, without re-running anything.

The router returns exactly the pairs of the strategy it picks, and records the
profile it picked from. So in a run that contains the router and every
strategy it can pick, the outcome of any other thresholds is a lookup:
re-route each task from its recorded profile and take that strategy's
recorded result (F1, pair counts, LLM calls). Use it to tune thresholds on the
dev split, then evaluate the chosen ones on test once.

The report also gives reference rows: the router as it ran, the router with
the current default thresholds, every fixed strategy, perfect routing (per
task, the best strategy, the cheapest on ties) and ``cascade`` if the run has
it. ``regret`` is perfect routing's F1 minus the row's, averaged over tasks,
so a negative value means better than any single strategy.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from semjoin.matchers import _parse_value, parse_spec
from semjoin.matchers.profiling import Profile
from semjoin.matchers.router import RouterMatcher, strategy_spec, strategy_specs
from semjoin.metrics import Counts, micro
from semjoin.summarize import load_results, to_markdown

OBJECTIVES = ("macro_f1", "micro_f1", "accuracy", "regret")

DEFAULT_GRID: dict[str, list[Any]] = {
    "acronym_min": [0.3, 0.5, 0.7],
    "code_len": [3.0, 4.0, 5.0],
    "code_uniformity": [0.6, 0.8, 0.9],
    "reorder_min": [0.1, 0.25, 0.5],
    "typo_cos": [0.4, 0.45, 0.5, 0.55, 0.6],
}

Record = dict[str, Any]


def parse_grid(text: str) -> dict[str, list[Any]]:
    """``"typo_cos=0.4,0.5;code_len=3,4"`` -> ``{"typo_cos": [0.4, 0.5], "code_len": [3, 4]}``."""
    grid = {}
    for item in filter(None, (s.strip() for s in text.split(";"))):
        key, sep, values = item.partition("=")
        if not sep:
            raise ValueError(f"bad grid item {item!r}; expected name=value1,value2")
        grid[key.strip()] = [_parse_value(v.strip()) for v in values.split(",") if v.strip()]
    return grid


def _row(label: str, picks: dict[str, Record], oracle_f1: dict[str, float], strategy_row: bool = True) -> dict[str, Any]:
    counts = micro(Counts(r["tp"], r["fp"], r["fn"]) for r in picks.values())
    return {
        "config": label,
        # Only meaningful for rows that pick one of the strategies per task.
        "accuracy": mean(r["f1"] == oracle_f1[k] for k, r in picks.items()) if strategy_row else None,
        "regret": mean(oracle_f1[k] - r["f1"] for k, r in picks.items()),
        "macro_f1": mean(r["f1"] for r in picks.values()),
        "micro_f1": counts.f1,
        "llm_calls": sum(r["llm_calls"] for r in picks.values()),
    }


def evaluate_settings(
    records: list[Record],
    router: str = "router",
    strategies: list[str] | None = None,
    grid: dict[str, list[Any]] | None = None,
    objective: str = "macro_f1",
) -> dict[str, Any]:
    """Reference rows and one row per grid setting, best first by ``objective``."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")
    name, base_kwargs = parse_spec(router)
    if name != "router":
        raise ValueError(f"{router!r} is not a router spec")
    strategies = strategies or strategy_specs(RouterMatcher(**base_kwargs).residual)
    grid = DEFAULT_GRID if grid is None else grid

    by_task: dict[str, dict[str, Record]] = defaultdict(dict)
    for r in records:
        by_task[f"{r['dataset']}/{r['task_id']}"][r["matcher"]] = r
    tasks = {
        k: v
        for k, v in by_task.items()
        if router in v and "profile" in v[router]["annotations"] and all(s in v for s in strategies)
    }
    if not tasks:
        raise ValueError(f"no task has {router!r} with a profile and all of {strategies}; run them together")
    try:
        profiles = {k: Profile(**v[router]["annotations"]["profile"]) for k, v in tasks.items()}
    except TypeError as e:
        raise ValueError("the recorded profiles come from another version of the router; run it again") from e
    oracle_f1 = {k: max(v[s]["f1"] for s in strategies) for k, v in tasks.items()}

    def picks_for(matcher: RouterMatcher) -> dict[str, Record]:
        picks = {}
        for k, v in tasks.items():
            spec = strategy_spec(matcher.decide(profiles[k])[1], matcher.residual)
            if spec not in v:
                raise ValueError(f"the router picks {spec!r} on {k}, which is not in the run")
            picks[k] = v[spec]
        return picks

    perfect = {
        k: min((v[s] for s in strategies if v[s]["f1"] == oracle_f1[k]), key=lambda r: r["llm_calls"])
        for k, v in tasks.items()
    }
    reference = [
        _row(f"{router} (as run)", {k: v[router] for k, v in tasks.items()}, oracle_f1),
        _row(f"{router} (current defaults)", picks_for(RouterMatcher(**base_kwargs)), oracle_f1),
        *(_row(f"always {s}", {k: v[s] for k, v in tasks.items()}, oracle_f1) for s in strategies),
        _row("perfect routing", perfect, oracle_f1),
    ]
    if all("cascade" in v for v in tasks.values()):
        reference.append(_row("cascade", {k: v["cascade"] for k, v in tasks.items()}, oracle_f1, strategy_row=False))

    settings = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        label = ",".join(f"{p}={v}" for p, v in params.items()) or "(defaults)"
        settings.append(_row(label, picks_for(RouterMatcher(**{**base_kwargs, **params})), oracle_f1))
    # Best first; ties go to fewer LLM calls, then to grid order.
    settings.sort(key=lambda row: (row[objective] if objective == "regret" else -row[objective], row["llm_calls"]))
    return {"tasks": len(tasks), "skipped": len(by_task) - len(tasks), "reference": reference, "settings": settings}


def tune(
    run_dir: Path,
    router: str = "router",
    strategies: list[str] | None = None,
    grid: dict[str, list[Any]] | None = None,
    objective: str = "macro_f1",
    top: int = 10,
) -> str:
    res = evaluate_settings(load_results(run_dir), router, strategies, grid, objective)
    best = res["settings"][:top]
    return (
        f"## Tuning {router} on {run_dir}\n\n"
        f"tasks={res['tasks']} skipped={res['skipped']} settings={len(res['settings'])} objective={objective}\n\n"
        "### Reference\n\n" + to_markdown(res["reference"])
        + f"\n### Best {len(best)} settings (ties: the split cannot tell them apart)\n\n" + to_markdown(best)
    )
