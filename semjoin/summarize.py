"""Aggregate a run directory into tables (and an optional plot).

Writes into the run directory:

    summary.md / summary.csv   per (matcher, dataset), plus an ALL row
    by_type.csv                per (matcher, mismatch type)
    routing.md                 only with --router: RQ1 routing accuracy
    routing.csv                only with --router: the same rows plus the router's profile
    cost_quality.png           F1 vs LLM calls, if matplotlib is installed
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from semjoin.metrics import Counts, micro

ALL = "ALL"


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "results.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    c = micro(Counts(r["tp"], r["fp"], r["fn"]) for r in records)
    hard = micro(Counts(r["hard_tp"], r["hard_fp"], r["hard_fn"]) for r in records)
    return {
        "tasks": len(records),
        "errors": sum(r["error"] is not None for r in records),
        "micro_p": c.precision,
        "micro_r": c.recall,
        "micro_f1": c.f1,
        "macro_f1": mean(r["f1"] for r in records),
        "hard_f1": hard.f1,
        "llm_calls": sum(r["llm_calls"] for r in records),
        "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in records),
        "runtime_s": sum(r["runtime_s"] for r in records),
    }


def by_dataset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in records:
        groups[(r["matcher"], r["dataset"])].append(r)
        groups[(r["matcher"], ALL)].append(r)
    return [
        {"matcher": m, "dataset": d, **_aggregate(rs)}
        for (m, d), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] == ALL, kv[0][1]))
    ]


def by_type(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in records:
        groups[(r["matcher"], r["mismatch_type"] or "unknown")].append(r)
    return [{"matcher": m, "mismatch_type": t, **_aggregate(rs)} for (m, t), rs in sorted(groups.items())]


def routing(records: list[dict[str, Any]], router: str, strategies: list[str]) -> dict[str, Any]:
    """Compare a router's chosen strategy with the per-task best strategy.

    The router must record its choice with ``ctx.annotate(strategy=<spec>)``,
    where <spec> is one of ``strategies``. A choice is correct if it ties the
    best F1 among the strategies on that task. Regret is best F1 minus the
    router's own F1.
    """
    if not strategies:
        raise ValueError("routing evaluation needs at least one strategy to compare against")
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_task[f"{r['dataset']}/{r['task_id']}"][r["matcher"]] = r
    rows = []
    for key, results in sorted(by_task.items()):
        if router not in results or not all(s in results for s in strategies):
            continue
        best_f1 = max(results[s]["f1"] for s in strategies)
        oracle = [s for s in strategies if results[s]["f1"] == best_f1]
        annotations = results[router]["annotations"]
        chosen = annotations.get("strategy")
        rows.append(
            {
                "task": key,
                "chosen": chosen,
                "rule": annotations.get("route_rule"),
                "oracle": "|".join(oracle),
                "correct": chosen in oracle,
                "router_f1": results[router]["f1"],
                "oracle_f1": best_f1,
                "regret": best_f1 - results[router]["f1"],
            }
        )
    return {
        "tasks": len(rows),
        "accuracy": float(mean(r["correct"] for r in rows)) if rows else float("nan"),
        "mean_regret": mean(r["regret"] for r in rows) if rows else float("nan"),
        # Macro F1 of the router and of perfect routing among the strategies (an upper bound).
        "router_f1": mean(r["router_f1"] for r in rows) if rows else float("nan"),
        "oracle_f1": mean(r["oracle_f1"] for r in rows) if rows else float("nan"),
        "rows": rows,
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def to_markdown(rows: Iterable[dict[str, Any]]) -> str:
    rows = list(rows)
    if not rows:
        return "_no results_\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_cost_quality(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    rows = [r for r in rows if r["dataset"] == ALL]
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in rows:
        ax.scatter(r["llm_calls"], r["micro_f1"])
        ax.annotate(r["matcher"], (r["llm_calls"], r["micro_f1"]), textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("LLM calls (all tasks)")
    ax.set_ylabel("micro F1")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def summarize(run_dir: Path, router: str | None = None, strategies: list[str] | None = None) -> str:
    records = load_results(run_dir)
    ds_rows = by_dataset(records)
    type_rows = by_type(records)
    write_csv(run_dir / "summary.csv", ds_rows)
    write_csv(run_dir / "by_type.csv", type_rows)
    report = "## By dataset\n\n" + to_markdown(ds_rows) + "\n## By mismatch type\n\n" + to_markdown(type_rows)
    if router:
        res = routing(records, router, strategies or [])
        routing_md = (
            f"## Routing: {router}\n\n"
            f"tasks={res['tasks']} accuracy={_fmt(res['accuracy'])} mean_regret={_fmt(res['mean_regret'])} "
            f"router_f1={_fmt(res['router_f1'])} oracle_f1={_fmt(res['oracle_f1'])}\n\n"
            + to_markdown(res["rows"])
        )
        (run_dir / "routing.md").write_text(routing_md)
        profiles = {
            f"{r['dataset']}/{r['task_id']}": r["annotations"].get("profile", {}) for r in records if r["matcher"] == router
        }
        # Same columns on every row: a task the router failed on has no profile.
        fields = list(dict.fromkeys(k for p in profiles.values() for k in p))
        write_csv(
            run_dir / "routing.csv",
            [{**row, **{k: profiles.get(row["task"], {}).get(k) for k in fields}} for row in res["rows"]],
        )
        report += "\n" + routing_md
    (run_dir / "summary.md").write_text(report)
    plot_cost_quality(run_dir / "cost_quality.png", ds_rows)
    return report
