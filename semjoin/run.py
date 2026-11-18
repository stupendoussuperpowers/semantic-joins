"""Run matchers over tasks and record one result line per (task, matcher).

Output layout of a run directory:

    config.json     what was run (matchers, tasks, LLM, git commit, ...)
    results.jsonl   metrics and cost per (task, matcher)
    errors.jsonl    false positive / false negative pairs per (task, matcher)
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from semjoin.core import JoinTask, RunContext
from semjoin.llm import LLMClient
from semjoin.matchers import build_matcher
from semjoin.metrics import hard_only, score


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate(task: JoinTask, matcher_spec: str, llm: LLMClient) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one matcher on one task; return (result record, error record)."""
    matcher = build_matcher(matcher_spec)
    ctx = RunContext(llm)
    error = None
    start = time.perf_counter()
    try:
        pred = set(matcher.match(task, ctx))
    except Exception:
        pred = set()
        error = traceback.format_exc()
    runtime = time.perf_counter() - start
    ctx.finish()

    counts = score(pred, task.gold)
    hard = score(hard_only(pred), hard_only(task.gold))
    base = {
        "dataset": task.dataset,
        "task_id": task.task_id,
        "split": task.split,
        "mismatch_type": task.mismatch_type,
        "matcher": matcher_spec,
    }
    record = {
        **base,
        "n_left": len(task.left),
        "n_right": len(task.right),
        "n_gold": len(task.gold),
        "n_pred": len(pred),
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
        "hard_tp": hard.tp,
        "hard_fp": hard.fp,
        "hard_fn": hard.fn,
        "llm_calls": sum(ctx.llm_calls.values()),
        "prompt_tokens": sum(ctx.prompt_tokens.values()),
        "completion_tokens": sum(ctx.completion_tokens.values()),
        "runtime_s": runtime,
        "stage_seconds": dict(ctx.stage_seconds),
        "stage_llm_calls": dict(ctx.llm_calls),
        "annotations": ctx.annotations,
        "error": error,
    }
    errors = {
        **base,
        "false_positives": sorted(pred - task.gold),
        "false_negatives": sorted(task.gold - pred),
    }
    return record, errors


def run(
    tasks: list[JoinTask],
    matcher_specs: list[str],
    llm_factory: Callable[[], LLMClient],
    out_dir: Path,
    config: dict[str, Any] | None = None,
) -> Path:
    for spec in matcher_specs:
        build_matcher(spec)  # fail fast on a bad spec
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                **(config or {}),
                "matchers": matcher_specs,
                "tasks": [t.key for t in tasks],
                "git_commit": _git_commit(),
                "python": platform.python_version(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=1,
        )
        + "\n"
    )
    llm = llm_factory()
    total = len(tasks) * len(matcher_specs)
    with (out_dir / "results.jsonl").open("w") as rf, (out_dir / "errors.jsonl").open("w") as ef:
        n = 0
        for task in tasks:
            for spec in matcher_specs:
                n += 1
                record, errors = evaluate(task, spec, llm)
                rf.write(json.dumps(record) + "\n")
                ef.write(json.dumps(errors) + "\n")
                rf.flush()
                ef.flush()
                status = "ERROR" if record["error"] else f"F1={record['f1']:.3f}"
                print(f"[{n}/{total}] {task.key} {spec}: {status}", file=sys.stderr)
    return out_dir
