"""Seeded slices of the task list, for cheap smoke runs before a full run.

* ``select_tasks``: at most N tasks per (dataset, mismatch type), so a slice
  still exercises every routing rule.
* ``slice_right``: at most N right values per task, keeping only the gold
  pairs whose right value survives. The left column is kept whole so the
  distractors stay. Recall on the slice is therefore comparable in kind, but
  not equal, to recall on the full task.

Both are deterministic, so a slice can be re-run.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import replace

from semjoin.core import JoinTask


def _rank(key: str, seed: int) -> str:
    return hashlib.sha256(f"slice:{seed}:{key}".encode()).hexdigest()


def select_tasks(tasks: list[JoinTask], max_per_group: int, seed: int = 0) -> list[JoinTask]:
    groups: dict[tuple[str, str | None], list[JoinTask]] = defaultdict(list)
    for t in tasks:
        groups[t.dataset, t.mismatch_type].append(t)
    keep = set()
    for group in groups.values():
        keep.update(t.key for t in sorted(group, key=lambda t: _rank(t.key, seed))[:max_per_group])
    return [t for t in tasks if t.key in keep]  # keep the original order


def slice_right(task: JoinTask, max_right: int, seed: int = 0) -> JoinTask:
    if len(task.right) <= max_right:
        return task
    rng = random.Random(f"slice:{seed}:{task.key}")
    kept = set(rng.sample(task.right, max_right))
    return replace(
        task,
        right=tuple(r for r in task.right if r in kept),
        gold=frozenset(p for p in task.gold if p[1] in kept),
    )
