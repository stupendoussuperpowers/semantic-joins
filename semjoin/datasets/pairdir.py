"""Read and write the task-directory format described in docs/data-format.md.

    <task_id>/left.csv    id,title
    <task_id>/right.csv   id,title
    <task_id>/gt.csv      id_l,title_l,id_r,title_r
    <task_id>/meta.json   optional, e.g. {"mismatch_type": "code"}

This is the Auto-FuzzyJoin benchmark layout; all datasets use it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from semjoin.core import JoinTask, Pair


def _read_titles(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return tuple(dict.fromkeys(row["title"] for row in csv.DictReader(f)))


def read_task(task_dir: Path, dataset: str) -> JoinTask:
    left = _read_titles(task_dir / "left.csv")
    right = _read_titles(task_dir / "right.csv")
    with (task_dir / "gt.csv").open(newline="", encoding="utf-8-sig") as f:
        gold = frozenset((row["title_l"], row["title_r"]) for row in csv.DictReader(f))
    left_set, right_set = set(left), set(right)
    bad = [p for p in gold if p[0] not in left_set or p[1] not in right_set]
    if bad:
        raise ValueError(f"{task_dir}: {len(bad)} gold pairs not in left/right, e.g. {bad[0]!r}")
    meta_path = task_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return JoinTask(
        dataset=dataset,
        task_id=task_dir.name,
        left=left,
        right=right,
        gold=gold,
        mismatch_type=meta.get("mismatch_type"),
    )


def read_dataset(root: Path, dataset: str) -> list[JoinTask]:
    if not root.exists():
        return []
    return [read_task(d, dataset) for d in sorted(root.iterdir()) if (d / "gt.csv").exists()]


def write_task(
    task_dir: Path,
    left: Iterable[str],
    right: Iterable[str],
    gold: Iterable[Pair],
    meta: dict[str, Any] | None = None,
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    left_ids = {v: i for i, v in enumerate(dict.fromkeys(left))}
    right_ids = {v: i for i, v in enumerate(dict.fromkeys(right))}
    for name, ids in (("left", left_ids), ("right", right_ids)):
        with (task_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "title"])
            w.writerows((i, v) for v, i in ids.items())
    with (task_dir / "gt.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id_l", "title_l", "id_r", "title_r"])
        for l, r in sorted(set(gold)):
            w.writerow([left_ids[l], l, right_ids[r], r])
    if meta:
        (task_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
