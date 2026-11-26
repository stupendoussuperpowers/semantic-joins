"""Checks run before data is pushed: ``semjoin validate``.

Per task (errors make the data unusable, warnings make it weak):

* error: the task does not load (bad CSV, gold pairs not in left/right)
* error: empty or whitespace-only values; no gold pairs
* error: an unknown ``mismatch_type``; a task missing from the split manifest
* warn:  no ``mismatch_type``; a ``mixed`` task without ``type_mix`` in meta
* warn:  fewer than ``min_right`` right values
* warn:  too few distractors: |left| / |left values in gold| < ``min_distractor_ratio``
  (not for held-out real-world datasets, which are used as published)

Across tasks, per mismatch type, a warning when dev or test has fewer tasks
than ``min_dev`` / ``min_test``, since routing rules cannot be tuned or
evaluated on a type that is missing from a split.

Also re-hashes every file in the raw cache (``data/raw/``).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from semjoin.datasets import DATASETS
from semjoin.datasets.pairdir import read_task
from semjoin.datasets.raw import RawCache
from semjoin.splits import HELDOUT_DATASETS

MISMATCH_TYPES = ("code", "alias", "typo", "reorder", "acronym", "fuzzy", "mixed")
UNTYPED = "untyped"


@dataclass
class Issue:
    level: str  # "error" or "warn"
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():5} {self.where}: {self.message}"


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    # (mismatch type, split) -> task count
    coverage: Counter = field(default_factory=Counter)
    n_tasks: int = 0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warn"]

    def ok(self, strict: bool = False) -> bool:
        return not self.errors and not (strict and self.warnings)

    def format(self) -> str:
        types = sorted({t for t, _ in self.coverage})
        splits = sorted({s for _, s in self.coverage})
        lines = [f"{self.n_tasks} tasks", "", "| type | " + " | ".join(splits) + " | total |"]
        lines.append("|---" * (len(splits) + 2) + "|")
        for t in types:
            row = [self.coverage[t, s] for s in splits]
            lines.append(f"| {t} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
        lines.append("")
        lines.extend(str(i) for i in self.issues)
        lines.append(f"{len(self.errors)} errors, {len(self.warnings)} warnings")
        return "\n".join(lines)


def validate_task_dir(
    task_dir: Path,
    dataset: str,
    manifest: dict[str, str] | None,
    min_right: int = 20,
    min_distractor_ratio: float = 1.5,
) -> tuple[list[Issue], str | None, str | None]:
    """Return (issues, mismatch type, split) for one task directory."""
    where = f"{dataset}/{task_dir.name}"
    issues: list[Issue] = []

    def add(level: str, message: str) -> None:
        issues.append(Issue(level, where, message))

    try:
        task = read_task(task_dir, dataset)
    except Exception as e:  # noqa: BLE001 -- any load failure is a validation error
        add("error", f"does not load: {e}")
        return issues, None, None

    meta_path = task_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    for side, values in (("left", task.left), ("right", task.right)):
        blank = sum(1 for v in values if not v.strip())
        if blank:
            add("error", f"{blank} empty values in {side}.csv")
    if not task.gold:
        add("error", "gt.csv has no pairs")

    mt = task.mismatch_type
    if mt is None:
        add("warn", "no mismatch_type in meta.json")
    elif mt not in MISMATCH_TYPES:
        add("error", f"unknown mismatch_type {mt!r}; expected one of {MISMATCH_TYPES}")
    elif mt == "mixed" and "type_mix" not in meta:
        add("warn", "mixed task without type_mix in meta.json")

    if len(task.right) < min_right:
        add("warn", f"only {len(task.right)} right values (< {min_right})")
    gold_left = {l for l, _ in task.gold}
    # Held-out real-world tables are used as published, so their overlap is not a defect.
    if dataset not in HELDOUT_DATASETS and gold_left and len(task.left) / len(gold_left) < min_distractor_ratio:
        add(
            "warn",
            f"few distractors: {len(task.left)} left values for {len(gold_left)} matched "
            f"(ratio < {min_distractor_ratio})",
        )

    split = None
    if manifest is not None:
        split = manifest.get(task.key)
        if split is None:
            add("error", "not in the split manifest; run `semjoin split`")
    return issues, mt, split


def validate(
    data_root: Path,
    datasets: list[str] | None = None,
    manifest: dict[str, str] | None = None,
    min_dev: int = 2,
    min_test: int = 4,
    **task_kw,
) -> Report:
    report = Report()
    for dataset in datasets or DATASETS:
        root = data_root / dataset
        if not root.exists():
            continue
        for task_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            if not (task_dir / "gt.csv").exists():
                report.issues.append(Issue("error", f"{dataset}/{task_dir.name}", "no gt.csv"))
                continue
            issues, mt, split = validate_task_dir(task_dir, dataset, manifest, **task_kw)
            report.issues.extend(issues)
            report.n_tasks += 1
            report.coverage[mt or UNTYPED, split or "unsplit"] += 1

    if manifest is not None:
        for t in MISMATCH_TYPES:
            if t in ("mixed", "fuzzy") and not any(k == t for k, _ in report.coverage):
                continue  # optional types: only checked once some exist
            for split, need in (("dev", min_dev), ("test", min_test)):
                have = report.coverage[t, split]
                if have < need:
                    report.issues.append(
                        Issue("warn", f"coverage/{t}", f"{have} {split} tasks (want >= {need})")
                    )

    raw = data_root / "raw"
    if raw.exists():
        for src in sorted(d.name for d in raw.iterdir() if d.is_dir()):
            for problem in RawCache(data_root, src).verify():
                report.issues.append(Issue("error", "raw", problem))
    return report
