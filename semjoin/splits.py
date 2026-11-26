"""Dev/test split assignment, frozen in a manifest file.

Tasks are stratified by (dataset, mismatch type). Within each stratum the
target dev count is ``round(dev_fraction * n)``, clamped so that a stratum of
two or more tasks has at least one task in each split. New tasks are ordered
by a seeded hash of their key, and the first ones go to dev until the
stratum reaches its target, counting tasks the manifest already assigned.

The result is written to ``splits.json`` and read back on every run. Existing
assignments never move, so adding tasks later is safe. Tasks of
``HELDOUT_DATASETS`` all go to test.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from semjoin.core import JoinTask

SPLITS = ("dev", "test")
# Real-world datasets kept out of dev entirely, so they are never tuned on.
HELDOUT_DATASETS = ("autojoin", "nyc")


def _rank(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def _target_dev(n: int, dev_fraction: float) -> int:
    target = round(dev_fraction * n)
    return min(max(target, 1), n - 1) if n >= 2 else target


def assign(
    task_keys: list[str],
    seed: int = 0,
    dev_fraction: float = 0.3,
    strata: dict[str, str] | None = None,
    existing: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assign splits to ``task_keys`` (which must not be in ``existing``).

    ``strata`` maps a key to its stratum (default: its dataset). ``existing``
    holds assignments already made, which count toward each stratum's target.
    """
    strata = strata or {}
    existing = existing or {}

    def stratum(key: str) -> str:
        return strata.get(key, key.split("/", 1)[0])

    new_by: dict[str, list[str]] = defaultdict(list)
    for key in task_keys:
        new_by[stratum(key)].append(key)
    old_dev: dict[str, int] = defaultdict(int)
    old_n: dict[str, int] = defaultdict(int)
    for key, split in existing.items():
        if key in strata or not strata:
            old_n[stratum(key)] += 1
            old_dev[stratum(key)] += split == "dev"

    out = {}
    for s, keys in new_by.items():
        keys = sorted(keys, key=lambda k: _rank(k, seed))
        need = _target_dev(old_n[s] + len(keys), dev_fraction) - old_dev[s]
        need = min(max(need, 0), len(keys))
        for i, key in enumerate(keys):
            out[key] = "dev" if i < need else "test"
    return dict(sorted(out.items()))


def stratum_of(task: JoinTask) -> str:
    return f"{task.dataset}/{task.mismatch_type or 'untyped'}"


def write_manifest(
    path: Path,
    tasks: list[JoinTask],
    seed: int = 0,
    dev_fraction: float = 0.3,
) -> dict[str, str]:
    """Assign splits to tasks not yet in the manifest; keep existing ones."""
    existing = load_manifest(path) if path.exists() else {}
    strata = {t.key: stratum_of(t) for t in tasks}
    new = [t for t in tasks if t.key not in existing]
    heldout = {t.key: "test" for t in new if t.dataset in HELDOUT_DATASETS}
    new_keys = [t.key for t in new if t.key not in heldout]
    assignment = {**existing, **heldout, **assign(new_keys, seed, dev_fraction, strata, existing)}
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "dev_fraction": dev_fraction,
                "stratify": ["dataset", "mismatch_type"],
                "tasks": dict(sorted(assignment.items())),
            },
            indent=1,
        )
        + "\n"
    )
    return assignment


def load_manifest(path: Path) -> dict[str, str]:
    return json.loads(path.read_text())["tasks"]
