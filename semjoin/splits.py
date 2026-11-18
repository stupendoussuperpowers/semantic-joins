"""Dev/test split assignment, frozen in a manifest file.

Within each dataset, tasks are ordered by a seeded hash of their key and the
first ``round(dev_fraction * n)`` go to dev. The result is written to
``splits.json`` and read back on every run, so adding tasks later never moves
existing ones between splits.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from semjoin.core import JoinTask

SPLITS = ("dev", "test")


def _rank(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def assign(task_keys: list[str], seed: int = 0, dev_fraction: float = 0.3) -> dict[str, str]:
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for key in task_keys:
        by_dataset[key.split("/", 1)[0]].append(key)
    out = {}
    for keys in by_dataset.values():
        keys = sorted(keys, key=lambda k: _rank(k, seed))
        n_dev = round(dev_fraction * len(keys))
        for i, key in enumerate(keys):
            out[key] = "dev" if i < n_dev else "test"
    return dict(sorted(out.items()))


def write_manifest(
    path: Path, tasks: list[JoinTask], seed: int = 0, dev_fraction: float = 0.3
) -> dict[str, str]:
    """Assign splits to tasks not yet in the manifest; keep existing ones."""
    existing = load_manifest(path) if path.exists() else {}
    new_keys = [t.key for t in tasks if t.key not in existing]
    assignment = {**existing, **assign(new_keys, seed, dev_fraction)}
    path.write_text(
        json.dumps(
            {"seed": seed, "dev_fraction": dev_fraction, "tasks": dict(sorted(assignment.items()))},
            indent=1,
        )
        + "\n"
    )
    return assignment


def load_manifest(path: Path) -> dict[str, str]:
    return json.loads(path.read_text())["tasks"]
