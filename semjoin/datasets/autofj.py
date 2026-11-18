"""Download the 50 Auto-FuzzyJoin benchmark tasks (Li et al., SIGMOD 2021)."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = "chu-data-lab/AutomaticFuzzyJoin"
# Pinned for reproducibility; the repo's last commit as of 2021-09-09.
COMMIT = "2e638b2dd17da41abf5ed71575cea94b1d175ccd"
BENCHMARK_PATH = "src/autofj/benchmark"
FILES = ("left.csv", "right.csv", "gt.csv")


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def list_tasks() -> list[str]:
    tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    prefix = BENCHMARK_PATH + "/"
    return sorted(
        {
            p["path"][len(prefix) :].split("/")[0]
            for p in tree["tree"]
            if p["path"].startswith(prefix) and p["path"].endswith("/gt.csv")
        }
    )


def download(out_dir: Path, names: list[str] | None = None) -> list[str]:
    names = names or list_tasks()
    for i, name in enumerate(names, 1):
        task_dir = out_dir / name
        if all((task_dir / f).exists() for f in FILES):
            continue
        print(f"[{i}/{len(names)}] {name}", file=sys.stderr)
        task_dir.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{BENCHMARK_PATH}/{name}/{f}"
            (task_dir / f).write_bytes(_get(url))
    return names
