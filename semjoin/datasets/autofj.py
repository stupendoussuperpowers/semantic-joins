"""The 50 Auto-FuzzyJoin benchmark tasks (Li et al., SIGMOD 2021).

``fetch`` downloads them into the raw cache (``data/raw/autofj/``);
``build`` copies them into ``data/autofj/`` and adds a ``meta.json``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from semjoin.datasets.pairdir import read_task
from semjoin.datasets.raw import RawCache
from semjoin.datasets.tagging import tag_task

SOURCE = "autofj"
REPO = "chu-data-lab/AutomaticFuzzyJoin"
# Pinned for reproducibility; the repo's last commit as of 2021-09-09.
COMMIT = "2e638b2dd17da41abf5ed71575cea94b1d175ccd"
BENCHMARK_PATH = "src/autofj/benchmark"
FILES = ("left.csv", "right.csv", "gt.csv")
TREE = "tree.json"


def _task_names(tree: dict) -> list[str]:
    prefix = BENCHMARK_PATH + "/"
    return sorted(
        {
            p["path"][len(prefix) :].split("/")[0]
            for p in tree["tree"]
            if p["path"].startswith(prefix) and p["path"].endswith("/gt.csv")
        }
    )


def fetch(data_root: Path, names: list[str] | None = None, refresh: bool = False, **kw) -> list[str]:
    cache = RawCache(data_root, SOURCE)
    cache.pin(repo=REPO, commit=COMMIT)
    tree_path = cache.fetch(
        TREE, f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1", refresh=refresh, **kw
    )
    names = names or _task_names(json.loads(tree_path.read_text()))
    for name in names:
        for f in FILES:
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{BENCHMARK_PATH}/{name}/{f}"
            cache.fetch(f"{name}/{f}", url, refresh=refresh, **kw)
    return names


def build(data_root: Path, out_dir: Path | None = None, names: list[str] | None = None) -> list[str]:
    cache = RawCache(data_root, SOURCE)
    out_dir = out_dir or data_root / SOURCE
    names = names or _task_names(json.loads(cache.require(TREE).read_text()))
    for name in names:
        task_dir = out_dir / name
        task_dir.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            shutil.copyfile(cache.require(f"{name}/{f}"), task_dir / f)
        gold = read_task(task_dir, SOURCE).gold
        mismatch_type, type_mix = tag_task(gold)
        meta = {
            "source": "autofj",
            "repo": REPO,
            "commit": COMMIT,
            "mismatch_type": mismatch_type,
            "type_mix": type_mix,
            "type_tagger": "semjoin.datasets.tagging",
        }
        (task_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return names
