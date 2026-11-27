"""The Auto-Join web-tables benchmark (Zhu, He, Chaudhuri, PVLDB 2017), held out.

31 pairs of real web tables (presidents, governors, parks, songs, staff
directories, ...) with value-level ground truth, as redistributed by DTT
(Dargahi Nobari and Rafiei, SIGMOD 2024) in ``data.tar.gz`` of their repo.
The same archive's ``FF-*`` FlashFill tables and ``names200`` are not used:
they are synthetic substring-extraction tasks, not real-world joins.

Per table, ``rows.txt`` names the join columns (``source column:target
column``). The source column becomes ``left``, the target column ``right``;
both keep their unmatched values. Gold pairs come from ``ground truth.csv``.
A few published gold values differ from their column only in whitespace
("New Mexico" vs. "NewMexico"); those are mapped to the column value, and
any other gold value missing from its column is dropped. Both counts are
recorded in ``meta.json``.

This is the real-world check: ``splits.json`` puts every task in ``test``
(``HELDOUT_DATASETS`` in ``semjoin/splits.py``), so it is never tuned on.
Types come from the heuristic tagger, for the per-type breakdown only.
"""

from __future__ import annotations

import csv
import io
import tarfile
from pathlib import Path

from semjoin.datasets.pairdir import write_task
from semjoin.datasets.raw import RawCache
from semjoin.datasets.tagging import tag_task

SOURCE = "autojoin"
REPO = "arashdn/dtt"
COMMIT = "a98cdc012a65d30891aba1f07d201052d9769487"  # main as of 2024-06-13
ARCHIVE = "data.tar.gz"  # a plain tar despite the name
PREFIX = "data/Datasets/FF_AJ/"


def fetch(data_root: Path, refresh: bool = False, **kw) -> None:
    cache = RawCache(data_root, SOURCE)
    cache.pin(repo=REPO, commit=COMMIT)
    cache.fetch(ARCHIVE, f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{ARCHIVE}", refresh=refresh, **kw)


def _column(text: str, name: str) -> list[str]:
    return [r[name].strip() for r in csv.DictReader(io.StringIO(text)) if (r.get(name) or "").strip()]


def _squash(value: str) -> str:
    return "".join(value.split())


def _resolve(value: str, column: set[str], squashed: dict[str, list[str]]) -> str | None:
    if value in column:
        return value
    candidates = squashed.get(_squash(value), [])
    return candidates[0] if len(candidates) == 1 else None


def read_table(files: dict[str, str]) -> tuple[list[str], list[str], set[tuple[str, str]], int, int]:
    """(left, right, gold, repaired, dropped) from one table's file contents."""
    source_col, target_col = files["rows.txt"].splitlines()[0].split(":", 1)
    left = _column(files["source.csv"], source_col)
    right = _column(files["target.csv"], target_col)
    left_set, right_set = set(left), set(right)
    squashed_l: dict[str, list[str]] = {}
    squashed_r: dict[str, list[str]] = {}
    for v in left_set:
        squashed_l.setdefault(_squash(v), []).append(v)
    for v in right_set:
        squashed_r.setdefault(_squash(v), []).append(v)
    gold, repaired, dropped = set(), 0, 0
    for row in csv.DictReader(io.StringIO(files["ground truth.csv"])):
        l, r = (row.get("source-" + source_col) or "").strip(), (row.get("target-" + target_col) or "").strip()
        if not l or not r:
            continue
        rl, rr = _resolve(l, left_set, squashed_l), _resolve(r, right_set, squashed_r)
        if rl is None or rr is None:
            dropped += 1
            continue
        repaired += (rl, rr) != (l, r)
        gold.add((rl, rr))
    return left, right, gold, repaired, dropped


def _tables(archive: Path) -> dict[str, dict[str, str]]:
    """table name -> {file name: text} for the AJ-* tables, read without extracting."""
    out: dict[str, dict[str, str]] = {}
    with tarfile.open(archive, "r:*") as tar:
        for m in tar.getmembers():
            if not m.isfile() or not m.name.startswith(PREFIX + "AJ-"):
                continue
            table, fname = m.name[len(PREFIX):].split("/", 1)
            if "/" in fname:
                continue
            out.setdefault(table, {})[fname] = tar.extractfile(m).read().decode("utf-8")
    return out


def build(data_root: Path, out_dir: Path | None = None) -> list[str]:
    out_dir = out_dir or data_root / SOURCE
    archive = RawCache(data_root, SOURCE).require(ARCHIVE)
    built = []
    for table, files in sorted(_tables(archive).items()):
        left, right, gold, repaired, dropped = read_table(files)
        mismatch_type, type_mix = tag_task(gold)
        task_id = table.removeprefix("AJ-")
        write_task(
            out_dir / task_id,
            left=left,
            right=right,
            gold=gold,
            meta={
                "source": "autojoin",
                "repo": REPO,
                "commit": COMMIT,
                "table": table,
                "join_columns": files["rows.txt"].splitlines()[0],
                "mismatch_type": mismatch_type,
                "type_mix": type_mix,
                "type_tagger": "semjoin.datasets.tagging",
                "gold_repaired": repaired,
                "gold_dropped": dropped,
            },
        )
        built.append(task_id)
    return built
