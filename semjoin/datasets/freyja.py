"""The FREYJA join-discovery benchmark (Maynou et al., 2024): fetch and candidates.

FREYJA's benchmark is a data lake of CSV files plus a ground truth of which
column pairs are *joinable*. It does not say which *values* match, so it
cannot be used as a task directly: value matches must be labeled by hand
(docs/data-format.md). This module covers the automatic part:

* ``fetch`` caches the lake (``datalake.zip``, ~34 MB, from the authors'
  share) and the column-level ground truth (pinned to a repo commit).
* ``candidates`` profiles every ground-truth column pair and ranks the ones
  worth labeling: string columns where a normalized equi-join leaves many
  values unmatched, i.e. where a semantic join has something to do.
"""

from __future__ import annotations

import csv
import difflib
import io
import random
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from semjoin.datasets.raw import RawCache
from semjoin.normalize import normalize

SOURCE = "freyja"
REPO = "dtak-upc/FREYJA"
COMMIT = "42b0532260895c0ed712ecf7dfb3b07a5ce44d0e"  # main as of 2026-07-02
GROUND_TRUTH = "freyja_ground_truth.csv"
LAKE_URL = "https://mydisk.cs.upc.edu/s/QHJbKcyeacxq35f/download"
LAKE = "datalake.zip"


def fetch(data_root: Path, names: list[str] | None = None, refresh: bool = False, **kw) -> list[str]:
    cache = RawCache(data_root, SOURCE)
    cache.pin(repo=REPO, commit=COMMIT, lake_url=LAKE_URL)
    gt_url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/ground_truths/{GROUND_TRUTH}"
    cache.fetch(GROUND_TRUTH, gt_url, refresh=refresh, **kw)
    cache.fetch(LAKE, LAKE_URL, refresh=refresh, **kw)
    return [GROUND_TRUTH, LAKE]


def ground_truth(data_root: Path) -> list[tuple[str, str, str, str]]:
    """(target table, target column, candidate table, candidate column) rows."""
    path = RawCache(data_root, SOURCE).require(GROUND_TRUTH)
    with path.open(newline="", encoding="utf-8") as f:
        return [
            (r["target_ds"], r["target_attr"], r["candidate_ds"], r["candidate_attr"]) for r in csv.DictReader(f)
        ]


class Lake:
    """Read columns straight out of the cached zip, without unpacking it."""

    def __init__(self, data_root: Path):
        self.zip = zipfile.ZipFile(RawCache(data_root, SOURCE).require(LAKE))
        self._by_name = {Path(n).name: n for n in self.zip.namelist() if n.endswith(".csv")}
        self._columns: dict[str, dict[str, list[str]]] = {}

    def tables(self) -> list[str]:
        return sorted(self._by_name)

    def column(self, table: str, column: str) -> list[str] | None:
        if table not in self._columns:
            member = self._by_name.get(table)
            if member is None:
                return None
            raw = self.zip.read(member)
            text = raw.decode("utf-8", errors="replace")
            try:
                dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            rows = csv.DictReader(io.StringIO(text), dialect=dialect)
            cols: dict[str, list[str]] = {}
            for row in rows:
                for k, v in row.items():
                    if k is not None and v is not None:
                        cols.setdefault(k.strip(), []).append(v.strip())
            self._columns[table] = cols
        return self._columns[table].get(column)


@dataclass
class Candidate:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    left_distinct: int
    right_distinct: int
    exact_overlap: float  # share of right values an exact join matches
    normalized_overlap: float  # ... after normalization
    unmatched_right: int  # distinct right values a normalized join leaves
    numeric_share: float  # share of right values that parse as numbers
    list_share: float  # share of right values that look like lists ("A, B, C")
    id_share: float  # share of right values that look like identifiers ("d23_1", "A-1043")
    near_miss: float  # share of sampled unmatched right values with a close left value
    examples_left: str
    examples_right: str


def _distinct(values: list[str]) -> list[str]:
    return sorted({v for v in values if v})


def _numeric_share(values: list[str]) -> float:
    def is_num(v: str) -> bool:
        try:
            float(v.replace(",", ""))
            return True
        except ValueError:
            return False

    return sum(map(is_num, values)) / len(values) if values else 0.0


def _near_miss(unmatched: list[str], left: list[str], sample: int = 40, cutoff: float = 0.75) -> float:
    """Share of (a seeded sample of) unmatched values with a close but unequal left value.

    High: the value exists on the other side in another form (typo, format),
    the case a semantic join is for. Low: the value is simply absent, or
    differs beyond string similarity (codes, aliases), which this cannot tell.
    """
    if not unmatched:
        return 0.0
    unmatched = [v for v in unmatched if len(v) >= 5]  # short strings are all "close"
    if not unmatched:
        return 0.0
    picked = random.Random(0).sample(unmatched, min(sample, len(unmatched)))
    pool = [normalize(v) for v in left]
    hits = sum(bool(difflib.get_close_matches(normalize(v), pool, n=1, cutoff=cutoff)) for v in picked)
    return round(hits / len(picked), 3)


_ID = re.compile(r"^[A-Za-z]{0,4}[_\-]?\d+([_\-.]\d+)*[A-Za-z]?$")


def _id_share(values: list[str]) -> float:
    return round(sum(1 for v in values if _ID.match(v)) / len(values), 3)


def _list_share(values: list[str]) -> float:
    return round(sum(1 for v in values if len([p for p in v.replace(";", ",").split(",") if p.strip()]) >= 3) / len(values), 3)


def candidates(data_root: Path) -> list[Candidate]:
    """Profile every ground-truth pair; most promising for labeling first.

    Promising: mostly non-numeric, not list-valued, and a normalized equi-join
    leaves between 10% and 90% of the right values unmatched (below that an
    equi-join suffices; above that the columns may not really join). Within
    those, pairs with more near-misses come first. Codes and aliases have no
    near-misses, so also skim further down the list for them.
    """
    lake = Lake(data_root)
    out = []
    seen = set()
    for lt, lc, rt, rc in ground_truth(data_root):
        key = tuple(sorted([(lt, lc), (rt, rc)]))
        if key in seen:
            continue
        seen.add(key)
        left, right = lake.column(lt, lc), lake.column(rt, rc)
        if not left or not right:
            continue
        left, right = _distinct(left), _distinct(right)
        if not left or not right:
            continue
        left_set, left_norm = set(left), {normalize(v) for v in left}
        unmatched = [v for v in right if normalize(v) not in left_norm]
        out.append(
            Candidate(
                lt, lc, rt, rc,
                left_distinct=len(left),
                right_distinct=len(right),
                exact_overlap=round(sum(v in left_set for v in right) / len(right), 3),
                normalized_overlap=round(1 - len(unmatched) / len(right), 3),
                unmatched_right=len(unmatched),
                numeric_share=round(_numeric_share(right), 3),
                list_share=_list_share(right),
                id_share=_id_share(right),
                near_miss=_near_miss(unmatched, left),
                examples_left=" | ".join(left[:: max(1, len(left) // 5)][:5]),
                examples_right=" | ".join(unmatched[:: max(1, len(unmatched) // 5)][:5]),
            )
        )

    return sorted(out, key=lambda c: (not promising(c), -c.near_miss, -c.unmatched_right))


def promising(c: Candidate) -> bool:
    return (
        c.numeric_share < 0.5
        and c.list_share < 0.2
        and c.id_share < 0.5
        and 0.1 <= 1 - c.normalized_overlap <= 0.9
        and c.unmatched_right >= 10
    )


def write_candidates(cands: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(cands[0])) if cands else ["left_table"])
        w.writeheader()
        w.writerows(asdict(c) for c in cands)
