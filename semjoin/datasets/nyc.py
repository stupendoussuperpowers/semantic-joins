"""Real-world joins from NYC Open Data (data.cityofnewyork.us), held out.

Six tasks whose gold comes from the data itself, so nothing is hand-labelled:

* ``agency_acronym``: agency name -> current or former acronym, same row of
  "NYC Agencies and Governance Organizations" (``DCWP``, ``DOE``, ``NYC DOT``)
* ``agency_former_name``: agency name -> alternate or former name, same row
  ("Office of Technology and Innovation" -> "Department of Information Technology
  and Telecommunications")
* ``school_name``: 2012 SAT results -> 2010 SAT results, same DBN
  ("NEW EXPLORATIONS INTO SCIENCE, TECHNOLOGY AND MATH HIGH SCHOOL" vs.
  "New Explorations into Sci, Tech and Math HS"). A DBN is reused when a
  school closes and another opens, so pairs sharing no distinctive word
  ("MARTA VALLE HIGH SCHOOL" vs. "CMSP HIGH SCHOOL") are dropped and counted.
* ``nta_code`` / ``nta_abbrev``: 2020 neighborhood tabulation area name -> its
  code (``BK0101``) / its official abbreviation (``Wllmsbrg``), same row
* ``building_address``: NYCHA address -> how building permit filings at DOB
  spell the same building (same BIN): ``1162 WASHINGTON AVENUE`` vs.
  ``1162 WASHINGTON AVE``, ``303 VERNON AV``. Only spellings with the same
  house number are gold; other numbers on the building (``301-305 VERNON
  AVENUE``) stay as near-miss distractors. Half the left buildings are not
  queried on the right, as further distractors.

Every task keeps its unmatched values. Where the type is known by construction
(codes, acronyms, former names) ``meta.json`` declares it; otherwise the
heuristic tagger assigns it. ``splits.json`` puts them all in
``test`` (``HELDOUT_DATASETS``). Open
Data tables are edited over time, so like ``data/wikidata/`` the built tasks
are committed, and ``meta.json`` records the dataset ids and fetch dates.
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
import re
from pathlib import Path
from urllib.parse import urlencode

from semjoin.datasets.pairdir import write_task
from semjoin.datasets.raw import RawCache
from semjoin.datasets.tagging import tag_task
from semjoin.normalize import normalize

SOURCE = "nyc"
RESOURCE = "https://data.cityofnewyork.us/resource"
# cache file -> (Socrata dataset id, SoQL query)
QUERIES = {
    "agencies": ("t3jq-9nkf", "SELECT name, acronym, alternate_or_former_names, alternate_or_former_acronyms LIMIT 5000"),
    "sat2012": ("f9bf-2cp4", "SELECT dbn, school_name LIMIT 5000"),
    "sat2010": ("zt9s-n5aj", "SELECT dbn, school_name LIMIT 5000"),
    "nta2020": ("9nt8-h7nd", "SELECT nta2020, ntaname, ntaabbrev LIMIT 5000"),
    "nycha": ("3ub5-4ph8", "SELECT address, bin LIMIT 10000"),
}
DOB = "ic3t-wcy2"  # DOB job application filings
N_BUILDINGS = 250  # queried on the DOB side
N_DISTRACTOR_BUILDINGS = 250  # left only
CHUNK = 50


def _url(dataset: str, query: str) -> str:
    return f"{RESOURCE}/{dataset}.csv?" + urlencode({"$query": query})


def _rows(path: Path) -> list[dict[str, str]]:
    return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(io.StringIO(path.read_text("utf-8")))]


def _split(value: str) -> list[str]:
    return [s.strip() for s in value.split(";") if s.strip()]


def _buildings(nycha: list[dict[str, str]], seed: int = 0) -> tuple[list[str], list[str]]:
    """(BINs queried on the DOB side, distractor BINs), deterministic in the NYCHA table."""
    bins = sorted({r["bin"] for r in nycha if r["bin"] and r["address"]})
    picked = random.Random(seed).sample(bins, min(len(bins), N_BUILDINGS + N_DISTRACTOR_BUILDINGS))
    return sorted(picked[:N_BUILDINGS]), sorted(picked[N_BUILDINGS:])


def _dob_chunks(bins: list[str]) -> dict[str, str]:
    """cache file -> DOB query, one per chunk of BINs; the name hashes the BINs so a
    changed NYCHA table never reuses a stale chunk."""
    out = {}
    for i in range(0, len(bins), CHUNK):
        chunk = bins[i : i + CHUNK]
        ids = ", ".join(f"'{b}'" for b in chunk)
        name = "dob/" + hashlib.sha256(",".join(chunk).encode()).hexdigest()[:12]
        out[name] = (
            f"SELECT bin__, house__, street_name WHERE bin__ in ({ids}) "
            "GROUP BY bin__, house__, street_name LIMIT 50000"
        )
    return out


def fetch(data_root: Path, refresh: bool = False, **kw) -> None:
    cache = RawCache(data_root, SOURCE)
    cache.pin(datasets={name: ds for name, (ds, _) in QUERIES.items()} | {"dob": DOB})
    for name, (ds, query) in QUERIES.items():
        cache.fetch(f"{name}.csv", _url(ds, query), refresh=refresh, **kw)
    queried, _ = _buildings(_rows(cache.path("nycha.csv")))
    for name, query in _dob_chunks(queried).items():
        cache.fetch(f"{name}.csv", _url(DOB, query), refresh=refresh, **kw)


def _same_row(rows, left_col: str, right_cols: tuple[str, ...]):
    left = [r[left_col] for r in rows if r[left_col]]
    gold = {(r[left_col], v) for r in rows if r[left_col] for c in right_cols for v in _split(r[c])}
    gold = {(l, r) for l, r in gold if normalize(l) != normalize(r)}
    return left, [r for _, r in sorted(gold)], gold, {}


_GENERIC = {"the", "of", "for", "and", "a", "an", "at", "in", "high", "school", "hs", "academy", "secondary", "campus"}


def _words(value: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in _GENERIC and len(w) >= 3}


def _share_word(a: str, b: str) -> bool:
    """Any distinctive word in common, allowing abbreviation by prefix (sci / science)."""
    return any(x.startswith(y) or y.startswith(x) for x in _words(a) for y in _words(b))


def _same_number(a: str, b: str) -> bool:
    return a.split()[0] == b.split()[0]


def _shared_key(left_rows, right_rows, key: str, left_col: str, right_col: str, keep=None):
    by_key: dict[str, set[str]] = {}
    for r in right_rows:
        if r[key] and r[right_col]:
            by_key.setdefault(r[key], set()).add(r[right_col])
    left = [r[left_col] for r in left_rows if r[left_col]]
    right = [r[right_col] for r in right_rows if r[right_col]]
    gold = {(r[left_col], v) for r in left_rows if r[left_col] for v in by_key.get(r[key], ())}
    kept = {p for p in gold if keep is None or keep(*p)}
    return left, right, kept, {"gold_dropped": len(gold) - len(kept)}


def tasks(cache: RawCache) -> dict[str, tuple[list[str], list[str], set, dict]]:
    """task id -> (left, right, gold, meta) from the cached tables."""

    def task(parts, **meta):
        left, right, gold, extra = parts
        return left, right, gold, {**meta, **extra}

    agencies = _rows(cache.require("agencies.csv"))
    nta = _rows(cache.require("nta2020.csv"))
    nycha = _rows(cache.require("nycha.csv"))
    queried, distractors = _buildings(nycha)
    dob = [
        {"bin": r["bin__"], "address": " ".join(f"{r['house__']} {r['street_name']}".split())}
        for name in _dob_chunks(queried)
        for r in _rows(cache.require(f"{name}.csv"))
        if r["house__"] and r["street_name"]
    ]
    buildings = set(queried) | set(distractors)
    ds = {name: ds for name, (ds, _) in QUERIES.items()}
    return {
        "agency_acronym": task(
            _same_row(agencies, "name", ("acronym", "alternate_or_former_acronyms")),
            datasets=[ds["agencies"]], gold_from="same row", mismatch_type="acronym",
        ),
        "agency_former_name": task(
            _same_row(agencies, "name", ("alternate_or_former_names",)),
            datasets=[ds["agencies"]], gold_from="same row", mismatch_type="alias",
        ),
        "school_name": task(
            _shared_key(_rows(cache.require("sat2012.csv")), _rows(cache.require("sat2010.csv")), "dbn",
                        "school_name", "school_name", keep=_share_word),
            datasets=[ds["sat2012"], ds["sat2010"]], gold_from="shared DBN, sharing a distinctive word",
        ),
        "nta_code": task(
            _same_row(nta, "ntaname", ("nta2020",)),
            datasets=[ds["nta2020"]], gold_from="same row", mismatch_type="code",
        ),
        "nta_abbrev": task(
            _same_row(nta, "ntaname", ("ntaabbrev",)),
            datasets=[ds["nta2020"]], gold_from="same row", mismatch_type="acronym",
        ),
        "building_address": task(
            _shared_key([r for r in nycha if r["bin"] in buildings], dob, "bin", "address", "address", keep=_same_number),
            datasets=[ds["nycha"], DOB], gold_from="shared BIN (same building), same house number",
        ),
    }


def build(data_root: Path, out_dir: Path | None = None) -> list[str]:
    out_dir = out_dir or data_root / SOURCE
    cache = RawCache(data_root, SOURCE)
    files = cache.manifest()["files"]
    fetched_at = max((f["fetched_at"] for f in files.values()), default=None)
    built = []
    for task_id, (left, right, gold, extra) in tasks(cache).items():
        tagged, type_mix = tag_task(gold)
        write_task(
            out_dir / task_id,
            left=left,
            right=right,
            gold=gold,
            meta={
                "source": "nyc-open-data",
                **{k: v for k, v in extra.items() if k != "mismatch_type"},
                "fetched_at": fetched_at,
                "mismatch_type": extra.pop("mismatch_type", tagged),
                "type_mix": type_mix,
                "type_tagger": "semjoin.datasets.tagging",
            },
        )
        built.append(task_id)
    return built
