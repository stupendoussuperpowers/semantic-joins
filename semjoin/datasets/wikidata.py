"""Real mapping tables from Wikidata, one task per table.

``fetch`` runs each SPARQL query in ``queries/wikidata/<table>.rq`` against
a public SPARQL endpoint (see ``ENDPOINTS``) and caches the JSON response in
``data/raw/wikidata/<table>.json``. Wikidata changes over time, so the cache
records the fetch date and each query's sha256, and the *built* tasks in
``data/wikidata/`` are committed to git: rebuilding from a later fetch would
give different tasks.

``build`` turns each cached table into a task:

1. clean rows: drop blank values, values that are bare Q-ids, non-Latin
   values (cross-script pairs are transliteration), right values
   that do not match the table's ``right_pattern``, and pairs whose two sides
   are equal after normalization (an equi-join finds those);
2. for ``reorder`` tables, derive the right value from the label and the
   family name: "Marie Curie" + "Curie" -> "Curie, Marie" or "CURIE Marie";
3. sample up to ``max_entities`` entities; their labels form the left column;
4. sample ``sample_fraction`` of those entities; their right values form the
   right column. The gold set is every cleaned pair between the two columns,
   so a right value shared by two entities is matched to both;
5. drop right values that equal some left value after normalization, as
   they would be ambiguous with an equi-join match.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from semjoin.datasets.pairdir import write_task
from semjoin.datasets.raw import USER_AGENT, RawCache
from semjoin.normalize import normalize

SOURCE = "wikidata"
# Two public endpoints over the same data. "wdqs" is Wikidata's own query
# service (60 s query limit, throttled during outages); "qlever" is the
# University of Freiburg's QLever mirror of Wikidata, which is much faster on
# these queries but needs explicit prefixes and ignores Blazegraph hints.
ENDPOINTS = {
    "wdqs": "https://query.wikidata.org/sparql",
    "qlever": "https://qlever.dev/api/wikidata",
}
DEFAULT_ENDPOINT = "qlever"
PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""
QUERY_DIR = Path(__file__).parent / "queries" / "wikidata"
_QID = re.compile(r"^Q\d+$")
_ACRONYM = r"^(?=(?:.*[A-Z]){2})[A-Za-z0-9&.\-]{2,10}$"  # no spaces, two or more capitals


@dataclass(frozen=True)
class Table:
    mismatch_type: str
    group: str
    right_pattern: str | None = None
    # reorder tables only: "comma" -> "Family, Given"; "upper" -> "FAMILY Given"
    reorder_style: str | None = None


TABLES: dict[str, Table] = {
    # code
    "airport_iata": Table("code", "airports", r"^[A-Z]{3}$"),
    "airline_iata": Table("code", "airlines", r"^[A-Z0-9]{2}$"),
    "element_symbol": Table("code", "elements", r"^[A-Z][a-z]{0,2}$"),
    "nyse_ticker": Table("code", "companies", r"^[A-Z][A-Z.\-]{0,5}$"),
    "icd10": Table("code", "diseases", r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$"),
    "ioc_country": Table("code", "countries", r"^[A-Z]{3}$"),
    # alias
    "drug_alias": Table("alias", "drugs"),
    "singer_birth_name": Table("alias", "people"),
    "writer_birth_name": Table("alias", "people"),
    "club_nickname": Table("alias", "football_clubs"),
    "bird_common_name": Table("alias", "species"),
    # acronym
    "intl_org_short": Table("acronym", "organizations", _ACRONYM),
    "university_short": Table("acronym", "universities", _ACRONYM),
    "gov_agency_short": Table("acronym", "agencies", _ACRONYM),
    "political_party_short": Table("acronym", "parties", _ACRONYM),
    "ngo_short": Table("acronym", "organizations", _ACRONYM),
    "research_institute_short": Table("acronym", "institutes", _ACRONYM),
    # reorder
    "politician_names": Table("reorder", "people", reorder_style="comma"),
    "footballer_names": Table("reorder", "people", reorder_style="upper"),
    "physicist_names": Table("reorder", "people", reorder_style="comma"),
    "painter_names": Table("reorder", "people", reorder_style="comma"),
    "actor_names": Table("reorder", "people", reorder_style="upper"),
}


def query_text(table: str) -> str:
    return (QUERY_DIR / f"{table}.rq").read_text()


def endpoint_query(query: str, endpoint: str) -> str:
    """The query as sent: explicit prefixes, and no Blazegraph hints outside wdqs."""
    if endpoint != "wdqs":
        query = "\n".join(line for line in query.splitlines() if "hint:" not in line)
    return PREFIXES + query


def _sparql_get(url: str, headers: dict[str, str] | None = None, attempts: int = 4) -> bytes:
    """GET with backoff on throttling (429), server errors and timeouts."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json", **(headers or {})},
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            json.loads(data)  # a timed-out query can return a truncated body with status 200
            return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429:
                raise
            if attempt == attempts:
                raise
            retry_after = getattr(e, "headers", None) and e.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 15 * attempt
            print(f"  {type(e).__name__}: {e}; retry {attempt}/{attempts - 1} in {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch(
    data_root: Path,
    names: list[str] | None = None,
    refresh: bool = False,
    getter=_sparql_get,
    pause: float = 2.0,
    endpoint: str = DEFAULT_ENDPOINT,
) -> list[str]:
    cache = RawCache(data_root, SOURCE)
    names = names or list(TABLES)
    queries = {}
    for i, name in enumerate(names):
        query = query_text(name)
        queries[name] = hashlib.sha256(query.encode()).hexdigest()
        rel = f"{name}.json"
        stale = cache.manifest().get("pinned", {}).get("queries", {}).get(name) != queries[name]
        tag = f"[{i + 1}/{len(names)}] {name}"
        if cache.has(rel) and not refresh and not stale:
            print(f"{tag}: cached", file=sys.stderr, flush=True)
            continue
        print(f"{tag}: querying...", file=sys.stderr, flush=True)
        start = time.monotonic()
        url = f"{ENDPOINTS[endpoint]}?{urllib.parse.urlencode({'query': endpoint_query(query, endpoint)})}"
        path = cache.fetch(rel, url, refresh=True, getter=getter)
        n_rows = len(json.loads(path.read_text())["results"]["bindings"])
        print(f"{tag}: {n_rows} rows in {time.monotonic() - start:.0f}s", file=sys.stderr, flush=True)
        _pin_query(cache, name, queries[name], endpoint)  # right away, so an interrupted run keeps it
        if pause and i + 1 < len(names):
            time.sleep(pause)  # be polite to the public endpoint
    return names


def _pin_query(cache: RawCache, name: str, query_sha: str, endpoint: str) -> None:
    pinned = cache.manifest().get("pinned", {})
    cache.pin(
        queries={**pinned.get("queries", {}), name: query_sha},
        endpoints={**pinned.get("endpoints", {}), name: endpoint},
    )


def _rows(cache: RawCache, name: str) -> list[dict[str, str]]:
    bindings = json.loads(cache.require(f"{name}.json").read_text())["results"]["bindings"]
    return [{k: v["value"].strip() for k, v in b.items()} for b in bindings]


def _is_latin(value: str) -> bool:
    """True if every letter is Latin once accents are stripped ("Kittilä" yes, "박아영" no)."""
    decomposed = unicodedata.normalize("NFKD", value)
    return all(c.isascii() for c in decomposed if c.isalpha())


def _reordered(label: str, family: str, style: str) -> str | None:
    if not label.endswith(" " + family):
        return None
    given = label[: -len(family) - 1].strip()
    if not given or "," in label:
        return None
    return f"{family}, {given}" if style == "comma" else f"{family.upper()} {given}"


def table_pairs(cache: RawCache, name: str) -> dict[str, set[tuple[str, str]]]:
    """Cleaned (left, right) pairs per entity id."""
    table = TABLES[name]
    pattern = re.compile(table.right_pattern) if table.right_pattern else None
    by_item: dict[str, set[tuple[str, str]]] = {}
    for row in _rows(cache, name):
        left = row.get("left", "")
        if table.reorder_style:
            right = _reordered(left, row.get("family", ""), table.reorder_style) or ""
        else:
            right = row.get("right", "")
        if not left or not right or _QID.match(left) or _QID.match(right):
            continue
        if not (_is_latin(left) and _is_latin(right)):
            continue  # cross-script pairs are transliteration, not the table's type
        if pattern and not pattern.match(right):
            continue
        if normalize(left) == normalize(right):
            continue
        by_item.setdefault(row["item"], set()).add((left, right))
    return by_item


def build(
    data_root: Path,
    out_dir: Path | None = None,
    names: list[str] | None = None,
    seed: int = 0,
    max_entities: int = 400,
    sample_fraction: float = 0.6,
    min_right: int = 20,
) -> list[str]:
    cache = RawCache(data_root, SOURCE)
    out_dir = out_dir or data_root / SOURCE
    manifest = cache.manifest()
    built = []
    if names is None:  # every cached table; explicitly named ones must be cached
        names = [n for n in TABLES if cache.has(f"{n}.json")]
        if not names:
            cache.require(f"{next(iter(TABLES))}.json")
    for name in names:
        table = TABLES[name]
        rng = random.Random(f"{seed}:{name}")
        by_item = table_pairs(cache, name)
        items = sorted(by_item)
        items = sorted(rng.sample(items, min(max_entities, len(items))))
        pairs = sorted({p for item in items for p in by_item[item]})
        left = sorted({l for l, _ in pairs})
        left_norm = {normalize(l) for l in left}
        pairs = [p for p in pairs if normalize(p[1]) not in left_norm]
        # Sample entities, not values, so that about 1 - sample_fraction of the
        # left column stays unmatched even when entities have several aliases.
        kept = {r for _, r in pairs}
        matched = rng.sample(items, round(sample_fraction * len(items)))
        right = {r for item in matched for _, r in by_item[item] if r in kept}
        if len(right) < min_right:
            print(f"skip {name}: {len(right)} right values (< {min_right})", file=sys.stderr)
            continue
        gold = [p for p in pairs if p[1] in right]
        entry = manifest["files"].get(f"{name}.json", {})
        write_task(
            out_dir / name,
            left=left,
            right=sorted(right),
            gold=gold,
            meta={
                "source": "wikidata",
                "mismatch_type": table.mismatch_type,
                "group": table.group,
                "fetched_at": entry.get("fetched_at"),
                "raw_sha256": entry.get("sha256"),
                "query_sha256": manifest.get("pinned", {}).get("queries", {}).get(name),
                "endpoint": manifest.get("pinned", {}).get("endpoints", {}).get(name),
                "seed": seed,
                "max_entities": max_entities,
                "sample_fraction": sample_fraction,
            },
        )
        built.append(name)
    return built
