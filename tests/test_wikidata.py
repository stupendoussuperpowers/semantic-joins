"""Wikidata fetch/build, offline: a fake getter stands in for the query service."""

import json
import urllib.parse

import pytest

from semjoin.datasets import wikidata
from semjoin.datasets.pairdir import read_task
from semjoin.datasets.raw import RawCache


def _bindings(rows):
    return json.dumps(
        {"results": {"bindings": [{k: {"type": "literal", "value": v} for k, v in r.items()} for r in rows]}}
    ).encode()


def _fake_service(tables):
    calls = []

    def getter(url, headers=None):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["query"][0]
        name = next(
            n for n in tables if wikidata.endpoint_query(wikidata.query_text(n), wikidata.DEFAULT_ENDPOINT) == query
        )
        calls.append(name)
        return _bindings(tables[name])

    return getter, calls


AIRPORTS = [{"item": f"Q{i}", "left": f"Airport {i}", "right": code} for i, code in enumerate(
    ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"] * 1
)] + [
    {"item": "Q100", "left": "Bad Code Airport", "right": "toolong"},  # fails right_pattern
    {"item": "Q101", "left": "Q101", "right": "KKK"},  # unlabeled entity
]
PEOPLE = [
    {"item": "Q1", "left": "Marie Curie", "family": "Curie"},
    {"item": "Q2", "left": "Niels Bohr", "family": "Bohr"},
    {"item": "Q3", "left": "J. Robert Oppenheimer", "family": "Oppenheimer"},
    {"item": "Q4", "left": "Aristotle", "family": "Aristotle"},  # label is the family name only
    {"item": "Q5", "left": "Bohr, Niels", "family": "Bohr"},  # already reordered
]
SINGERS = [
    {"item": "Q1", "left": "Bob Dylan", "right": "Robert Allen Zimmerman"},
    {"item": "Q2", "left": "Sting", "right": "Gordon Sumner"},
    {"item": "Q3", "left": "Adele", "right": "Adele"},  # equal: an equi-join finds it
    {"item": "Q4", "left": "Prince", "right": "Prince Rogers Nelson"},
    {"item": "Q5", "left": "Prince Rogers Nelson", "right": "Skipper"},  # left value reused as a right value
    {"item": "Q6", "left": "Ah-young", "right": "박아영"},  # cross-script
]


def test_fetch_caches_and_refetches_only_on_query_change(tmp_path, monkeypatch):
    getter, calls = _fake_service({"airport_iata": AIRPORTS})
    wikidata.fetch(tmp_path, ["airport_iata"], getter=getter, pause=0)
    wikidata.fetch(tmp_path, ["airport_iata"], getter=getter, pause=0)
    assert calls == ["airport_iata"]
    pinned = RawCache(tmp_path, "wikidata").manifest()["pinned"]
    assert set(pinned["queries"]) == {"airport_iata"}

    real = wikidata.query_text
    monkeypatch.setattr(wikidata, "query_text", lambda n: real(n) + "\n# edited")
    getter, calls = _fake_service({"airport_iata": AIRPORTS})
    wikidata.fetch(tmp_path, ["airport_iata"], getter=getter, pause=0)
    assert calls == ["airport_iata"]


def test_every_table_has_a_query_file():
    for name in wikidata.TABLES:
        assert "SELECT" in wikidata.query_text(name)


def test_table_pairs_cleaning(tmp_path):
    getter, _ = _fake_service({"airport_iata": AIRPORTS, "physicist_names": PEOPLE, "footballer_names": PEOPLE})
    wikidata.fetch(tmp_path, ["airport_iata", "physicist_names", "footballer_names"], getter=getter, pause=0)
    cache = RawCache(tmp_path, "wikidata")

    airports = wikidata.table_pairs(cache, "airport_iata")
    assert len(airports) == 10 and "Q100" not in airports and "Q101" not in airports

    comma = {p for ps in wikidata.table_pairs(cache, "physicist_names").values() for p in ps}
    assert comma == {
        ("Marie Curie", "Curie, Marie"),
        ("Niels Bohr", "Bohr, Niels"),
        ("J. Robert Oppenheimer", "Oppenheimer, J. Robert"),
    }
    upper = {p for ps in wikidata.table_pairs(cache, "footballer_names").values() for p in ps}
    assert ("Marie Curie", "CURIE Marie") in upper


def test_build_gold_is_consistent_and_deterministic(tmp_path):
    getter, _ = _fake_service({"airport_iata": AIRPORTS, "singer_birth_name": SINGERS})
    wikidata.fetch(tmp_path, ["airport_iata", "singer_birth_name"], getter=getter, pause=0)
    a = wikidata.build(tmp_path, tmp_path / "a", sample_fraction=1.0, min_right=1)
    b = wikidata.build(tmp_path, tmp_path / "b", sample_fraction=1.0, min_right=1)
    assert a == b == ["airport_iata", "singer_birth_name"]
    for name in a:
        for f in ("left.csv", "right.csv", "gt.csv", "meta.json"):
            assert (tmp_path / "a" / name / f).read_text() == (tmp_path / "b" / name / f).read_text()

    singers = read_task(tmp_path / "a" / "singer_birth_name", "wikidata")
    assert ("Bob Dylan", "Robert Allen Zimmerman") in singers.gold
    assert "Adele" not in singers.right  # equal pair dropped
    assert "Prince Rogers Nelson" not in singers.right  # equals a left value
    assert singers.mismatch_type == "alias"
    assert "박아영" not in singers.right
    meta = json.loads((tmp_path / "a" / "singer_birth_name" / "meta.json").read_text())
    assert meta["group"] == "people" and meta["raw_sha256"] and meta["query_sha256"]


def test_build_skips_small_tables(tmp_path):
    getter, _ = _fake_service({"airport_iata": AIRPORTS})
    wikidata.fetch(tmp_path, ["airport_iata"], getter=getter, pause=0)
    assert wikidata.build(tmp_path, names=["airport_iata"], min_right=20) == []


def test_build_requires_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="semjoin download-wikidata"):
        wikidata.build(tmp_path, names=["airport_iata"])


def test_interrupted_fetch_keeps_finished_tables(tmp_path):
    good, calls = _fake_service({"airport_iata": AIRPORTS})

    def failing(url, headers=None):
        if "P229" in urllib.parse.unquote_plus(url):  # airline_iata
            raise RuntimeError("throttled")
        return good(url, headers)

    with pytest.raises(RuntimeError):
        wikidata.fetch(tmp_path, ["airport_iata", "airline_iata"], getter=failing, pause=0)
    getter, calls = _fake_service({"airport_iata": AIRPORTS, "airline_iata": AIRPORTS})
    wikidata.fetch(tmp_path, ["airport_iata", "airline_iata"], getter=getter, pause=0)
    assert calls == ["airline_iata"]


def test_endpoint_query_adds_prefixes_and_drops_hints_off_wdqs():
    q = wikidata.query_text("physicist_names")
    assert "hint:" in q
    qlever = wikidata.endpoint_query(q, "qlever")
    assert qlever.startswith("PREFIX wd:") and "hint:" not in qlever
    assert "hint:" in wikidata.endpoint_query(q, "wdqs")
