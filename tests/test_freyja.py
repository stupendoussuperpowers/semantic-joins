"""FREYJA fetch and candidate ranking, offline with a tiny fake lake."""

import csv
import io
import zipfile

from semjoin.datasets import freyja


def _csv(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _fake_getter():
    countries = ["Germany", "France", "Iran", "Korea, Republic of", "Bahamas", "Egypt", "Japan", "Spain",
                 "Netherlands", "Portugal", "Morocco", "Tunisia"]
    other = ["Germany", "Frace", "Iran (Islamic Republic of)", "Korea Republic", "The Bahamas", "Egypt",
             "Japn", "Spain", "Netherland", "Portugall", "Moroco", "Tunesia", "Atlantis"]
    lake = io.BytesIO()
    with zipfile.ZipFile(lake, "w") as z:
        z.writestr("datalake/a.csv", _csv(["country", "pop"], [(c, i) for i, c in enumerate(countries)]))
        z.writestr("datalake/b.csv", _csv(["nation"], [(c,) for c in other]))
        z.writestr("datalake/ids_a.csv", _csv(["id"], [(f"d{i}",) for i in range(30)]))
        z.writestr("datalake/ids_b.csv", _csv(["id"], [(f"d{i}",) for i in range(15, 60)]))
        z.writestr("datalake/lists.csv", _csv(["c"], [(f"Germany, France, {c}",) for c in countries[2:7]] +[(c,) for c in other[:5]]))
    gt = _csv(
        ["target_ds", "target_attr", "candidate_ds", "candidate_attr"],
        [("a.csv", "country", "b.csv", "nation"), ("b.csv", "nation", "a.csv", "country"),
         ("ids_a.csv", "id", "ids_b.csv", "id"), ("a.csv", "country", "lists.csv", "c"),
         ("a.csv", "country", "missing.csv", "x")],
    ).encode()

    def getter(url, headers=None):
        return lake.getvalue() if url == freyja.LAKE_URL else gt

    return getter


def test_fetch_then_rank_candidates(tmp_path):
    freyja.fetch(tmp_path, getter=_fake_getter())
    lake = freyja.Lake(tmp_path)
    assert lake.column("a.csv", "country")[0] == "Germany"
    assert lake.column("a.csv", "nope") is None and lake.column("missing.csv", "x") is None

    cands = freyja.candidates(tmp_path)
    by = {(c.left_table, c.right_table): c for c in cands}
    assert len(cands) == 3  # the mirrored a/b row is deduplicated, missing.csv skipped
    best = cands[0]
    assert {best.left_table, best.right_table} == {"a.csv", "b.csv"} and freyja.promising(best)
    assert best.near_miss > 0.5
    assert by["ids_a.csv", "ids_b.csv"].id_share == 1.0 and not freyja.promising(by["ids_a.csv", "ids_b.csv"])
    assert by["a.csv", "lists.csv"].list_share > 0.2 and not freyja.promising(by["a.csv", "lists.csv"])

    out = tmp_path / "cands.csv"
    freyja.write_candidates(cands, out)
    assert len(list(csv.DictReader(out.open()))) == 3
