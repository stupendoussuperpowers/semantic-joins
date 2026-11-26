"""NYC Open Data tasks, offline with a fake Socrata service."""

import json
from urllib.parse import parse_qs, urlparse

from semjoin.datasets import load_tasks, nyc
from semjoin.datasets.pairdir import read_task
from semjoin.splits import write_manifest

TABLES = {
    "t3jq-9nkf": "name,acronym,alternate_or_former_names,alternate_or_former_acronyms\n"
    "Department of Consumer and Worker Protection,DCWP,Department of Consumer Affairs,DCA\n"
    "Mayor's Advance Team,,,\n",
    "f9bf-2cp4": "dbn,school_name\n01M001,FOOD AND FINANCE HIGH SCHOOL\n01M002,MARTA VALLE HIGH SCHOOL\n01M003,NEW SCHOOL\n",
    "zt9s-n5aj": "dbn,school_name\n01M001,FOOD AND FNANCE HS\n01M002,CMSP HIGH SCHOOL\n",
    "9nt8-h7nd": "nta2020,ntaname,ntaabbrev\nBK0101,Greenpoint,Grnpt\n",
    "3ub5-4ph8": "address,bin\n" + "".join(f"{100 + i} MAIN STREET,{i:07d}\n" for i in range(600)),
}


def _dob(url):
    query = parse_qs(urlparse(url).query)["$query"][0]
    bins = [b.strip(" '") for b in query.split("in (")[1].split(")")[0].split(",")]
    rows = "".join(f"{b},{100 + int(b)},MAIN ST\n{b},{100 + int(b)}-{102 + int(b)},MAIN ST\n" for b in bins)
    return "bin__,house__,street_name\n" + rows


def fake_get(url, headers=None):
    dataset = urlparse(url).path.rsplit("/", 1)[1].removesuffix(".csv")
    return (_dob(url) if dataset == nyc.DOB else TABLES[dataset]).encode()


def test_build_derives_gold_from_the_data(tmp_path):
    nyc.fetch(tmp_path, getter=fake_get)
    assert set(nyc.build(tmp_path)) == {
        "agency_acronym", "agency_former_name", "school_name", "nta_code", "nta_abbrev", "building_address",
    }
    out = tmp_path / "nyc"
    acr = read_task(out / "agency_acronym", "nyc")
    assert acr.gold == {("Department of Consumer and Worker Protection", r) for r in ("DCWP", "DCA")}
    assert "Mayor's Advance Team" in acr.left  # unmatched kept
    assert json.loads((out / "nta_code" / "meta.json").read_text())["mismatch_type"] == "code"

    school = read_task(out / "school_name", "nyc")
    # a reused DBN with no word in common is dropped
    assert school.gold == {("FOOD AND FINANCE HIGH SCHOOL", "FOOD AND FNANCE HS")}
    assert json.loads((out / "school_name" / "meta.json").read_text())["gold_dropped"] == 1

    addr = read_task(out / "building_address", "nyc")
    assert len(addr.left) == nyc.N_BUILDINGS + nyc.N_DISTRACTOR_BUILDINGS
    assert all(l.split()[0] == r.split()[0] for l, r in addr.gold)  # same house number only
    assert len(addr.gold) == nyc.N_BUILDINGS and len(addr.right) == 2 * nyc.N_BUILDINGS

    assignment = write_manifest(tmp_path / "splits.json", load_tasks(["nyc"], tmp_path), dev_fraction=0.9)
    assert set(assignment.values()) == {"test"}
