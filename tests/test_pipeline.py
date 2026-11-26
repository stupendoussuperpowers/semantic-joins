"""Phase 0 dataset infrastructure: raw cache, build, validate, slicing, stratified splits.

Everything here runs offline on the committed fixtures in tests/fixtures/data/.
"""

import json
import shutil
from pathlib import Path

import pytest

from semjoin.cli import main
from semjoin.datasets import autofj, load_tasks
from semjoin.datasets.pairdir import write_task
from semjoin.datasets.raw import RawCache
from semjoin.slicing import select_tasks, slice_right
from semjoin.splits import assign, load_manifest, write_manifest
from semjoin.validate import validate

FIXTURES = Path(__file__).parent / "fixtures" / "data"
ALL = ["autofj", "mapping", "freyja"]


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(FIXTURES, root)
    return root


# --- raw cache -------------------------------------------------------------


def test_raw_cache_fetches_once_and_verifies(tmp_path):
    calls = []

    def getter(url, headers=None):
        calls.append(url)
        return b"id,title\n0,x\n"

    cache = RawCache(tmp_path, "src")
    cache.fetch("a/left.csv", "http://example/a", getter=getter)
    cache.fetch("a/left.csv", "http://example/a", getter=getter)
    assert calls == ["http://example/a"]
    entry = cache.manifest()["files"]["a/left.csv"]
    assert entry["url"] == "http://example/a" and entry["bytes"] == 13
    assert cache.verify() == []

    cache.path("a/left.csv").write_text("tampered")
    assert cache.verify() == ["src/a/left.csv: sha256 mismatch"]
    cache.fetch("a/left.csv", "http://example/a", getter=getter, refresh=True)
    assert cache.verify() == [] and len(calls) == 2


def test_require_names_the_fetch_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="semjoin download-autofj"):
        RawCache(tmp_path, "autofj").require("tree.json")


def test_autofj_fetch_then_build_offline(tmp_path):
    src = FIXTURES / "autofj" / "ToyFilm"
    tree = {"tree": [{"path": f"{autofj.BENCHMARK_PATH}/ToyFilm/gt.csv"}]}

    def getter(url, headers=None):
        if "git/trees" in url:
            return json.dumps(tree).encode()
        return (src / url.rsplit("/", 1)[1]).read_bytes()

    assert autofj.fetch(tmp_path, getter=getter) == ["ToyFilm"]
    assert autofj.build(tmp_path) == ["ToyFilm"]
    task_dir = tmp_path / "autofj" / "ToyFilm"
    assert (task_dir / "gt.csv").read_bytes() == (src / "gt.csv").read_bytes()
    assert json.loads((task_dir / "meta.json").read_text())["commit"] == autofj.COMMIT


# --- validate --------------------------------------------------------------


def _manifest(data):
    path = data.parent / "splits.json"
    write_manifest(path, load_tasks(ALL, data))
    return load_manifest(path)


def test_fixtures_validate_clean(data):
    report = validate(data, ALL, _manifest(data), min_dev=0, min_test=0, min_right=3)
    assert report.errors == [] and report.warnings == [], report.format()
    assert report.n_tasks == 5


def test_validate_flags_problems(data):
    write_task(data / "mapping" / "blank", ["a", " "], ["b"], [("a", "b")], {"mismatch_type": "codes"})
    write_task(data / "mapping" / "untyped", ["a"], ["b"], [("a", "b")])
    (data / "mapping" / "nogt").mkdir()
    report = validate(data, ALL, {}, min_right=3)
    msgs = [str(i) for i in report.issues]
    assert any("mapping/blank" in m and "empty values in left.csv" in m for m in msgs)
    assert any("mapping/blank" in m and "unknown mismatch_type 'codes'" in m for m in msgs)
    assert any("mapping/untyped" in m and "no mismatch_type" in m for m in msgs)
    assert any("mapping/untyped" in m and "few distractors" in m for m in msgs)
    assert any("mapping/nogt" in m and "no gt.csv" in m for m in msgs)
    assert any("not in the split manifest" in m for m in msgs)
    assert any(m.startswith("WARN  coverage/acronym") for m in msgs)
    assert not report.ok()


def test_validate_catches_tampered_raw_cache(data):
    RawCache(data, "src").put("f.csv", b"abc")
    (data / "raw" / "src" / "f.csv").write_text("xyz")
    report = validate(data, ALL, None, min_right=3)
    assert [str(i) for i in report.errors] == ["ERROR raw: src/f.csv: sha256 mismatch"]


def test_cli_validate_exit_codes(data, tmp_path):
    common = ["--data-root", str(data), "--manifest", str(tmp_path / "splits.json")]
    assert main([*common, "split"]) == 0
    assert main([*common, "validate"]) == 0  # warnings alone do not fail
    assert main([*common, "validate", "--strict"]) == 1  # the tiny fixtures warn
    write_task(data / "mapping" / "broken", ["a", " "], ["b"], [("a", "b")], {"mismatch_type": "code"})
    assert main([*common, "validate"]) == 1  # an error fails


# --- slicing ---------------------------------------------------------------


def test_select_tasks_per_dataset_and_type(data):
    tasks = load_tasks(ALL, data)
    for i in range(3):
        write_task(data / "mapping" / f"extra_code{i}", ["x", "y"], ["x"], [("x", "x")], {"mismatch_type": "code"})
    tasks = load_tasks(ALL, data)
    picked = select_tasks(tasks, 1)
    groups = [(t.dataset, t.mismatch_type) for t in picked]
    assert len(groups) == len(set(groups)) == len({(t.dataset, t.mismatch_type) for t in tasks})
    assert picked == select_tasks(tasks, 1)
    assert len(select_tasks(tasks, 10)) == len(tasks)


def test_slice_right_keeps_gold_consistent(data):
    task = next(t for t in load_tasks(["mapping"], data) if t.task_id == "toy_typo")
    small = slice_right(task, 2, seed=1)
    assert len(small.right) == 2 and small.left == task.left
    assert small.gold == {p for p in task.gold if p[1] in small.right}
    assert slice_right(task, 2, seed=1) == small
    assert slice_right(task, 100) is task


# --- stratified splits -----------------------------------------------------


def test_assign_unstratified_matches_old_behaviour():
    keys = [f"ds/t{i}" for i in range(20)] + [f"other/t{i}" for i in range(10)]
    a = assign(keys, seed=0, dev_fraction=0.3)
    assert sum(v == "dev" for k, v in a.items() if k.startswith("ds/")) == 6
    assert sum(v == "dev" for k, v in a.items() if k.startswith("other/")) == 3


def test_assign_stratifies_and_keeps_small_strata_in_both_splits():
    keys = [f"m/code{i}" for i in range(10)] + ["m/typo0", "m/typo1"] + ["m/acr0"]
    strata = {k: k.rstrip("0123456789") for k in keys}
    a = assign(keys, strata=strata)
    assert sum(a[k] == "dev" for k in keys if "code" in k) == 3
    assert sorted(a[k] for k in ("m/typo0", "m/typo1")) == ["dev", "test"]
    assert a["m/acr0"] == "test"


def test_assign_counts_existing_assignments():
    strata = {f"m/t{i}": "m/typo" for i in range(4)}
    existing = {"m/t0": "dev"}
    a = assign(["m/t1", "m/t2", "m/t3"], strata=strata, existing=existing)
    assert list(a.values()).count("dev") == 0  # target round(0.3*4)=1 is already met


# --- end to end ------------------------------------------------------------


def test_cli_build_validate_run_on_slice(data, tmp_path):
    common = ["--data-root", str(data), "--manifest", str(tmp_path / "splits.json")]
    assert main([*common, "build-mapping"]) == 0
    assert main([*common, "split"]) == 0
    assert main([*common, "validate"]) == 0
    out = tmp_path / "run"
    rc = main(
        [*common, "run", "--split", "all", "--matchers", "normalized;fuzzy;router:residual=false",
         "--max-tasks-per-type", "1", "--max-right", "5", "--out", str(out), "--no-summary"]
    )
    assert rc == 0
    config = json.loads((out / "config.json").read_text())
    assert config["slice"] == {"max_tasks_per_type": 1, "max_right": 5}
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert rows and all(r["error"] is None for r in rows)
    assert all(r["n_right"] <= 5 for r in rows)
    # one task per (dataset, type) group among the fixtures plus the freshly built mapping tasks
    groups = {(r["dataset"], r["mismatch_type"]) for r in rows}
    assert len({(r["dataset"], r["mismatch_type"], r["task_id"]) for r in rows}) == len(groups)
