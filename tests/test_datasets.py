import pytest

from semjoin.datasets import load_tasks
from semjoin.datasets.mapping import build
from semjoin.datasets.pairdir import read_task, write_task
from semjoin.splits import assign, load_manifest, write_manifest


def test_pairdir_roundtrip(tmp_path):
    gold = {("New York", "NY"), ("New York", "N.Y."), ("Texas", "TX")}
    write_task(tmp_path / "states", ["New York", "Texas", "Ohio"], ["NY", "N.Y.", "TX"], gold, {"mismatch_type": "code"})
    task = read_task(tmp_path / "states", "mapping")
    assert task.key == "mapping/states"
    assert task.left == ("New York", "Texas", "Ohio")
    assert task.gold == gold
    assert task.mismatch_type == "code"


def test_gold_must_reference_columns(tmp_path):
    write_task(tmp_path / "t", ["a"], ["b"], [("a", "b")])
    gt = tmp_path / "t" / "gt.csv"
    gt.write_text(gt.read_text() + "9,zzz,0,b\n")
    with pytest.raises(ValueError, match="not in left/right"):
        read_task(tmp_path / "t", "x")


def test_mapping_build_is_deterministic(tmp_path):
    build(tmp_path / "a", seed=1)
    build(tmp_path / "b", seed=1)
    for task_dir in (tmp_path / "a").iterdir():
        for f in ("left.csv", "right.csv", "gt.csv"):
            assert (task_dir / f).read_text() == (tmp_path / "b" / task_dir.name / f).read_text()
    task = read_task(tmp_path / "a" / "country_alpha2", "mapping")
    assert "Germany" in task.left
    assert all(len(code) == 2 for _, code in task.gold)
    assert task.mismatch_type == "code"


def test_split_assignment_and_manifest_stability(tmp_path):
    keys = [f"ds/t{i}" for i in range(20)] + [f"other/t{i}" for i in range(10)]
    a = assign(keys, seed=0, dev_fraction=0.3)
    assert a == assign(keys, seed=0, dev_fraction=0.3)
    assert sum(v == "dev" for k, v in a.items() if k.startswith("ds/")) == 6
    assert sum(v == "dev" for k, v in a.items() if k.startswith("other/")) == 3

    for name in ("t1", "t2", "t3"):
        write_task(tmp_path / "data" / "mapping" / name, ["a"], ["a"], [("a", "a")])
    manifest = tmp_path / "splits.json"
    first = write_manifest(manifest, load_tasks(["mapping"], tmp_path / "data"))
    write_task(tmp_path / "data" / "mapping" / "t4", ["a"], ["a"], [("a", "a")])
    second = write_manifest(manifest, load_tasks(["mapping"], tmp_path / "data"))
    assert {k: second[k] for k in first} == first
    assert "mapping/t4" in load_manifest(manifest)


def test_load_tasks_requires_manifest_entries(tmp_path):
    write_task(tmp_path / "mapping" / "t1", ["a"], ["a"], [("a", "a")])
    with pytest.raises(ValueError, match="semjoin split"):
        load_tasks(["mapping"], tmp_path, manifest={})
    tasks = load_tasks(["mapping"], tmp_path, manifest={"mapping/t1": "test"}, split="test")
    assert [t.split for t in tasks] == ["test"]
