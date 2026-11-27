import json
import random

from semjoin.datasets import mixed
from semjoin.datasets.pairdir import read_task, write_task
from semjoin.normalize import normalize


def test_build_task_meets_quotas_and_keeps_values_unambiguous():
    entities = {f"Name {i}": {"typo": f"Nmae {i}"} for i in range(20)}
    for i in range(6):  # only 6 entities have a code
        entities[f"Name {i}"]["code"] = f"C{i}"
    entities["Name 19"]["code"] = "Name 0"  # equals a left value: never usable
    left, gold, mix = mixed.build_task(entities, {"code": 0.5, "typo": 0.5}, random.Random(0), 0.5)
    assert len(left) == 20 and len(gold) == 10
    assert mix == {"code": 5, "typo": 5}
    assert len({r for _, r in gold}) == len(gold)
    left_norm = {normalize(l) for l in left}
    assert all(normalize(r) not in left_norm for _, r in gold)
    # a type with too few supporters leaves a deficit the other type fills
    _, gold, mix = mixed.build_task(entities, {"code": 0.9, "typo": 0.1}, random.Random(0), 0.5)
    assert len(gold) == 10 and mix["code"] == 6


def test_build_is_deterministic_and_skips_missing_wikidata(tmp_path):
    a = mixed.build(tmp_path, tmp_path / "a")
    b = mixed.build(tmp_path, tmp_path / "b")
    assert a == b and "mixed_country" in a and "mixed_airports" not in a
    for task_id in a:
        for f in ("left.csv", "right.csv", "gt.csv", "meta.json"):
            assert (tmp_path / "a" / task_id / f).read_text() == (tmp_path / "b" / task_id / f).read_text()


def test_built_tasks_record_their_mix(tmp_path):
    for task_id in mixed.build(tmp_path, tmp_path / "out"):
        task = read_task(tmp_path / "out" / task_id, "mixed")
        meta = json.loads((tmp_path / "out" / task_id / "meta.json").read_text())
        assert task.mismatch_type == "mixed"
        assert sum(meta["type_mix"].values()) == len(task.gold)
        assert len(meta["type_mix"]) >= 2
        for t, w in meta["weights"].items():
            assert abs(meta["type_mix"].get(t, 0) / len(task.gold) - w) <= 0.05, (task_id, t)


def test_wikidata_sourced_task(tmp_path):
    names = [f"Airport {c}" for c in "ABCDEFGHIJ"]
    gold = [(n, f"X{n[-1]}{n[-1]}") for n in names[:6]]
    write_task(tmp_path / "wikidata" / "airport_iata", names, [r for _, r in gold], gold, {"mismatch_type": "code"})
    assert mixed.build(tmp_path, names=["mixed_airports"]) == ["mixed_airports"]
    task = read_task(tmp_path / "mapping" / "mixed_airports", "mapping")
    assert set(task.left) == set(names)
    codes = {r for _, r in task.gold if r.startswith("X")}
    assert codes and codes <= {r for _, r in gold}
