import json

import pytest

from semjoin.cli import build_parser
from semjoin.matchers.profiling import Profile
from semjoin.matchers.router import strategy_specs
from semjoin.tune import evaluate_settings, parse_grid, tune

BASE = dict(
    n_left=10, n_right=10, n_sample=10, left_len=10.0, right_len=10.0, left_uniformity=0.2,
    right_uniformity=0.2, qgram_cos=0.0, token_jaccard=0.0, token_set_equal=0.0, acronym_rate=0.0,
)


def rec(task, matcher, f1, calls, annotations=None):
    return {
        "dataset": "d", "task_id": task, "matcher": matcher, "f1": f1, "tp": 0, "fp": 0, "fn": 0,
        "llm_calls": calls, "annotations": annotations or {},
    }


def run_records():
    """t1 is clearly fuzzy (autofj wins); t2 is borderline at qgram_cos=0.45 (llm wins)."""
    records = []
    for task, cos, f1 in (("t1", 0.7, {"autofj": 0.9, "llm": 0.5}), ("t2", 0.45, {"autofj": 0.3, "llm": 0.8})):
        for spec in strategy_specs():
            name = spec.split(":")[0]
            records.append(rec(task, spec, f1.get(name, 0.1), 5 if name == "llm" else 1))
        # As run with the default typo_cos=0.5: t1 -> autofj, t2 -> the llm fallback.
        chosen = "autofj" if cos >= 0.5 else "llm"
        profile = Profile(**{**BASE, "qgram_cos": cos}).as_dict()
        records.append(rec(task, "router", f1[chosen], 5 if chosen == "llm" else 1, {"profile": profile}))
    return records


def test_rerouting_reproduces_the_run_and_scores_settings():
    res = evaluate_settings(run_records(), grid={"typo_cos": [0.4, 0.5]})
    ref = {row["config"]: row for row in res["reference"]}
    assert {**ref["router (as run)"], "config": ""} == {**ref["router (current defaults)"], "config": ""}
    assert (ref["perfect routing"]["accuracy"], ref["perfect routing"]["regret"]) == (1, 0)
    assert ref["always autofj:residual=true"]["accuracy"] == 0.5
    best, worse = res["settings"]
    assert best["config"] == "typo_cos=0.5" and best["accuracy"] == 1 and best["macro_f1"] == pytest.approx(0.85)
    assert worse["config"] == "typo_cos=0.4" and worse["macro_f1"] == pytest.approx(0.6) and worse["llm_calls"] == 2


def test_objectives_and_errors():
    res = evaluate_settings(run_records(), grid={"typo_cos": [0.4, 0.5]}, objective="regret")
    assert res["settings"][0]["config"] == "typo_cos=0.5"
    with pytest.raises(ValueError):
        evaluate_settings(run_records(), objective="speed")
    with pytest.raises(ValueError):
        evaluate_settings(run_records(), router="cascade")
    stale = run_records()
    for r in stale:
        r["annotations"].get("profile", {}).pop("acronym_rate", None)
    with pytest.raises(ValueError, match="another version"):
        evaluate_settings(stale)


def test_tune_report_from_a_run_dir(tmp_path):
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run_records()))
    report = tune(tmp_path, grid={"typo_cos": [0.5]})
    assert "tasks=2 skipped=0 settings=1" in report and "| perfect routing |" in report


def test_parse_grid_and_cli():
    assert parse_grid("typo_cos=0.4,0.5; code_len=3") == {"typo_cos": [0.4, 0.5], "code_len": [3]}
    with pytest.raises(ValueError):
        parse_grid("typo_cos")
    args = build_parser().parse_args(["tune", "runs/x", "--grid", "typo_cos=0.4", "--objective", "accuracy"])
    assert (args.grid, args.objective, args.router, args.top) == ("typo_cos=0.4", "accuracy", "router", 10)
