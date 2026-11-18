import json

from semjoin.core import JoinTask, RunContext
from semjoin.llm import FakeLLM
from semjoin.matchers import MATCHERS, build_matcher, parse_spec
from semjoin.run import run
from semjoin.summarize import routing, summarize

TASK = JoinTask(
    dataset="mapping",
    task_id="toy",
    left=("Germany", "France", "United Kingdom"),
    right=("germany", "Frnace", "UK"),
    gold=frozenset({("Germany", "germany"), ("France", "Frnace"), ("United Kingdom", "UK")}),
    mismatch_type="mixed",
    split="dev",
)


class EchoLLMMatcher:
    """Asks the (fake) LLM about each right value; used to check accounting."""

    def match(self, task, ctx: RunContext):
        with ctx.stage("normalize"):
            pairs = build_matcher("normalized").match(task, ctx)
        with ctx.stage("llm_residual"):
            for r in task.right:
                if ctx.llm.complete(f"match {r}").text == "United Kingdom":
                    pairs.add(("United Kingdom", r))
        ctx.annotate(strategy="normalized")
        return pairs


class BrokenMatcher:
    def match(self, task, ctx):
        raise RuntimeError("boom")


def test_parse_spec():
    assert parse_spec("fuzzy:threshold=0.7,q=2") == ("fuzzy", {"threshold": 0.7, "q": 2})
    assert parse_spec("exact") == ("exact", {})


def test_baselines():
    ctx = RunContext(FakeLLM())
    assert build_matcher("exact").match(TASK, ctx) == set()
    assert build_matcher("normalized").match(TASK, ctx) == {("Germany", "germany")}
    fuzzy = build_matcher("fuzzy:threshold=0.3").match(TASK, ctx)
    assert ("France", "Frnace") in fuzzy


def test_run_records_cost_and_errors(tmp_path, monkeypatch):
    monkeypatch.setitem(MATCHERS, "echo", EchoLLMMatcher)
    monkeypatch.setitem(MATCHERS, "broken", BrokenMatcher)
    llm = FakeLLM(lambda p: "United Kingdom" if p.endswith("UK") else "none")
    out = run([TASK], ["normalized", "echo", "broken"], lambda: llm, tmp_path / "run")

    records = {r["matcher"]: r for r in map(json.loads, (out / "results.jsonl").read_text().splitlines())}
    echo = records["echo"]
    assert echo["llm_calls"] == 3
    assert echo["stage_llm_calls"] == {"llm_residual": 3}
    assert set(echo["stage_seconds"]) >= {"normalize", "llm_residual"}
    assert (echo["tp"], echo["fp"], echo["fn"]) == (2, 0, 1)
    assert echo["hard_tp"] == 1 and echo["hard_fn"] == 1
    assert echo["annotations"] == {"strategy": "normalized"}
    assert records["broken"]["error"] and "boom" in records["broken"]["error"]

    errors = [json.loads(l) for l in (out / "errors.jsonl").read_text().splitlines()]
    assert {"matcher": "echo", "false_negatives": [["France", "Frnace"]]}.items() <= errors[1].items()

    report = summarize(out, router="echo", strategies=["normalized"])
    assert "## By dataset" in report and (out / "summary.csv").exists()
    assert "accuracy=1.000" in report


def test_routing_regret():
    rec = lambda m, f1, ann=None: {"dataset": "d", "task_id": "t", "matcher": m, "f1": f1, "annotations": ann or {}}
    res = routing([rec("a", 0.9), rec("b", 0.5), rec("r", 0.5, {"strategy": "b"})], "r", ["a", "b"])
    assert res["accuracy"] == 0 and abs(res["mean_regret"] - 0.4) < 1e-9
