import re

import pytest

from semjoin.core import JoinTask, RunContext
from semjoin.llm import FakeLLM
from semjoin.matchers import build_matcher
from semjoin.matchers.acronym import initialisms
from semjoin.matchers.pipeline import parse_json_object, pick_examples
from semjoin.matchers.transform import learn_programs, run_program, tokens


def task(left, right, gold, task_id="t"):
    return JoinTask("d", task_id, tuple(left), tuple(right), frozenset(gold))


def run(spec, t, llm=None):
    ctx = RunContext(llm or FakeLLM())
    return build_matcher(spec).match(t, ctx), ctx


COUNTRIES = ["Germany", "France", "Spain", "Italy", "Norway", "Sweden", "Poland", "Austria"]
CODES = {"DE": "Germany", "FR": "France", "ES": "Spain", "IT": "Italy", "NO": "Norway"}


def test_acronym_both_directions():
    assert {"ny"} == initialisms("New York")
    assert "uk" in initialisms("the United Kingdom") and "usa" in initialisms("United States of America")
    t = task(["New York", "UK", "Ohio"], ["NY", "United Kingdom", "OH"], {("New York", "NY"), ("UK", "United Kingdom")})
    pairs, _ = run("acronym", t)
    assert pairs == t.gold


def test_acronym_ambiguous_is_skipped():
    t = task(["New York", "North Yemen"], ["NY"], set())
    assert run("acronym", t)[0] == set()
    assert len(run("acronym:unique=false", t)[0]) == 2


def test_transform_programs():
    programs = learn_programs([("John Smith", "J Smith"), ("Mary Jones", "M Jones"), ("Ann Lee", "A Lee")])
    assert programs and programs[0][1] == 3
    best = programs[0][0]
    assert run_program(best, tokens("Peter Parker")) == "pparker"


def test_transform_bag_join_and_learned_programs():
    left = ["John Smith", "Mary Jones", "Ann Lee", "Peter Parker", "Bruce Wayne"]
    right = ["Smith, John", "Jones, Mary", "P. Parker", "B. Wayne"]
    gold = {("John Smith", "Smith, John"), ("Mary Jones", "Jones, Mary"), ("Peter Parker", "P. Parker"), ("Bruce Wayne", "B. Wayne")}
    pairs, ctx = run("transform:example_sim=0.4", task(left, right, gold))
    assert {("John Smith", "Smith, John"), ("Mary Jones", "Jones, Mary")} <= pairs
    assert pairs <= gold


def test_autofj_joins_typos_and_reports_configs():
    left = ["Germany", "France", "Spain", "Italy", "Norway", "Sweden", "Portugal", "Netherlands"]
    right = ["Germnay", "Frnace", "Portugall", "Netherland"]
    gold = {("Germany", "Germnay"), ("France", "Frnace"), ("Portugal", "Portugall"), ("Netherlands", "Netherland")}
    pairs, ctx = run("autofj", task(left, right, gold))
    # Transposition typos in short names land too far away for the 2d-ball test
    # on an 8-value reference table; the near-misses are joined.
    assert {("Portugal", "Portugall"), ("Netherlands", "Netherland")} <= pairs <= gold
    assert ctx.annotations["autofj_configs"]


def test_autofj_rejects_dense_reference_neighbourhoods():
    left = [f"{y} Florida Gators football team" for y in range(2000, 2012)]
    # With the year, the year token separates the seasons: a clean, safe join.
    assert run("autofj", task(left, ["2008 Florida Gators football"], set()))[0] == {(left[8], "2008 Florida Gators football")}
    # Without it, every season is equally close, the 2d-ball around the nearest
    # one holds all twelve, and the estimated precision (1/12) is far below target.
    assert run("autofj", task(left, ["Florida Gators football"], set()))[0] == set()


def test_parse_json_object():
    assert parse_json_object('```json\n{"b1": 3, "b2": null}\n```') == {"b1": 3, "b2": None}
    assert parse_json_object('sure: "b1": 2, "b2": [1, 4] done') == {"b1": 2, "b2": [1, 4]}
    assert parse_json_object("nope") == {}


def test_pick_examples_prefers_hard_pairs():
    seeds = {("France", "france"), ("Germany", "DE"), ("Spain", "ES")}
    assert pick_examples(seeds, 2) == [("Germany", "DE"), ("Spain", "ES")]


def oracle_residual(gold_r2l):
    """Answers residual prompts (both layouts) from a right -> left dict."""

    def respond(prompt):
        answer = {}
        if prompt.startswith("Column A values:"):
            a = {v: int(n) for n, v in re.findall(r'^(\d+)\. "(.*)"$', prompt, re.M)}
            for bid, v in re.findall(r'^(b\d+)\. "(.*)"$', prompt, re.M):
                answer[bid] = a.get(gold_r2l.get(v))
        else:
            for block in prompt.split("\n\n"):
                m = re.match(r'(b\d+)\. "(.*)"', block)
                if m:
                    cands = {v: int(n) for n, v in re.findall(r'^\s+(\d+)\. "(.*)"$', block, re.M)}
                    answer[m.group(1)] = cands.get(gold_r2l.get(m.group(2)))
        import json

        return json.dumps(answer)

    return respond


def test_llm_residual_shared_list():
    right = list(CODES) + ["Germnay"]
    gold = {(l, r) for r, l in CODES.items()} | {("Germany", "Germnay")}
    llm = FakeLLM(oracle_residual({r: l for l, r in gold}))
    pairs, ctx = run("llm:batch_size=2", task(COUNTRIES, right, gold), llm)
    assert pairs == gold
    assert ctx.llm_calls == {"llm_residual": 3}
    assert ctx.annotations["llm_residual_mode"] == "shared"


def test_llm_residual_candidates():
    right = ["Germnay", "Frnace", "Swedn", "DE"]
    gold = {("Germany", "Germnay"), ("France", "Frnace"), ("Sweden", "Swedn"), ("Germany", "DE")}
    llm = FakeLLM(oracle_residual({r: l for l, r in gold}))
    pairs, ctx = run("llm:full_list_max=0,k=3", task(COUNTRIES, right, gold), llm)
    # Candidate retrieval is lexical, so a code cannot reach its name.
    assert pairs == gold - {("Germany", "DE")}
    assert ctx.annotations["llm_residual_mode"] == "candidates"


def test_llm_gen_generates_and_joins():
    def respond(prompt):
        import json

        src = re.findall(r'^(\d+)\. "(.*)"$', prompt, re.M)
        return json.dumps({n: CODES.get(v, "Germanyy" if v == "XX" else None) for n, v in src})

    right = list(CODES) + ["XX", "QQ"]
    gold = {(l, r) for r, l in CODES.items()}
    pairs, ctx = run("llm_gen:batch_size=4,fuzzy_threshold=0.5", task(COUNTRIES, right, gold), FakeLLM(respond))
    # "Germanyy" is resolved to Germany by the fuzzy fallback (a false positive here).
    assert pairs == gold | {("Germany", "XX")}
    assert ctx.llm_calls == {"gen": 2}


def test_cascade_sends_only_residual_to_llm():
    left = COUNTRIES + ["United Kingdom"]
    right = ["germany", "Frnace", "UK", "IT"]
    gold = {("Germany", "germany"), ("France", "Frnace"), ("United Kingdom", "UK"), ("Italy", "IT")}
    llm = FakeLLM(oracle_residual({r: l for l, r in gold}))
    pairs, ctx = run("cascade", task(left, right, gold), llm)
    assert pairs == gold
    assert ctx.llm_calls == {"llm_residual": 1}
    assert '"IT"' in llm.prompts[0] and '"UK"' not in llm.prompts[0].split("Column B values:")[1]
    assert ctx.annotations["step_pairs"]["normalize"] == 1


def test_pipeline_kwarg_routing():
    m = build_matcher("pipeline:steps=autofj+acronym,residual=true,autofj_target=0.8,acronym_max_len=5,llm_batch_size=7")
    assert m.steps[0][1].target == 0.8 and m.steps[1][1].max_len == 5 and m.residual_step.batch_size == 7
    assert build_matcher("autofj:target=0.7").steps[0][1].target == 0.7
    with pytest.raises(ValueError, match="prefixed"):
        build_matcher("cascade:target=0.8")
    with pytest.raises(ValueError, match="unknown argument"):
        build_matcher("acronym:bogus=1")
    with pytest.raises(ValueError, match="unknown step"):
        build_matcher("pipeline:steps=nope")
