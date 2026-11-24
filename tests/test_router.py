import csv

import pytest

from semjoin.core import JoinTask, RunContext
from semjoin.llm import FakeLLM
from semjoin.matchers import build_matcher
from semjoin.matchers.profiling import Profile, profile, shape, uniformity
from semjoin.matchers.router import RouterMatcher, strategy_specs
from semjoin.run import run
from semjoin.summarize import load_results, routing, summarize

COUNTRIES = [
    "Germany", "France", "Spain", "Italy", "Norway", "Sweden", "Poland", "Austria",
    "United Kingdom", "New Zealand", "South Africa", "Costa Rica", "United States",
]
# One toy column pair per routing rule: (left, right, gold).
CASES = {
    "code": (COUNTRIES, ["DE", "FR", "ES", "IT"], {("Germany", "DE"), ("France", "FR"), ("Spain", "ES"), ("Italy", "IT")}),
    "acronym": (
        COUNTRIES,
        ["UK", "NZ", "SA", "CR"],
        {("United Kingdom", "UK"), ("New Zealand", "NZ"), ("South Africa", "SA"), ("Costa Rica", "CR")},
    ),
    "reorder": (
        ["John Smith", "Mary Jones", "Ann Lee", "Peter Parker"],
        ["Smith, John", "Jones, Mary", "Lee, Ann"],
        {("John Smith", "Smith, John"), ("Mary Jones", "Jones, Mary"), ("Ann Lee", "Lee, Ann")},
    ),
    "typo": (
        COUNTRIES,
        ["Untied Kingdom", "New Zeland", "South Afrca", "Unted States"],
        {
            ("United Kingdom", "Untied Kingdom"),
            ("New Zealand", "New Zeland"),
            ("South Africa", "South Afrca"),
            ("United States", "Unted States"),
        },
    ),
    "fallback": (
        ["Germany", "Netherlands", "United States", "Japan"],
        ["Deutschland", "Holland", "America", "Nippon"],
        {("Germany", "Deutschland"), ("Netherlands", "Holland"), ("United States", "America"), ("Japan", "Nippon")},
    ),
}


def task(name):
    left, right, gold = CASES[name]
    return JoinTask("toy", name, tuple(left), tuple(right), frozenset(gold))


def make_profile(**overrides):
    base = dict(
        n_left=10, n_right=10, n_sample=10, left_len=10.0, right_len=10.0, left_uniformity=0.2,
        right_uniformity=0.2, qgram_cos=0.0, token_jaccard=0.0, token_set_equal=0.0, acronym_rate=0.0,
    )
    return Profile(**{**base, **overrides})


def test_shape_and_uniformity():
    assert shape("DE") == "AA" and shape(" 111-11-1111 ") == "999-99-9999"
    assert uniformity(["DE", "FR", "USA"]) == pytest.approx(2 / 3)
    assert uniformity([]) == 0.0


def test_profile_statistics():
    code = profile(*CASES["code"][:2])
    assert (code.right_len, code.right_uniformity, code.acronym_rate) == (2.0, 1.0, 0.0)
    assert profile(*CASES["acronym"][:2]).acronym_rate == 1.0
    reorder = profile(*CASES["reorder"][:2])
    assert reorder.token_set_equal == 1.0 and reorder.token_jaccard == 1.0
    typo, alias = profile(*CASES["typo"][:2]), profile(*CASES["fallback"][:2])
    assert typo.token_set_equal == 0.0 and typo.qgram_cos > 0.6 > alias.qgram_cos


def test_profile_samples_right_values_deterministically():
    right = [f"value {i}" for i in range(50)]
    p = profile(COUNTRIES, right, sample=10, seed=3)
    assert (p.n_right, p.n_sample) == (50, 10)
    assert p == profile(COUNTRIES, right, sample=10, seed=3)
    assert profile(COUNTRIES, []).n_sample == 0


@pytest.mark.parametrize("name", CASES)
def test_each_toy_pair_fires_its_rule(name):
    t = task(name)
    rule, _ = RouterMatcher().route(profile(t.left, t.right))
    assert rule == name


def test_route_order_and_thresholds():
    r = RouterMatcher()
    assert r.route(make_profile()) == ("fallback", "llm")
    # Initialisms are short and uniform too; the acronym rule must win.
    assert r.route(make_profile(right_len=2.0, right_uniformity=1.0, acronym_rate=0.9)) == ("acronym", "acronym")
    assert r.route(make_profile(right_len=2.0, right_uniformity=1.0, acronym_rate=0.1)) == ("code", "llm_gen")
    # The code side may be either column.
    assert r.route(make_profile(left_len=3.0, left_uniformity=0.9)) == ("code", "llm_gen")
    # Reordered tokens can share q-grams too; the reorder rule must win.
    assert r.route(make_profile(token_set_equal=0.8, qgram_cos=0.9)) == ("reorder", "transform")
    assert r.route(make_profile(qgram_cos=0.7)) == ("typo", "autofj")
    strict = build_matcher("router:typo_cos=0.8,fallback=autofj")
    assert strict.route(make_profile(qgram_cos=0.7)) == ("fallback", "autofj")


@pytest.mark.parametrize("name", CASES)
def test_router_returns_the_chosen_strategys_pairs(name):
    t = task(name)
    ctx = RunContext(FakeLLM())
    pairs = build_matcher("router").match(t, ctx)
    spec = ctx.annotations["strategy"]
    assert spec in strategy_specs()
    assert ctx.annotations["route_rule"] == name
    assert ctx.annotations["profile"]["n_right"] == len(t.right)
    assert pairs == build_matcher(spec).match(t, RunContext(FakeLLM()))


def test_profiling_calls_no_llm():
    ctx = RunContext(FakeLLM())
    pairs = build_matcher("router:residual=false").match(task("typo"), ctx)
    assert ctx.annotations["strategy"] == "autofj"
    assert sum(ctx.llm_calls.values()) == 0 and "profile" in ctx.stage_seconds
    assert pairs == build_matcher("autofj").match(task("typo"), RunContext(FakeLLM()))


def test_nothing_unmatched_needs_no_llm():
    t = JoinTask("toy", "same", ("New York", "Ohio"), ("new york", "OHIO"), frozenset({("New York", "new york"), ("Ohio", "OHIO")}))
    ctx = RunContext(FakeLLM())
    assert build_matcher("router").match(t, ctx) == t.gold
    assert ctx.annotations["route_rule"] == "nothing_unmatched" and not ctx.llm_calls


def test_strategy_specs_build():
    assert strategy_specs() == [
        "acronym:residual=true", "llm_gen:residual=true", "transform:residual=true", "autofj:residual=true", "llm",
    ]
    assert strategy_specs(residual=False) == ["acronym", "llm_gen", "transform", "autofj", "llm"]
    for spec in strategy_specs() + strategy_specs(residual=False):
        build_matcher(spec)
    with pytest.raises(ValueError):
        build_matcher("router:fallback=fuzzy")


def test_summarize_router_against_its_strategies(tmp_path):
    specs = strategy_specs()
    out = run([task("code"), task("reorder")], ["router", *specs], FakeLLM, tmp_path / "run")
    assert "## Routing: router" in summarize(out, router="router", strategies=specs)
    records = load_results(out)
    f1 = {(r["task_id"], r["matcher"]): r["f1"] for r in records}
    res = routing(records, "router", specs)
    assert res["tasks"] == 2
    for row in res["rows"]:
        assert row["chosen"] in specs
        assert row["router_f1"] == f1[(row["task"].split("/", 1)[1], row["chosen"])]
    with (out / "routing.csv").open() as f:
        header = next(csv.reader(f))
    assert {"rule", "qgram_cos", "acronym_rate"} <= set(header)
