import pytest

from semjoin.metrics import Counts, hard_only, micro, score


def test_score_many_to_many():
    gold = {("Germany", "DE"), ("Germany", "GER"), ("France", "FR")}
    pred = {("Germany", "DE"), ("Germany", "GER"), ("Spain", "FR")}
    c = score(pred, gold)
    assert (c.tp, c.fp, c.fn) == (2, 1, 1)
    assert c.precision == pytest.approx(2 / 3)
    assert c.recall == pytest.approx(2 / 3)
    assert c.f1 == pytest.approx(2 / 3)


def test_empty_conventions():
    assert score(set(), {("a", "b")}).f1 == 0.0
    nothing = score(set(), set())
    assert (nothing.precision, nothing.recall, nothing.f1) == (1.0, 1.0, 1.0)


def test_micro_pools_counts():
    total = micro([Counts(1, 0, 1), Counts(3, 1, 0)])
    assert total == Counts(4, 1, 1)


def test_hard_only_drops_normalized_matches():
    pairs = {("Qing Dynasty", "Qing dynasty"), ("Burma", "Myanmar"), ("São Tomé", "Sao Tome")}
    assert hard_only(pairs) == {("Burma", "Myanmar")}
