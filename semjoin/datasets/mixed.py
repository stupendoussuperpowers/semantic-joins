"""Mixed-type tasks: one column pair whose matches need different strategies.

Each task takes one domain (countries, currencies, airports, ...) and gives
every entity the variants it supports, e.g. Germany -> {code: "DEU",
alias: "Federal Republic of Germany", typo: "Germnay", reorder: None}. A
``sample_fraction`` of the entities is matched; each matched entity gets one
variant type, drawn by the task's ``weights`` from the types it supports.
The left column holds every entity's name, so the unmatched ones are
distractors.

This is the case routing cannot win outright: a router picks one strategy
per task, so it is the test of RQ2 (router vs. cascade). The realized type
counts go to ``type_mix`` in ``meta.json``; they differ a little from the
weights because not every entity supports every type.

Sources are offline or committed, so building needs no
network: pycountry, and the built tasks in ``data/wikidata/``. Tasks whose
Wikidata source is missing are skipped with a warning. They are part of the
``mapping`` dataset (task ids ``mixed_*``) and built by ``semjoin build-mapping``.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pycountry

from semjoin.datasets.mapping import _acronym, _reorder, _typo
from semjoin.datasets.pairdir import read_task, write_task
from semjoin.normalize import normalize

# entity name -> {mismatch type: right value}
Entities = dict[str, dict[str, str]]


@dataclass(frozen=True)
class Spec:
    entities: Callable[[Path, random.Random], Entities | None]
    weights: dict[str, float]
    source: str


def _synthetic(name: str, rng: random.Random, kinds=("typo", "reorder", "acronym")) -> dict[str, str]:
    makers = {"typo": _typo, "reorder": _reorder, "acronym": _acronym}
    out = {}
    for kind in kinds:
        if kind not in makers:
            continue  # real variants (code, alias) come from the source, not from here
        value = makers[kind](name, rng)
        if value is not None:
            out[kind] = value
    return out


def _countries(code_attr: str, kinds: tuple[str, ...]):
    def entities(data_root: Path, rng: random.Random) -> Entities:
        out = {}
        for c in sorted(pycountry.countries, key=lambda c: c.alpha_2):
            variants = {"code": getattr(c, code_attr), **_synthetic(c.name, rng, kinds)}
            official = getattr(c, "official_name", None)
            if "alias" in kinds and official and official != c.name:
                variants["alias"] = official
            out[c.name] = variants
        return out

    return entities


def _currencies(data_root: Path, rng: random.Random) -> Entities:
    return {
        c.name: {"code": c.alpha_3, **_synthetic(c.name, rng, ("typo", "reorder"))}
        for c in sorted(pycountry.currencies, key=lambda c: c.alpha_3)
    }


def _languages(data_root: Path, rng: random.Random) -> Entities:
    langs = sorted((l for l in pycountry.languages if hasattr(l, "alpha_2")), key=lambda l: l.alpha_2)
    return {l.name: {"code": l.alpha_2, **_synthetic(l.name, rng, ("typo",))} for l in langs}


def _subdivisions(data_root: Path, rng: random.Random) -> Entities:
    out = {}
    for cc in ("IN", "MX", "ZA", "NG", "AU", "CA"):
        for s in sorted(pycountry.subdivisions.get(country_code=cc), key=lambda s: s.code):
            out.setdefault(s.name, {"code": s.code, **_synthetic(s.name, rng)})
    return out


def _from_wikidata(task_id: str, real_type: str):
    """Entities from a committed Wikidata task: its gold value as ``real_type``, plus typos.

    The task's unmatched left values are entities too: they can still get a typo.
    """

    def entities(data_root: Path, rng: random.Random) -> Entities | None:
        task_dir = data_root / "wikidata" / task_id
        if not (task_dir / "gt.csv").exists():
            return None
        task = read_task(task_dir, "wikidata")
        real: dict[str, list[str]] = {}
        for l, r in sorted(task.gold):
            real.setdefault(l, []).append(r)
        out = {}
        for name in task.left:
            variants = _synthetic(name, rng, ("typo",))
            if name in real:
                variants[real_type] = rng.choice(real[name])
            out[name] = variants
        return out

    return entities


SPECS: dict[str, Spec] = {
    "mixed_country": Spec(
        _countries("alpha_3", ("alias", "typo", "reorder")),
        {"code": 0.4, "alias": 0.2, "typo": 0.2, "reorder": 0.2},
        "pycountry",
    ),
    "mixed_country_code_typo": Spec(_countries("alpha_2", ("typo",)), {"code": 0.5, "typo": 0.5}, "pycountry"),
    "mixed_currency": Spec(_currencies, {"code": 0.4, "typo": 0.3, "reorder": 0.3}, "pycountry"),
    "mixed_language": Spec(_languages, {"code": 0.7, "typo": 0.3}, "pycountry"),
    "mixed_subdivision": Spec(
        _subdivisions, {"code": 0.4, "typo": 0.3, "acronym": 0.15, "reorder": 0.15}, "pycountry"
    ),
    "mixed_airports": Spec(_from_wikidata("airport_iata", "code"), {"code": 0.6, "typo": 0.4}, "wikidata"),
    "mixed_physicists": Spec(_from_wikidata("physicist_names", "reorder"), {"reorder": 0.5, "typo": 0.5}, "wikidata"),
    "mixed_ngos": Spec(_from_wikidata("ngo_short", "acronym"), {"acronym": 0.6, "typo": 0.4}, "wikidata"),
    "mixed_drugs": Spec(_from_wikidata("drug_alias", "alias"), {"alias": 0.5, "typo": 0.5}, "wikidata"),
}


def build_task(
    entities: Entities, weights: dict[str, float], rng: random.Random, sample_fraction: float
) -> tuple[list[str], list[tuple[str, str]], dict[str, int]]:
    """Return (left values, gold pairs, realized type counts)."""
    left = sorted(entities)
    left_norm = {normalize(n) for n in left}
    # A value usable as a right value: not equal to any name (an equi-join or a
    # different entity would claim it) and produced by exactly one entity.
    produced = Counter(v for variants in entities.values() for v in set(variants.values()))

    def usable(value: str) -> bool:
        return normalize(value) not in left_norm and produced[value] == 1

    supporters = {
        t: [n for n in left if t in entities[n] and usable(entities[n][t])] for t in sorted(weights)
    }
    n_total = min(
        len({n for names in supporters.values() for n in names}), round(sample_fraction * len(left))
    )
    # Fill each type's quota from the entities that support it, scarcest type
    # first so common types (typo) do not use up the entities a rare type needs.
    # A type with too few supporters leaves a deficit that the other types fill.
    used: set[str] = set()
    picks: list[tuple[str, str]] = []  # (name, type)
    for t in sorted(weights, key=lambda t: (len(supporters[t]), t)):
        pool = [n for n in supporters[t] if n not in used]
        take = rng.sample(pool, min(len(pool), round(weights[t] * n_total)))
        used.update(take)
        picks.extend((n, t) for n in take)
    spare = [(n, t) for t in sorted(weights) for n in supporters[t] if n not in used]
    rng.shuffle(spare)
    for n, t in spare:
        if len(picks) >= n_total:
            break
        if n not in used:
            used.add(n)
            picks.append((n, t))

    gold = sorted((n, entities[n][t]) for n, t in picks)
    counts = Counter(t for _, t in picks)
    return left, gold, dict(counts.most_common())


def build(
    data_root: Path,
    out_dir: Path | None = None,
    names: list[str] | None = None,
    seed: int = 0,
    sample_fraction: float = 0.6,
) -> list[str]:
    out_dir = out_dir or data_root / "mapping"
    built = []
    for task_id in names or list(SPECS):
        spec = SPECS[task_id]
        rng = random.Random(f"{seed}:{task_id}")
        entities = spec.entities(data_root, rng)
        if entities is None:
            print(f"skip {task_id}: build the wikidata dataset first", file=sys.stderr)
            continue
        left, gold, type_mix = build_task(entities, spec.weights, rng, sample_fraction)
        write_task(
            out_dir / task_id,
            left=left,
            right=sorted({r for _, r in gold}),
            gold=gold,
            meta={
                "mismatch_type": "mixed",
                "type_mix": type_mix,
                "weights": spec.weights,
                "source": spec.source,
                "seed": seed,
                "sample_fraction": sample_fraction,
                "synthetic": True,
            },
        )
        built.append(task_id)
    return built
