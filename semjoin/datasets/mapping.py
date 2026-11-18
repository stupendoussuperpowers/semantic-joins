"""Build the constructed mapping-table dataset from pycountry.

Two kinds of tasks:

* real mappings (code <-> name, name <-> official name) taken directly from
  ISO tables, e.g. DE <-> Germany, NY <-> New York;
* synthetic perturbations of country names (typos, token reordering,
  initialisms), seeded so they are reproducible.

For each task a random ``sample_fraction`` of the mapping rows forms the right
column, while the left column holds every name in the table, so left-side
values without a match act as distractors.
"""

from __future__ import annotations

import importlib.metadata
import random
import string
from pathlib import Path
from typing import Callable

import pycountry

from semjoin.datasets.pairdir import write_task

_STOPWORDS = {"of", "and", "the", "da", "de", "du"}


def _countries() -> list:
    return sorted(pycountry.countries, key=lambda c: c.alpha_2)


def _country_names() -> list[str]:
    return [c.name for c in _countries()]


def _code_tables() -> dict[str, tuple[list[tuple[str, str]], str]]:
    """task_id -> ([(left name, right value)], mismatch_type)."""
    us = sorted(pycountry.subdivisions.get(country_code="US"), key=lambda s: s.code)
    langs = sorted((l for l in pycountry.languages if hasattr(l, "alpha_2")), key=lambda l: l.alpha_2)
    return {
        "country_alpha2": ([(c.name, c.alpha_2) for c in _countries()], "code"),
        "country_alpha3": ([(c.name, c.alpha_3) for c in _countries()], "code"),
        "language_alpha2": ([(l.name, l.alpha_2) for l in langs], "code"),
        "currency_alpha3": (
            [(c.name, c.alpha_3) for c in sorted(pycountry.currencies, key=lambda c: c.alpha_3)],
            "code",
        ),
        "us_state_postal": ([(s.name, s.code.removeprefix("US-")) for s in us], "code"),
        "country_official_name": (
            [(c.name, c.official_name) for c in _countries() if hasattr(c, "official_name")],
            "alias",
        ),
    }


def _typo(value: str, rng: random.Random) -> str | None:
    chars = list(value)
    for _ in range(rng.randint(1, 2)):
        i = rng.randrange(len(chars))
        op = rng.choice(("delete", "insert", "substitute", "swap"))
        if op == "delete" and len(chars) > 3:
            del chars[i]
        elif op == "insert":
            chars.insert(i, rng.choice(string.ascii_lowercase))
        elif op == "swap" and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        else:
            chars[i] = rng.choice(string.ascii_lowercase)
    out = "".join(chars)
    return out if out != value else None


def _reorder(value: str, rng: random.Random) -> str | None:
    # "Smith, John" style: move the last token to the front.
    tokens = value.split()
    if len(tokens) < 2 or "," in value:
        return None
    return f"{tokens[-1]}, {' '.join(tokens[:-1])}"


def _acronym(value: str, rng: random.Random) -> str | None:
    words = [w for w in value.replace(",", " ").replace("-", " ").split() if w.lower() not in _STOPWORDS]
    if len(words) < 2:
        return None
    return "".join(w[0].upper() for w in words)


_PERTURBATIONS: dict[str, tuple[Callable[[str, random.Random], str | None], str]] = {
    "country_typo": (_typo, "typo"),
    "country_reorder": (_reorder, "reorder"),
    "country_acronym": (_acronym, "acronym"),
}


def build(out_dir: Path, seed: int = 0, sample_fraction: float = 0.6) -> list[str]:
    meta_base = {
        "source": "pycountry",
        "pycountry_version": importlib.metadata.version("pycountry"),
        "seed": seed,
        "sample_fraction": sample_fraction,
    }
    built = []

    for task_id, (rows, mismatch_type) in _code_tables().items():
        rng = random.Random(f"{seed}:{task_id}")
        sample = rng.sample(rows, round(sample_fraction * len(rows)))
        write_task(
            out_dir / task_id,
            left=[name for name, _ in rows],
            right=[value for _, value in sample],
            gold=sample,
            meta={**meta_base, "mismatch_type": mismatch_type},
        )
        built.append(task_id)

    names = _country_names()
    name_set = set(names)
    for task_id, (perturb, mismatch_type) in _PERTURBATIONS.items():
        rng = random.Random(f"{seed}:{task_id}")
        pairs = []
        for name in names:
            value = perturb(name, rng)
            # Drop perturbations that collide with a real name.
            if value is not None and value not in name_set:
                pairs.append((name, value))
        sample = rng.sample(pairs, round(sample_fraction * len(pairs)))
        write_task(
            out_dir / task_id,
            left=names,
            right=[value for _, value in sample],
            gold=sample,
            meta={**meta_base, "mismatch_type": mismatch_type, "synthetic": True},
        )
        built.append(task_id)

    return built
