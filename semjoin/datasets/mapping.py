"""Build the constructed mapping-table dataset from pycountry.

Three kinds of tasks:

* real mappings (code <-> name, name <-> official name) taken directly from
  ISO tables, e.g. DE <-> Germany, NY <-> New York;
* ISO 3166-2 subdivision codes per country (``subdiv_<cc>``, e.g. Maharashtra
  <-> IN-MH). The left column adds a second country's subdivision names as
  distractors; the full code with its country prefix keeps them unambiguous;
* synthetic perturbations (typos, token reordering, initialisms) of four name
  domains: countries, subdivisions, languages and currencies, seeded so they
  are reproducible. Acronyms that more than one name produces are dropped.

For each task a random ``sample_fraction`` of the mapping rows forms the right
column, while the left column holds every name in the table, so left-side
values without a match act as distractors. A task whose right column would
have fewer than ``MIN_RIGHT`` values is skipped.
"""

from __future__ import annotations

import importlib.metadata
import random
import string
from collections import Counter
from pathlib import Path
from typing import Callable

import pycountry

from semjoin.datasets.pairdir import write_task

_STOPWORDS = {"of", "and", "the", "da", "de", "du"}
MIN_RIGHT = 20

# subdiv_<cc> tasks: (country, country whose subdivision names are the distractors).
_SUBDIV_TASKS = (
    ("IN", "BD"),
    ("MX", "AR"),
    ("NG", "GH"),
    ("CN", "TW"),
    ("ES", "PT"),
    ("BR", "CO"),
    ("IT", "CH"),
    ("JP", "KR"),
)
# Countries whose subdivision names form the "subdivision" perturbation domain.
_SUBDIV_POOL = ("US", "CA", "AU", "IN", "BR", "MX", "ZA", "NG", "GB", "DE")
_SUBDIV_POOL_SIZE = 300


def _countries() -> list:
    return sorted(pycountry.countries, key=lambda c: c.alpha_2)


def _country_names() -> list[str]:
    return [c.name for c in _countries()]


def _subdivision_names(country_code: str) -> list[str]:
    subs = pycountry.subdivisions.get(country_code=country_code) or []
    return sorted({s.name for s in subs})


def _subdivision_tables() -> dict[str, tuple[list[tuple[str, str]], list[str]]]:
    """task_id -> ([(name, ISO 3166-2 code)], distractor names)."""
    out = {}
    for cc, other in _SUBDIV_TASKS:
        subs = sorted(pycountry.subdivisions.get(country_code=cc), key=lambda s: s.code)
        out[f"subdiv_{cc.lower()}"] = ([(s.name, s.code) for s in subs], _subdivision_names(other))
    return out


def _name_domains() -> dict[str, list[str]]:
    pool = sorted({n for cc in _SUBDIV_POOL for n in _subdivision_names(cc)})
    pool = sorted(random.Random("subdivision_pool").sample(pool, min(_SUBDIV_POOL_SIZE, len(pool))))
    langs = sorted({l.name for l in pycountry.languages if hasattr(l, "alpha_2")})
    return {
        "country": _country_names(),
        "subdivision": pool,
        "language": langs,
        "currency": sorted({c.name for c in pycountry.currencies}),
    }


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
    "typo": (_typo, "typo"),
    "reorder": (_reorder, "reorder"),
    "acronym": (_acronym, "acronym"),
}


def _write_sampled(out_dir, task_id, left, rows, sample_fraction, meta, rng=None) -> bool:
    rng = rng or random.Random(f"{meta['seed']}:{task_id}")
    n = round(sample_fraction * len(rows))
    if n < MIN_RIGHT:
        return False
    sample = rng.sample(rows, n)
    write_task(out_dir / task_id, left=left, right=[v for _, v in sample], gold=sample, meta=meta)
    return True


def _perturbed_pairs(names: list[str], perturb, rng: random.Random) -> list[tuple[str, str]]:
    name_set = set(names)
    pairs = []
    for name in names:
        value = perturb(name, rng)
        # Drop perturbations that collide with a real name.
        if value is not None and value not in name_set:
            pairs.append((name, value))
    # Drop values that more than one name produces (e.g. two names with the
    # same initials): the gold would otherwise miss a valid match.
    produced = Counter(v for _, v in pairs)
    return [(n, v) for n, v in pairs if produced[v] == 1]


def build(out_dir: Path, seed: int = 0, sample_fraction: float = 0.6) -> list[str]:
    meta_base = {
        "source": "pycountry",
        "pycountry_version": importlib.metadata.version("pycountry"),
        "seed": seed,
        "sample_fraction": sample_fraction,
    }
    built = []

    for task_id, (rows, mismatch_type) in _code_tables().items():
        meta = {**meta_base, "mismatch_type": mismatch_type}
        if _write_sampled(out_dir, task_id, [name for name, _ in rows], rows, sample_fraction, meta):
            built.append(task_id)

    for task_id, (rows, distractors) in _subdivision_tables().items():
        meta = {**meta_base, "mismatch_type": "code", "sample_fraction": 0.8, "distractors": len(distractors)}
        left = [name for name, _ in rows] + distractors
        if _write_sampled(out_dir, task_id, left, rows, 0.8, meta):
            built.append(task_id)

    for domain, names in _name_domains().items():
        for kind, (perturb, mismatch_type) in _PERTURBATIONS.items():
            task_id = f"{domain}_{kind}"
            # One generator for perturbing and sampling, as the original country tasks did.
            rng = random.Random(f"{seed}:{task_id}")
            pairs = _perturbed_pairs(names, perturb, rng)
            meta = {**meta_base, "mismatch_type": mismatch_type, "synthetic": True, "domain": domain}
            if _write_sampled(out_dir, task_id, names, pairs, sample_fraction, meta, rng):
                built.append(task_id)

    return built
