"""Batched LLM matching for the values no cheaper step matched (proposal step 4).

Right values are sent ``batch_size`` at a time. Each prompt is seeded with up
to ``n_examples`` confident pairs already found for this column pair (by the
normalized join and earlier steps), preferring pairs that differ after
normalization, so the model sees how the two columns relate.

Two prompt layouts, chosen per task:

* **shared list** (``len(left) <= full_list_max``): the whole left column is
  listed once per prompt and the model picks left numbers. Needed for
  code <-> value columns, where q-gram retrieval cannot find "Germany" for "DE".
* **candidates** (larger left columns): each right value gets its own
  shortlist of the ``k`` most similar left values (max of char 3-gram and word
  TF-IDF cosine) and the model picks from it.

The model answers with one JSON object, e.g. ``{"b1": 3, "b2": null}``; a list
of numbers is accepted for many-to-many matches. Unparseable or out-of-range
answers are treated as "no match".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from semjoin.core import JoinTask, Pair, RunContext
from semjoin.matchers.utils_matching import normalize_first, parse_json_object, pick_examples, unmatched
from semjoin.matchers.utils_similarity import top_k

SYSTEM = (
    "You are a careful data integration assistant. You decide which values from two "
    "table columns refer to the same real-world entity. Reply with a single JSON object and nothing else."
)


def _q(value: str, max_chars: int = 120) -> str:
    v = value if len(value) <= max_chars else value[: max_chars - 3] + "..."
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _examples_block(examples: list[Pair]) -> str:
    if not examples:
        return ""
    lines = "\n".join(f"  B {_q(r)} matches A {_q(l)}" for l, r in examples)
    return f"Examples of matching values from this pair of columns:\n{lines}\n\n"


def _as_indices(answer: Any) -> list[int]:
    items = answer if isinstance(answer, list) else [answer]
    out = []
    for item in items:
        if isinstance(item, bool) or item is None:
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


class LLMMatcher:
    """``llm``: normalized join, then this LLM pass on everything left unmatched."""

    def __init__(self, batch_size: int = 20, k: int = 10, full_list_max: int = 300, n_examples: int = 5):
        self.batch_size = batch_size
        self.k = k
        self.full_list_max = full_list_max
        self.n_examples = n_examples

    def shared_prompt(self, left: Sequence[str], batch: Sequence[str], examples: list[Pair]) -> str:
        a_list = "\n".join(f"{i}. {_q(v)}" for i, v in enumerate(left, 1))
        b_list = "\n".join(f"b{i}. {_q(v)}" for i, v in enumerate(batch, 1))
        return (
            "Column A values:\n"
            f"{a_list}\n\n"
            f"{_examples_block(examples)}"
            "For each column B value below, give the number of the column A value that refers to the "
            "same entity, or null if none does. Values may differ by codes, abbreviations, aliases, "
            "typos, formatting or word order.\n\n"
            "Column B values:\n"
            f"{b_list}\n\n"
            'Answer as JSON mapping each B id to an A number or null, e.g. {"b1": 12, "b2": null}.'
        )

    def candidates_prompt(self, left: Sequence[str], batch: Sequence[str], cands: list[list[int]], examples: list[Pair]) -> str:
        blocks = []
        for i, (v, cs) in enumerate(zip(batch, cands), 1):
            opts = "\n".join(f"    {j}. {_q(left[c])}" for j, c in enumerate(cs, 1))
            blocks.append(f"b{i}. {_q(v)}\n  candidates:\n{opts}")
        return (
            f"{_examples_block(examples)}"
            "For each value below, give the number of the candidate that refers to the same entity, "
            "or null if none does. Values may differ by codes, abbreviations, aliases, typos, "
            "formatting or word order. Do not pick a candidate that is only related or similar.\n\n"
            + "\n\n".join(blocks)
            + '\n\nAnswer as JSON mapping each id to a candidate number or null, e.g. {"b1": 2, "b2": null}.'
        )

    def match(self, task: JoinTask, ctx: RunContext) -> set[Pair]:
        pairs, _ = normalize_first(task, ctx)
        ctx.annotate(strategy="llm")
        return add_residual(task, ctx, pairs, self)

    def find_pairs(self, left: Sequence[str], right: Sequence[str], ctx: RunContext, seeds: set[Pair]) -> set[Pair]:
        if not right or not left:
            return set()
        examples = pick_examples(seeds, self.n_examples)
        shared = len(left) <= self.full_list_max
        cands = None if shared else [[j for j, _ in row] for row in top_k(left, right, self.k)]
        pairs: set[Pair] = set()
        for start in range(0, len(right), self.batch_size):
            batch = right[start : start + self.batch_size]
            if shared:
                prompt = self.shared_prompt(left, batch, examples)
                options = [list(range(len(left)))] * len(batch)
            else:
                options = cands[start : start + self.batch_size]
                if not any(options):
                    continue
                prompt = self.candidates_prompt(left, batch, options, examples)
            answer = parse_json_object(ctx.llm.complete(prompt, system=SYSTEM).text)
            for i, (r, opts) in enumerate(zip(batch, options), 1):
                for n in _as_indices(answer.get(f"b{i}", answer.get(str(i)))):
                    if 1 <= n <= len(opts):
                        pairs.add((left[opts[n - 1]], r))
        ctx.annotate(llm_residual_mode="shared" if shared else "candidates")
        return pairs


def add_residual(task: JoinTask, ctx: RunContext, pairs: set[Pair], llm: LLMMatcher | None = None) -> set[Pair]:
    """Send the right values not in ``pairs`` to the LLM; ``pairs`` seed the few-shot examples."""
    remaining = unmatched(task.right, pairs)
    if not remaining:
        return pairs
    with ctx.stage("llm_residual"):
        return pairs | (llm or LLMMatcher()).find_pairs(list(task.left), remaining, ctx, pairs)
