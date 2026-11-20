"""Example-driven LLM transformation in the style of DTT (Dargahi Nobari & Rafiei, SIGMOD 2024).

DTT turns join values of one column into the format of the other with a
language model prompted by a few example pairs, then joins the generated
values against the target column. Here:

1. Few-shot examples come from the confident pairs found so far (no gold
   labels). When there are none, as in code <-> name columns where nothing
   equi-joins, the prompt shows a sample of target values instead so the
   model can see the target format.
2. Source values are sent ``batch_size`` at a time; the model answers with a
   JSON object mapping each input number to its transformed value.
3. Each output is joined to the target column on the normalized value, and
   otherwise to its nearest target value by char 3-gram TF-IDF cosine if
   that is at least ``fuzzy_threshold``.

``direction=right_to_left`` (the default) rewrites right values into the left
column's format; in the Auto-FuzzyJoin layout the left column is the clean
reference table. ``direction=left_to_right`` does the opposite, and then only
right values still unmatched are joinable.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from semjoin.core import JoinTask, Pair, RunContext
from semjoin.matchers.llm_residual import _q, add_residual
from semjoin.matchers.utils_matching import normalize_first, parse_json_object, pick_examples
from semjoin.matchers.utils_similarity import top_k
from semjoin.normalize import normalize

SYSTEM = (
    "You are a careful data transformation assistant. You rewrite values from one table column "
    "into the format used by another column. Reply with a single JSON object and nothing else."
)


class LLMGenerateMatcher:
    def __init__(
        self,
        batch_size: int = 20,
        n_examples: int = 5,
        n_samples: int = 15,
        fuzzy_threshold: float = 0.8,
        direction: str = "right_to_left",
        seed: int = 0,
        residual: bool = False,
    ):
        if direction not in ("right_to_left", "left_to_right"):
            raise ValueError("direction must be right_to_left or left_to_right")
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.n_samples = n_samples
        self.fuzzy_threshold = fuzzy_threshold
        self.direction = direction
        self.seed = seed
        self.residual = residual

    def prompt(self, inputs: Sequence[str], examples: list[tuple[str, str]], samples: Sequence[str]) -> str:
        parts = []
        if samples:
            parts.append("Sample values from the target column:\n" + "\n".join(f"  {_q(v)}" for v in samples))
        if examples:
            parts.append(
                "Examples of source values and their target values:\n"
                + "\n".join(f"  {_q(s)} -> {_q(t)}" for s, t in examples)
            )
        items = "\n".join(f"{i}. {_q(v)}" for i, v in enumerate(inputs, 1))
        parts.append(
            "Rewrite each source value below into the value the target column would use for the same "
            "entity (for example expand a code or abbreviation, fix a typo, or reorder words). If you do "
            "not know, give null.\n\n"
            f"Source values:\n{items}"
        )
        parts.append('Answer as JSON mapping each number to the target value, e.g. {"1": "...", "2": null}.')
        return "\n\n".join(parts)

    def match(self, task: JoinTask, ctx: RunContext) -> set[Pair]:
        pairs, remaining = normalize_first(task, ctx)
        if remaining:
            with ctx.stage("llm_gen"):
                pairs |= self.find_pairs(list(task.left), remaining, ctx, pairs)
        if self.residual:
            pairs = add_residual(task, ctx, pairs)
        ctx.annotate(strategy="llm_gen")
        return pairs

    def find_pairs(self, left: Sequence[str], right: Sequence[str], ctx: RunContext, seeds: set[Pair]) -> set[Pair]:
        r2l = self.direction == "right_to_left"
        src, tgt = (right, left) if r2l else (left, right)
        if not src or not tgt:
            return set()
        examples = [((r, l) if r2l else (l, r)) for l, r in pick_examples(seeds, self.n_examples)]
        rng = random.Random(self.seed)
        samples = rng.sample(list(tgt), min(self.n_samples, len(tgt))) if len(examples) < self.n_examples else []

        outputs: dict[str, str] = {}
        for start in range(0, len(src), self.batch_size):
            batch = src[start : start + self.batch_size]
            answer = parse_json_object(ctx.llm.complete(self.prompt(batch, examples, samples), system=SYSTEM).text)
            for i, v in enumerate(batch, 1):
                out = answer.get(str(i))
                if isinstance(out, str) and out.strip():
                    outputs[v] = out.strip()

        tgt_by_norm: dict[str, list[str]] = defaultdict(list)
        for v in tgt:
            tgt_by_norm[normalize(v)].append(v)
        pairs: set[Pair] = set()
        unresolved = []
        for s, out in outputs.items():
            hits = tgt_by_norm.get(normalize(out))
            if hits:
                pairs.update((s, t) for t in hits)
            else:
                unresolved.append((s, out))
        if unresolved:
            nearest = top_k(tgt, [out for _, out in unresolved], k=1, analyzers=("char3",))
            for (s, _), cand in zip(unresolved, nearest):
                if cand and cand[0][1] >= self.fuzzy_threshold:
                    pairs.add((s, tgt[cand[0][0]]))
        ctx.annotate(llm_gen_outputs=len(outputs), llm_gen_unresolved=len(unresolved))
        return {(t, s) for s, t in pairs} if r2l else pairs
