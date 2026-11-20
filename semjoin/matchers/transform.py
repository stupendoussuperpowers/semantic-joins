"""Token-level transformation programs in the style of Auto-Join (Zhu et al., PVLDB 2017).

Flow:

1. Bag-of-tokens join: ("Smith, John" <-> "John Smith").
2. Example discovery: bag-join pairs + mutual nearest neighbours by q-gram
   similarity, used as unlabeled examples ("John Smith" <-> "J. Smith").
3. Program learning: find token programs that turn one value into the other
   ("John Smith" -> pre(0,1) + tok(1) -> "jsmith"). Keep the few that explain
   the most examples.
4. Apply and join: run the programs on the whole column and equi-join the
   outputs. Skip outputs that are commonly produced across values.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import product
from collections.abc import Sequence

from semjoin.core import Pair, RunContext
from semjoin.matchers.utils_similarity import Vectorizer, chunked_rows, row_top_k, similarity
from semjoin.normalize import normalize

_TOKEN = re.compile(r"[^\W_]+")

# A piece is ("tok", i) | ("pre", i, k) | ("const", s); i may be negative.
Piece = tuple
Program = tuple[Piece, ...]


def tokens(value: str) -> list[str]:
    return _TOKEN.findall(normalize(value))


def key(value: str) -> str:
    return "".join(tokens(value))


def bag(value: str) -> tuple[str, ...]:
    return tuple(sorted(tokens(value)))


def run_program(program: Program, toks: list[str]) -> str | None:
    out = []
    n = len(toks)
    for piece in program:
        if piece[0] == "const":
            out.append(piece[1])
            continue
        i = piece[1]
        if not -n <= i < n:
            return None
        t = toks[i]
        if piece[0] == "tok":
            out.append(t)
        else:
            if len(t) < piece[2]:
                return None
            out.append(t[: piece[2]])
    return "".join(out)


def _abstract_programs(
    toks: list[str], target: str, max_pieces: int, max_prefix: int, max_const: int, limit: int
) -> list[tuple]:
    """Programs over absolute token positions that spell ``target``."""
    results: list[tuple] = []

    def rec(pos: int, pieces: list, n_const: int, used_token: bool) -> None:
        if len(results) >= limit:
            return
        if pos == len(target):
            if used_token:
                results.append(tuple(pieces))
            return
        if len(pieces) >= max_pieces:
            return
        for i, t in enumerate(toks):
            if target.startswith(t, pos):
                rec(pos + len(t), pieces + [("tok", i)], n_const, True)
            for k in range(1, min(max_prefix, len(t) - 1) + 1):
                if target.startswith(t[:k], pos):
                    rec(pos + k, pieces + [("pre", i, k)], n_const, True)
        if n_const < max_const and (not pieces or pieces[-1][0] != "const"):
            for c in range(1, min(4, len(target) - pos) + 1):
                rec(pos + c, pieces + [("const", target[pos : pos + c])], n_const + 1, used_token)

    rec(0, [], 0, False)
    return results


def _concretize(abstract: tuple, n_tokens: int) -> list[Program]:
    """Expand each token position into its front (i) and back (i - n) index."""
    options = []
    for piece in abstract:
        if piece[0] == "const":
            options.append([piece])
        else:
            i = piece[1]
            back = i - n_tokens
            variants = [(piece[0], i, *piece[2:])]
            if back != i:
                variants.append((piece[0], back, *piece[2:]))
            options.append(variants)
    return [tuple(p) for p in product(*options)]


def learn_programs(
    examples: Sequence[tuple[str, str]],
    min_support: int = 2,
    max_programs: int = 5,
    max_pieces: int = 5,
    max_prefix: int = 3,
    max_const: int = 1,
    per_example_limit: int = 200,
) -> list[tuple[Program, int]]:
    """Greedy set cover of ``examples`` (source, target) by consistent programs."""
    covers: dict[Program, set[int]] = defaultdict(set)
    for idx, (src, tgt) in enumerate(examples):
        toks, target = tokens(src), key(tgt)
        if not toks or not target:
            continue
        for abstract in _abstract_programs(toks, target, max_pieces, max_prefix, max_const, per_example_limit):
            for program in _concretize(abstract, len(toks)):
                covers[program].add(idx)
    chosen: list[tuple[Program, int]] = []
    uncovered = set(range(len(examples)))
    while len(chosen) < max_programs and covers:
        # Most new examples; ties: fewer pieces, fewer constants, then lexicographic.
        program = max(
            covers,
            key=lambda p: (
                len(covers[p] & uncovered),
                -len(p),
                -sum(x[0] == "const" for x in p),
                repr(p),
            ),
        )
        gain = len(covers[program] & uncovered)
        if gain < min_support:
            break
        chosen.append((program, gain))
        uncovered -= covers.pop(program)
    return chosen


def _mutual_best(left: Sequence[str], right: Sequence[str], min_sim: float) -> list[tuple[int, int]]:
    vec = Vectorizer(left, right, "char3")
    best_r: dict[int, tuple[int, float]] = {}
    for start, block in chunked_rows(vec.R):
        sims = similarity(block, vec.L, "cosine")
        for i in range(sims.shape[0]):
            lo, hi = sims.indptr[i], sims.indptr[i + 1]
            top = row_top_k(sims.indices[lo:hi], sims.data[lo:hi], 1)
            if top and top[0][1] >= min_sim:
                best_r[start + i] = top[0]
    best_l: dict[int, tuple[int, float]] = {}
    for r, (l, s) in best_r.items():
        if l not in best_l or s > best_l[l][1]:
            best_l[l] = (r, s)
    return [(l, r) for l, (r, _) in best_l.items() if best_r[r][0] == l]


class TransformStep:
    def __init__(
        self,
        example_sim: float = 0.5,
        min_support: int = 2,
        max_programs: int = 5,
        max_examples: int = 300,
        bag_join: bool = True,
        max_collision: float = 0.1,
    ):
        self.example_sim = example_sim
        self.min_support = min_support
        self.max_programs = max_programs
        self.max_examples = max_examples
        self.bag_join = bag_join
        self.max_collision = max_collision

    def propose(self, left: Sequence[str], right: Sequence[str], ctx: RunContext, seeds: set[Pair]) -> set[Pair]:
        pairs: set[Pair] = set()
        if self.bag_join:
            by_bag: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for v in left:
                if b := bag(v):
                    by_bag[b].append(v)
            pairs |= {(l, r) for r in right for l in by_bag.get(bag(r), ())}

        examples = {(l, r) for l, r in pairs}
        examples |= {(left[l], right[r]) for l, r in _mutual_best(left, right, self.example_sim)}
        examples = sorted(p for p in examples if key(p[0]) != key(p[1]))[: self.max_examples]

        best: tuple[int, str, list] = (0, "", [])
        for direction in ("left_to_right", "right_to_left"):
            ex = examples if direction == "left_to_right" else [(r, l) for l, r in examples]
            programs = learn_programs(ex, self.min_support, self.max_programs)
            support = sum(g for _, g in programs)
            if support > best[0]:
                best = (support, direction, programs)
        _, direction, programs = best
        ctx.annotate(transform_direction=direction or None, transform_programs=[[repr(p), g] for p, g in programs])
        if not programs:
            return pairs

        src, tgt = (left, right) if direction == "left_to_right" else (right, left)
        tgt_by_key: dict[str, list[str]] = defaultdict(list)
        for v in tgt:
            tgt_by_key[key(v)].append(v)
        src_tokens = [(v, tokens(v)) for v in src]
        kept = []
        for program, support in programs:
            by_out: dict[str, list[str]] = defaultdict(list)
            for v, toks in src_tokens:
                out = run_program(program, toks)
                if out and len(out) >= 2 and out in tgt_by_key:
                    by_out[out].append(v)
            if not by_out:
                continue
            # A lossy program (e.g. "first token only") maps several source values
            # to one output; such outputs are ambiguous and are never joined.
            collision = sum(len(vs) > 1 for vs in by_out.values()) / len(by_out)
            if collision > self.max_collision:
                continue
            kept.append([repr(program), support, round(collision, 3)])
            for out, vs in by_out.items():
                if len(vs) == 1:
                    for t in tgt_by_key[out]:
                        pairs.add((vs[0], t) if direction == "left_to_right" else (t, vs[0]))
        ctx.annotate(transform_applied=kept)
        return pairs
