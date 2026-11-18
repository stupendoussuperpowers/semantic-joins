"""Core types shared by datasets, matchers and the runner."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Protocol

if TYPE_CHECKING:
    from semjoin.llm import LLMClient

# A predicted or gold match: (left value, right value). Matching is
# many-to-many, so a value may appear in any number of pairs.
Pair = tuple[str, str]

UNSTAGED = "unstaged"


@dataclass(frozen=True)
class JoinTask:
    """One column pair to join, with its exhaustive set of true matches."""

    dataset: str
    task_id: str
    left: tuple[str, ...]
    right: tuple[str, ...]
    gold: frozenset[Pair]
    # e.g. "code", "typo", "reorder", "acronym"; None when unknown or mixed.
    mismatch_type: str | None = None
    split: str | None = None

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.task_id}"


class Matcher(Protocol):
    """A join strategy. Returns every (left, right) pair it believes matches."""

    def match(self, task: JoinTask, ctx: RunContext) -> set[Pair]: ...


class RunContext:
    """Per-(task, matcher) handle for LLM access and cost accounting.

    Matchers must call the LLM through ``ctx.llm`` so calls and tokens are
    counted, and should wrap pipeline phases in ``with ctx.stage(name):``.
    Stage times are exclusive: time spent in a nested stage is charged to the
    inner stage only.
    """

    def __init__(self, llm: LLMClient):
        from semjoin.llm import CountingLLM

        self.stage_seconds: dict[str, float] = defaultdict(float)
        self.llm_calls: dict[str, int] = defaultdict(int)
        self.prompt_tokens: dict[str, int] = defaultdict(int)
        self.completion_tokens: dict[str, int] = defaultdict(int)
        self.annotations: dict[str, Any] = {}
        self.llm = CountingLLM(llm, self)
        self._current = UNSTAGED
        self._since = time.perf_counter()

    @property
    def current_stage(self) -> str:
        return self._current

    def _charge(self) -> None:
        now = time.perf_counter()
        self.stage_seconds[self._current] += now - self._since
        self._since = now

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self._charge()
        prev, self._current = self._current, name
        try:
            yield
        finally:
            self._charge()
            self._current = prev

    def annotate(self, **values: Any) -> None:
        """Attach JSON-serializable metadata to the result record.

        A router should call ``ctx.annotate(strategy=...)`` so routing
        accuracy can be computed by ``semjoin summarize --router``.
        """
        self.annotations.update(values)

    def finish(self) -> None:
        self._charge()
