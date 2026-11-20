"""LLM clients. Matchers only see ``CountingLLM`` via ``RunContext.llm``."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from semjoin.core import RunContext


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse: ...


class CountingLLM:
    """Wraps a client and charges every call to the context's current stage."""

    def __init__(self, inner: LLMClient, ctx: RunContext):
        self._inner = inner
        self._ctx = ctx

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        resp = self._inner.complete(prompt, system=system)
        stage = self._ctx.current_stage
        self._ctx.llm_calls[stage] += 1
        self._ctx.prompt_tokens[stage] += resp.prompt_tokens
        self._ctx.completion_tokens[stage] += resp.completion_tokens
        return resp


class FakeLLM:
    """Offline stand-in for tests and dry runs.

    ``responder`` maps a prompt to the reply text; the default replies with an
    empty string. Token counts are whitespace word counts.
    """

    def __init__(self, responder: Callable[[str], str] | None = None):
        self._responder = responder or (lambda prompt: "")
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        text = self._responder(prompt)
        n_prompt = len(prompt.split()) + (len(system.split()) if system else 0)
        return LLMResponse(text, n_prompt, len(text.split()))


class OllamaClient:
    """Calls a local or remote Ollama server's /api/chat endpoint.

    Defaults come from SEMJOIN_OLLAMA_URL, SEMJOIN_OLLAMA_MODEL and
    SEMJOIN_OLLAMA_NUM_CTX. Temperature is 0 so runs are as repeatable as the
    model allows. ``num_ctx`` is set explicitly because Ollama's default
    context window is small and it truncates longer prompts silently; the
    shared-list and candidate prompts of the LLM matchers run to a few
    thousand tokens.
    """

    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        timeout: float = 300.0,
        retries: int = 2,
        num_ctx: int | None = None,
    ):
        self.model = model or os.environ.get("SEMJOIN_OLLAMA_MODEL", "qwen2.5:7b")
        self.url = (url or os.environ.get("SEMJOIN_OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.num_ctx = num_ctx or int(os.environ.get("SEMJOIN_OLLAMA_NUM_CTX", "8192"))

    def complete(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": self.num_ctx},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                break
            except OSError:
                if attempt == self.retries:
                    raise
                time.sleep(2**attempt)
        return LLMResponse(
            text=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
