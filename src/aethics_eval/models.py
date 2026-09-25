# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Model provider protocol for the evaluation engine (VOS-1).

An open-source evaluation library cannot hardcode one inference vendor, and a
module-level client singleton is untestable. Every provider satisfies one small
``Model`` protocol; the engine and benchmarks take an injected ``Model`` instead of
importing a global client.

The protocol is deliberately minimal — ``generate``, ``sequence_logprob``, and a
``capabilities`` descriptor. Capability negotiation is honest: a model that cannot
return log-probabilities advertises ``supports_logprobs=False``, and the benchmarks
mark the result ``not_measured`` rather than substituting a proxy/length score.

Implementations:
- ``HFInferenceModel``   — HuggingFace Inference API (existing behaviour).
- ``OpenAICompatModel``  — any OpenAI-shaped endpoint (vLLM, Together, ...).
- ``LocalHFModel``       — open-weight model loaded locally; exact log-probs.
- ``FakeModel``          — scriptable, no network, so the suite runs offline.

``LocalHFModel`` additionally exposes the richer shared-token log-prob methods
(``conditional_logprob`` / ``pair_stereotype_logprobs``) that the bias benchmarks
use for exact scoring; models without them fall back to ``sequence_logprob``.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# Transient HTTP statuses worth retrying (rate-limit / upstream unavailable).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRYABLE_TEXT = (
    "429",
    "too many requests",
    "rate limit",
    "500",
    "502",
    "503",
    "504",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "timed out",
)


def _is_retryable(exc: Exception) -> bool:
    """True for transient failures (throttling / upstream hiccups), not for
    permanent ones like 404 (model not found) or 401 (auth)."""
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if code in _RETRYABLE_STATUS:
        return True
    s = str(exc).lower()
    return any(k in s for k in _RETRYABLE_TEXT)


def _call_with_retry(fn: Callable, max_retries: int, backoff: float):
    """Run sync ``fn``, retrying transient errors with exponential backoff +
    jitter. Non-retryable errors and the final attempt propagate. Runs inside an
    executor thread, so ``time.sleep`` blocks only that worker, not the loop."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            time.sleep(min(backoff * (2**attempt), 8.0) + random.uniform(0, 0.25))
            attempt += 1


@dataclass
class Generation:
    """The result of a single generation request."""

    text: str
    success: bool = True
    error: Optional[str] = None
    method: Optional[str] = None


@dataclass(frozen=True)
class Capabilities:
    """What a provider can do, so callers negotiate instead of assuming."""

    supports_logprobs: bool = False
    supports_echo: bool = False
    max_context: int = 2048


@runtime_checkable
class Model(Protocol):
    """The interface every inference provider satisfies."""

    async def generate(self, prompt: str, **params: Any) -> Generation: ...

    async def sequence_logprob(self, text: str) -> Optional[float]: ...

    @property
    def capabilities(self) -> Capabilities: ...


# ── HuggingFace Inference API ───────────────────────────────────────


class HFInferenceModel:
    """A self-contained HuggingFace Inference API provider.

    Talks to the API directly via ``huggingface_hub`` — no dependency on the
    product's client — so the package stays import-clean for the SDK boundary.
    Generation uses the chat/messages API (``chat_completion``), which is what HF
    Inference Providers serve today; the raw ``text_generation`` task is retired
    for most hosted models. HF serverless does not return token log-probs, so
    ``supports_logprobs`` defaults to False; set it True for a TGI/dedicated
    endpoint that returns prefill logprobs. A raw ``huggingface_hub``-style client
    (anything exposing ``text_generation``) may be injected for offline tests.
    """

    def __init__(
        self,
        model_name: str,
        token: Optional[str] = None,
        timeout: int = 20,
        supports_logprobs: bool = False,
        max_context: int = 2048,
        normalize: bool = False,
        client: Any = None,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        self.model_name = model_name
        self._token = token
        self._timeout = timeout
        self._normalize = normalize
        self._client = client
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._caps = Capabilities(
            supports_logprobs=supports_logprobs,
            supports_echo=False,
            max_context=max_context,
        )

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    def _make_client(self, timeout: Optional[int] = None):
        if self._client is not None:
            return self._client
        from huggingface_hub import InferenceClient

        return InferenceClient(token=self._token, timeout=timeout)

    async def generate(self, prompt: str, **params: Any) -> Generation:
        from .eval_config import EVAL_INFERENCE_PARAMS as P

        loop = asyncio.get_event_loop()
        client = self._make_client()

        def _call():
            try:
                # HF Inference Providers serve hosted models through the chat/
                # messages API; the raw text_generation task is retired for most
                # of them ("not supported for task text-generation"). Wrap the
                # prompt as a single user turn.
                out = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    max_tokens=params.get("max_new_tokens", P.max_new_tokens),
                    temperature=0,  # deterministic for reproducible evals
                )
            except StopIteration as e:
                # huggingface_hub raises a bare StopIteration on auth/provider
                # resolution failure. Raised inside an executor thread it
                # corrupts the asyncio future boundary rather than propagating,
                # so the honest failure path below never runs — convert it.
                raise RuntimeError(
                    "HF chat_completion failed (auth or provider resolution)"
                ) from e
            choices = getattr(out, "choices", None) or []
            if not choices:
                return ""
            return getattr(choices[0].message, "content", None) or ""

        # Retry transient throttling (429/503) with backoff so a burst of eval
        # calls doesn't silently collapse to not_measured under rate limits.
        def _gen():
            return _call_with_retry(_call, self._max_retries, self._retry_backoff)

        try:
            text = await loop.run_in_executor(None, _gen)
            text = text.strip() if text else ""
            return Generation(text=text, success=bool(text), method="chat_completion")
        except Exception as e:
            return Generation(text="", success=False, error=str(e))

    async def sequence_logprob(self, text: str) -> Optional[float]:
        # Providers that don't support decoder_input_details tend to hang rather
        # than error, so time-bound the call and fall back to None (not_measured).
        loop = asyncio.get_event_loop()
        client = self._make_client(timeout=self._timeout)

        def _logprob():
            try:
                out = client.text_generation(
                    text,
                    model=self.model_name,
                    max_new_tokens=1,
                    details=True,
                    decoder_input_details=True,
                    do_sample=False,
                    return_full_text=False,
                )
            except StopIteration as e:
                # See generate(): convert so it can cross the executor boundary
                # and be handled as a None (not_measured) result.
                raise RuntimeError(
                    "HF sequence_logprob failed (auth or provider resolution)"
                ) from e
            details = getattr(out, "details", None)
            prefill = getattr(details, "prefill", None) if details else None
            if not prefill:
                return None
            lps = [
                t.logprob for t in prefill if getattr(t, "logprob", None) is not None
            ]
            if not lps:
                return None
            return sum(lps) / len(lps) if self._normalize else sum(lps)

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _logprob), timeout=self._timeout
            )
        except Exception:
            return None


# ── OpenAI-compatible endpoints (vLLM / Together / ...) ─────────────


class OpenAICompatModel:
    """Wraps an OpenAI-shaped endpoint via ``ExternalModelClient``.

    Log-probs are only available on the completions endpoint with echo+logprobs;
    the generic client doesn't request them, so ``supports_logprobs`` defaults to
    False and ``sequence_logprob`` returns None (→ not_measured).
    """

    def __init__(
        self,
        client: Any,
        supports_logprobs: bool = False,
        max_context: int = 4096,
    ):
        self._client = client
        self._caps = Capabilities(
            supports_logprobs=supports_logprobs,
            supports_echo=supports_logprobs,
            max_context=max_context,
        )

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    async def generate(self, prompt: str, **params: Any) -> Generation:
        loop = asyncio.get_event_loop()
        r = await loop.run_in_executor(
            None, lambda: self._client.generate(prompt, **params)
        )
        if r.get("success"):
            return Generation(
                text=str(r.get("text") or ""), success=True, method="openai_compat"
            )
        return Generation(text="", success=False, error=r.get("error"))

    async def sequence_logprob(self, text: str) -> Optional[float]:
        # Honest not_measured until an echo+logprobs completions path is wired.
        return None


# ── Local open-weight model (exact log-probs) ───────────────────────


class LocalHFModel:
    """Loads an open-weight model locally for generation and *exact* log-probs.

    Wraps ``LogprobScorer`` (from SC-1), so it also exposes ``conditional_logprob``
    and ``pair_stereotype_logprobs`` — the shared-token pseudo-log-likelihood the
    bias benchmarks use to reproduce published numbers.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        revision: Optional[str] = None,
        scorer: Any = None,
        max_new_tokens: int = 100,
        max_context: int = 1024,
        normalize: bool = False,
    ):
        if scorer is None:
            from .logprob_scorer import LogprobScorer

            if model_name is None:
                scorer = LogprobScorer()
            else:
                scorer = LogprobScorer(model_name=model_name, revision=revision)
        self._scorer = scorer
        self._max_new_tokens = max_new_tokens
        self._normalize = normalize
        self._caps = Capabilities(
            supports_logprobs=True, supports_echo=True, max_context=max_context
        )

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    async def generate(self, prompt: str, **params: Any) -> Generation:
        loop = asyncio.get_event_loop()
        max_new = params.get("max_new_tokens", self._max_new_tokens)
        text = await loop.run_in_executor(
            None, lambda: self._scorer.generate(prompt, max_new_tokens=max_new)
        )
        return Generation(text=text, success=True, method="local_hf")

    async def sequence_logprob(self, text: str) -> Optional[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._scorer.sequence_logprob(text, self._normalize)
        )

    async def conditional_logprob(
        self, context: str, continuation: str, normalize: bool = False
    ) -> Optional[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._scorer.conditional_logprob(context, continuation, normalize),
        )

    async def pair_stereotype_logprobs(self, text_a: str, text_b: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._scorer.pair_stereotype_logprobs(text_a, text_b)
        )


# ── Fake model for offline tests ────────────────────────────────────


class FakeModel:
    """Scriptable model so the whole suite runs with no network and no API key.

    ``responses`` and ``logprobs`` may be a dict keyed by text, or a callable.
    ``pair_logprobs`` (dict keyed by (a, b) or callable) lets tests drive the
    shared-token path deterministically.
    """

    def __init__(
        self,
        responses: Any = None,
        logprobs: Any = None,
        pair_logprobs: Any = None,
        default_response: str = "",
        default_logprob: Optional[float] = None,
        capabilities: Optional[Capabilities] = None,
    ):
        self._responses = responses if responses is not None else {}
        self._logprobs = logprobs
        self._pair_logprobs = pair_logprobs
        self._default_response = default_response
        self._default_logprob = default_logprob
        self._caps = capabilities or Capabilities(
            supports_logprobs=(logprobs is not None or pair_logprobs is not None)
        )

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    async def generate(self, prompt: str, **params: Any) -> Generation:
        r = self._responses
        text = r(prompt) if callable(r) else r.get(prompt, self._default_response)
        return Generation(text=str(text), success=True, method="fake")

    async def sequence_logprob(self, text: str) -> Optional[float]:
        lp = self._logprobs
        if lp is None:
            return self._default_logprob
        return lp(text) if callable(lp) else lp.get(text, self._default_logprob)

    async def conditional_logprob(
        self, context: str, continuation: str, normalize: bool = False
    ) -> Optional[float]:
        return await self.sequence_logprob(context + continuation)

    async def pair_stereotype_logprobs(self, text_a: str, text_b: str):
        pl = self._pair_logprobs
        if pl is None:
            la = await self.sequence_logprob(text_a)
            lb = await self.sequence_logprob(text_b)
            return None if la is None or lb is None else (la, lb)
        return pl((text_a, text_b)) if callable(pl) else pl.get((text_a, text_b))
