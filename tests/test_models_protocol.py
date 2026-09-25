# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-1 — the Model provider protocol and its implementations.

These are fast and fully offline: they exercise FakeModel and check every provider
class structurally satisfies the Model protocol, so the whole suite can run with no
network and no API key. The real local-model behaviour (generation + exact
log-probs) is covered by a gated tiny-model test.
"""

import asyncio
import os

import pytest

from aethics_eval.models import (
    Capabilities,
    FakeModel,
    Generation,
    HFInferenceModel,
    LocalHFModel,
    Model,
    OpenAICompatModel,
)


def _run(coro):
    return asyncio.run(coro)


# ── FakeModel drives the whole interface offline ────────────────────


def test_fakemodel_generate_and_logprob():
    m = FakeModel(
        responses={"hello": "world"},
        logprobs={"a probable sentence": -1.0},
        default_response="(none)",
        default_logprob=-9.0,
    )
    g = _run(m.generate("hello"))
    assert isinstance(g, Generation) and g.text == "world" and g.success
    assert _run(m.generate("unknown")).text == "(none)"
    assert _run(m.sequence_logprob("a probable sentence")) == -1.0
    assert _run(m.sequence_logprob("unseen")) == -9.0
    assert m.capabilities.supports_logprobs is True


def test_fakemodel_callable_responses():
    m = FakeModel(responses=lambda p: p.upper(), logprobs=lambda t: -len(t))
    assert _run(m.generate("abc")).text == "ABC"
    assert _run(m.sequence_logprob("abcd")) == -4


def test_fakemodel_no_logprobs_is_honest():
    m = FakeModel(responses={"x": "y"})
    assert m.capabilities.supports_logprobs is False
    assert _run(m.sequence_logprob("x")) is None


def test_fakemodel_pair_logprobs_for_shared_token_path():
    m = FakeModel(pair_logprobs={("more", "less"): (-2.0, -5.0)})
    assert _run(m.pair_stereotype_logprobs("more", "less")) == (-2.0, -5.0)


# ── Every implementation satisfies the protocol ─────────────────────


def test_all_providers_satisfy_protocol():
    # runtime_checkable Protocol => isinstance checks the required methods exist.
    fake = FakeModel()
    hf = HFInferenceModel("openai-community/gpt2")
    oai = OpenAICompatModel(client=object())
    for m in (fake, hf, oai):
        assert isinstance(m, Model), f"{type(m).__name__} does not satisfy Model"
        assert isinstance(m.capabilities, Capabilities)


def test_capability_defaults_are_honest():
    # HF serverless: no logprobs by default; local: exact logprobs.
    assert HFInferenceModel("m").capabilities.supports_logprobs is False
    assert OpenAICompatModel(client=object()).capabilities.supports_logprobs is False


# ── HFInferenceModel is self-contained: no core.hf_client import ─────


def test_hf_model_is_import_clean():
    # The SDK boundary (POS-1 contract): the provider must not reach back into
    # the product's client. It builds its own huggingface_hub client instead.
    import inspect

    import aethics_eval.models as models_mod

    src = inspect.getsource(models_mod)
    assert "hf_client" not in src, "models.py must not import the product's hf_client"


def test_hf_model_uses_injected_raw_client():
    # A raw huggingface_hub-style client can be injected, so both paths are
    # testable with no network: generate() calls chat_completion (the hosted
    # path), sequence_logprob() reads prefill logprobs off text_generation.
    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _ChatResult:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Tok:
        def __init__(self, lp):
            self.logprob = lp

    class _Details:
        def __init__(self):
            self.prefill = [_Tok(-1.0), _Tok(-2.0)]

    class _Result(str):
        details = _Details()

    class StubClient:
        def chat_completion(self, messages, model=None, **kwargs):
            return _ChatResult(f"gen:{messages[0]['content']}")

        def text_generation(self, prompt, model=None, details=False, **kwargs):
            return _Result("") if details else f"gen:{prompt}"

    m = HFInferenceModel("m", client=StubClient())
    assert _run(m.generate("hi")).text == "gen:hi"
    assert _run(m.generate("hi")).method == "chat_completion"
    # sum of prefill logprobs (-1.0 + -2.0) with normalize False
    assert _run(m.sequence_logprob("hi")) == -3.0


def test_hf_model_survives_stopiteration_from_client():
    # huggingface_hub raises a bare StopIteration on auth/provider-resolution
    # failure. Run inside an executor thread, that corrupts the asyncio future
    # boundary instead of propagating — so without the guard, the call hangs
    # instead of returning the honest failure. Verify both paths degrade cleanly:
    # generate → Generation(success=False), sequence_logprob → None.
    class StopClient:
        def chat_completion(self, *args, **kwargs):
            raise StopIteration

        def text_generation(self, *args, **kwargs):
            raise StopIteration

    m = HFInferenceModel("m", client=StopClient())
    g = _run(m.generate("hi"))
    assert g.success is False
    assert g.text == ""
    assert g.error  # a real error message, not a hang
    assert _run(m.sequence_logprob("hi")) is None


# ── Retry/backoff on transient throttling (429 / 503) ───────────────


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _ChatResult:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _FlakyClient:
    """Fails the first `fail_times` calls with `exc`, then succeeds."""

    def __init__(self, fail_times, exc):
        self.calls = 0
        self.fail_times = fail_times
        self.exc = exc

    def chat_completion(self, messages, model=None, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return _ChatResult(f"gen:{messages[0]['content']}")


def test_generate_retries_transient_throttle_then_succeeds():
    c = _FlakyClient(fail_times=2, exc=Exception("429 Client Error: Too Many Requests"))
    m = HFInferenceModel("m", client=c, max_retries=3, retry_backoff=0)  # no real sleep
    g = _run(m.generate("hi"))
    assert g.success is True and g.text == "gen:hi"
    assert c.calls == 3  # 2 throttles + 1 success


def test_generate_gives_up_cleanly_after_max_retries():
    c = _FlakyClient(fail_times=99, exc=Exception("503 Service Unavailable"))
    m = HFInferenceModel("m", client=c, max_retries=2, retry_backoff=0)
    g = _run(m.generate("hi"))
    assert g.success is False and g.error  # honest failure, no hang
    assert c.calls == 3  # initial + 2 retries


def test_generate_does_not_retry_permanent_errors():
    # 404 (model not found) is permanent — retrying wastes time and quota.
    c = _FlakyClient(fail_times=99, exc=Exception("404 Client Error: Not Found"))
    m = HFInferenceModel("m", client=c, max_retries=3, retry_backoff=0)
    g = _run(m.generate("hi"))
    assert g.success is False
    assert c.calls == 1  # no retry


# ── Gated: real local model (tiny) generates + scores ───────────────


@pytest.mark.skipif(
    not os.environ.get("VOS1_LOCAL_MODEL"),
    reason="loads a tiny local model; set VOS1_LOCAL_MODEL=1 to run",
)
def test_local_hf_model_generates_and_scores():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from aethics_eval.logprob_scorer import LogprobScorer

    m = LocalHFModel(
        scorer=LogprobScorer(model_name="sshleifer/tiny-gpt2", revision=None)
    )
    assert m.capabilities.supports_logprobs is True
    g = _run(m.generate("The weather is", max_new_tokens=3))
    assert isinstance(g.text, str)
    lp = _run(m.sequence_logprob("The doctor finished the shift."))
    assert isinstance(lp, float) and lp < 0
