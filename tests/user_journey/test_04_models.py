# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Model providers: the ``Model`` protocol, ``FakeModel``, bringing your own
model, and the ``provider:model_id`` strings the CLI accepts."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Optional

import httpx
import pytest
import typer

from aethics_eval import cli
from aethics_eval.models import (
    Capabilities,
    FakeModel,
    Generation,
    HFInferenceModel,
    Model,
    OpenAICompatModel,
)


def run(coro):
    return asyncio.run(coro)


# ── FakeModel ───────────────────────────────────────────────────────


def test_fake_model_scripted_responses_and_default() -> None:
    m = FakeModel(responses={"hi": "hello"}, default_response="?")
    assert run(m.generate("hi")).text == "hello"
    assert run(m.generate("other")).text == "?"
    assert run(m.generate("x")).success is True


def test_fake_model_callable_responses() -> None:
    m = FakeModel(responses=lambda p: p.upper())
    assert run(m.generate("shout")).text == "SHOUT"


def test_fake_model_without_logprobs_is_honest_about_it() -> None:
    m = FakeModel()
    assert m.capabilities.supports_logprobs is False
    assert run(m.sequence_logprob("anything")) is None
    assert run(m.pair_stereotype_logprobs("a", "b")) is None


def test_fake_model_with_logprobs() -> None:
    m = FakeModel(logprobs={"a": -1.0, "b": -2.0})
    assert m.capabilities.supports_logprobs is True
    assert run(m.sequence_logprob("a")) == -1.0
    assert run(m.pair_stereotype_logprobs("a", "b")) == (-1.0, -2.0)
    assert run(m.conditional_logprob("", "a")) == -1.0


def test_fake_model_pair_logprobs_override() -> None:
    m = FakeModel(pair_logprobs={("x", "y"): (-0.5, -0.7)})
    assert run(m.pair_stereotype_logprobs("x", "y")) == (-0.5, -0.7)


# ── Bring your own model ────────────────────────────────────────────


class MyModel:
    """What a user writes to plug in their own inference stack."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(supports_logprobs=True)

    async def generate(self, prompt: str, **params: Any) -> Generation:
        self.calls.append(prompt)
        return Generation(text="ok")

    async def sequence_logprob(self, text: str) -> Optional[float]:
        return -float(len(text))


def test_custom_class_satisfies_model_protocol() -> None:
    assert isinstance(MyModel(), Model)
    assert isinstance(FakeModel(), Model)


def test_custom_model_runs_a_generation_task(offline_benchmarks) -> None:
    from aethics_eval.tasks import get_task, run_task

    model = MyModel()
    res = run(run_task(get_task("bold"), model, limit=6))
    assert res.score == 1.0  # constant reply → no cross-domain disparity
    assert len(model.calls) == 6


@pytest.mark.parametrize("task", ["crows_pairs", "stereoset", "winobias"])
def test_protocol_only_model_with_logprobs_is_measured(
    offline_benchmarks, task
) -> None:
    from aethics_eval.tasks import get_task, run_task

    res = run(run_task(get_task(task), MyModel(), limit=4))
    assert res.score is not None


# ── OpenAI-compatible provider, against a mock endpoint ─────────────


def _patch_httpx(monkeypatch, handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    real = httpx.Client

    def client(*a, **kw):
        kw["transport"] = httpx.MockTransport(wrapped)
        return real(*a, **kw)

    monkeypatch.setattr(httpx, "Client", client)
    return seen


def test_openai_provider_sends_deterministic_request(monkeypatch) -> None:
    monkeypatch.setenv("AETHICS_API_KEY", "sk-test-123")
    seen = _patch_httpx(
        monkeypatch,
        lambda r: httpx.Response(
            200, json={"choices": [{"message": {"content": "a reply"}}]}
        ),
    )
    model = cli.build_model("openai:my-model", base_url="http://llm.local/v1/")
    assert isinstance(model, OpenAICompatModel)
    assert model.capabilities.supports_logprobs is False

    g = run(model.generate("hello"))
    assert g.success and g.text == "a reply"

    (req,) = seen
    assert str(req.url) == "http://llm.local/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-test-123"
    import json

    body = json.loads(req.content)
    assert body["model"] == "my-model"
    assert body["temperature"] == 0.0
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_openai_provider_error_never_leaks_the_key(monkeypatch) -> None:
    monkeypatch.setenv("AETHICS_API_KEY", "sk-secret-xyz")
    _patch_httpx(
        monkeypatch,
        lambda r: httpx.Response(401, json={"error": "bad key sk-secret-xyz"}),
    )
    model = cli.build_model("openai:m", base_url="http://llm.local/v1")
    g = run(model.generate("hello"))
    assert g.success is False
    assert "401" in (g.error or "")
    assert "sk-secret-xyz" not in (g.error or "")


def test_openai_provider_falls_back_to_openai_api_key(monkeypatch) -> None:
    monkeypatch.delenv("AETHICS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")
    seen = _patch_httpx(
        monkeypatch,
        lambda r: httpx.Response(
            200, json={"choices": [{"message": {"content": "x"}}]}
        ),
    )
    run(cli.build_model("openai:m", base_url="http://h/v1").generate("p"))
    assert seen[0].headers["Authorization"] == "Bearer sk-fallback"


# ── HF provider with an injected client (no network) ────────────────


class _FakeHFClient:
    def chat_completion(self, messages, model, max_tokens, temperature):
        class _Msg:
            content = f"echo: {messages[0]['content']}"

        class _Choice:
            message = _Msg()

        class _Out:
            choices: ClassVar = [_Choice()]

        return _Out()


def test_hf_provider_with_injected_client() -> None:
    m = HFInferenceModel("some/model", client=_FakeHFClient())
    assert m.capabilities.supports_logprobs is False
    g = run(m.generate("hi"))
    assert g.success and g.text == "echo: hi"


# ── Model spec strings ──────────────────────────────────────────────


def test_build_model_fake_uses_text_after_colon() -> None:
    m = cli.build_model("fake:I refuse.")
    assert run(m.generate("anything")).text == "I refuse."


def test_build_model_hf_returns_hf_provider() -> None:
    assert isinstance(cli.build_model("hf:gpt2"), HFInferenceModel)


@pytest.mark.parametrize(
    "spec", ["no-colon", "nosuch:model", "hf:", "local:", "openai:"]
)
def test_bad_model_specs_are_usage_errors(spec) -> None:
    with pytest.raises(typer.BadParameter):
        cli.build_model(spec)


def test_openai_without_base_url_is_refused() -> None:
    with pytest.raises(typer.BadParameter, match="base-url"):
        cli.build_model("openai:gpt-4o")
