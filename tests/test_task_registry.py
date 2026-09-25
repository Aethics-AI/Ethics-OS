# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-3 Part A — the Task abstraction, registry, and runner.

Offline: a scriptable dummy task + stub model, so the registry/runner contract is
covered without any network or dataset.
"""

import asyncio

import pytest

from aethics_eval.tasks import (
    Request,
    Sample,
    SampleScore,
    Task,
    TaskResult,
    dispatch,
    get_task,
    list_tasks,
    register_task,
    run_task,
    task_names,
)


def _run(coro):
    return asyncio.run(coro)


# ── A dummy task registered once for the whole module ───────────────


@register_task(
    "dummy_mean",
    metric="mean_logprob",
    requires_logprobs=True,
    citation="none",
    licence="MIT",
    description="averages a per-sample logprob; test-only",
)
class DummyTask(Task):
    def load(self, limit=None):
        n = 3 if limit is None else min(limit, 3)
        return [Sample(id=str(i), data={"text": f"t{i}"}, target=i) for i in range(n)]

    def build_request(self, sample):
        return Request(method="sequence_logprob", args=(sample.data["text"],))

    def score(self, sample, response):
        return SampleScore(
            sample_id=sample.id,
            value=float(response.value) if response.success else 0.0,
            measured=response.success,
        )

    def aggregate(self, scores):
        measured = [s.value for s in scores if s.measured]
        score = sum(measured) / len(measured) if measured else None
        return TaskResult(
            task="dummy_mean",
            score=score,
            samples_tested=len(measured),
            passed=bool(score is not None and score >= 0.0),
            threshold=0.0,
            details={"n": len(scores)},
        )


class StubModel:
    """Returns a fixed logprob per text; unknown texts return None (unmeasured)."""

    def __init__(self, table):
        self._table = table

    async def sequence_logprob(self, text):
        return self._table.get(text)


# ── Registry ────────────────────────────────────────────────────────


def test_task_is_registered_and_listed():
    assert "dummy_mean" in task_names()
    metas = {m.name: m for m in list_tasks()}
    assert "dummy_mean" in metas
    m = metas["dummy_mean"]
    assert m.metric == "mean_logprob"
    assert m.requires_logprobs is True
    assert m.licence == "MIT"


def test_get_task_instantiates():
    t = get_task("dummy_mean")
    assert isinstance(t, Task)
    assert t.meta.name == "dummy_mean"


def test_get_unknown_task_raises_with_available_list():
    with pytest.raises(KeyError) as exc:
        get_task("does_not_exist")
    assert "dummy_mean" in str(exc.value)  # error lists what IS available


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):

        @register_task("dummy_mean", metric="x", requires_logprobs=False)
        class _Dup(Task):
            def load(self, limit=None):
                return []

            def build_request(self, sample):
                return Request(method="sequence_logprob")

            def score(self, sample, response):
                return SampleScore(sample.id, 0.0)

            def aggregate(self, scores):
                return TaskResult("dup", None, 0, False)


def test_register_non_task_raises():
    with pytest.raises(TypeError):

        @register_task("not_a_task", metric="x", requires_logprobs=False)
        class _NotATask:  # doesn't subclass Task
            pass


# ── Runner + dispatch ───────────────────────────────────────────────


def test_run_task_end_to_end():
    model = StubModel({"t0": -1.0, "t1": -2.0, "t2": -3.0})
    result = _run(run_task(get_task("dummy_mean"), model))
    assert isinstance(result, TaskResult)
    assert result.samples_tested == 3
    assert result.score == pytest.approx((-1.0 - 2.0 - 3.0) / 3)


def test_run_task_respects_limit():
    model = StubModel({"t0": -1.0, "t1": -2.0, "t2": -3.0})
    result = _run(run_task(get_task("dummy_mean"), model, limit=2))
    assert result.samples_tested == 2


def test_dispatch_missing_response_is_not_measured():
    # t2 not in the table → sequence_logprob returns None → unsuccessful → excluded.
    model = StubModel({"t0": -1.0, "t1": -2.0})  # no t2
    result = _run(run_task(get_task("dummy_mean"), model))
    assert result.samples_tested == 2  # only the two measured
    assert result.details["n"] == 3  # but all three were attempted


def test_dispatch_missing_method_fails_cleanly():
    class NoLogprob:
        pass  # has no sequence_logprob

    resp = _run(dispatch(NoLogprob(), Request(method="sequence_logprob", args=("x",))))
    assert resp.success is False and resp.value is None and resp.error


def test_dispatch_exception_fails_cleanly():
    class Boom:
        async def sequence_logprob(self, text):
            raise RuntimeError("down")

    resp = _run(dispatch(Boom(), Request(method="sequence_logprob", args=("x",))))
    assert resp.success is False and "down" in resp.error


def test_task_result_to_dict_matches_legacy_shape():
    # Byte-identical parity depends on the same keys as the old BenchmarkResult.
    r = TaskResult(
        task="WinoBias", score=0.875, samples_tested=50, passed=True, threshold=0.7
    )
    d = r.to_dict()
    assert set(d) == {
        "benchmark_name",
        "score",
        "details",
        "samples_tested",
        "passed",
        "threshold",
    }
    assert d["benchmark_name"] == "WinoBias" and d["score"] == 0.875
