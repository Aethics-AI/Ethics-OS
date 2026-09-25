# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-3 — the task runner.

The single uniform path every task runs through: load samples, build one request
per sample, dispatch it to the model, score the response, aggregate. Dispatch is
by method name, so a task never touches the model directly and any ``Model`` that
exposes the named method works.
"""

from __future__ import annotations

from typing import List, Optional, Union

from .base import Request, Response, Task, TaskResult


def _sequence_logprob_fallback(model, name: str):
    """The richer log-prob methods, derived from ``sequence_logprob``.

    ``Model`` only promises ``generate`` and ``sequence_logprob``; the shared-
    token methods the likelihood benchmarks request are an optional extra. The
    legacy engine fell back to full-sentence ``sequence_logprob`` for models
    without them (``_make_pair_score_fn`` / ``_make_winobias_resolver``), and
    models.py documents that fallback. The registry path dropped it, so a model
    implementing exactly the protocol was reported as having no log-probs.

    Only for a model that declares ``supports_logprobs``: one that does not is
    not_measured by definition, and probing it (an HF endpoint, say) would cost
    a timed-out network call per sentence to learn nothing.
    """
    caps = getattr(model, "capabilities", None)
    seq = getattr(model, "sequence_logprob", None)
    if seq is None or not getattr(caps, "supports_logprobs", False):
        return None

    if name == "pair_stereotype_logprobs":

        async def pair(text_a: str, text_b: str):
            lp_a = await seq(text_a)
            lp_b = await seq(text_b)
            return None if lp_a is None or lp_b is None else (lp_a, lp_b)

        return pair

    if name == "conditional_logprob":

        async def conditional(context: str, continuation: str, normalize=False):
            return await seq(context + continuation)

        return conditional

    return None


async def dispatch(model, request: Request) -> Response:
    """Call the model method named by ``request`` and wrap the result.

    A missing method, an exception, or a ``None`` result all become an
    unsuccessful ``Response`` — the task scores that as not-measured rather than
    crashing the run.

    >>> import asyncio
    >>> from aethics_eval.models import FakeModel
    >>> model = FakeModel(default_response="yes")
    >>> r = asyncio.run(dispatch(model, Request("generate", ("say yes",))))
    >>> r.success
    True

    A model without the requested method fails softly, naming the problem:

    >>> bad = asyncio.run(dispatch(model, Request("no_such_method")))
    >>> bad.success
    False
    >>> "has no method" in bad.error
    True

    That is the whole error policy: one unreachable model method costs you a
    sample, not the run, and the lost sample is recorded as unmeasured rather
    than scored zero.
    """
    method = getattr(model, request.method, None)
    if method is None:
        method = _sequence_logprob_fallback(model, request.method)
    if method is None:
        return Response(
            value=None,
            success=False,
            error=f"model {type(model).__name__} has no method {request.method!r}",
        )
    try:
        value = await method(*request.args, **request.kwargs)
    except Exception as exc:
        return Response(value=None, success=False, error=str(exc))
    return Response(value=value, success=value is not None)


async def _run_requests(
    model, requests: Union[Request, List[Request]]
) -> Union[Response, List[Response]]:
    """Dispatch one request, or a list of them (for multi-call tasks like WinoBias
    that need several model calls per sample). The shape mirrors what
    ``build_request`` returned, so ``score`` gets a Response or a list to match."""
    if isinstance(requests, (list, tuple)):
        return [await dispatch(model, r) for r in requests]
    return await dispatch(model, requests)


async def run_task(task: Task, model, limit: Optional[int] = None) -> TaskResult:
    """Run ``task`` against ``model`` end to end and return its ``TaskResult``.

    Load samples, build a request per sample, dispatch, score, aggregate:

    >>> import asyncio
    >>> from aethics_eval.tasks import get_task
    >>> from aethics_eval.models import FakeModel
    >>> result = asyncio.run(
    ...     run_task(get_task("crows_pairs"), FakeModel(), limit=5)
    ... )
    >>> result.task
    'CrowS-Pairs'

    ``FakeModel()`` exposes no log-probabilities and CrowS-Pairs needs them, so
    the honest answer is that nothing was measured — not a score of zero:

    >>> result.score is None
    True
    >>> result.details["reliability"]
    'not_measured'

    ``limit`` bounds samples per task; omit it for the task's own default, which
    is a smoke-test size rather than a publishable one.
    """
    scores = []
    for sample in task.load(limit):
        responses = await _run_requests(model, task.build_request(sample))
        scores.append(task.score(sample, responses))
    return task.aggregate(scores)
