# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-3 — the Task abstraction.

A benchmark is a `Task`: it loads samples, turns each into a `Request` (one model
call), scores the `Response`, and aggregates the per-sample `SampleScore`s into a
`TaskResult`. Tasks are model-agnostic — a `Request` only names which `Model`
method to call and with what arguments; the runner dispatches it. This is what
lets a third party add a benchmark without touching engine code, and lets the
whole suite run through one uniform path (the byte-identical parity guard).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union

from aethics_eval.scoring import ScoredMetric


@dataclass
class Sample:
    """One benchmark item. ``data`` holds task-specific fields; ``target`` the gold
    answer, if the task has one.

    ``id`` must be stable across runs: it is hashed into the evidence tree, so a
    sample that changes identity between runs breaks ``aethics verify``. Prefer
    the dataset's own identifier over an enumeration index.

    >>> s = Sample(id="crows-0042", data={"prompt": "Is water wet?"})
    >>> s.id
    'crows-0042'
    >>> s.target is None
    True
    """

    id: str
    data: Dict[str, Any] = field(default_factory=dict)
    target: Any = None


@dataclass
class Request:
    """A single model call a task needs for a sample.

    ``method`` names the ``Model`` method to invoke (e.g. ``sequence_logprob``,
    ``pair_stereotype_logprobs``, ``conditional_logprob``, ``generate``); the
    runner dispatches to it with ``args``/``kwargs``. Keeping the method name
    declarative (rather than the task calling the model itself) is what makes
    tasks model-agnostic and the run path uniform.

    >>> r = Request(method="generate", args=("Is water wet?",))
    >>> r.method
    'generate'
    >>> r.args
    ('Is water wet?',)

    A model lacking the named method is not an error at build time — the runner
    turns it into an unsuccessful ``Response``, which the task scores as
    not-measured.
    """

    method: str
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """The model's answer to a Request. ``success`` is False when the call failed
    or returned nothing usable — the task then scores it as not-measured.

    >>> ok = Response(value="yes")
    >>> ok.success
    True
    >>> failed = Response(success=False, error="connection timed out")
    >>> failed.value is None
    True
    >>> failed.error
    'connection timed out'
    """

    value: Any = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class SampleScore:
    """A task's per-sample numeric contribution. ``measured`` is False when the
    sample could not be scored (e.g. the model call failed).

    >>> SampleScore("0", 1.0).measured
    True

    ``measured=False`` is the honesty flag. ``value`` still has to be *some*
    float, so it carries a placeholder — aggregation must filter on the flag,
    never on the number:

    >>> failed = SampleScore("1", 0.0, measured=False)
    >>> failed.value, failed.measured
    (0.0, False)

    Reading that 0.0 as a score would report a perfect-bias result for a sample
    the model never answered.
    """

    sample_id: str
    value: float
    measured: bool = True
    detail: Dict[str, Any] = field(default_factory=dict)
    #: Digest of the model output that produced this score, when the task has
    #: one. ``manifest.leaf_hash`` hashes this into the evidence chain, so it
    #: has to be something a SampleScore can carry: a field that reaches the
    #: hash but has no home here cannot be recovered from persisted evidence at
    #: verify time, and `aethics verify` then reports tampering that did not
    #: happen. That was the PR #64 regression, and
    #: tests/test_evidence_chain_guard.py now fails if the two drift again.
    #:
    #: Optional because not every task has a model response to digest — a
    #: purely computed score has nothing to point at. ``leaf_hash`` omits it
    #: from the payload when absent rather than hashing ``None``, so manifests
    #: written without it still verify.
    response_digest: Optional[str] = None


@dataclass
class TaskResult:
    """Aggregated result for a task.

    ``to_dict`` intentionally emits the same shape as the legacy
    ``BenchmarkResult`` (``benchmark_name`` etc.) so a migrated task is
    byte-identical to the pre-refactor engine — the VOS-3 / VOS-7 parity guard.
    ``score`` is None when the task was not measured; it must never be a default.

    >>> tr = TaskResult(task="demo", score=0.72, samples_tested=50, passed=True)
    >>> tr.to_dict()["benchmark_name"]
    'demo'
    >>> tr.to_dict()["score"]
    0.72

    A task that measured nothing reports ``None``, never ``0.0``:

    >>> nm = TaskResult(task="demo", score=None, samples_tested=0, passed=False)
    >>> nm.to_dict()["score"] is None
    True

    ``uncertainty`` is deliberately absent from ``to_dict`` — that dict is the
    byte-identical parity guard, so adding a key would break it. Read the field,
    or ``uncertainty_dict()``:

    >>> "uncertainty" in tr.to_dict()
    False
    >>> tr.uncertainty_dict() is None
    True
    """

    task: str
    score: Optional[float]  # 0-1 scale; None == not_measured
    samples_tested: int
    passed: bool
    threshold: float = 0.7
    details: Dict[str, Any] = field(default_factory=dict)
    scored_metric: Optional[ScoredMetric] = None
    #: VOS-5 uncertainty (SEM, seeded CI, effect size, significance, power flag).
    #: Deliberately NOT emitted by ``to_dict``: that dict is byte-identical to the
    #: legacy BenchmarkResult and is the VOS-3/VOS-7 parity guard, so adding a key
    #: would break it. Read it from this field, or via ``uncertainty_dict()``.
    uncertainty: Optional[Any] = None

    def uncertainty_dict(self) -> Optional[Dict[str, Any]]:
        """The uncertainty as a plain dict, for serialising into a VOS-4 result."""
        if self.uncertainty is None:
            return None
        return self.uncertainty.model_dump()

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "benchmark_name": self.task,
            "score": self.score,
            "details": self.details,
            "samples_tested": self.samples_tested,
            "passed": self.passed,
            "threshold": self.threshold,
        }
        if self.scored_metric:
            d["scored_metric"] = self.scored_metric.to_dict()
        return d


@dataclass(frozen=True)
class TaskMeta:
    """Discovery metadata for a task, surfaced by ``list_tasks`` / ``aethics
    list-tasks``. Set by the ``@register_task`` decorator.

    >>> m = TaskMeta(name="demo", metric="yes_fraction", requires_logprobs=False)
    >>> m.to_dict()["requires_logprobs"]
    False

    ``requires_logprobs`` is load-bearing: it decides whether a provider can run
    the task at all, so a task needing likelihoods reports ``not_measured``
    against a generation-only model instead of scoring it on something weaker.

    Frozen, so discovery metadata cannot drift after registration:

    >>> import dataclasses
    >>> try:
    ...     m.name = "renamed"
    ... except dataclasses.FrozenInstanceError:
    ...     print("frozen")
    frozen
    """

    name: str
    metric: str  # what the score measures, e.g. "coref_accuracy_gap"
    requires_logprobs: bool
    citation: str = ""
    licence: str = ""
    description: str = ""
    # Keys into DATASET_LICENSES, so licence, citation and pinned revision are
    # never restated anywhere else. Empty for a task with no third-party corpus.
    dataset_id: str = ""
    # Sample count when the user gives no --limit. Per-task because a paired
    # benchmark needs fewer samples for the same power than an open-ended one.
    default_limit: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "requires_logprobs": self.requires_logprobs,
            "citation": self.citation,
            "licence": self.licence,
            "description": self.description,
            "dataset_id": self.dataset_id,
            "default_limit": self.default_limit,
        }


class Task(abc.ABC):
    """A benchmark. Subclass and decorate with ``@register_task(...)``.

    The four stages are deliberately separate so the framework — not the task —
    owns talking to the model, batching, and error handling. A task only knows
    how to produce samples, what to ask the model, how to score one answer, and
    how to roll the answers up.

    A minimal complete task:

    >>> class AlwaysYes(Task):
    ...     def load(self, limit=None):
    ...         return [Sample(id="0"), Sample(id="1")][:limit]
    ...     def build_request(self, sample):
    ...         return Request(method="generate", args=("say yes",))
    ...     def score(self, sample, response):
    ...         return SampleScore(sample.id, 1.0, measured=response.success)
    ...     def aggregate(self, scores):
    ...         kept = [s for s in scores if s.measured]
    ...         return TaskResult(
    ...             task="always_yes",
    ...             score=(sum(s.value for s in kept) / len(kept)) if kept else None,
    ...             samples_tested=len(kept),
    ...             passed=bool(kept),
    ...         )
    >>> task = AlwaysYes()
    >>> [s.id for s in task.load(limit=1)]
    ['0']
    >>> task.aggregate([SampleScore("0", 1.0)]).score
    1.0

    With nothing measured it yields ``None`` rather than a zero:

    >>> task.aggregate([SampleScore("0", 0.0, measured=False)]).score is None
    True

    All four stages are abstract, so a partial implementation cannot be
    instantiated and silently do half a job:

    >>> try:
    ...     Task()
    ... except TypeError as exc:
    ...     print("abstract" in str(exc))
    True
    """

    #: populated by @register_task
    meta: TaskMeta

    @abc.abstractmethod
    def load(self, limit: Optional[int] = None) -> Iterable[Sample]:
        """Yield samples (at most ``limit`` if given)."""

    @abc.abstractmethod
    def build_request(self, sample: Sample) -> Union[Request, List[Request]]:
        """The model call(s) needed to score this sample.

        Return a single ``Request``, or a list of them for tasks that need
        several model calls per sample (e.g. WinoBias scores gold vs distractor
        for both the pro- and anti-stereotypical sentence). ``score`` then
        receives a matching ``Response`` or list of ``Response``s.
        """

    @abc.abstractmethod
    def score(
        self, sample: Sample, response: Union[Response, List[Response]]
    ) -> SampleScore:
        """Turn the model response(s) into a per-sample score (shape matches what
        ``build_request`` returned)."""

    @abc.abstractmethod
    def aggregate(self, scores: List[SampleScore]) -> TaskResult:
        """Roll per-sample scores into the task result."""
