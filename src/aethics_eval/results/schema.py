# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-4 — the versioned result schema.

Every evaluation serialises to a ``RunResult``: a JSON contract with a top-level
``schema_version`` that downstream tooling, version diffing, third-party
dashboards, and the VOS-7 SaaS all consume. It carries everything needed to
trust and reproduce a number — task id + version, the pinned dataset revision,
the model + provider + params, the methodology-manifest fingerprint, the metric
value with its uncertainty, sample count, method, reliability, and UTC
timestamps.

Compatibility (SemVer on ``schema_version``): additive changes are a MINOR bump;
any field removal or semantic change is a MAJOR bump. Loaders ignore unknown
fields so a v1 reader tolerates a v1.x result (see results/loader.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

#: Bump per the compatibility policy above.
#: 1.3.0 — POS-11 added optional SampleScore.response_digest. Additive, so a
#: 1.0.0 reader still loads a 1.3.0 result and ignores it. The field is not
#: cosmetic: the manifest hashes it into every evidence leaf, and its absence
#: here is what made `aethics verify` fail on untouched results.
#: 1.1.0 — VOS-5 added optional Uncertainty fields (standard_error, sample_size,
#: underpowered). Purely additive, so a 1.0.0 reader still loads a 1.1.0 result.
SCHEMA_VERSION = "1.3.0"


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp (serialises to ISO-8601 with offset).

    Aware, not naive — a bare local timestamp on an audit record is unusable
    evidence once it crosses a timezone.

    >>> utc_now().tzinfo is not None
    True
    """
    return datetime.now(timezone.utc)


class _Base(BaseModel):
    # Ignore unknown fields so a v1 loader tolerates additive (minor) results;
    # disable the protected `model_` namespace so we can use a `model` field.
    model_config = ConfigDict(extra="ignore", protected_namespaces=())


class SampleScore(_Base):
    """One sample's contribution to a task's metric.

    ``measured=False`` marks a sample the model never answered. ``value`` is
    still a float either way, so filter on the flag rather than the number.

    >>> SampleScore(sample_id="0", value=1.0).measured
    True
    >>> SampleScore(sample_id="1", value=0.0, measured=False).measured
    False
    """

    sample_id: str
    value: float
    measured: bool = True
    #: Digest of the model output behind this score, when the task produced one.
    #:
    #: This has to exist here because `manifest.leaf_hash` hashes it into the
    #: evidence chain. Without it, `extra="ignore"` on this model silently drops
    #: the digest during serialisation, so the recorded root is computed over a
    #: field the persisted evidence no longer carries and `aethics verify`
    #: reports tampering on a result nobody touched.
    #:
    #: That is the PR #64 regression. It was only half fixed: the task-layer
    #: dataclass and this model both need the field, and this one was missed.
    #: tests/test_evidence_chain_guard.py now fails if either drifts again.
    response_digest: Optional[str] = None


class Uncertainty(_Base):
    """Statistical uncertainty around a metric value (VOS-5).

    A point estimate is not a claim: 0.61 from 30 samples and 0.61 from 3,000 are
    different statements. Every field is optional — a statistic that cannot be
    computed is absent, never a fabricated 0.0 that would read as perfect
    precision.

    ``underpowered`` is a flag, not a filter — an underpowered result is still
    reported, just never silently weighted as though it were well-powered.

    >>> u = Uncertainty()
    >>> u.standard_error is None and u.confidence_interval is None
    True
    >>> u.sample_size, u.underpowered
    (0, False)
    >>> u = Uncertainty(method="bootstrap_1000", standard_error=0.04,
    ...                 confidence_interval=(0.55, 0.60), sample_size=1508)
    >>> u.confidence_interval
    (0.55, 0.6)
    """

    method: Optional[str] = None  # e.g. "bootstrap_1000", "paired_permutation"
    standard_error: Optional[float] = None  # SEM; renders as 0.61 ± 0.04
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None  # e.g. Cohen's d
    p_value: Optional[float] = None
    sample_size: int = 0
    #: True when sample_size is below the task's EVAL_MANIFEST.min_samples
    #: threshold — reported and flagged, never silently averaged in as though it
    #: carried the same weight as a well-powered run.
    underpowered: bool = False


class DatasetInfo(_Base):
    """The dataset a task ran on, with a pinned revision for reproducibility.

    ``revision`` is required. A dataset name alone does not identify data —
    upstream can change it under you between runs.

    >>> d = DatasetInfo(name="nyu-mll/crows-pairs", revision="8aaac11c")
    >>> d.revision
    '8aaac11c'
    """

    name: str
    revision: str


class ModelInfo(_Base):
    """The model under evaluation.

    ``params`` records the inference settings the run actually used, so a
    sampled run cannot later be mistaken for a deterministic one.

    >>> m = ModelInfo(id="gpt2", provider="local", params={"temperature": 0.0})
    >>> m.id, m.provider
    ('gpt2', 'local')
    """

    id: str
    provider: str
    params: Dict[str, Any] = Field(default_factory=dict)


class TaskResult(_Base):
    """One task's result within a run.

    ``metric_value`` is None when the task was not measured — never a fabricated
    default (the honesty rule carried through from F1/F6). ``reliability`` and
    ``method`` say how much to trust it.

    An unmeasured task carries no value at all — the distinction a ``0.0``
    would destroy.

    >>> t = TaskResult(
    ...     task_id="crows_pairs", task_version="1.0.0",
    ...     dataset=DatasetInfo(name="nyu-mll/crows-pairs", revision="8aaac11c"),
    ...     metric="stereotype_preference_rate", metric_value=0.5773,
    ...     sample_count=1508, method="sequence_logprob", reliability="direct",
    ... )
    >>> t.metric_value
    0.5773
    >>> nm = TaskResult(
    ...     task_id="crows_pairs", task_version="1.0.0",
    ...     dataset=DatasetInfo(name="nyu-mll/crows-pairs", revision="8aaac11c"),
    ...     metric="stereotype_preference_rate", metric_value=None,
    ...     method="sequence_logprob", reliability="not_measured",
    ... )
    >>> nm.metric_value is None
    True
    """

    task_id: str
    task_version: str
    dataset: DatasetInfo
    metric: str
    metric_value: Optional[float] = None  # None == not_measured
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    sample_count: int = 0
    method: str
    reliability: str
    passed: Optional[bool] = None
    sample_scores: Optional[List[SampleScore]] = None
    # Free-form, task-specific extras: per-domain breakdowns, sample results,
    # the methodology string. Deliberately untyped - every benchmark reports
    # something different here, and forcing that into the contract would either
    # bloat it or discard it. Nothing downstream may rely on a key in here; the
    # typed fields above are the promise.
    details: Optional[Dict[str, Any]] = None


class RunSummary(_Base):
    """Headline numbers across every task in a run.

    ``mean_score`` is None rather than 0.0 when nothing was measured: a mean of
    no measurements is not a score, and reporting 0.0 would make a run that
    failed to measure anything look like a run that measured a catastrophe.

    ``underpowered_tasks`` names the tasks whose sample count fell below the
    threshold in eval_config, so a consumer can see at the top level that a
    headline number rests on too little data.

    >>> s = RunSummary(tasks_run=3, tasks_measured=2, mean_score=0.61,
    ...                underpowered_tasks=["bold"])
    >>> s.mean_score
    0.61
    >>> s.underpowered_tasks
    ['bold']

    A run that measured nothing reports no mean, rather than 0.0:

    >>> RunSummary(tasks_run=2, tasks_measured=0).mean_score is None
    True
    """

    tasks_run: int = 0
    tasks_measured: int = 0
    mean_score: Optional[float] = None
    underpowered_tasks: List[str] = Field(default_factory=list)


class RunResult(_Base):
    """A complete evaluation run — the top-level serialised contract.

    This is what third-party tooling and the SaaS product read, so its shape is
    a promise governed by SemVer (see docs/result-schema-compatibility.md).

    Unknown fields are ignored rather than rejected, which is what makes
    additive (MINOR) changes safe for older readers.

    ``aethics eval`` writes exactly this shape. It previously wrote a separate
    envelope stamped ``schema_version 0.1.0``, so the published contract had no
    producer a user could invoke and ``aethics validate`` checked the wrong
    document.

    >>> r = RunResult(
    ...     run_id="abc123",
    ...     model=ModelInfo(id="gpt2", provider="local"),
    ...     manifest_fingerprint="266b57eaa394ea0b",
    ... )
    >>> r.schema_version == SCHEMA_VERSION
    True
    >>> r.tasks
    []
    >>> '"run_id": "abc123"' in r.to_json()
    True
    >>> r2 = RunResult.model_validate({
    ...     "run_id": "abc123",
    ...     "model": {"id": "gpt2", "provider": "local"},
    ...     "manifest_fingerprint": "266b57eaa394ea0b",
    ...     "a_field_from_the_future": 42,
    ... })
    >>> hasattr(r2, "a_field_from_the_future")
    False
    """

    schema_version: str = SCHEMA_VERSION
    run_id: str
    model: ModelInfo
    manifest_fingerprint: str  # EVAL_MANIFEST.fingerprint()
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    tasks: List[TaskResult] = Field(default_factory=list)
    # Headline numbers across all tasks. Optional because a RunResult
    # constructed by hand (a test, a downstream tool) need not compute them.
    summary: Optional[RunSummary] = None
    # How the run was configured: methodology fingerprint and version,
    # inference params, dataset revisions, seeds, offline flag.
    provenance: Optional[Dict[str, Any]] = None
    # The POS-5 run manifest: run hash, per-task evidence roots, seeds.
    # `aethics verify` recomputes this from the per-sample evidence in
    # tasks[].sample_scores.
    manifest: Optional[Dict[str, Any]] = None

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Canonical JSON serialisation (stable key order for golden files)."""
        return self.model_dump_json(indent=indent)
