# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-5 — the run manifest: determinism and verifiable evidence.

F5 was that ``audit_hash`` covered only the final scores. An auditor handed a
result could confirm the totals had not been edited, and nothing else — the
per-sample evidence those totals were computed from was outside the hash. So
"the score is 0.61" was checkable; "the score is 0.61 *because of these 200
samples*" was not.

This module closes that. Every sample becomes a leaf in a Merkle tree, each
task is the root of its own subtree, and the run hash is computed over those
roots together with the inputs that determined them — methodology fingerprint,
dataset revisions and content hashes, model id and parameters, library version,
and seeds.

Two consequences, and both are the point:

* **Two runs on the same seed and the same pinned inputs produce the same run
  hash.** If they do not, something moved that should not have.
* **Editing any single sample changes the run hash.** Not the sample's own
  record — the run hash. A tampered result cannot be made to verify by also
  editing the totals to match, because the totals are not what is hashed.

Why a real tree rather than hashing a concatenation of everything: a tree lets
you prove one sample belongs to a run without disclosing the other 199. That
matters for a benchmark whose licence forbids redistributing the data — we can
evidence a score without shipping the corpus. ``inclusion_proof`` and
``verify_inclusion`` implement it.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

MANIFEST_VERSION = "1.0.0"

# Domain separation. Hashing a leaf and hashing an internal node with the same
# function lets an attacker present an internal node as a leaf ("second
# preimage"). Distinct prefixes make the two spaces disjoint.
_LEAF_PREFIX = b"\x00aethics-leaf:"
_NODE_PREFIX = b"\x01aethics-node:"
_ROOT_PREFIX = b"\x02aethics-root:"


def _sha256(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def _canonical(obj: Any) -> bytes:
    """Deterministic bytes for any JSON-able value.

    ``sort_keys`` so dict insertion order cannot change a hash, and
    ``separators`` without spaces so formatting cannot either. ``default=str``
    keeps a stray non-JSON type from raising at the moment we are trying to
    record evidence — a degraded but stable representation beats a crash here.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True
    ).encode("utf-8")


# ── Merkle tree ─────────────────────────────────────────────────────


def response_digest(responses: Any) -> str:
    """Digest of what the model actually returned for one sample.

    The Merkle tree originally covered ``{id, value, measured}`` - the *score*
    of each sample, but nothing about the text that produced it. So an auditor
    could verify that the per-sample scores add up to the reported metric, and
    could not tell whether those scores described the responses stored beside
    them: editing a recorded model output left every hash intact.

    Hashing the response closes that. A digest rather than the text itself
    because the result file is not an archive - responses can be long, and can
    carry whatever the model said - but a digest is enough to prove that a
    stored response is the one that was scored.

    Accepts a single Response or a list (multi-call tasks like WinoBias), and
    reads them structurally so a failed call and an empty one stay distinct.
    """
    items = responses if isinstance(responses, (list, tuple)) else [responses]
    payload = [
        {
            "value": getattr(r, "value", None),
            "success": bool(getattr(r, "success", True)),
            "error": getattr(r, "error", None),
        }
        for r in items
    ]
    return _sha256(_LEAF_PREFIX, _canonical(payload))


def leaf_hash(
    sample_id: str,
    value: Any,
    measured: bool = True,
    response_digest: Optional[str] = None,
) -> str:
    """Hash one sample's evidence.

    ``measured`` is included deliberately: a sample that was skipped and a
    sample that scored 0.0 are different claims, and flipping one to the other
    must change the hash.

    ``response_digest`` covers the model output that produced the score. It is
    omitted from the hashed payload when absent rather than hashed as ``None``,
    so a manifest written before responses were covered still verifies against
    its recorded root instead of failing as if it had been tampered with.
    """
    payload: Dict[str, Any] = {
        "id": sample_id,
        "value": value,
        "measured": measured,
    }
    if response_digest is not None:
        payload["response_digest"] = response_digest
    return _sha256(_LEAF_PREFIX, _canonical(payload))


def _pair(left: str, right: str) -> str:
    return _sha256(_NODE_PREFIX, left.encode(), right.encode())


def merkle_root(leaves: Sequence[str]) -> str:
    """Root of a binary Merkle tree over ``leaves``, in the order given.

    An odd node at any level is promoted rather than duplicated. Duplicating it
    is the more common construction and is the source of CVE-2012-2459 in
    Bitcoin — two different leaf sets can produce the same root. Promotion has
    no such collision.

    An empty tree hashes the empty marker rather than returning "", so "no
    samples" is a statable fact with a hash of its own, not an absence.
    """
    if not leaves:
        return _sha256(_NODE_PREFIX, b"empty")

    level = list(leaves)
    while len(level) > 1:
        nxt: List[str] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_pair(level[i], level[i + 1]))
        if len(level) % 2:
            nxt.append(level[-1])  # promote, do not duplicate
        level = nxt
    return level[0]


def inclusion_proof(leaves: Sequence[str], index: int) -> List[Tuple[str, str]]:
    """Sibling path proving ``leaves[index]`` is in the tree.

    Returns [(side, hash), ...] where side is "L" or "R" — the position of the
    *sibling*. Lets a third party confirm one sample belongs to a run without
    seeing the rest, which is what makes evidence shareable for a dataset we
    are not allowed to redistribute.
    """
    if not 0 <= index < len(leaves):
        raise IndexError(f"index {index} outside 0..{len(leaves) - 1}")

    proof: List[Tuple[str, str]] = []
    level = list(leaves)
    idx = index

    while len(level) > 1:
        nxt: List[str] = []
        next_idx: Optional[int] = None

        # Pair up. Whichever pair contains our node contributes its sibling to
        # the proof, and our position in the next level is that pair's index.
        for i in range(0, len(level) - 1, 2):
            if i == idx:
                proof.append(("R", level[i + 1]))
                next_idx = len(nxt)
            elif i + 1 == idx:
                proof.append(("L", level[i]))
                next_idx = len(nxt)
            nxt.append(_pair(level[i], level[i + 1]))

        # An odd trailing node is promoted unchanged. If that is our node it
        # gains no sibling at this level, so nothing is added to the proof.
        if len(level) % 2:
            if idx == len(level) - 1:
                next_idx = len(nxt)
            nxt.append(level[-1])

        if next_idx is None:  # pragma: no cover — unreachable by construction
            # Not an assert: those vanish under `python -O`, and an invariant
            # guarding hash correctness should not be optional.
            raise RuntimeError(
                f"lost track of index {index} while walking a tree of "
                f"{len(leaves)} leaves — the proof would be silently wrong"
            )
        idx = next_idx
        level = nxt

    return proof


def verify_inclusion(leaf: str, proof: Sequence[Tuple[str, str]], root: str) -> bool:
    """Replay a sibling path and check it reaches ``root``."""
    current = leaf
    for side, sibling in proof:
        current = _pair(sibling, current) if side == "L" else _pair(current, sibling)
    return current == root


# ── The manifest ────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskEvidence:
    """One task's evidence root and what it covers."""

    task_id: str
    evidence_root: str
    sample_count: int
    measured_count: int
    metric_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetRecord:
    """A dataset as actually loaded — not as the registry describes it."""

    name: str
    revision: Optional[str]
    content_hash: Optional[str] = None
    row_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to reproduce a run and check it was not edited."""

    manifest_version: str
    run_id: str
    methodology_fingerprint: str
    library_version: str
    seeds: Dict[str, int]
    model: Dict[str, Any]
    datasets: List[DatasetRecord]
    tasks: List[TaskEvidence]
    run_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "run_id": self.run_id,
            "methodology_fingerprint": self.methodology_fingerprint,
            "library_version": self.library_version,
            "seeds": dict(sorted(self.seeds.items())),
            "model": self.model,
            "datasets": [d.to_dict() for d in self.datasets],
            "tasks": [t.to_dict() for t in self.tasks],
            "run_hash": self.run_hash,
        }


def compute_run_hash(
    *,
    methodology_fingerprint: str,
    library_version: str,
    seeds: Dict[str, int],
    model: Dict[str, Any],
    datasets: Sequence[DatasetRecord],
    tasks: Sequence[TaskEvidence],
) -> str:
    """The root hash: inputs plus evidence roots.

    Deliberately does **not** include run_id or timestamps. Two runs of the same
    thing should hash the same; if the wall clock changed the answer, the hash
    would prove nothing about reproducibility.

    Everything is sorted before hashing, so neither dict ordering nor the order
    tasks happened to finish in can move the result.
    """
    payload = {
        "methodology_fingerprint": methodology_fingerprint,
        "library_version": library_version,
        "seeds": dict(sorted(seeds.items())),
        "model": model,
        "datasets": sorted(
            (d.to_dict() for d in datasets), key=lambda d: str(d["name"])
        ),
        "tasks": sorted((t.to_dict() for t in tasks), key=lambda t: str(t["task_id"])),
    }
    return _sha256(_ROOT_PREFIX, _canonical(payload))


def build_manifest(
    *,
    run_id: str,
    methodology_fingerprint: str,
    library_version: str,
    seeds: Dict[str, int],
    model: Dict[str, Any],
    datasets: Sequence[DatasetRecord],
    task_samples: Dict[str, Sequence[Dict[str, Any]]],
    task_metrics: Optional[Dict[str, Optional[float]]] = None,
) -> RunManifest:
    """Build a manifest from a run's per-sample evidence.

    ``task_samples`` maps task id to that task's samples, each
    ``{"sample_id": str, "value": float, "measured": bool, "response_digest": str | None}``. Samples are sorted
    by id before hashing so a task that runs concurrently still produces a
    stable root.
    """
    metrics = task_metrics or {}
    evidence: List[TaskEvidence] = []

    for task_id in sorted(task_samples):
        samples = sorted(
            task_samples[task_id], key=lambda s: str(s.get("sample_id", ""))
        )
        leaves = [
            leaf_hash(
                str(s.get("sample_id", "")),
                s.get("value"),
                bool(s.get("measured", True)),
                s.get("response_digest"),
            )
            for s in samples
        ]
        evidence.append(
            TaskEvidence(
                task_id=task_id,
                evidence_root=merkle_root(leaves),
                sample_count=len(samples),
                measured_count=sum(1 for s in samples if s.get("measured", True)),
                metric_value=metrics.get(task_id),
            )
        )

    run_hash = compute_run_hash(
        methodology_fingerprint=methodology_fingerprint,
        library_version=library_version,
        seeds=seeds,
        model=model,
        datasets=datasets,
        tasks=evidence,
    )

    return RunManifest(
        manifest_version=MANIFEST_VERSION,
        run_id=run_id,
        methodology_fingerprint=methodology_fingerprint,
        library_version=library_version,
        seeds=dict(sorted(seeds.items())),
        model=model,
        datasets=list(datasets),
        tasks=evidence,
        run_hash=run_hash,
    )


# ── Verification ────────────────────────────────────────────────────


@dataclass
class VerificationReport:
    """The outcome of re-deriving a manifest from the evidence beside it."""

    ok: bool
    run_hash_expected: str = ""
    run_hash_computed: str = ""
    problems: List[str] = field(default_factory=list)
    tasks_checked: int = 0
    samples_checked: int = 0

    def summary(self) -> str:
        if self.ok:
            return (
                f"verified: run hash {self.run_hash_computed[:16]}… over "
                f"{self.samples_checked} samples across {self.tasks_checked} task(s)"
            )
        return f"FAILED: {len(self.problems)} problem(s)"


def _same_metric(reported: Any, hashed: Any) -> bool:
    """Whether two recorded scores are the same score.

    ``None`` means not measured, and is only equal to ``None`` — a task that
    reports no score and one that reports 0.0 are different claims, and
    collapsing them here would undo the distinction the rest of the library
    exists to keep.

    Numbers are compared with a tolerance because a score can make a round trip
    through JSON on either side; the tolerance is far tighter than any edit
    worth catching.
    """
    if reported is None or hashed is None:
        return reported is None and hashed is None
    if isinstance(reported, bool) or isinstance(hashed, bool):
        return reported is hashed
    if isinstance(reported, (int, float)) and isinstance(hashed, (int, float)):
        return math.isclose(float(reported), float(hashed), rel_tol=1e-9, abs_tol=1e-12)
    return bool(reported == hashed)


def verify_manifest(
    manifest: Dict[str, Any],
    task_samples: Dict[str, Sequence[Dict[str, Any]]],
    reported_metrics: Optional[Dict[str, Any]] = None,
) -> VerificationReport:
    """Recompute every evidence root and the run hash, and compare.

    This is what makes the manifest worth having: it is not a value we trust
    because it is written down, it is a value anyone can re-derive from the
    evidence sitting next to it.

    ``reported_metrics`` maps task id to the score the *result file* displays.
    The manifest keeps its own copy of each score and the run hash is computed
    over that copy, so without this the two can disagree and everything still
    verifies: edit ``tasks[].metric_value``, leave the samples and the manifest
    alone, and the hashes all match while the file reports a different number
    than the one that was hashed. That is the reading a caller is least likely
    to expect, because it is the number `aethics show` prints.

    Callers that pass nothing check the evidence only, which is what a bare
    manifest with no result envelope around it can support.
    """
    problems: List[str] = []
    recorded_tasks = {t["task_id"]: t for t in manifest.get("tasks", [])}

    samples_checked = 0
    rebuilt: List[TaskEvidence] = []

    for task_id in sorted(recorded_tasks):
        recorded = recorded_tasks[task_id]
        samples = task_samples.get(task_id)

        if samples is None:
            problems.append(
                f"{task_id}: manifest records evidence but the result carries no "
                f"samples for it — the evidence cannot be checked"
            )
            continue

        ordered = sorted(samples, key=lambda s: str(s.get("sample_id", "")))
        leaves = [
            leaf_hash(
                str(s.get("sample_id", "")),
                s.get("value"),
                bool(s.get("measured", True)),
                s.get("response_digest"),
            )
            for s in ordered
        ]
        root = merkle_root(leaves)
        samples_checked += len(ordered)

        if root != recorded.get("evidence_root"):
            problems.append(
                f"{task_id}: evidence root mismatch — recorded "
                f"{str(recorded.get('evidence_root'))[:16]}…, computed {root[:16]}…. "
                f"A sample was added, removed or edited."
            )
        if len(ordered) != recorded.get("sample_count"):
            problems.append(
                f"{task_id}: sample count mismatch — recorded "
                f"{recorded.get('sample_count')}, found {len(ordered)}"
            )

        # The score the file reports, against the one the run hash covers.
        # Compared as written rather than recomputed from the leaves: how a
        # metric aggregates is the task's business (BOLD means its samples,
        # others do not), and re-deriving it here would put a second, silently
        # diverging implementation of every metric inside the verifier.
        if reported_metrics is not None and task_id in reported_metrics:
            reported = reported_metrics[task_id]
            hashed = recorded.get("metric_value")
            if not _same_metric(reported, hashed):
                problems.append(
                    f"{task_id}: reported score does not match the evidence — "
                    f"the file says {reported}, the manifest hashed {hashed}. "
                    f"The score was edited after the run."
                )

        rebuilt.append(
            TaskEvidence(
                task_id=task_id,
                evidence_root=root,
                sample_count=len(ordered),
                measured_count=sum(1 for s in ordered if s.get("measured", True)),
                metric_value=recorded.get("metric_value"),
            )
        )

    for task_id in sorted(set(task_samples) - set(recorded_tasks)):
        problems.append(
            f"{task_id}: result carries samples the manifest does not cover — "
            f"evidence was appended after the manifest was written"
        )

    computed = compute_run_hash(
        methodology_fingerprint=manifest.get("methodology_fingerprint", ""),
        library_version=manifest.get("library_version", ""),
        seeds=manifest.get("seeds", {}),
        model=manifest.get("model", {}),
        datasets=[DatasetRecord(**d) for d in manifest.get("datasets", [])],
        tasks=rebuilt,
    )
    expected = manifest.get("run_hash", "")

    if computed != expected:
        problems.append(
            f"run hash mismatch — recorded {expected[:16]}…, computed {computed[:16]}…"
        )

    return VerificationReport(
        ok=not problems,
        run_hash_expected=expected,
        run_hash_computed=computed,
        problems=problems,
        tasks_checked=len(rebuilt),
        samples_checked=samples_checked,
    )


# ── Seeding and evidence collection ─────────────────────────────────


DEFAULT_SEED = 42


def set_global_seeds(seed: int) -> Dict[str, int]:
    """Seed every RNG a run might touch, and report what was set.

    Returns the record that goes into the manifest, so the seeds a run claims
    are the seeds it actually applied rather than the ones someone meant to.

    torch and numpy are seeded only if already imported. Importing torch here
    to seed it would pull 2GB into a run that never needed it, and a library
    that quietly changes global RNG state for a dependency the caller is not
    using is worse than one that does less.
    """
    import random

    random.seed(seed)
    applied: Dict[str, int] = {"python_random": seed}

    # numpy is a core dependency, so seed it unconditionally. An earlier
    # version only seeded it "if already imported", which made the applied-seed
    # record depend on import order: the first run in a process recorded
    # {python_random}, the second {numpy, python_random} because loading a
    # dataset had imported numpy in between. That fed into the run hash and made
    # two identical runs hash differently — exactly the non-determinism this
    # module exists to rule out.
    try:
        import numpy

        numpy.random.seed(seed)
        applied["numpy"] = seed
    except Exception:  # pragma: no cover — numpy is a declared dependency
        pass

    # torch is in the [local] extra, not core. Seeding it is conditional on it
    # being present, which is a property of the environment rather than of
    # import order — stable across runs in the same install. It is recorded but
    # deliberately excluded from the hash (see canonical_seeds).
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.manual_seed(seed)
            applied["torch"] = seed
        except Exception:  # pragma: no cover — a stub or partial import
            pass

    return applied


def canonical_seeds(seed: int) -> Dict[str, int]:
    """What goes into the run hash: the seed that was requested.

    The full applied map is recorded in the result for a human to read, but the
    hash covers the requested value only. Otherwise the hash would move with
    which optional RNGs happened to be present, and two runs of the same
    evaluation on the same data would stop agreeing for a reason that has
    nothing to do with the evaluation.
    """
    return {"seed": seed}


async def run_task_collecting_evidence(
    task: Any, model: Any, limit: Optional[int] = None
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Run a task and keep the per-sample scores, not just the aggregate.

    ``tasks.runner.run_task`` discards them once ``aggregate`` has run, which
    is right for its purpose and wrong for ours — the manifest exists to hash
    exactly the evidence that gets thrown away there. Rather than change that
    function and its parity guarantees, this walks the same public four-stage
    API and keeps a copy.

    Returns ``(TaskResult, [{sample_id, value, measured, response_digest}, ...])``.
    """
    from .tasks.runner import _run_requests

    scores = []
    evidence: List[Dict[str, Any]] = []

    for sample in task.load(limit):
        responses = await _run_requests(model, task.build_request(sample))
        score = task.score(sample, responses)
        scores.append(score)
        evidence.append(
            {
                "sample_id": str(getattr(score, "sample_id", sample.id)),
                "value": getattr(score, "value", None),
                "measured": bool(getattr(score, "measured", True)),
                "response_digest": response_digest(responses),
            }
        )

    return task.aggregate(scores), evidence
