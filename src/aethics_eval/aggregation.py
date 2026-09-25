# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""How requirement-level results roll up into dimension scores and one readiness
score.

This is the aggregation half of the scoring: benchmarks and classifiers produce
measurements, something maps those measurements onto requirements, and this
module turns the requirement results into the figures a report shows.

Which requirements exist, which legal provisions they come from and which
dimension each one belongs to are **inputs**, not part of this module. Any
catalogue works, provided each result says which dimension it belongs to, what
its status is and what it scored (see :class:`RequirementResult`).

The six dimensions are the standardised assessment dimensions of the
cross-framework analysis, chosen to be jurisdiction-neutral and grounded in
international human rights language.

Two rules carry most of the weight here.

**Unmeasured is not zero.** Only results with ``status == "measured"``
contribute to a score. A requirement evidenced by documentation rather than by
probing the model, one with no test data, or one measured on too little data is
an absence of evidence. Averaging it in as a number would turn "we do not know"
into "the model did badly". A dimension with nothing measured scores ``None``,
never ``0.0``.

**A readiness score needs all six dimensions.** A single figure travels without
its context. Averaging the five dimensions that scored overstates readiness, and
the omission cannot be seen in the number. Substituting 0 for the missing one
reports a measurement that never happened. So the readiness score is withheld,
with the missing dimensions named, unless every dimension scored. The
per-dimension scores are still reported in full.

>>> results = [
...     RequirementResult(Dimension.SECURITY, "measured", 0.9),
...     RequirementResult(Dimension.SECURITY, "measured", 0.7),
...     RequirementResult(Dimension.SECURITY, "requires_attestation"),
... ]
>>> s = score_dimensions(results)[Dimension.SECURITY]
>>> s.score, s.scored, s.requirements
(8.0, 2, 3)
>>> readiness_from_scores(score_dimensions(results)).score is None
True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol, Sequence, Tuple


class Dimension(Enum):
    """The six assessment dimensions, named as the source analysis gives them."""

    RIGHT_TO_LIBERTY = "right to liberty"
    DIGITAL_INCLUSION = "digital inclusion"
    DATA_GOVERNANCE = "data governance"
    TRANSPARENCY = "transparency"
    HUMAN_OVERSIGHT = "human oversight"
    SECURITY = "security"

    @property
    def slug(self) -> str:
        """Machine-friendly form, for serialised output."""
        return self.value.replace(" ", "_")


#: Reporting order. Explicit so output has a stable sequence rather than
#: whatever the enum happens to iterate.
DIMENSION_ORDER: Tuple[Dimension, ...] = (
    Dimension.RIGHT_TO_LIBERTY,
    Dimension.DIGITAL_INCLUSION,
    Dimension.DATA_GOVERNANCE,
    Dimension.TRANSPARENCY,
    Dimension.HUMAN_OVERSIGHT,
    Dimension.SECURITY,
)

DIMENSION_DESCRIPTION: Dict[Dimension, str] = {
    Dimension.RIGHT_TO_LIBERTY: (
        "Impact on individual freedom and autonomy — whether a system "
        "constrains what a person may do or decide."
    ),
    Dimension.DIGITAL_INCLUSION: (
        "Accessibility and inclusivity — whether a system serves people "
        "equitably across groups rather than working well only for some."
    ),
    Dimension.DATA_GOVERNANCE: (
        "Data quality, privacy and management practices across the lifecycle."
    ),
    Dimension.TRANSPARENCY: (
        "Explainability and disclosure — whether affected people can understand "
        "that AI is involved and, in proportion to the stakes, how it operates."
    ),
    Dimension.HUMAN_OVERSIGHT: (
        "Human control and intervention capability over consequential decisions."
    ),
    Dimension.SECURITY: (
        "Cybersecurity, robustness and safety of the system and its outputs."
    ),
}

#: Upper bound of the dimension and readiness scales.
SCALE_MAX = 10.0

#: Statuses that contribute to a score. Only "measured".
SCORING_STATUSES = frozenset({"measured"})

#: Evidence exists but is too thin to score on. Counted separately so the
#: reason for a small denominator is visible.
LOW_CONFIDENCE_STATUSES = frozenset({"low_coverage"})

#: Evidenced by documentation, not by probing the model.
ATTESTATION_STATUS = "requires_attestation"

#: Every dimension carries the same weight. The source analysis states no
#: weighting, and inventing one would put an unsourced judgment about the
#: relative importance of human rights protections into a headline number. Held
#: as data so the choice is visible and contestable.
DIMENSION_WEIGHT: Dict[Dimension, float] = {d: 1.0 for d in DIMENSION_ORDER}


class ScoredRequirement(Protocol):
    """What aggregation needs from one requirement's result.

    ``score`` is 0.0–1.0 and is read only when ``status`` is ``"measured"``.
    For any other status it is ignored, so a catalogue that stores ``0.0`` for
    an unmeasured requirement cannot leak that zero into a score.
    """

    @property
    def dimension(self) -> Dimension: ...

    @property
    def status(self) -> str: ...

    @property
    def score(self) -> Optional[float]: ...


@dataclass(frozen=True)
class RequirementResult:
    """A plain :class:`ScoredRequirement`, for callers with no type of their own."""

    dimension: Dimension
    status: str
    score: Optional[float] = None


@dataclass(frozen=True)
class DimensionScore:
    """One dimension's score, with the counts needed to read it honestly.

    ``4.0 from 1 of 6 requirements`` and ``4.0 from 8 of 8`` are different
    claims, so the counts travel with the number.
    """

    dimension: Dimension
    #: None when nothing was measured. Never 0.0 as a stand-in.
    score: Optional[float]
    requirements: int
    scored: int
    attestation_only: int
    not_measured: int
    low_coverage: int

    @property
    def measured(self) -> bool:
        return self.scored > 0

    @property
    def status(self) -> str:
        if self.requirements == 0:
            return "no_requirements"
        if not self.measured:
            return "not_measured"
        if self.scored < self.requirements:
            return "partially_measured"
        return "measured"

    @property
    def coverage(self) -> Optional[float]:
        """Fraction of this dimension's requirements the score speaks for."""
        if self.requirements == 0:
            return None
        return round(self.scored / self.requirements, 2)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "dimension": self.dimension.value,
            "slug": self.dimension.slug,
            "description": DIMENSION_DESCRIPTION[self.dimension],
            "status": self.status,
            "requirements": self.requirements,
            "scored": self.scored,
            "attestation_only": self.attestation_only,
            "not_measured": self.not_measured,
            "low_coverage": self.low_coverage,
            "coverage": self.coverage,
            # Present and null rather than absent: a missing key lets a
            # consumer's `.get("score", 0)` turn an absence into a zero.
            "score": self.score,
            "scale": f"0-{SCALE_MAX:.0f}",
        }
        if not self.measured:
            payload["note"] = (
                "Not measured. No requirement on this dimension produced a "
                "measurement, so there is no score. This is not a score of "
                "zero, which would mean the requirements were measured and "
                "failed."
            )
        elif self.scored < self.requirements:
            payload["note"] = (
                f"Scored on {self.scored} of {self.requirements} requirements. "
                f"The rest could not be measured and are excluded rather than "
                f"counted as zero, so this score speaks for part of the "
                f"dimension."
            )
        return payload


def score_dimension(
    dimension: Dimension, results: Sequence[ScoredRequirement]
) -> DimensionScore:
    """Score one dimension: the mean of its measured results, out of 10."""
    scored: List[float] = []
    attestation = unmeasured = low_coverage = 0

    for result in results:
        if result.status in SCORING_STATUSES:
            if result.score is None:
                raise ValueError(
                    f"a {dimension.value} result has status 'measured' but no "
                    "score. A measured result must carry the value measured."
                )
            if not 0.0 <= result.score <= 1.0:
                raise ValueError(
                    f"requirement scores are 0.0-1.0; got {result.score!r} on "
                    f"{dimension.value}"
                )
            scored.append(result.score)
        elif result.status == ATTESTATION_STATUS:
            attestation += 1
        elif result.status in LOW_CONFIDENCE_STATUSES:
            low_coverage += 1
        else:
            unmeasured += 1

    score = round(SCALE_MAX * sum(scored) / len(scored), 1) if scored else None
    return DimensionScore(
        dimension=dimension,
        score=score,
        requirements=len(results),
        scored=len(scored),
        attestation_only=attestation,
        not_measured=unmeasured,
        low_coverage=low_coverage,
    )


def score_dimensions(
    results: Sequence[ScoredRequirement],
) -> Dict[Dimension, DimensionScore]:
    """Score every dimension from a flat list of requirement results."""
    buckets: Dict[Dimension, List[ScoredRequirement]] = {d: [] for d in DIMENSION_ORDER}
    for result in results:
        buckets[result.dimension].append(result)
    return {d: score_dimension(d, buckets[d]) for d in DIMENSION_ORDER}


def dimension_score_report(results: Sequence[ScoredRequirement]) -> Dict[str, object]:
    """Serialisable per-dimension scores, in reporting order."""
    scores = score_dimensions(results)
    return {
        "scale": {
            "min": 0,
            "max": SCALE_MAX,
            "note": (
                "A dimension's score is the mean of its measured requirement "
                "scores, on a 0-10 scale. Only requirements that produced a "
                "measurement contribute; attestation-only requirements and "
                "those with no or insufficient data are excluded rather than "
                "counted as zero."
            ),
        },
        "dimensions": [scores[d].to_dict() for d in DIMENSION_ORDER],
        "not_measured": [
            d.value for d in DIMENSION_ORDER if scores[d].status == "not_measured"
        ],
    }


@dataclass(frozen=True)
class Readiness:
    """A readiness score, or a recorded refusal to produce one."""

    #: None when coverage is incomplete. Never a partial average, never 0.0.
    score: Optional[float]
    dimensions_scored: int
    dimensions_total: int
    #: Dimensions with no score, in reporting order. The reason for a None.
    missing: Tuple[Dimension, ...]

    @property
    def available(self) -> bool:
        return self.score is not None

    @property
    def status(self) -> str:
        if self.available:
            return "complete"
        if self.dimensions_scored == 0:
            return "nothing_measured"
        return "incomplete_coverage"

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "score": self.score,
            "scale": f"0-{SCALE_MAX:.0f}",
            "status": self.status,
            "dimensions_scored": self.dimensions_scored,
            "dimensions_total": self.dimensions_total,
            "weighting": "equal",
        }
        if self.available:
            return payload

        payload["missing_dimensions"] = [d.value for d in self.missing]
        if self.dimensions_scored == 0:
            payload["note"] = (
                "No readiness score. No dimension produced a score, so there is "
                "nothing to combine."
            )
        else:
            missing = ", ".join(d.value for d in self.missing)
            payload["note"] = (
                f"No readiness score. {self.dimensions_scored} of "
                f"{self.dimensions_total} dimensions were scored; {missing} "
                f"could not be measured. A composite over the remainder would "
                f"read as an assessment of the whole, so it is withheld. The "
                f"per-dimension scores are reported in full."
            )
        return payload


def readiness_from_scores(scores: Dict[Dimension, DimensionScore]) -> Readiness:
    """Combine dimension scores into one readiness figure, or decline to."""
    missing = tuple(
        d for d in DIMENSION_ORDER if scores.get(d) is None or scores[d].score is None
    )
    scored = tuple(d for d in DIMENSION_ORDER if d not in missing)

    if missing:
        return Readiness(
            score=None,
            dimensions_scored=len(scored),
            dimensions_total=len(DIMENSION_ORDER),
            missing=missing,
        )

    total_weight = sum(DIMENSION_WEIGHT[d] for d in scored)
    weighted = sum((scores[d].score or 0.0) * DIMENSION_WEIGHT[d] for d in scored)
    return Readiness(
        score=round(weighted / total_weight, 1),
        dimensions_scored=len(scored),
        dimensions_total=len(DIMENSION_ORDER),
        missing=(),
    )


def readiness_for(results: Sequence[ScoredRequirement]) -> Readiness:
    """Readiness for one assessment, scoring the dimensions on the way."""
    return readiness_from_scores(score_dimensions(results))
