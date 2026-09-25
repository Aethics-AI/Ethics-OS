<!--
SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
SPDX-License-Identifier: Apache-2.0
-->

# From results to a readiness score

Benchmarks produce measurements. A catalogue of requirements decides which
measurements bear on which requirement. This page covers the last step: how
requirement results become a 0–10 score for each of six dimensions, and when
those scores become a single readiness figure.

The implementation is [`aethics_eval.aggregation`](https://github.com/Aethics-AI/Ethics-OS/blob/main/src/aethics_eval/aggregation.py).
It does not contain a requirement catalogue: which requirements exist, which
provisions they come from and which dimension each belongs to are inputs. The
AETHICS certification platform supplies its own catalogue. You can supply
yours.

## The six dimensions

| Dimension | Covers |
|---|---|
| Right to liberty | Impact on individual freedom and autonomy |
| Digital inclusion | Whether a system serves people equitably across groups |
| Data governance | Data quality, privacy and management across the lifecycle |
| Transparency | Whether affected people can tell AI is involved, and how it operates |
| Human oversight | Human control over consequential decisions |
| Security | Cybersecurity, robustness and safety |

They are jurisdiction-neutral by design. Every requirement in a catalogue
belongs to exactly one.

## Input

Each requirement result carries three things:

| Field | Meaning |
|---|---|
| `dimension` | Which of the six it belongs to |
| `status` | `measured`, `requires_attestation`, `low_coverage`, or anything else for "not measured" |
| `score` | 0.0–1.0, read only when `status` is `measured` |

```python
from aethics_eval.aggregation import (
    Dimension,
    RequirementResult,
    readiness_for,
    score_dimensions,
)

results = [
    RequirementResult(Dimension.SECURITY, "measured", 0.9),
    RequirementResult(Dimension.SECURITY, "measured", 0.7),
    RequirementResult(Dimension.HUMAN_OVERSIGHT, "requires_attestation"),
]
score_dimensions(results)[Dimension.SECURITY].score  # 8.0
readiness_for(results).score  # None: four dimensions have no score
```

Any object with those three attributes works; `RequirementResult` is a
convenience.

## Dimension score

**The mean of the dimension's measured results, times ten.** Security with
results 0.9 and 0.7 scores 8.0.

Only `measured` results count. The others are excluded and counted separately:

- **`requires_attestation`**: the requirement is about organisational process
  (risk management procedures, oversight structures) and is evidenced by
  documents, not by probing the model.
- **`low_coverage`**: measured, but on too little data to trust.
- **anything else**: no measurement at all.

None of these is a low score, and averaging them in as zero would turn "we do
not know" into "the model did badly". A dimension with no measured results
scores `null`, never `0`. A genuine `0.0` result is still a zero: the
requirement was measured and failed.

The counts travel with every score (`requirements`, `scored`,
`attestation_only`, `not_measured`, `low_coverage`, `coverage`), so 4.0 from
one of six requirements does not read the same as 4.0 from eight of eight.

## Readiness score

**The equal-weighted mean of the six dimension scores, and only when all six
have one.** Otherwise the readiness score is `null` and the missing dimensions
are named.

Two alternatives were rejected:

- **Average the dimensions that scored.** Overstates readiness, and the omission
  is invisible in the number. A single figure ends up in slides and procurement
  forms without the caveat that qualified it.
- **Count a missing dimension as 0.** Reports a measurement that never happened.

Withholding the headline number does not withhold the findings: every dimension
score is still reported.

**Weighting is equal** because the source analysis states no weighting, and
choosing one would put an unsourced judgment about the relative importance of,
say, human oversight into the headline number. The weights are held as data
(`DIMENSION_WEIGHT`) so the choice is visible and can be challenged.

## Disagreeing with this

Open an issue. The rules above are meant to be contested: that is why they are
published.
