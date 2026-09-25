<!--
SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
SPDX-License-Identifier: Apache-2.0
-->

# Claims register

Every public claim we make about how scoring works, with its status and the
code that backs it. If a claim about this toolkit is not here, we do not make
it. If a claim here has no backing, it is not `Implemented`.

This register covers the open-source scoring engine only. Claims about the
AETHICS certification platform, such as which legal frameworks it maps to, are
out of scope here.

## Status vocabulary

| Status | Means |
|---|---|
| **Implemented** | Backed by code in this repository that runs today. |
| **In development** | Built, not yet released. |
| **Target architecture** | Specified, not yet built. |

A claim with no backing code is **Target architecture**, however confidently
anyone phrases it.

## How to use this file

**Adding or changing a claim.** Add a row. Name the file and, where useful, the
symbol that backs it, so a reader can go and check. "The engine does it" is not
a backing reference.

**Changing scoring or methodology.** Update the affected rows in the same pull
request.

---

## Register

### Evaluation and scoring

| Claim | Status | Backed by |
|---|---|---|
| Models are evaluated against four published bias and safety benchmarks: CrowS-Pairs, StereoSet, WinoBias, BOLD | Implemented | `src/aethics_eval/tasks/benchmarks/` (one module per benchmark); `aethics list-tasks` |
| Scores carry a bootstrap confidence interval | Implemented | `scoring.py::bootstrap_confidence_interval` (seeded, 1,000 resamples) |
| A task that cannot be measured reports null and is excluded from aggregates, never recorded as zero | Implemented | `scoring.py::ScoredMetric.reliability`; engine aggregation |
| Every result carries a measurement-status qualifier so the basis of a score is visible | Implemented | `ScoredMetric.reliability`: `direct` / `derived` / `proxy` / `not_measured` |
| Toxicity and refusal judgments record which method produced them, so a heuristic fallback is never presented as a classifier result | Implemented | `toxicity_classifier.py`, `refusal_classifier.py`; `tests/test_safety_method_stamping.py` |
| Per-sample results are hashed into a tamper-evident chain that can be independently re-verified | Implemented | `manifest.py`: `leaf_hash` (one leaf per sample, including `measured` and the response digest), `merkle_root`, `verify_manifest`. Inclusion proofs via `inclusion_proof` / `verify_inclusion`. Held to the claim by `tests/test_evidence_chain_guard.py`, which fails if a field enters the leaf hash without a home on `SampleScore`. `verify` also compares each task's reported score against the hashed copy, so a result cannot claim a score its own evidence contradicts. |

> **`verify` did not always cover the reported score.** The manifest keeps its
> own copy of each task's score and the run hash is computed over that copy, but
> nothing used to compare it to the `metric_value` the result file displays.
> Editing only that left every hash intact. The per-sample chain was never
> affected: editing a sample was always caught. The two copies are now compared
> and disagreement fails verification.
>
> **The "including the response digest" part was briefly untrue.** For a period
> the published `SampleScore` had no `response_digest` field, so it was dropped
> during serialisation while the manifest carried on hashing it, and `aethics
> verify` failed on results nobody had touched. Recorded here because a register
> is only worth having if it says when a row was wrong.

> **Do not cite `scoring.py::hash_audit_results` for the evidence-chain claim.**
> It hashes the whole result blob, final scores included, as a single SHA-256.
> That detects casual corruption, but it is not a per-sample evidence chain: an
> edited sample can still hash correctly if the totals are adjusted to match,
> and it cannot answer "was *this* sample in the run", which is what an auditor
> asks. `manifest.py` is the real thing.

### Aggregation

| Claim | Status | Backed by |
|---|---|---|
| Six assessment dimensions: right to liberty, digital inclusion, data governance, transparency, human oversight, security | Implemented | `aggregation.py::Dimension`, `DIMENSION_ORDER` |
| Each dimension carries a 0–10 score, the mean of its **measured** requirement results | Implemented | `aggregation.py::score_dimension`. Attestation-only, no-data and low-coverage results are excluded rather than averaged in as zeros. A dimension with nothing measured scores `null`, never 0.0. Each score carries the counts it was computed from. |
| A single readiness score over the six dimensions, emitted only when all six scored | Implemented **as a withholding rule** | `aggregation.py::readiness_from_scores`. Otherwise `null`, with the missing dimensions named. Weighting is equal and held as data (`DIMENSION_WEIGHT`). |

### Validity

| Claim | Status | Backed by |
|---|---|---|
| Our implementations reproduce published results for CrowS-Pairs and WinoBias, within a documented tolerance | Implemented | `validity.py::VALIDITY_TARGETS`; `docs/methodology.md`; `pytest -m validity` |
| StereoSet is **not** claimed to reproduce | Implemented (as a documented exclusion) | `validity.py`, `excluded_reason`. The published figure is measured on a test split that was never publicly released; HuggingFace ships validation only. |

### Not yet built

| Claim | Status | Backed by |
|---|---|---|
| Local, pinned toxicity classifier, so safety scores reproduce exactly | **Target architecture** | ⚠️ `toxicity_classifier.py` calls the HuggingFace Inference API today, with a heuristic fallback. Results record which ran, but a remote model can change under you. |
| Fairness metrics: demographic parity, equalized odds | **Target architecture** | ⚠️ No backing code. |

### Packaging and release

| Claim | Status | Backed by |
|---|---|---|
| The evaluation toolkit is released open source under Apache-2.0 | Implemented | `LICENSE`; SPDX headers throughout |
| Python 3.10, 3.11 and 3.12 are supported | Implemented | `pyproject.toml`; CI matrix across three platforms |
