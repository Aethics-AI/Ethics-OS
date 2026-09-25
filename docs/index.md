# aethics-eval

Reproducible bias and safety evaluation for language models.

!!! warning "Alpha"

    Version `0.1.0`. The API is not stable and the package is not on PyPI.
    Pin an exact revision if you depend on it.

Four published bias benchmarks — **WinoBias, StereoSet, CrowS-Pairs and
BOLD** — with provenance and honest reporting treated as features rather than
paperwork.

## Start here

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)** — install to a verified result in about five minutes
- **[Model providers](model-providers.md)** — what each provider can and cannot measure
- **[Writing a task](writing-a-task.md)** — add your own benchmark
- **[Result schema](result-schema.md)** — every field, and what it does not mean
- **[Methodology](methodology.md)** — what each benchmark measures, and how close we get to the published numbers
- **[Aggregation](aggregation.md)** — how requirement results become dimension scores and a readiness score

</div>

## The three ideas

**A measurement that failed is not a score of zero.** A model with no
log-probabilities cannot be scored on CrowS-Pairs, so the result says `null` and
the task is excluded from aggregates. Conflating "could not measure" with
"measured badly" is the single easiest way for an evaluation tool to mislead.

**A number should be re-checkable.** Per-sample scores are hashed into a per-task
Merkle tree; `aethics verify` recomputes the whole structure from the evidence.
Editing a sample and adjusting the totals does not survive it, because the
totals are not what is hashed.

**Uncertainty belongs next to the estimate.** The confidence interval is printed
beside the score, not buried. At the default sample size it is wide enough to
make the point: the [quickstart](quickstart.md) shows the same model scoring
0.64 on 50 pairs and 0.577 on 1,508.

## What this is not

Not a general evaluation framework. Four bias benchmarks, deliberately narrow.
For broad capability evaluation use
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) or
[HELM](https://crfm.stanford.edu/helm/) — both far more mature and much wider in
scope.

Come here when the provenance of a number matters as much as the number.
