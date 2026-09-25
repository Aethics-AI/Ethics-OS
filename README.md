# Ethics-OS

The open scoring engine behind AETHICS AI certification, packaged as
`aethics-eval`: reproducible bias and safety evaluation for language models.

AETHICS certifies AI systems against regulations such as the EU AI Act. The
certification platform is a hosted service. The **scoring** it relies on, meaning
how a model's behaviour becomes a number, is published here so anyone can read it,
run it and challenge it.

> **Status: alpha (`0.1.0`).** The API is not stable and the package is not
> on PyPI yet. The scoring is the same code the certification platform runs,
> but pin an exact revision if you depend on it.

## What's open, and what isn't

| Open, in this repository | Closed, in the certification platform |
|---|---|
| Benchmark implementations: WinoBias, StereoSet, CrowS-Pairs, BOLD | The requirement catalogue: which legal provisions we assess, and against which thresholds |
| Safety scoring: toxicity and refusal classification | How measurements map onto those requirements |
| Statistics: confidence intervals, effect sizes, significance tests, power checks | Resolving conflicts between frameworks |
| Aggregation: six dimension scores (0–10) and the readiness score | Certification decisions and issued certificates |
| Evidence hashing and independent verification of any result | Dashboard, accounts, billing, hosted API, customer data |
| The result schema the platform consumes | |

The line is deliberate. **How we measure and how scores combine** is public,
because a measurement nobody can inspect is a claim, not evidence. **Which laws
a measurement counts towards** is our product.

## What this is

A small evaluation library for four published bias benchmarks — **WinoBias,
StereoSet, CrowS-Pairs and BOLD** — that treats *provenance* and *honest
reporting* as features rather than paperwork.

Four things it does that motivated writing it:

- **A task that cannot be measured reports `null`, never `0.0`.** A model with
  no log-probabilities cannot be scored on CrowS-Pairs, and the result says so.
  Unmeasured tasks are excluded from aggregates rather than counted as zeros.
- **Every run is hashable and re-checkable.** Per-sample scores are hashed into
  a per-task Merkle tree; `aethics verify` recomputes the whole thing from the
  evidence. Editing a sample and fixing up the totals does not survive it,
  because the totals are not what is hashed.
- **Datasets are pinned to a revision and carry their licence.** `aethics
  list-tasks` prints both. No benchmark data is vendored — three of the four
  datasets are ShareAlike, so redistributing them would impose that licence on
  your project.
- **Scores roll up without inventing anything.** Requirement results become a
  0–10 score for each of six dimensions (right to liberty, digital inclusion,
  data governance, transparency, human oversight, security) from *measured*
  results only. The single readiness score is withheld, with the missing
  dimensions named, unless all six were measured. See
  [aggregation](docs/aggregation.md).

## Install

```bash
pip install aethics-eval
```

**Not yet — that command does not work today.** The name is not on PyPI, and
is not reserved either. Until it is, install from source:

```bash
git clone https://github.com/Aethics-AI/Ethics-OS.git
pip install ./Ethics-OS
```

Python 3.10–3.12, tested on Linux, macOS and Windows. The core install pulls no
PyTorch. Scoring local open-weight models needs the extra:

```bash
pip install "aethics-eval[local]"   # ~2 GB, adds torch + transformers
```

## Quickstart

Everything here runs on the core install — no model download, no GPU, no API
key. The `fake:` provider returns the text after the colon for every prompt,
which is enough to exercise the whole pipeline end to end:

```bash
# what you can run, with licences and whether each needs logprobs
aethics list-tasks

# score a model and write the result plus its evidence
aethics eval --model "fake:I cannot help with that." --tasks bold --limit 5 -o out.json

aethics show out.json      # human-readable table
aethics verify out.json    # recompute the hashes from the evidence
```

```console
$ aethics show out.json
model        : fake:I cannot help with that.
methodology  : 3.0.0
fingerprint  : 266b57eaa394ea0b...

TASK                      SCORE  SAMPLES  NOTE
--------------------------------------------------------------------
bold                 1.0000 ± 0        5  underpowered
--------------------------------------------------------------------
mean                     1.0000  (1/1 measured)
```

Note the `underpowered` flag: five samples cannot support a claim, and the tool
says so rather than letting you quote the number. That is the point of the
project more than any individual score.

### Then a real model

Scoring open weights locally needs the `[local]` extra (~2 GB, adds torch):

```bash
pip install "aethics-eval[local]"
aethics eval --model local:gpt2 --tasks crows_pairs --limit 200 -o gpt2.json
```

Or point it at any OpenAI-shaped endpoint — vLLM, Together, or the API itself —
with no extra install: `--model openai:gpt-4o` (needs `OPENAI_API_KEY`). See
[model providers](docs/model-providers.md).

Full walkthrough: **[docs/quickstart.md](docs/quickstart.md)**.

## An honest example

GPT-2 on the full CrowS-Pairs set, on a laptop CPU. This is a real run, not an
illustration:

```console
$ aethics eval --model local:gpt2 --tasks crows_pairs --limit 1508 -o gpt2.json
  crows_pairs: running (limit=1508)...
  crows_pairs: score=0.8455
```

```json
{
  "stereotype_preference_rate": 0.5773,
  "confidence_interval_95": [0.5537, 0.6025],
  "total_pairs": 1508
}
```

GPT-2 preferred the more-stereotypical sentence in **57.7%** of pairs, where
50% would mean no preference. The 95% interval is [0.554, 0.603] — it excludes
0.5, so the effect is real, and it overlaps the figure the CrowS-Pairs authors
report for GPT-2, which suggests the scoring reproduces the published method
rather than merely producing a number. Two minutes, no GPU.

Now the part that usually gets left out. The same command at `--limit 50`:

```json
{
  "stereotype_preference_rate": 0.64,
  "confidence_interval_95": [0.5, 0.78],
  "total_pairs": 50
}
```

**0.64 — a worse-looking model, from the same weights.** The interval touches
0.5, so that run establishes nothing at all. The small sample was not a preview
of the large one, it was noise, and the number moved 0.06 in the opposite
direction once there was enough data to say anything.

This is why the confidence interval is printed next to the score instead of
buried in the JSON, and why the default `--limit` of 30–50 is documented as a
smoke test rather than a result.

## How it compares

`aethics-eval` is small and narrow. It is not a general evaluation framework
and is not trying to be:

| If you want… | Use |
|---|---|
| Broad capability benchmarks, hundreds of tasks, a large community | **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** |
| A wide, multi-metric standardised evaluation across many models | **[HELM](https://crfm.stanford.edu/helm/)** |
| Four bias benchmarks with pinned data, per-run hashes you can recheck, and results that refuse to guess | **this** |

Being straight about the trade: those projects cover vastly more ground, are
far more mature, and have had many more eyes on them. If you need general model
evaluation, use one of them. Come here if the provenance of the number matters
as much as the number — audit trails, regulatory evidence, anything where
someone may later ask *how exactly did you get this*.

We have not benchmarked against them for correctness, and where the same
benchmark exists in both, theirs has been more widely validated. Treat any
disagreement as ours to explain.

## Library use

```python
from aethics_eval import EVAL_MANIFEST, bootstrap_confidence_interval

# Prove two runs shared a methodology.
print(EVAL_MANIFEST.fingerprint()[:16])  # 266b57eaa394ea0b

# The statistics used throughout, usable on their own.
lo, hi = bootstrap_confidence_interval([1, 0, 1, 1, 0, 1])
```

The bootstrap seeds itself internally, so the interval is deterministic for a
given input — you do not need to seed anything, and `--seed` does not change
it. `--seed` governs which samples are drawn, not the resampling inside the
statistics.

## Documentation

| | |
|---|---|
| [Quickstart](docs/quickstart.md) | Install to verified result |
| [Model providers](docs/model-providers.md) | `local:` / `hf:` / `openai:` / `fake:`, and what each can measure |
| [Methodology](docs/methodology.md) | What each benchmark measures, reproduction results, deviations |
| [Aggregation](docs/aggregation.md) | How requirement results become dimension scores and a readiness score |
| [Writing a task](docs/writing-a-task.md) | Add your own benchmark |
| [Result schema](docs/result-schema.md) | Every field, and what it does not mean |
| [Datasets](docs/datasets.md) | Licences, citations, pinned revisions |
| [Claims register](docs/CLAIMS.md) | Every public claim we make, and the code behind it |
| [FAQ](docs/faq.md) | |

## Contributing

Issues and pull requests are welcome, especially ones that dispute a
measurement. See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues
privately as described in [SECURITY.md](SECURITY.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

CI runs lint, format, types, licence headers, the import contract and the test
suite across 3.10/3.11/3.12 × Linux/macOS/Windows.

The suite includes an import-linter contract asserting this package never
imports the application it was extracted from. That is not a style rule — it is
what keeps the library standalone.

## Licence

**Apache-2.0.** See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Apache-2.0 rather than MIT for the express patent grant, which is what
enterprise legal teams look for when adopting an evaluation tool into a
compliance workflow.

Dataset licences are a separate matter: this package redistributes no dataset,
and each is fetched from its original source under its own licence. Several are
CC-BY-SA-4.0, which is precisely why they are loaded rather than vendored — the
ShareAlike obligation does not propagate to this package or to yours. See
[docs/datasets.md](docs/datasets.md), and cite the benchmark authors if you
publish results.
