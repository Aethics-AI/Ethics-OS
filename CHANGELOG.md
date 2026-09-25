# Changelog

All notable changes to `aethics-eval` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from `0.1.0` onward. Pre-`1.0`, the public API may change in a MINOR release;
the **result schema** is versioned separately and more strictly — see
[docs/result-schema-compatibility.md](docs/result-schema-compatibility.md).

## [Unreleased]

Nothing released yet. Everything below is in `0.1.0.dev0` and has not been
published to PyPI.

### Added

- **`aethics_eval.aggregation`**: how requirement results roll up into six
  dimension scores (0–10) and a single readiness score. Only measured results
  count; a dimension with nothing measured scores `null`, and readiness is
  withheld unless all six dimensions scored. The requirement catalogue is an
  input, not part of the package. See [docs/aggregation.md](docs/aggregation.md).
- **Apache-2.0 licence.** `LICENSE`, `NOTICE`, PEP 639 metadata
  (`License-Expression: Apache-2.0`), the OSI classifier, and an SPDX header on
  every source file. Chosen over MIT for the express patent grant. The package
  previously declared `NOASSERTION`, and the release pipeline refused to publish
  in that state.
- **DCO sign-off for contributions**, enforced in CI. Contributors keep their
  copyright and license their work under Apache-2.0; there is no CLA and no
  assignment.
- **The `aethics` command line** — `eval`, `list-tasks`, `cache`, `validate`,
  `verify` and `show`. Results go to stdout and progress to stderr, so output
  pipes cleanly.
- **Run manifest and verification.** Per-sample scores are hashed into a
  per-task Merkle tree; `aethics verify` recomputes the whole structure from the
  evidence. Editing a sample and adjusting the totals does not survive it,
  because the totals are not what is hashed.
- **`--seed`**, recorded in the manifest, so a run can be repeated.
- **Task registry and plugin discovery.** A benchmark is a `Task` with four
  methods; third-party packages register via the `aethics_eval.tasks`
  entry-point group. See [docs/writing-a-task.md](docs/writing-a-task.md).
- **Versioned result schema** (`RunResult`), published as
  `schemas/result-v1.json`, with a typed loader that refuses a newer MAJOR
  version rather than parsing it halfway.
- **Uncertainty on every metric** — standard error, seeded bootstrap intervals,
  paired significance tests, and an `underpowered` flag. `aethics show` renders
  `0.61 ± 0.04`.
- **Pinned, cached, honest dataset loading.** Every dataset is fetched at a
  fixed revision and cached; the loader never substitutes different data when
  the pinned revision is unavailable.
- **Documentation site** (MkDocs Material) — quickstart, model providers,
  writing a task, result schema, datasets, FAQ.
- **CI** across Python 3.10/3.11/3.12 on Linux, macOS and Windows, with lint,
  formatting, types, licence headers, the import boundary contract, and a
  coverage floor that ratchets up.
- **Community documentation** — this file, plus contributing, governance,
  security and code of conduct.

### Changed

- **Benchmark scoring now uses likelihood, not generation length.** CrowS-Pairs
  and StereoSet compare the log-probability a model assigns to paired sentences.
  The previous length-based proxy sat near chance and did not reproduce
  published figures.
- **WinoBias scoring preserves the pronoun**, which is the entire point of the
  matched pro/anti design. The earlier gender-blind method reported a zero gap
  by construction.
- **`aethics_eval` is a standalone package** (`src/` layout), extracted from the
  AETHICS application. An import-linter contract enforces that it never imports
  the product it came from.

### Fixed

The findings that motivated the extraction. Each is a case of a number being
reported that was not measured:

- **Fabricated scores from zero data.** A missing `repetition_penalty` made every
  inference call raise, leaving audits with no responses and default scores.
  Scores are now `null` when nothing was measured.
- **Neutral defaults for unmeasured results.** A model-derived score with no data
  behind it now reports `null` rather than a midpoint.
- **Aggregates counted unmeasured sub-tests as zero.** They are excluded.
- **Refusal detection could not tell a refusal from an apology.** "Sorry, I can't
  help" and "Sorry, here's how to..." are now distinguished, and every judgement
  is stamped with the method that produced it (`classifier` / `keyword` /
  `not_measured`) so a keyword score never wears a classifier's badge.
- **Transient throttling (429/503) was treated as a failed measurement.** Now
  retried with backoff.

### Security

- **API keys are never accepted as a command-line flag** — `argv` is visible in
  shell history and to `ps`, and config files get committed. Keys are read from
  `AETHICS_API_KEY` or `OPENAI_API_KEY` only, and a test asserts the flag cannot
  be reintroduced.
- **Non-200 HTTP responses report a status code only**, because auth error
  bodies sometimes echo the submitted key back and result files get shared.

### Known issues

Documented rather than hidden; each is tracked:

- **`ScoredMetric.value` and `confidence_interval` are not on the same scale**
  and describe different quantities — e.g. `value: 84.55` beside
  `[0.5537, 0.6025]`. Do not render "value ± interval" from it. Use
  `details.confidence_interval_95` against `details.stereotype_preference_rate`.
- **`p_value` and `cohens_d` are computed on raw log-probabilities**, not on the
  metric they sit beside in the result.
- **Plugin tasks are invisible to the CLI.** They register and run through the
  library, but `aethics list-tasks` and `aethics eval` still read a hardcoded
  table of the four built-in benchmarks.
- **The CLI emits its own result envelope** (`schema_version 0.1.0`), not the
  published `RunResult` shape. `aethics validate` checks the CLI envelope, not
  `schemas/result-v1.json`.
- **The `hf:` provider requires `HF_TOKEN`**, and a missing token surfaces as a
  dataset-diversity error rather than an authentication failure.
- Nothing is on PyPI yet; the name `aethics-eval` is unregistered until the
  first publish claims it.

[Unreleased]: https://github.com/Aethics-AI/Ethics-OS/commits/main
