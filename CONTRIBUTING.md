# Contributing to aethics-eval

Thanks for considering it. This document covers getting set up, what we expect
in a pull request, and how to propose a new benchmark.

> **Alpha.** The API is not stable, the package is not yet on PyPI, and no
> licence has been assigned. Some of what follows will change.

## Setting up

```bash
git clone https://github.com/Aethics-AI/Ethics-OS.git
cd Ethics-OS
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Python 3.10, 3.11 and 3.12 are supported and tested on Linux, macOS and Windows.

Local open-weight inference (needed to run the log-probability benchmarks) is a
separate extra, because it pulls PyTorch and costs about 2 GB:

```bash
pip install -e ".[local]"
```

You do not need it to work on most of the codebase. If you are changing scoring
or benchmark logic, you probably do.

## Running the checks

Everything CI runs, in the order it runs it:

```bash
ruff check .                    # lint
ruff format --check .           # formatting
mypy --config-file pyproject.toml
python scripts/apply_spdx_headers.py --check
lint-imports                    # the SDK must not import the parent product
pytest
```

`pytest` alone is fine while iterating. Run the rest before opening a PR — CI
will tell you anyway, but it is a slower way to find out.

Two of these are worth explaining, because they look like style rules and are
not:

- **`lint-imports`** enforces that `aethics_eval` never imports the FastAPI
  application it was extracted from. That boundary is the reason this package
  can exist standalone. A failure here is a design problem, not a lint nit.
- **`apply_spdx_headers.py`** checks every source file carries its licence
  header. Files without one are a legal problem for anyone redistributing.

## Pull request expectations

**Say what you verified, not that it works.** "Ran the suite, 273 passed" beats
"tested". If you could not test something, say which part and why — that is
useful information, not an admission.

**One concern per PR.** A bug fix plus a rename plus a refactor is three PRs
that arrived together and will be reviewed as none of them.

**Tests for behaviour changes.** Especially for the honesty rules below; those
have regressed before, quietly.

**Don't weaken a gate to make it pass.** Lowering the coverage floor, adding a
lint suppression, or deleting an assertion to get green is the same act as
turning the check off. If a gate is wrong, say so in the PR and argue it.

### The honesty rules

These are the ones we care most about, and the ones most easily broken by
accident:

- **A measurement that did not happen reports `None`, never `0.0`.** A failed
  measurement and a score of zero are different claims about a model.
- **Unmeasured results are excluded from aggregates**, not counted as zeros.
- **A proxy is labelled a proxy.** `reliability` travels with every number so a
  keyword heuristic is never mistaken for a classifier.
- **Uncertainty ships with the estimate.** A bare point estimate invites
  over-reading; the docs show the same model scoring 0.64 on 50 samples and
  0.577 on 1,508.

If your change makes one of these harder to hold, that is worth discussing in
the PR before the code review starts.

### Commit messages

Explain *why*, not *what* — the diff already says what. If you found something
surprising on the way, put it in the message. The next person to touch that
code is likely to hit the same surprise.

### Sign your commits (DCO)

Every commit must carry a `Signed-off-by:` line. `git commit -s` adds it:

```
Signed-off-by: Jane Doe <jane@example.com>
```

That line is the [Developer Certificate of Origin](https://developercertificate.org/)
— a statement that you wrote the change, or have the right to submit it under
the project licence. It is not a copyright assignment: **you keep your
copyright** and license the contribution under Apache-2.0. There is no CLA and
nothing to sign separately.

DCO was chosen over a CLA deliberately. A CLA would preserve our freedom to
relicense later, at the cost of a signing flow that deters exactly the
drive-by methodology fixes this project most wants. The consequence is worth
stating plainly: the project cannot be relicensed without every contributor's
agreement, and we consider that an acceptable constraint.

A CI check enforces the sign-off. If you forget it:

```bash
git commit --amend -s          # last commit
git rebase --signoff origin/main   # a whole branch
```

## Proposing a new benchmark task

The mechanism is documented in [docs/writing-a-task.md](docs/writing-a-task.md).
A task is a class with four methods, and it can live in your own package — you
do not need to change this repository to add one.

Before writing code, **open an issue using the "New benchmark task" template**.
Adding a benchmark is a methodological claim as much as a technical one, and it
is much cheaper to discuss the claim before the implementation exists. The
template asks the questions we will ask anyway:

- What does the benchmark measure, and what published work defines it?
- What licence is the dataset under, and is it redistributable?
- What does a score mean — and what does it *not* mean?
- What happens when a model cannot be measured on it?

That last one is not a formality. Three of our four benchmarks require token
log-probabilities and simply cannot run against a chat API; a task that quietly
substitutes a weaker method when the real one is unavailable is the failure mode
this project exists to avoid.

### Dataset licensing

If your task loads a dataset, check its licence **against the authors' original
source**, not the HuggingFace card. We found a card claiming CC-BY-4.0 for a
dataset the authors publish under CC-BY-SA-4.0 — a difference that decides
whether it can be vendored at all.

Three of our four datasets are ShareAlike, which is why this package ships no
benchmark data and loads everything at runtime. See [docs/datasets.md](docs/datasets.md).

## Reporting a methodology problem

If you think a score is wrong — not the code, the *method* — please open a
"Methodology challenge" issue. That template exists because we would rather hear
it from you than have a customer find it. Scrutiny of the measurements is the
most valuable contribution this project can receive.

## Security

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).

## Code of conduct

Participation is covered by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
