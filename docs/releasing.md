# Releasing

How a version gets cut, what the version number promises, and what it does not.

## Stability: what `0.1.0` means

**This is alpha software. Pin an exact version.**

`0.1.0` is the first release, not a stable one. Concretely:

| | Promise |
|---|---|
| **Python API** | May change in any release before `1.0.0`. Functions can be renamed, arguments can change meaning. |
| **CLI flags and output** | May change. Do not parse the human-readable `show` output; use the JSON. |
| **Result schema** | **Stronger promise** — versioned separately and governed by SemVer from day one. See [result-schema-compatibility.md](result-schema-compatibility.md). |
| **Scores themselves** | A methodology change can alter what a number *means*. When that happens the methodology fingerprint changes, so results from before and after are distinguishable rather than silently mixed. |

That last row is the one that matters most and gets least attention. An
evaluation library that quietly rescales a metric breaks nothing loudly: your
dashboards keep working, your numbers keep arriving, and they mean something
different. The fingerprint exists so you can detect it; compare
`provenance.methodology_fingerprint` between runs before comparing scores.

Under SemVer, `0.x` versions are permitted to break things in a MINOR bump, and
this project will use that latitude while the API settles. If you need
stability now, pin the exact version:

```
aethics-eval==0.1.0
```

not `>=0.1.0`.

### What is not implemented

Stated plainly, because a version number does not convey it:

- Three of the four benchmarks require token log-probabilities and cannot run
  against chat APIs. They report `not_measured`, honestly, but they do not
  produce a number.
- Plugin tasks register with the library but are invisible to the CLI.
- The CLI emits its own result envelope, not the published `RunResult` schema.

See the Known issues section of [CHANGELOG.md](https://github.com/Aethics-AI/Ethics-OS/blob/main/CHANGELOG.md).

## Versioning

The version lives in **one place**: `__version__` in
`src/aethics_eval/__init__.py`. `pyproject.toml` declares it dynamic and
hatchling reads it from there.

It used to be written in both files, which is a release hazard rather than an
untidiness — they drift, and then the wheel's metadata describes a different
version from the code inside it. `tests/test_release_metadata.py` fails if
anyone reintroduces a static version.

Pre-releases use PEP 440 suffixes (`0.2.0rc1`, `0.2.0a1`). The pipeline treats
anything with a pre-release or `.dev` suffix as **TestPyPI only** — it will not
reach PyPI, so an rc cannot be released by accident.

## Cutting a release

1. Update `__version__`.
2. Move the `[Unreleased]` section of `CHANGELOG.md` under the new version with
   a date.
3. Merge that to the default branch.
4. Tag and push:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

The tag must match `__version__` exactly. If it does not, the pipeline stops
before building — a tag that disagrees with the package publishes something
other than what the tag claims, and there is no good way to guess which one was
intended.

### Dry run

Two ways, both of which build, check metadata, install the wheel in a clean
environment, smoke-test the CLI, generate the SBOM, and then stop:

- Run the **Release** workflow manually with `dry_run` ticked (the default).
- Or push a tag ending in `-dryrun`:

  ```bash
  git tag v0.1.0-dryrun && git push origin v0.1.0-dryrun
  ```

The suffix is stripped before the tag is compared to the package version, so
`v0.1.0-dryrun` still verifies that the tag matches `0.1.0`.

The tag form exists because `workflow_dispatch` only appears once the workflow
is on the default branch, and because the first real tag is a bad moment to
discover the pipeline is broken. Dry-run tags are cheap; use them.

## What the pipeline does

```
guard → build → TestPyPI → install FROM TestPyPI → PyPI
```

**guard** checks the tag matches the version, decides whether this is a
pre-release, and enforces the licence gate.

**build** builds the wheel and sdist, runs `twine check --strict`, then installs
the wheel into a clean virtual environment *outside the source tree* and runs
the quickstart against it. An editable install hides packaging mistakes — a
module missing from the wheel still imports from the source directory — so the
only meaningful test is one where the source is not reachable.

**install from TestPyPI** is the step that matters, and is separate from
publishing on purpose. Building a wheel proves it builds. Installing it from an
index in a clean environment proves it is usable, which is a different claim.

**PyPI** runs last, only for non-pre-release versions, behind a GitHub
Environment that can require a human approval.

## The licence gate

**The pipeline refuses to publish while the declared licence is
`NOASSERTION`.** It is now `Apache-2.0`, so the gate passes — but the gate stays,
because the state it guards against is reachable again by editing one line.

`NOASSERTION` is the SPDX identifier for "no licence has been granted" — not
public domain, not permissive. Publishing in that state distributes code that
nobody has permission to use, and unlike almost everything else in this
pipeline it cannot be undone: PyPI does not allow deleting a version, only
yanking it, and a yanked version is still downloadable.

The gate is a hard failure rather than a warning for that reason.

Dry runs are exempt, so the pipeline can be developed and tested before the
licence question is settled.

## Trusted publishing (OIDC)

There are **no API tokens in repository secrets**. Publishing uses PyPI's
trusted publishing: PyPI is configured to trust a specific repository, workflow
file and environment, and mints a short-lived token for each run. A long-lived
token stored in secrets is a credential that leaks eventually, and it grants
upload rights to anyone who obtains it.

**This needs configuring once, per index, before the first publish** — it is not
something the workflow can do for itself:

| Setting | Value |
|---|---|
| Owner | `Aethics-AI` |
| Repository | `Ethics-OS` |
| Workflow | `release.yml` |
| Environment | `testpypi`, and separately `pypi` |

Set up at <https://pypi.org/manage/account/publishing/> and the TestPyPI
equivalent. Both require the project name to be registered first, which is why
reserving `aethics-eval` is a prerequisite rather than a nicety.

Add a required reviewer to the `pypi` environment. That is the last human gate
before something becomes permanent.

## SBOM

Each release generates a CycloneDX SBOM (`sbom.cyclonedx.json`), attached as a
build artifact rather than uploaded to the index.

It is generated from the **runtime** environment — the wheel installed with its
dependencies and nothing else. An SBOM produced from a development environment
lists pytest, ruff and mypy as though the shipped package depended on them,
which is false, and an SBOM's entire value is being accurate about what you are
shipping. The build asserts those tools are absent.

The SBOM records `Apache-2.0` for `aethics-eval` itself, so consumers running
licence-compliance tooling see a properly licensed component. While the licence
was `NOASSERTION` it recorded none, and those consumers saw an unlicensed
dependency — which is what an unassigned licence actually looks like downstream.
