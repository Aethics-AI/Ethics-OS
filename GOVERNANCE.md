# Governance

How decisions get made, who can merge, and how a contributor becomes a
maintainer.

> [!IMPORTANT]
> **This document is provisional.** It describes how the project actually works
> today, which is not the same as how AETHICS has decided it should work. Two
> items below are open decisions marked **OPEN**, and they belong to AETHICS
> rather than to whoever writes this file.
>
> They were written down as questions rather than guessed at, because a
> governance document that invents its own authority is worse than one that
> admits it is unfinished.
>
> Two have since been decided by AETHICS and are recorded below as decisions:
> the **licence** (Apache-2.0) and **contributor agreements** (DCO). What
> remains open is the maintainer model — who is a maintainer, how someone
> becomes one, and who decides. That one still says OPEN because it still is.

## Current state, honestly

The project is pre-1.0, developed inside AETHICS, and has not yet accepted an
external contribution. Everything below reflects a small internal team, and
should be revisited before the repository is public.

**Something worth stating plainly:** as of writing, PRs in this repository have
been merged without review. That is how the CLI and the result schema drifted
apart without either author noticing, and how 34 files reached the shared branch
with no licence header. The review requirement in the next section is a response
to that, not boilerplate.

## Decisions

| Decision | Made by | How |
|---|---|---|
| Bug fixes, docs, tests, refactors | Any maintainer | PR + one approving review |
| New benchmark task | Maintainers | Issue first ("New benchmark task"), then PR |
| **Methodology change** (how a score is computed) | Maintainers, with the reasoning written down | Issue + PR; the PR must state what the number meant before and after |
| Public API change | Maintainers | PR, with the schema compatibility policy applied |
| Result schema change | Maintainers | Per [docs/result-schema-compatibility.md](docs/result-schema-compatibility.md) — SemVer, and the drift test enforces regeneration |
| Licence, trademark, publishing | **AETHICS** | Not a maintainer decision |

A methodology change deserves the extra ceremony because it silently changes
what past numbers meant. Rescaling a metric breaks nothing loudly, which is
precisely what makes it dangerous.

## Merging

- **No self-merges without review.** At least one approving review from someone
  who did not write the change.
- **CI must be green.** The `CI passed` check is required; it aggregates lint,
  types, licence headers, the import contract, and the test matrix.
- **Do not weaken a gate to merge.** Lowering the coverage floor or adding a
  suppression to get green is the same act as deleting a failing test. If the
  gate is wrong, change it deliberately, in its own PR, with the reasoning.

**OPEN — who exactly holds merge rights on the public repository, and does the
"no self-merge" rule apply to AETHICS staff?** The current answer is "the
internal team", which will not survive contact with outside contributors.

## Maintainers

**OPEN — the maintainer list, and how someone joins it.**

There is no published list yet. Before the repository is public this needs: who
is a maintainer, what a contributor has to do to become one, and who decides.
Absent that, "maintainers" above means "AETHICS staff on the project", which is
not a governance model, just a description.

## Contributor agreements: DCO

**Decided: DCO.** Every commit carries a `Signed-off-by:` line, enforced by a CI
check. See [CONTRIBUTING.md](CONTRIBUTING.md) for the mechanics.

A CLA was the alternative. It would have preserved our freedom to relicense
later, at the cost of a signing flow that measurably deters drive-by
contributions — which, for a project whose most valuable outside input is a
methodology correction from someone who read the source once, is the wrong
trade.

The cost is real and worth naming: **the project cannot be relicensed without
the agreement of everyone who has contributed.** That is accepted deliberately,
not overlooked.

Nothing is applied retroactively. The check runs on pull requests opened from
the point it landed; earlier contributions are not being revisited.

## Licence

**Apache-2.0**, declared in `pyproject.toml` and carried as an SPDX header on
every source file. Chosen for the express patent grant.

Contributions are accepted under the DCO (see [CONTRIBUTING.md](CONTRIBUTING.md)),
which means contributors retain their copyright and license their work under
Apache-2.0 rather than assigning it. A consequence worth stating plainly: the
project cannot be relicensed without the agreement of everyone who has
contributed. That is a deliberate constraint, not an oversight.

Dataset licences are a separate matter and are documented in
[docs/datasets.md](docs/datasets.md).

## Changing this document

Same as any other change: a PR, reviewed. The two **OPEN** items are the
exception — those need an answer from AETHICS first, and a PR that fills one in
should say who decided and when.
