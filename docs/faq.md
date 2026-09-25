# FAQ

## Why does my task say `not_measured` instead of giving a score?

Because it could not be measured, and saying `0.0` would be a lie.

The usual cause is a provider that cannot supply token log-probabilities.
`winobias`, `stereoset` and `crows_pairs` all score by comparing the likelihood
a model assigns to two sentences; without log-probs there is nothing to compare.
Only `local:` supplies them today — see [model providers](model-providers.md).

A `not_measured` task is excluded from `mean_score` rather than counted as zero,
so one unmeasurable benchmark cannot drag an average down and look like a
finding.

## Can I score an OpenAI / Anthropic / hosted model?

For `bold`, yes — use `openai:` with `--base-url`. For the other three, no, and
nothing can fix that from our side: chat APIs do not return the log-probabilities
those benchmarks need. You will get one measured task and three honest
`not_measured` rows.

If that is unsatisfying, it is worth sitting with — "how biased is this API" is
substantially harder to answer than "how biased is this open-weight model", and
tools that return a confident number for both are usually hiding the difference.

## Why is there no `--api-key` flag?

Because command-line arguments end up in shell history and are visible to
anything that can run `ps`, and keys in config files get committed. An
environment variable is the only one of the three not routinely written down.

Set `AETHICS_API_KEY` (or `OPENAI_API_KEY`). A test asserts the flag does not
exist, so it cannot reappear by accident.

## `hf:` gives me "Insufficient domain diversity" — what did I do wrong?

Almost certainly nothing about domains. That message usually means every
generation failed, and the most common cause is a missing or invalid
`HF_TOKEN` — unauthenticated requests get a `401`.

Note the variable is `HF_TOKEN`. `HUGGINGFACE_TOKEN` is read by nothing.

The misleading diagnosis is a known bug; the run is still honest in that it
reports `not_measured` rather than inventing a score.

## Two runs of the same command gave different numbers. Is it broken?

Check the sample count first. At the default `--limit` (30–50) the confidence
intervals are wide enough that ordinary resampling noise moves the headline
figure noticeably. The [quickstart](quickstart.md) shows GPT-2 scoring 0.64 at
50 pairs and 0.577 at 1,508 — the small run was not a preview of the large one.

If the sample count and `--seed` are identical and the numbers still differ,
that is a genuine bug. Compare `manifest.run_hash`: identical inputs should give
an identical hash, and if they do not, please report it.

## What is the difference between `validate` and `verify`?

`validate` asks *is this file well-formed and honestly reported* — envelope
complete, provenance present, no score asserted without evidence behind it.

`verify` asks *is this the result that was actually produced* — it re-hashes
every per-sample score, rebuilds each task's Merkle root, and recomputes the run
hash. Editing a sample and adjusting the totals to match does not survive it,
because the totals are not what is hashed.

Each leaf covers the sample id, its score, whether it was measured, and a digest
of the model's response. Including the response matters: without it the chain
proved the scores added up, but not that they described the answers stored
beside them, so a recorded response could be swapped while every hash stayed
valid.

What `verify` does **not** do: it cannot tell you the scoring was *correct*, only
that nothing has changed since the run. A wrong scorer produces a perfectly
verifiable wrong answer. Correctness is what `docs/methodology.md` and the
nightly validity job are for.

Both exit `0` on success and `1` on failure.

## Does `validate` check the published JSON Schema?

Yes. `validate` checks the file against `schemas/result-v1.json` and then
applies the honesty rules a JSON Schema cannot express: a score must have
samples behind it, a measured metric must carry uncertainty, and a mean must
not be reported over nothing measured.

It previously checked only the CLI's own `0.1.0` envelope, so a file that did
not satisfy the published schema was still reported "valid".

## Why does a refusing model score 1.0 on `bold`?

Because it never produces the toxic continuation the benchmark looks for. That
is a correct measurement and a useless model.

It is the clearest illustration of why a benchmark number is not a safety
verdict: the way to score perfectly on a toxicity benchmark is to say nothing at
all.

## Why does `score` disagree with the rate in `details`?

`score` is normalised so that 1.0 is ideal behaviour, and for the paired
benchmarks ideal means *no preference* — a rate of 0.5. So
`score = 1 - |rate - 0.5| * 2`, and a rate of 0.577 becomes 0.845.

The raw rate is the interesting number. The score is the comparable one.

Separately, `scored_metric.value` is the same figure on a 0–100 scale, so
`0.8455` and `84.55` are one number twice.

## Can I use these datasets in my own project?

Read [datasets.md](datasets.md) first. **Three of the four are ShareAlike**
(CC-BY-SA-4.0), which imposes obligations on derived work. That is why this
package ships no benchmark data and loads everything at runtime — vendoring it
would push those obligations onto you.

Every licence was checked against the authors' original source rather than the
HuggingFace card. At least one card was wrong.

## Why is `p_value` insignificant when the confidence interval says otherwise?

They are measuring different things. The interval is a bootstrap over the
per-pair outcomes and describes the headline rate. `p_value` and `cohens_d` are
computed on raw sentence log-probabilities and test whether the two sentence
sets differ in mean log-likelihood.

Use the interval for "is this preference real". The adjacency is a known
reporting flaw. See [result schema](result-schema.md).

## Is my custom task supposed to be missing from `aethics list-tasks`?

Unfortunately yes, for now. Plugin tasks register with the library correctly and
run through the Python API, but the CLI still reads a hardcoded table of the four
built-ins. Known gap — see [writing a task](writing-a-task.md).

## What licence is this under?

Apache-2.0. The choice was the patent grant: MIT is shorter and marginally
friendlier to casual contributors, but it grants no patent rights, and that is
the first question enterprise counsel asks about a tool being adopted into a
compliance workflow.

It carried `NOASSERTION` through the alpha — the SPDX identifier for *no licence
granted*, which was the honest label while the decision was open, since claiming
a licence not yet agreed would have been worse than admitting none.

Dataset licences are separate and unaffected; see [datasets](datasets.md).

## Why isn't it on PyPI?

Publishing is a later ticket (POS-9). Install from a checkout meanwhile; see the
[quickstart](quickstart.md).

## Do I need a GPU?

No. The GPT-2 runs in these docs were done on a laptop CPU — about 2 minutes for
the full 1,508-pair CrowS-Pairs set. Larger models will want one.
