# Result schema

What `aethics eval` writes, field by field — and, just as importantly, what
some of those fields do **not** mean.

!!! info "One schema"

    `aethics eval` writes the `RunResult` model published as
    [`schemas/result-v1.json`](https://github.com/Aethics-AI/Ethics-OS/blob/main/schemas/result-v1.json),
    currently **`1.3.0`**. The same contract serves third-party tooling, the
    AETHICS certification platform, and the CLI, and `aethics validate` checks output against it.

    There were briefly two shapes: the CLI wrote its own envelope stamped
    `0.1.0`, which did not satisfy the published schema, while `validate`
    reported it "valid" because it was checking the other document. Tooling
    written against `result-v1.json` in that period still works — the change
    was additive.

    Versioning policy: [result-schema-compatibility.md](result-schema-compatibility.md).

## Top level

```json
{
  "schema_version": "1.2.0",
  "run_id": "9f2c...",
  "model":      { "id": "gpt2", "provider": "local", "params": { ... } },
  "manifest_fingerprint": "266b57ea...",
  "created_at": "2026-01-02T03:04:05Z",
  "completed_at": "2026-01-02T03:09:10Z",
  "tasks":      [ ... ],
  "summary":    { ... },
  "manifest":   { ... },
  "provenance": { ... }
}
```



```json
{ "mean_score": 0.8455, "tasks_measured": 1, "tasks_run": 1 }
```

`mean_score` averages **only measured tasks**, and is `null` when nothing was
measured. `tasks_measured` < `tasks_run` means something was skipped — always
worth looking at before quoting `mean_score`.

## `tasks.<task_id>`

```json
{
  "benchmark_name": "CrowS-Pairs",
  "score": 0.8455,
  "samples_tested": 1508,
  "passed": true,
  "threshold": 0.7,
  "details": { ... },
  "scored_metric": { ... }
}
```

| Field | Meaning |
|---|---|
| `score` | **`0.0`–`1.0`, higher is better. `null` when not measured — never `0.0`.** |
| `samples_tested` | Samples actually *measured*, not the number requested. |
| `passed` | `score >= threshold`. A convenience, not a verdict. |
| `threshold` | The bar `passed` used. |
| `details` | Benchmark-specific — the numbers you probably came for. |
| `scored_metric` | The same result in the canonical statistics shape. |

`score` is a normalised "closeness to ideal" figure, not the raw metric. For
CrowS-Pairs, ideal behaviour is a stereotype-preference rate of 0.5, so
`score = 1 - |rate - 0.5| * 2`. A rate of 0.577 gives 0.845. **The raw rate is
the interesting number; the score is the comparable one.** Read `details` for
the former.

### `details` — worked example

```json
{
  "stereotype_preference_rate": 0.5773,
  "confidence_interval_95": [0.5537, 0.6025],
  "total_pairs": 1508,
  "cohens_d": 0.0218,
  "p_value": 0.5332,
  "categories": { "race-color": 516, "gender": 262, "...": 0 },
  "data_source": "nyu-mll/crows-pairs (1,508 pairs via GitHub CSV)",
  "methodology": "Sequence log-likelihood — fraction of pairs where ...",
  "sample_results": [ ... ]
}
```

!!! danger "`p_value` and `cohens_d` do not describe the headline rate"

    `confidence_interval_95` is a bootstrap over the per-pair outcomes, so it
    **does** describe `stereotype_preference_rate`. Use it to ask "is this
    preference real".

    `cohens_d` and `p_value` are computed on the raw sentence
    log-probabilities. They test whether the two sets of sentences differ in
    mean log-likelihood — a different question. In the run above the interval
    excludes 0.5 (the preference is real) while `p_value` reads 0.53. Quoting
    that p-value as evidence about the rate would be wrong.

    This adjacency is a known reporting flaw and is being fixed.

`sample_results` holds the first five scored samples for eyeballing. It is a
sample of the evidence, not all of it — the full set is under `evidence`.

### `scored_metric`

```json
{
  "value": 84.55,
  "confidence_interval": [0.5537, 0.6025],
  "effect_size": 0.0218,
  "p_value": 0.5332,
  "sample_size": 1508,
  "reliability": "direct"
}
```

!!! danger "`value` and `confidence_interval` are not on the same scale"

    That block is copied verbatim from a real run, and it does not mean what it
    appears to mean.

    - `value` is `84.55` — the **score** (`0.8455`) on the canonical 0–100 scale.
    - `confidence_interval` is `[0.5537, 0.6025]` — the **preference rate**
      interval, left on its native 0–1 scale.

    Two different quantities, two different scales, in one object. `84.55` does
    not lie inside `[0.55, 0.60]`, and anything that renders "value ± interval"
    from this will produce nonsense.

    Until this is fixed, take the interval from
    `details.confidence_interval_95` and read it against
    `details.stereotype_preference_rate`, both of which are on the same 0–1
    scale and do describe each other.

`reliability` declares how the number was obtained: `direct` (measured as
intended), `derived` (computed from other measurements), `proxy` (a weaker
stand-in, e.g. keyword matching rather than a classifier), or `not_measured`.
Only `direct`, `proxy` and `not_measured` are currently emitted.

It travels with the number so a proxy can never be mistaken for a direct
measurement — the distinction the F7 finding was about.

## `evidence`

```json
{ "crows_pairs": [ { "sample_id": "0", "value": 1.0, "measured": true } ] }
```

One entry per sample. `measured: false` marks a sample the model did not
answer; those carry `value: 0.0` as a placeholder and are excluded from
aggregation — the flag, not the value, is what counts.

This is what `aethics verify` re-hashes.

## `manifest`

The POS-5 run manifest — everything needed to prove the run.

```json
{
  "manifest_version": "1.0.0",
  "run_id": "eb44e8774b4543b6b3a0c74614b67d80",
  "run_hash": "818009af327bcf1d...",
  "library_version": "0.1.0",
  "methodology_fingerprint": "266b57eaa394ea0b...",
  "model": { "spec": "local:gpt2", "params": { "temperature": 0.0, "...": 0 } },
  "seeds": { "seed": 42 },
  "datasets": [ { "name": "nyu-mll/crows-pairs", "revision": "8aaac11c...", "content_hash": null, "row_count": null } ],
  "tasks": [ { "task_id": "crows_pairs", "evidence_root": "492fdeea...", "metric_value": 0.8455, "sample_count": 1508, "measured_count": 1508 } ]
}
```

| Field | Meaning |
|---|---|
| `run_id` | Unique per run. Changes every time; not a fingerprint. |
| `run_hash` | Hash over the evidence roots plus the run's inputs. **Two runs with the same hash produced the same results from the same inputs.** |
| `methodology_fingerprint` | Hash of the scoring methodology. Equal fingerprints mean the same method — even across different models or data. |
| `evidence_root` | Merkle root over that task's per-sample scores. |
| `seeds` | The seed actually used, so a run can be repeated. |
| `datasets[].revision` | The pinned revision genuinely fetched. |

`content_hash` and `row_count` are `null` above because this run loaded from
cache without recomputing them; they are populated when available and should
not be treated as guaranteed.

Merkle detail worth knowing if you reimplement the check: leaves, internal
nodes and the root use distinct domain prefixes, and an odd node is **promoted**
rather than duplicated, which avoids the CVE-2012-2459 style collision where two
different trees hash alike.

## `provenance`

```json
{
  "methodology_version": "3.0.0",
  "methodology_fingerprint": "266b57eaa394ea0b...",
  "inference_params": { "temperature": 0.0, "do_sample": false, "...": 0 },
  "seeds": { "python_random": 42, "numpy": 42 },
  "offline": false,
  "datasets": { "nyu-mll/crows-pairs": { "revision": "8aaac11c...", "license": "CC-BY-SA-4.0", "pinned": true } }
}
```

Overlaps `manifest` deliberately: `manifest` is what gets hashed, `provenance`
is what a human reads. The licence is here because a number derived from a
ShareAlike dataset carries obligations, and by the time someone asks, the run is
usually long finished.

`offline: true` means datasets came only from cache.

## Reading a result in Python

```python
import json

with open("gpt2.json") as fh:
    result = json.load(fh)

for task_id, task in result["tasks"].items():
    if task["score"] is None:
        print(f"{task_id}: not measured")  # never treat this as zero
    else:
        print(f"{task_id}: {task['score']:.4f}")
```

For results produced through the library API, prefer `load_result()` from
`aethics_eval.results`, which enforces the VOS-4 compatibility policy — refusing
a newer MAJOR version rather than parsing it halfway.

## See also

- [Compatibility policy](result-schema-compatibility.md) — SemVer rules for the published schema
- [Quickstart](quickstart.md) — `validate` and `verify` in use
- [Datasets](datasets.md) — licences behind `provenance.datasets`
