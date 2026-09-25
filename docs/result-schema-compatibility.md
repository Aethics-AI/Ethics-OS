# Result schema — compatibility policy

Every evaluation result serialises with a top-level `schema_version`, currently
**`1.3.0`**. The published contract is [`schemas/result-v1.json`](https://github.com/Aethics-AI/Ethics-OS/blob/main/schemas/result-v1.json),
generated from the `RunResult` model.

The schema has two customers from day one — third-party tooling reading our
output, and the AETHICS certification platform consuming it — so the
version is a promise, not a label.

## Versioning rules (SemVer)

| Change | Bump | Example |
|---|---|---|
| Add an optional field | **MINOR** | adding `uncertainty.method` |
| Add a new optional object/section | **MINOR** | adding a `run_metadata` block |
| Relax a constraint (required → optional) | **MINOR** | making `passed` nullable |
| Remove or rename a field | **MAJOR** | dropping `manifest_fingerprint` |
| Change a field's type | **MAJOR** | `sample_count` int → string |
| Change what a field *means* | **MAJOR** | `metric_value` 0-1 → 0-100 |
| Tighten a constraint (optional → required) | **MAJOR** | requiring `completed_at` |
| Change a default that alters interpretation | **MAJOR** | `reliability` default flip |

The rule of thumb: **if an existing v1 reader would misinterpret the result, it's
a MAJOR bump.** If the reader can safely ignore what's new, it's MINOR.

Note that a *semantic* change is major even when the JSON shape is unchanged —
silently rescaling a metric is the most damaging change we could ship, because
nothing breaks loudly.

## What readers must do

`load_result()` implements this policy:

- **Newer MAJOR is refused** — a `2.x` result raises `IncompatibleSchemaVersion`
  rather than being partially parsed.
- **Newer MINOR is accepted** — a `1.7.0` result loads under a `1.0.0` reader.
- **Unknown fields are ignored** (`extra="ignore"`), which is what makes additive
  changes safe.

So a v1 reader keeps working for the entire v1 line, and fails loudly — never
silently — the moment a v2 result appears.

## Changing the schema

1. Edit the models in `src/aethics_eval/results/schema.py`.
2. Bump `SCHEMA_VERSION` per the table above.
3. Regenerate the published schema:
   ```bash
   python -m scripts.export_schema
   ```
4. Update the golden file if the serialisation legitimately changed, and note the
   change here.

CI enforces steps 3–4: a drift test fails if `schemas/result-v1.json` no longer
matches the model, and the golden-file test fails if the serialisation changed
without the golden being updated. Both are deliberate speed bumps — the schema
should never change by accident.

## Version history

| Version | Change |
|---|---|
| `1.3.0` | **MINOR (additive)** — POS-11 added optional `SampleScore.response_digest`. A `1.0.0` reader still loads a `1.3.0` result and ignores it. Not cosmetic: `manifest.leaf_hash` hashes this field into every evidence leaf, so while the model had no place for it the digest was silently dropped on serialisation and `aethics verify` reported tampering on results nobody had touched. |
| `1.2.0` | **MINOR (additive)** — the CLI moved onto this contract. Added optional `summary` (`RunSummary`, including `underpowered_tasks`), `provenance`, `manifest`, and `TaskResult.details`. Nothing was removed or re-meant, so a `1.0.0` reader still loads a `1.2.0` result. Per-sample evidence now travels in `tasks[].sample_scores`, where the model already had a place for it. |
| `1.1.0` | **MINOR (additive)** — VOS-5 added optional `Uncertainty` fields: `standard_error`, `sample_size`, `underpowered`. No field was removed, retyped, or re-meant, so a `1.0.0` reader still loads a `1.1.0` result and simply ignores the new fields. |
| `1.0.0` | Initial schema: `RunResult` / `TaskResult` / `SampleScore`, with task + dataset versioning, model + provider + params, manifest fingerprint, metric value (`null` when not measured), uncertainty, sample count, method, reliability, UTC timestamps. |
