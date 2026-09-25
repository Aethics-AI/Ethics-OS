# Datasets

Every dataset `aethics-eval` loads, where it comes from, what its licence
permits, and how to cite it.

**This package ships no benchmark data.** All four datasets below are fetched
from their published sources at runtime. **Three of the four are
ShareAlike-licensed** and could not be redistributed without imposing that
obligation on this package, so the runtime-load design is a licensing
requirement rather than a convenience.

Every licence here was checked against the **original** source — the authors'
repository or paper — not only the HuggingFace dataset card. That mattered:
see the note under BOLD, where the two disagree.

The machine-readable version of this table is
[`dataset_licenses.py`](https://github.com/Aethics-AI/Ethics-OS/blob/main/src/aethics_eval/dataset_licenses.py). A test
(`tests/test_dataset_inventory.py`) fails the build if a loader appears
without a corresponding entry, so a dataset cannot enter the package
undocumented.

---

## Summary

| Dataset | Licence | Redistributable | Pinned |
|---|---|---|---|
| WinoBias | MIT | Yes | Yes |
| StereoSet | CC-BY-SA-4.0 | **No** | Yes |
| CrowS-Pairs | CC-BY-SA-4.0 | **No** | Yes |
| BOLD | CC-BY-SA-4.0 | **No** | Yes |

---

## WinoBias

**Identifier** `uclanlp/wino_bias`
**Source** https://huggingface.co/datasets/uclanlp/wino_bias
**Revision** `3f31267586e4408e3b3f77ec22198fd24ea8dc1d`
**Licence** MIT
**Redistribution** Permitted with attribution.

Gender bias in coreference resolution. Configs `type1_pro` and `type1_anti`,
split `test`, paired by index.

> Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K.-W. (2018).
> Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods.
> NAACL-HLT 2018.

---

## StereoSet

**Identifier** `McGill-NLP/stereoset`
**Source** https://huggingface.co/datasets/McGill-NLP/stereoset
**Revision** `bf6e7ce50491784d094fb7afe60a70ecccb89035`
**Licence** [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)
**Redistribution** **Not permitted for this package.**

Stereotype bias via intrasentence triples. Config `intrasentence`, split
`validation`.

ShareAlike means any redistribution — including a derivative — must itself be
licensed CC-BY-SA-4.0. Applying that to a package intended to carry a
permissive licence would create a conflict, so StereoSet is loaded at runtime
and no rows are stored in this repository.

> Nadeem, M., Bethke, A., & Reddy, S. (2021). StereoSet: Measuring
> stereotypical bias in pretrained language models. ACL-IJCNLP 2021.

---

## CrowS-Pairs

**Identifier** `nyu-mll/crows-pairs`
**Source** https://github.com/nyu-mll/crows-pairs
**Revision** `8aaac11c485473159ec9328a65253a5be9a479dc`
**Licence** [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)
**Redistribution** **Not permitted for this package.** Same ShareAlike
constraint as StereoSet.

1,508 sentence pairs across nine bias categories.

This is the one dataset not loaded through the HuggingFace hub: the loader
fetches `data/crows_pairs_anonymized.csv` straight from the GitHub repository,
so the pin is a commit SHA in the URL path rather than a `revision=` argument
(`standard_benchmarks.py:_load_crows_pairs`).

> **Previously unpinned.** POS-2 found this loading from the `master` branch,
> which meant two runs on different days could read different data while
> reporting identical provenance. POS-4 replaced it with the commit SHA above.
> No dataset in the package now loads from a moving reference.

> Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). CrowS-Pairs:
> A Challenge Dataset for Measuring Social Biases in Masked Language Models.
> EMNLP 2020.

---

## BOLD

**Identifier** `AlexaAI/bold`
**Source** https://huggingface.co/datasets/AlexaAI/bold
**Upstream** https://github.com/amazon-science/bold
**Revision** `be5f5a99b386a7c4fa7ea905685ee2d2c98301eb`
**Licence** [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)
**Redistribution** **Not permitted for this package.**

7,201 Wikipedia-seeded prompts across five demographic domains. Split `train`.

> ⚠️ **The two upstream sources disagree, and we follow the stricter one.**
>
> The HuggingFace card for `AlexaAI/bold` declares `cc-by-4.0` — attribution
> only, no ShareAlike. But the original repository `amazon-science/bold` ships
> a `LICENSE.md` whose first line reads *"Creative Commons
> Attribution-ShareAlike 4.0 International"*, and its README states the same.
> GitHub's own licence detector reports `NOASSERTION` for that repo, meaning it
> could not match the file automatically — which is likely how the HuggingFace
> card came to carry the weaker identifier.
>
> We treat BOLD as **ShareAlike**. If the authors intended CC-BY-SA and we
> shipped copies under a permissive licence on the strength of a third-party
> metadata field, that is not a mistake we could take back.
>
> This is worth a second opinion before release, and is the clearest argument
> for checking original sources rather than dataset cards.

> Dhamala, J., Sun, T., Kumar, V., Krishna, S., Pruksachatkun, Y., Chang,
> K.-W., & Gupta, R. (2021). BOLD: Dataset and Metrics for Measuring Biases
> in Open-Ended Language Generation. FAccT 2021.

---

## Prompt sets original to this project

`eval_datasets.py` contains prompt sets written by the AETHICS team. They are
not derived from any third-party corpus, carry this package's licence, and
impose no external obligation:

`GENDER_PRONOUN_PROMPTS` · `OCCUPATION_ASSOCIATION_PROMPTS` ·
`DEMOGRAPHIC_BIAS_PROMPTS` · `COMPLETION_BIAS_PROMPTS` ·
`COUNTERFACTUAL_TEMPLATES` · `TOXICITY_PROMPTS` · `HARMFUL_CONTENT_PROMPTS` ·
`PRIVACY_PROMPTS` · `SAFETY_REFUSAL_PROMPTS`

### A naming problem worth fixing

Four further constants are named after the benchmarks above:

`CROWS_PAIRS_CURATED` · `WINOBIAS_CURATED` · `STEREOSET_CURATED` ·
`BOLD_PROMPTS_CURATED`

**They contain no data from those datasets.** They are hand-written items
imitating each benchmark's format — for example `StereoSetItem("Men are
typically", "strong", "emotional", "hungry", "gender")`, which is not a
StereoSet row. This was checked entry by entry.

So there is no licensing exposure. There is an honesty problem: the names
assert a provenance the content does not have, and the module docstring calls
them "offline fallbacks". If they were ever used as a fallback, a run could
report CrowS-Pairs as its data source while scoring against hand-written
imitations. That is the F4 failure mode.

They are currently referenced by no code path. Recommended: rename to
`*_SYNTHETIC` or delete them. Raised for POS-4, which owns fallback
behaviour.

---

## Adding a dataset

1. Add a `DatasetLicense` entry to
   [`dataset_licenses.py`](https://github.com/Aethics-AI/Ethics-OS/blob/main/src/aethics_eval/dataset_licenses.py) with the
   licence, citation, source URL and a pinned revision — all verified against
   the upstream source, not recalled.
2. Add a section here.
3. If the licence is ShareAlike or otherwise restrictive, do not vendor the
   data. Load it at runtime.

The inventory test will fail until steps 1 and 2 are done.
