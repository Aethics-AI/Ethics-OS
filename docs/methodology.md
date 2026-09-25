# Methodology and validity

This document states what our benchmark implementations measure, how that compares to the published numbers, and every place we depart from the published methodology. It is the evidence that our implementations are *correct*, not merely that they run.

It is generated from `src/aethics_eval/validity.py`, which is also what the `-m validity` tests assert against, so the claims below cannot drift from what is actually checked.

## Reference model

All numbers below are for **`openai-community/gpt2`**, pinned to revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

GPT-2 is the conventional reference for these benchmarks: it is small, deterministic under greedy decoding, and the model the published figures were computed against. Pinning the exact commit matters — a reproduction claim against a moving model is not a claim.

## Summary

| Benchmark | Published | Ours | Δ | Tolerance | Within? |
|---|---|---|---|---|---|
| `crows_pairs` | 0.601 | 0.5933 ± 0.04 | -0.0077 | ±0.05 | ✅ |
| `stereoset` | 0.604 | 0.43 | — | — | *no claim — see below* |
| `winobias` | 0.13 | 0.1 ± 0.036 | -0.03 | ±0.08 | ✅ |
| `bold` | — | — | — | — | *no claim — see below* |

Our figures carry a standard error because a reproduction claim without one is not falsifiable: whether 0.61 reproduces 0.60 depends entirely on whether our figure is ±0.01 or ±0.15.

## Per benchmark

### crows_pairs

**Metric.** stereotype_score (fraction preferring the more-stereotypical sentence; ideal 0.50)

**Published.** 0.601 — Nangia, Vania, Bhalerao & Bowman (2020), *CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models*, EMNLP 2020 — GPT-2 stereotype score ≈ 60%.

**Ours.** 0.5933 ± 0.04 at n=150 (Δ -0.0077).

**Tolerance.** ±0.05 absolute. The published figure is reported to the percent, and our sample is a subset of the 1,508 pairs, so binomial sampling noise alone is ~±0.04 at n=150. A tighter band would fail on sampling variation rather than on a real implementation difference.

**Deviations from the published methodology.**

- *Shared-token pseudo-log-likelihood over the pair, rather than scoring each sentence independently.*  
  CrowS-Pairs sentences in a pair differ by only a few tokens. Conditioning on the shared tokens isolates the stereotype-bearing difference, which is the comparison the paper describes.

- *Summed log-probability, not length-normalised (LIKELIHOOD_NORMALIZE = False).*  
  Minimal pairs differ by a token or two, so raw sequence probability is the standard comparison; normalising would divide by near-identical lengths and add noise without removing bias.

### stereoset

**Metric.** stereotype_score (intrasentence; ideal 0.50)

**No reproduction claim — we cannot evaluate on the split the published number describes.**

The paper's 60.4 is measured on the StereoSet **test set**, which was never publicly released — it is held out behind the authors' leaderboard. The HuggingFace distribution (`McGill-NLP/stereoset`) ships **only a `validation` split**, for both the intrasentence and intersentence configs. So any number we compute is on a different set of examples from the one the published figure describes, and comparing them is not a like-for-like reproduction regardless of how we score.

On the validation split we measure ~0.43. That is reproducible, not sampling noise (0.43 at n=150, 0.44 at n=400), and it is not a scoring artefact: four methods were tested and all land in the same region, well below the published figure.

| Scoring method | Our result (validation split) |
|---|---|
| Shared-token PLL (what we ship) | 0.440 |
| Full-sentence likelihood | 0.365 |
| Length-normalised full-sentence | 0.365 |
| Attribute-term only (the paper's stated method) | 0.427 |

Two corrections were made to this document during the investigation, both worth recording. First, the target was originally listed as 0.603 — that is GPT2-**medium**'s **intersentence** score from the same table, not base GPT-2 intrasentence, which is 60.4. Second, an earlier draft blamed our shared-token deviation; that was tested and is wrong, since the paper's own attribute-only method gives 0.427, no closer than ours. Label handling was also verified against the dataset's per-annotator labels and `target` field.

So the discrepancy is **explained**: it is a split mismatch, not an implementation defect. We could still make no honest reproduction claim without access to the test set, so StereoSet ships as a measurement without a claim, while CrowS-Pairs and WinoBias — whose published numbers are computed on data we actually have — carry reproduction claims.

**Published work.** Nadeem, Bethke & Reddy (2021), *StereoSet: Measuring stereotypical bias in pretrained language models*, ACL 2021, Table 4 — GPT2 (base), Intrasentence Task, stereotype score **60.4** on the **test set**. (For reference the same table gives GPT2-medium 62.9 and GPT2-large 63.9 intrasentence; the intersentence column reads 57.6 / 60.3 / 66.2.)

### winobias

**Metric.** coreference accuracy gap between the pro- and anti-stereotypical sets (ideal 0.0)

**Published.** 0.13 — Zhao, Wang, Yatskar, Ordonez & Chang (2018), *Gender Bias in Coreference Resolution*, NAACL 2018 — systems score substantially higher on pro- than anti-stereotypical sentences; the gap is the reported bias measure.

**Ours.** 0.1 ± 0.036 at n=100 (Δ -0.03).

**Tolerance.** ±0.08 absolute — wider than the stereotype-score benchmarks, and deliberately so. The published gap is for dedicated coreference systems, not for a language model scored by likelihood, so this is a same-direction, same-order-of-magnitude claim rather than a like-for-like replication. See the deviation below.

**Deviations from the published methodology.**

- *Coreference is resolved by comparing the likelihood of the two candidate occupations completing a probe, not by a trained coreference system.*  
  WinoBias was published against coreference systems. We measure the same quantity (pro/anti accuracy gap) on a language model, which is the standard adaptation for LM evaluation — but it is an adaptation, so the tolerance is wider and the claim is about the direction and magnitude of the gap, not an exact match.

- *The probe keeps the gendered pronoun in the context.*  
  Pro- and anti-stereotypical sentences differ only in pronoun gender. Removing it would erase exactly the variable WinoBias measures.

### bold

**Metric.** cross-domain sentiment consistency

**No reproduction claim — there is no comparable published number.** BOLD's published results are per-domain sentiment/regard/toxicity distributions produced by specific external classifiers (VADER, a regard classifier, Perspective API). Our score is a single cross-domain consistency figure computed with our own toxicity heuristic, so there is no like-for-like published number to reproduce. Claiming one would be dishonest; we state the absence instead.

**Published work.** Dhamala et al. (2021), *BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation*, FAccT 2021.

## Reproducing this

```bash
pip install -e '.[local,dev]'   # local extra pulls torch + transformers
pytest -m validity -s           # downloads the pinned model, runs the table
```

The validity suite is excluded from the default run (it downloads a model and does real inference) and runs nightly instead. All statistics are seeded, so a rerun on the same revision reproduces the same numbers.
