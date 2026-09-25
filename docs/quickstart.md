# Quickstart

About five minutes, most of it install. By the end you will have produced a
result file, read it, and verified it has not been tampered with.

Every command and every output on this page was run against a clean virtual
environment. If something here does not match what you see, that is a bug in
the docs — please [open an issue](https://github.com/Aethics-AI/Ethics-OS/issues).

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install aethics-eval
```

!!! warning "Not published yet"

    `aethics-eval` is **not on PyPI** at the time of writing, and the package
    name is not yet reserved. Until then, install from a
    checkout:

    ```bash
    git clone https://github.com/Aethics-AI/Ethics-OS.git
    pip install ./Ethics-OS
    ```

Python 3.10, 3.11 and 3.12 are supported and tested on Linux, macOS and Windows.

The core install is deliberately light — no PyTorch. Check:

```console
$ python -c "import importlib.util as u; print(u.find_spec('torch'))"
None
```

## 2. See what you can run

```console
$ aethics list-tasks
TASK           LICENCE        LOGPROBS  PINNED  DATASET
------------------------------------------------------------------------------
bold           CC-BY-SA-4.0   no        yes     AlexaAI/bold
crows_pairs    CC-BY-SA-4.0   yes       yes     nyu-mll/crows-pairs
stereoset      CC-BY-SA-4.0   yes       yes     McGill-NLP/stereoset
winobias       MIT            yes       yes     uclanlp/wino_bias

4 tasks. `aethics list-tasks --json` for citations.
```

`LOGPROBS: yes` means the benchmark needs token log-probabilities, which
restricts which providers can run it — see
[model providers](model-providers.md). `PINNED` means the dataset is fetched at
a fixed revision, so the data cannot change under you between runs.

## 3. Your first run

Start with the `fake:` provider. It needs no model, no credentials and no
GPU — it returns the text after the colon for every prompt:

```console
$ aethics eval --model "fake:I cannot help with that." --tasks bold --limit 5 --output first.json
model : fake:I cannot help with that.
tasks : bold
seed  : 42

  bold: running (limit=5)...
  bold: score=1.0

wrote first.json
run hash: 8989f2946d37bf3201770b9a20078b852595f6f5860fd2ae7cfdba9aee5d4b2b
```

The benchmark dataset is downloaded on this first run and cached afterwards, so
you do need a network connection here even though the model is a stub. Use
`aethics cache` to see what has been stored and `--offline` to forbid fetching.

**That 1.0 is not a good score, it is a meaningless one.** A model that refuses
every prompt never produces a toxic continuation, so it scores perfectly on
`bold` by saying nothing. Keep it in mind the next time you see a single
benchmark number quoted as a safety result.

## 4. Read the result

```console
$ aethics show first.json
model        : fake:I cannot help with that.
methodology  : 3.0.0
fingerprint  : 266b57eaa394ea0b...

TASK                      SCORE  SAMPLES  NOTE
--------------------------------------------------------------------
bold                 1.0000 ± 0        5  underpowered
--------------------------------------------------------------------
mean                     1.0000  (1/1 measured)
```

Note `underpowered`: five samples cannot support a conclusion, and the tool
says so rather than leaving you to notice. The `fingerprint` is a hash of the
methodology — two runs sharing it were scored the same way.

## 5. A real model, and real uncertainty

For actual numbers you need log-probabilities, which means local open weights:

```bash
pip install "aethics-eval[local]"
```

This pulls PyTorch (~2 GB). Then:

```console
$ aethics eval --model local:gpt2 --tasks crows_pairs --limit 50 --output gpt2.json
model : local:gpt2
tasks : crows_pairs
seed  : 42

  crows_pairs: running (limit=50)...
  crows_pairs: score=0.72

wrote gpt2.json
run hash: 059b8c6dc053ffeb6a62e6db1ed745b217b173d67a879696c2ea24bd4cf76929
```

That took about 11 seconds on a laptop CPU, including loading GPT-2.

Now look at what the number is actually worth:

```console
$ aethics show gpt2.json
TASK                      SCORE  SAMPLES  NOTE
--------------------------------------------------------------------
crows_pairs       0.7200 ± 0.14       50
--------------------------------------------------------------------
mean                     0.7200  (1/1 measured)
```

Underneath, in `gpt2.json`:

```json
{
  "stereotype_preference_rate": 0.64,
  "confidence_interval_95": [0.5, 0.78],
  "total_pairs": 50
}
```

GPT-2 preferred the more-stereotypical sentence in 64% of pairs, where 50%
would be no preference.

**But the 95% interval runs from 0.50 to 0.78.** It touches chance, so 50 pairs
cannot establish that this model has any preference at all. The honest reading
is "suggestive, and far too small to conclude anything".

This is the whole reason the interval is printed next to the score rather than
in a footnote. The full set is 1,508 pairs:

```console
$ aethics eval --model local:gpt2 --tasks crows_pairs --limit 1508 --output gpt2-full.json
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

Two minutes of laptop CPU, and now the interval excludes 0.50 — GPT-2 really
does prefer the more-stereotypical sentence. The rate also lands in the region
the CrowS-Pairs authors report for GPT-2, which is a decent sign the scoring is
doing what it claims rather than merely producing a number.

Notice the rate *fell* from 0.64 to 0.58 as the sample grew. The small run was
not a preview of the large one; it was noise.

!!! warning "`p_value` and `cohens_d` do not describe the rate"

    The result also carries `p_value` and `cohens_d`. These are computed on the
    raw sentence log-probabilities — they test whether the two sets of sentences
    differ in mean log-likelihood, **not** whether the preference rate differs
    from 0.5. They sit next to `stereotype_preference_rate` in the JSON and it
    is easy to read them as commenting on it. They do not.

    For "is this preference real", use the confidence interval, which is a
    bootstrap over the per-pair outcomes and does describe the rate.

!!! note "The default limit is small"

    With no `--limit`, tasks run 30–50 samples (`crows_pairs` and `bold` use 30;
    `winobias` and `stereoset` use 50). That is sized for a quick smoke test,
    **not** for a number you would publish. Set `--limit` explicitly for
    anything real.

## 6. When something cannot be measured

Ask a model without log-probabilities to run a benchmark that needs them:

```console
$ aethics eval --model "fake:hello" --tasks crows_pairs --limit 5 --output nm.json
  crows_pairs: running (limit=5)...
  crows_pairs: not_measured

no task produced a measurement — see errors above

$ aethics show nm.json
TASK                      SCORE  SAMPLES  NOTE
--------------------------------------------------------------------
crows_pairs        not_measured        0
--------------------------------------------------------------------
mean                        n/a  (0/1 measured)

No task produced a measurement. This is not a score of zero.
```

`score` is `null` in the JSON — never `0.0`. A failed measurement and a score of
zero are different claims about a model, and conflating them is how an
evaluation tool starts lying. Unmeasured tasks are excluded from the mean rather
than counted as zeros.

## 7. Check the result is genuine

```console
$ aethics validate first.json
first.json: valid (schema_version 1.2.0)
```

`validate` asks whether the file is well-formed and honestly reported.
`verify` asks a harder question — whether this is the result that was actually
produced:

```console
$ aethics verify first.json
run hash recorded : 8989f2946d37bf3201770b9a20078b852595f6f5860fd2ae7cfdba9aee5d4b2b
run hash computed : 8989f2946d37bf3201770b9a20078b852595f6f5860fd2ae7cfdba9aee5d4b2b
tasks checked     : 1
samples checked   : 5

VERIFIED — the evidence matches the recorded hashes.
```

Every per-sample score is re-hashed, each task's evidence tree is rebuilt, and
the run hash is recomputed from those roots. Editing a sample and adjusting the
totals to match does not survive this, because the totals are not what is
hashed.

Both commands exit `0` on success and `1` on failure, so they work in CI.

## Where next

- [Model providers](model-providers.md) — scoring a hosted model, and what it costs you in coverage
- [Writing a task](writing-a-task.md) — add your own benchmark
- [Result schema](result-schema.md) — every field explained
- [Datasets](datasets.md) — licences, citations, pinned revisions
- [FAQ](faq.md)
