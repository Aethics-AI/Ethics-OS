# Model providers

A model is named `provider:model_id` — `local:gpt2`, `hf:meta-llama/Llama-3.1-8B`,
`openai:gpt-4o-mini`. The prefix decides how the model is reached; the rest is
passed through as the model identifier.

## The short version

| Prefix | Reaches | Needs | Log-probs? | Can measure |
|---|---|---|---|---|
| `local:` | Open-weight model on your machine | `[local]` extra (~2 GB) | **yes**, exact | all 4 benchmarks |
| `hf:` | HuggingFace Inference Providers | `HF_TOKEN` | no | `bold` only |
| `openai:` | Any OpenAI-shaped `/chat/completions` | `--base-url` + key in env | no | `bold` only |
| `fake:` | Nothing — a scripted stub | nothing | no | `bold` only |

**Read the last column before choosing.** Three of the four benchmarks
(`winobias`, `stereoset`, `crows_pairs`) score by comparing the likelihood a
model assigns to two sentences. That needs token log-probabilities. A provider
that cannot supply them cannot run those benchmarks, and `aethics-eval` will
tell you so — it reports `not_measured` rather than substituting a weaker proxy
and calling it the same number.

Today that means **`local:` is the only provider that can measure three of the
four benchmarks.** This is a real limitation, not a temporary gap in the docs.

## `local:` — open weights, exact log-probabilities

```bash
pip install "aethics-eval[local]"
aethics eval --model local:gpt2 --tasks crows_pairs --limit 200
```

Loads the model with `transformers` and scores with exact conditional
log-probabilities. The `[local]` extra pulls PyTorch, so it costs about 2 GB;
this is why it is an extra rather than a core dependency, and why an
API-scored run never pays for it.

The model is downloaded once and cached by `transformers` (under
`~/.cache/huggingface`), so subsequent runs need no network.

## `hf:` — HuggingFace Inference Providers

```bash
export HF_TOKEN=hf_...
aethics eval --model hf:gpt2 --tasks bold --limit 50
```

**The token is required, not optional.** Unauthenticated requests to the
inference router return `401`. The variable is `HF_TOKEN`, which is what
`huggingface_hub` reads — not `HUGGINGFACE_TOKEN`, which nothing reads.

Two things to know before you rely on this provider:

- **No log-probabilities.** HF's serverless inference does not return them, so
  `winobias`, `stereoset` and `crows_pairs` all report `not_measured`. If you
  run a dedicated TGI endpoint that returns prefill log-probs, the underlying
  `HFInferenceModel` accepts `supports_logprobs=True` — the CLI does not expose
  that flag yet.
- **A missing or invalid token currently surfaces badly.** Generation failures
  are counted as unmeasured samples, so on `bold` you will see
  `Insufficient domain diversity: 0 domain(s) with data` rather than an
  authentication error. The run is honest — it reports `not_measured` and never
  invents a score — but the *reason* it gives points at the dataset instead of
  your credentials. If you see that message, check `HF_TOKEN` first. This is
  tracked as a bug; the diagnosis should name the real cause.

## `openai:` — any OpenAI-compatible endpoint

```bash
export AETHICS_API_KEY=sk-...
aethics eval --model openai:gpt-4o-mini \
  --base-url https://api.openai.com/v1 \
  --tasks bold --limit 50
```

Works against anything serving an OpenAI-shaped `/chat/completions` — OpenAI,
vLLM, Together, a local llama.cpp server. `--base-url` is required; there is no
default endpoint, because guessing where to send your prompts is not a
reasonable default.

Sampling is pinned to `temperature=0.0` to match `EVAL_INFERENCE_PARAMS`. An
evaluation that samples is not reproducible, so this is not configurable.

### How the key is read, and why

The key comes from `AETHICS_API_KEY`, or `OPENAI_API_KEY` if that is unset.

**There is deliberately no `--api-key` flag.** Command-line arguments are
recorded in shell history and are visible to any process that can run `ps`, and
a key in a config file gets committed. An environment variable is the only one
of the three that is not routinely written down somewhere. A test asserts the
flag does not exist, so it cannot be added back by accident.

For the same reason, a non-200 response reports only its status code. Auth
error bodies sometimes echo the submitted key back, and a result file is
something people attach to tickets.

If no key is found the run proceeds unauthenticated and says so on stderr,
which is usually what you want against a local vLLM server and never what you
want against a hosted one.

## `fake:` — a scripted stub

```bash
aethics eval --model "fake:I cannot help with that." --tasks bold --limit 5
```

Returns the text after the colon for every prompt. No network, no credentials,
no model. It exists so the CLI and the test suite can be exercised offline, and
so you can see the shape of a result before committing to a real run.

**Do not read anything into its scores.** A model that refuses every prompt
scores 1.0 on `bold`, because it never produces the toxic continuation the
benchmark is looking for. That is a correct measurement of a useless model, and
a good illustration of why a single benchmark number is not a safety claim.

## Choosing a provider

- **Trying the tool out?** `fake:` — instant, offline.
- **Want real numbers on the bias benchmarks?** `local:` — it is the only
  provider that can measure them.
- **Auditing a hosted model you cannot download?** `hf:` or `openai:`, and
  accept that you get `bold` and three `not_measured` rows. That is the honest
  answer to "how biased is this API", given the API does not expose what the
  measurement needs.

## See also

- [Quickstart](quickstart.md) — a first run in about a minute
- [Result schema](result-schema.md) — what comes back, field by field
- [Writing a task](writing-a-task.md) — the `Model` protocol a task talks to
