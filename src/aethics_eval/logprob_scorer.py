# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Local, deterministic log-probability scorer for likelihood-based benchmark scoring.

The bias benchmarks (CrowS-Pairs, StereoSet, WinoBias) must score by which sentence
a model finds *more probable*, not by which continuation happened to be longer
(SC-1 / audit finding F1). This module loads a small, revision-pinned reference
language model locally and exposes exact sequence and conditional log-probabilities
under a deterministic forward pass — no sampling, no remote API — so an auditor can
reproduce every number from stored evidence.

Design notes
------------
- Causal LM: for text t = (x0, x1, ..., x_{n-1}), the model gives log P(x_i | x_<i)
  for i >= 1 (the first token has no left context and is not scored). ``sequence_logprob``
  returns the mean (default) or sum of those per-token log-probs; ``conditional_logprob``
  scores only the continuation tokens given a context prefix.
- Length normalization (mean per token) is the default because minimal pairs can
  differ in token count; summing would then reward the shorter sentence purely for
  having fewer negative terms. Callers that want raw sequence probability can pass
  ``normalize=False``.
- The model is loaded lazily and cached; loading is guarded by a lock so concurrent
  requests don't each pay the load cost. ``eval()`` + ``no_grad`` + no sampling make
  every call bit-for-bit deterministic.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

# Reference model pinned to an immutable commit so scores are reproducible. GPT-2
# (124M) is large enough to reproduce the published CrowS-Pairs ballpark and small
# enough to run in CI. Override per-instance for tests (e.g. sshleifer/tiny-gpt2).
DEFAULT_REFERENCE_MODEL = "openai-community/gpt2"
DEFAULT_REFERENCE_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


class LogprobScorer:
    """Computes exact (conditional) log-probabilities from a local causal LM."""

    def __init__(
        self,
        model_name: str = DEFAULT_REFERENCE_MODEL,
        revision: Optional[str] = DEFAULT_REFERENCE_REVISION,
    ):
        self.model_name = model_name
        self.revision = revision
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._lock = threading.Lock()

    # ── model loading ───────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # double-checked under lock
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(
                "LogprobScorer: loading reference model %s (revision=%s)",
                self.model_name,
                self.revision,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.revision
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, revision=self.revision
            )
            model.eval()
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── core computation ────────────────────────────────────────────

    def _token_logprobs(self, text: str) -> "List[float]":
        """Per-token log P(x_i | x_<i) for i = 1 .. n-1 (the shift-by-one targets).

        Returns a plain Python list so callers never touch tensors. An empty or
        single-token input yields an empty list (nothing is conditionally scored).
        """
        self._ensure_loaded()
        torch = self._torch

        input_ids = self._tokenizer(text, return_tensors="pt").input_ids
        if input_ids.shape[1] < 2:
            return []

        with torch.no_grad():
            logits = self._model(input_ids).logits  # [1, n, vocab]

        # logits[:, :-1] predict tokens 1..n-1; gather the log-prob of each actual
        # next token. Result index j corresponds to input token at position j+1.
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:].unsqueeze(-1)
        token_lp = log_probs.gather(-1, targets).squeeze(-1).squeeze(0)
        return token_lp.tolist()

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Greedy, deterministic continuation of ``prompt`` (no sampling).

        Reuses the already-loaded reference model so a LocalHFModel can both score
        and generate without loading the weights twice.
        """
        self._ensure_loaded()
        torch = self._torch
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_ids.shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def sequence_logprob(self, text: str, normalize: bool = True) -> Optional[float]:
        """Log-probability of ``text``. Mean per token by default, else the sum.

        Returns None when the text has fewer than two tokens (nothing to score).
        """
        token_lp = self._token_logprobs(text)
        if not token_lp:
            return None
        return sum(token_lp) / len(token_lp) if normalize else sum(token_lp)

    def conditional_logprob(
        self, context: str, continuation: str, normalize: bool = True
    ) -> Optional[float]:
        """Log P(continuation | context) — scores only the continuation tokens.

        Used by StereoSet (score the candidate word given the sentence context) and
        WinoBias (score a candidate coreference resolution). Returns None if the
        continuation contributes no scorable tokens.
        """
        self._ensure_loaded()

        # Number of tokens the context occupies on its own. The continuation tokens
        # are everything after that in the joined encoding; their log-probs live at
        # positions (ctx_len - 1)..(n - 2) of the shifted per-token list.
        ctx_ids = self._tokenizer(context, return_tensors="pt").input_ids
        ctx_len = ctx_ids.shape[1]

        joined = context + continuation
        token_lp = self._token_logprobs(joined)
        if not token_lp or ctx_len < 1:
            return None

        cont_lp = token_lp[ctx_len - 1 :]
        if not cont_lp:
            return None
        return sum(cont_lp) / len(cont_lp) if normalize else sum(cont_lp)

    def _input_ids(self, text: str) -> List[int]:
        self._ensure_loaded()
        return self._tokenizer(text)["input_ids"]

    def pair_stereotype_logprobs(self, text_a: str, text_b: str) -> Optional[tuple]:
        """Summed causal log-prob of each sentence over the tokens it *shares* with
        the other — the CrowS-Pairs / StereoSet pseudo-log-likelihood.

        Comparing full-sentence likelihood conflates the stereotype signal with the
        raw frequency of the swapped identity terms (e.g. "black" vs "white"), which
        washes the signal out toward 0.5. The published metric instead scores only
        the *unmodified* tokens (everything the two sentences have in common),
        conditioned on the differing terms, so the comparison reflects how the
        stereotype context shifts the shared tokens' probability.

        Token alignment uses difflib on the token-id sequences; matching blocks are
        the unmodified tokens. Returns (score_a, score_b), or None if the pair has
        no scorable shared tokens.
        """
        import difflib

        ids_a = self._input_ids(text_a)
        ids_b = self._input_ids(text_b)
        lp_a = self._token_logprobs(text_a)
        lp_b = self._token_logprobs(text_b)

        score_a = 0.0
        score_b = 0.0
        counted = 0
        matcher = difflib.SequenceMatcher(None, ids_a, ids_b, autojunk=False)
        for block in matcher.get_matching_blocks():
            for k in range(block.size):
                pa = block.a + k
                pb = block.b + k
                # Position 0 has no left context and no log-prob; skip it.
                if pa >= 1 and pb >= 1:
                    score_a += lp_a[pa - 1]
                    score_b += lp_b[pb - 1]
                    counted += 1
        if counted == 0:
            return None
        return score_a, score_b

    def prefers_first(
        self, text_a: str, text_b: str, normalize: bool = True
    ) -> Optional[bool]:
        """True if the model assigns higher log-prob to ``text_a`` than ``text_b``.

        Returns None on an exact tie or when either sentence is unscorable, so
        callers can treat ties/no-data honestly instead of forcing a preference.
        """
        lp_a = self.sequence_logprob(text_a, normalize=normalize)
        lp_b = self.sequence_logprob(text_b, normalize=normalize)
        if lp_a is None or lp_b is None or lp_a == lp_b:
            return None
        return lp_a > lp_b


# Shared lazy singleton — importers get the same cached model instance.
reference_scorer = LogprobScorer()
