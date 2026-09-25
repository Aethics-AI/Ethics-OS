# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-6 Part A — the validity targets are well-formed.

Fast tests (no model). These guard the *document's* integrity: every claim has a
citation, a justified tolerance, and either a published number or an explicit
reason we make no claim. docs/methodology.md is generated from these constants,
so a malformed target would produce a misleading methods section.
"""

import pytest

from aethics_eval.validity import (
    REFERENCE_MODEL,
    REFERENCE_REVISION,
    VALIDITY_TARGETS,
    claimed_targets,
    target_for,
)


def test_reference_model_is_pinned_to_an_exact_revision():
    # A reproduction claim against a moving model is not a claim.
    assert REFERENCE_MODEL == "openai-community/gpt2"
    assert len(REFERENCE_REVISION) == 40, "expected a full commit SHA"
    assert all(c in "0123456789abcdef" for c in REFERENCE_REVISION)


def test_claimed_benchmarks_are_the_ones_that_actually_reproduce():
    """CrowS-Pairs and WinoBias reproduce their published numbers; StereoSet does
    not (our shared-token scoring measures a different quantity — see its
    excluded_reason) and BOLD has no comparable published figure. A benchmark is
    listed as claimed only if it genuinely reproduces."""
    assert {t.task for t in claimed_targets()} == {"crows_pairs", "winobias"}


@pytest.mark.parametrize(
    "target", [t for t in VALIDITY_TARGETS if not t.claimed], ids=lambda t: t.task
)
def test_every_excluded_benchmark_explains_itself(target):
    # An absent claim must be explained, not merely omitted.
    assert target.excluded_reason and len(target.excluded_reason) > 40
    assert target.tolerance is None  # nothing to be "within"


@pytest.mark.parametrize("target", claimed_targets(), ids=lambda t: t.task)
def test_every_claim_is_complete(target):
    assert target.published_value is not None, "a claim needs a published number"
    assert target.tolerance is not None and target.tolerance > 0
    assert target.published_source.strip(), "a claim needs a citation"
    assert target.tolerance_rationale.strip(), "a tolerance needs a justification"
    assert target.sample_size > 0
    assert target.metric.strip()


@pytest.mark.parametrize("target", VALIDITY_TARGETS, ids=lambda t: t.task)
def test_every_deviation_is_justified(target):
    for dev in target.deviations:
        assert dev.what.strip() and dev.why.strip(), (
            f"{target.task}: a deviation must state both what and why"
        )


def test_excluded_benchmark_states_its_reason():
    bold = target_for("bold")
    assert not bold.claimed
    assert bold.published_value is None  # no number invented
    assert bold.excluded_reason and len(bold.excluded_reason) > 40


def test_within_tolerance_logic():
    t = target_for("crows_pairs")  # 0.601 ± 0.05
    assert t.within_tolerance(0.601)
    assert t.within_tolerance(0.65)
    assert t.within_tolerance(0.552)
    assert not t.within_tolerance(0.70)


def test_excluded_target_never_reports_within_tolerance():
    # No published number means nothing can be "within tolerance" of it.
    assert target_for("bold").within_tolerance(0.5) is False


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        target_for("not_a_benchmark")
