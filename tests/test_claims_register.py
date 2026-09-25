# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
The claims register says true things.

The register is only worth having if its rows can be checked. These tests check
the ones that are checkable from code: that the evidence-chain claim points at
the implementation that actually backs it, that every symbol an aggregation row
cites exists, and that the status vocabulary has not drifted.

They are deliberately narrow. Verifying every row is a person's job; this is the
subset a machine can hold.
"""

from pathlib import Path

import pytest

_REGISTER = Path(__file__).resolve().parent.parent / "docs" / "CLAIMS.md"


@pytest.fixture(scope="module")
def register() -> str:
    assert _REGISTER.exists(), f"{_REGISTER} is missing"
    return _REGISTER.read_text(encoding="utf-8")


# ── The evidence chain points at the right implementation ───────────


def test_evidence_chain_row_cites_the_manifest(register: str) -> None:
    """The per-sample evidence-chain claim must cite manifest.py.

    An earlier version of this register cited scoring.py::hash_audit_results,
    which hashes the whole result blob including final scores. That is not a
    per-sample chain: an edited sample can still hash correctly if the totals
    are adjusted, and it cannot answer "was this sample in the run".
    """
    row = next(
        (
            line
            for line in register.splitlines()
            if "tamper-evident chain" in line and line.lstrip().startswith("|")
        ),
        None,
    )
    assert row is not None, "the evidence-chain row is missing from the register"

    for symbol in ("manifest.py", "leaf_hash", "merkle_root", "verify_manifest"):
        assert symbol in row, (
            f"the evidence-chain row does not cite {symbol}. It must point at the "
            "Merkle implementation that actually backs the claim."
        )


def test_no_row_cites_hash_audit_results_for_the_evidence_chain(
    register: str,
) -> None:
    """hash_audit_results may be discussed, but never as the backing for the
    per-sample evidence chain. Guarding the table rows specifically, so the
    explanatory note above them is still allowed to name it."""
    offenders = [
        line
        for line in register.splitlines()
        if line.lstrip().startswith("|")
        and "hash_audit_results" in line
        and "not" not in line.lower()
    ]
    assert not offenders, (
        "a register row cites hash_audit_results as backing:\n  "
        + "\n  ".join(offenders)
    )


# ── Aggregation rows cite symbols that exist ────────────────────────


@pytest.mark.parametrize(
    "symbol",
    [
        "Dimension",
        "DIMENSION_ORDER",
        "score_dimension",
        "readiness_from_scores",
        "DIMENSION_WEIGHT",
    ],
)
def test_aggregation_rows_cite_real_symbols(register: str, symbol: str) -> None:
    from aethics_eval import aggregation

    assert f"aggregation.py::{symbol}" in register or f"`{symbol}`" in register, (
        f"the register no longer cites {symbol}"
    )
    assert hasattr(aggregation, symbol), (
        f"the register cites aggregation.py::{symbol}, which does not exist"
    )


# ── Vocabulary and shape ────────────────────────────────────────────


@pytest.mark.parametrize(
    "label", ["Implemented", "In development", "Target architecture"]
)
def test_status_vocabulary_is_policy_section_1_2(register: str, label: str) -> None:
    """The three status labels are a fixed vocabulary; renaming one silently
    changes what every row means."""
    assert label in register, (
        f"the status label '{label}' no longer appears in the register. The "
        "vocabulary is fixed — do not rename it."
    )
