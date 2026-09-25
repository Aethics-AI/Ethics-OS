# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Licence and provenance registry for every dataset this package loads.

Publishing code that downloads benchmark data means publishing a claim about
what that data is and what its terms are. This module is the single machine
readable source of that claim; ``docs/datasets.md`` is generated from it and
``tests/test_dataset_inventory.py`` fails CI if a loader appears without a
corresponding entry here.

Every field is verified against the upstream source rather than recalled.
``revision`` values are the upstream commit SHAs current at the time of
writing; POS-4 wires them into the loaders so runs are reproducible.

Nothing in this package vendors third-party benchmark data. See the note on
ORIGINAL_PROMPT_SETS at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetLicense:
    """Provenance and terms for one upstream dataset."""

    dataset_id: str  # identifier as it appears in the loader call
    name: str
    source_url: str
    revision: Optional[str]  # upstream commit SHA to pin (POS-4)
    license_id: str  # SPDX identifier where one exists
    license_url: str
    citation: str
    redistributable: bool  # may we ship copies of the data itself?
    notes: str = ""


# ── Upstream benchmark datasets ─────────────────────────────────────
# Loaded at runtime from their published sources. We ship no copies.

DATASET_LICENSES: Dict[str, DatasetLicense] = {
    "uclanlp/wino_bias": DatasetLicense(
        dataset_id="uclanlp/wino_bias",
        name="WinoBias",
        source_url="https://huggingface.co/datasets/uclanlp/wino_bias",
        revision="3f31267586e4408e3b3f77ec22198fd24ea8dc1d",
        license_id="MIT",
        license_url="https://huggingface.co/datasets/uclanlp/wino_bias",
        citation=(
            "Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K.-W. (2018). "
            "Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods. "
            "NAACL-HLT 2018."
        ),
        redistributable=True,
        notes="Permissive. Configs used: type1_pro and type1_anti, split=test.",
    ),
    "McGill-NLP/stereoset": DatasetLicense(
        dataset_id="McGill-NLP/stereoset",
        name="StereoSet",
        source_url="https://huggingface.co/datasets/McGill-NLP/stereoset",
        revision="bf6e7ce50491784d094fb7afe60a70ecccb89035",
        license_id="CC-BY-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        citation=(
            "Nadeem, M., Bethke, A., & Reddy, S. (2021). StereoSet: Measuring "
            "stereotypical bias in pretrained language models. ACL-IJCNLP 2021."
        ),
        redistributable=False,
        notes=(
            "ShareAlike. Redistributing the data, or a derivative of it, would "
            "oblige us to license that redistribution under CC-BY-SA-4.0 — which "
            "would conflict with a permissive licence on this package. We "
            "therefore load it at runtime and ship no copies. Config used: "
            "intrasentence, split=validation."
        ),
    ),
    "nyu-mll/crows-pairs": DatasetLicense(
        dataset_id="nyu-mll/crows-pairs",
        name="CrowS-Pairs",
        source_url="https://github.com/nyu-mll/crows-pairs",
        revision="8aaac11c485473159ec9328a65253a5be9a479dc",
        license_id="CC-BY-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        citation=(
            "Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). "
            "CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in "
            "Masked Language Models. EMNLP 2020."
        ),
        redistributable=False,
        notes=(
            "ShareAlike, same constraint as StereoSet. Not on the Hub — loaded "
            "from a raw GitHub CSV. POS-2 flagged that this read from the "
            "master branch, so two runs could silently see different data; "
            "POS-4 pinned it to 8aaac11c, the tip of master as of 2021-11-03 "
            "('fixed dead link to data in readme'). The repository has not "
            "moved since. The URL now embeds the SHA rather than a branch name."
        ),
    ),
    "AlexaAI/bold": DatasetLicense(
        dataset_id="AlexaAI/bold",
        name="BOLD (Bias in Open-Ended Language Generation Dataset)",
        source_url="https://huggingface.co/datasets/AlexaAI/bold",
        revision="be5f5a99b386a7c4fa7ea905685ee2d2c98301eb",
        license_id="CC-BY-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        citation=(
            "Dhamala, J., Sun, T., Kumar, V., Krishna, S., Pruksachatkun, Y., "
            "Chang, K.-W., & Gupta, R. (2021). BOLD: Dataset and Metrics for "
            "Measuring Biases in Open-Ended Language Generation. FAccT 2021."
        ),
        redistributable=False,
        notes=(
            "UPSTREAM SOURCES DISAGREE — we follow the stricter one. The "
            "HuggingFace card for AlexaAI/bold declares cc-by-4.0, but the "
            "original repository amazon-science/bold ships a LICENSE.md whose "
            "first line reads 'Creative Commons Attribution-ShareAlike 4.0 "
            "International', and its README says the same. GitHub's own licence "
            "detector reports NOASSERTION for that repo, i.e. it could not "
            "match the file automatically. Treating BOLD as ShareAlike is the "
            "safe reading: if the upstream authors intended CC-BY-SA and we "
            "shipped copies under a permissive licence, that is not a mistake "
            "we could take back. split=train."
        ),
    ),
}


# ── Prompt sets original to this project ────────────────────────────
# Written by the AETHICS team, not derived from any third-party corpus, so
# they carry this package's licence and no external obligation.
#
# This is a mapping rather than a list on purpose. It is the exemption list
# for the vendoring guard in tests/test_dataset_inventory.py, and an exemption
# should cost something: adding a name here requires writing down where the
# content actually came from, so the claim is on the record and reviewable.
# A bare tuple could be extended in a diff nobody reads.

ORIGINAL_PROMPT_SETS: Dict[str, str] = {
    "GENDER_PRONOUN_PROMPTS": "Written in-house; generated from an occupation list.",
    "OCCUPATION_ASSOCIATION_PROMPTS": "Written in-house; generated from an occupation list.",
    "DEMOGRAPHIC_BIAS_PROMPTS": "Written in-house from templates.",
    "COMPLETION_BIAS_PROMPTS": "Written in-house.",
    "COUNTERFACTUAL_TEMPLATES": "Written in-house.",
    "TOXICITY_PROMPTS": "Written in-house.",
    "HARMFUL_CONTENT_PROMPTS": "Written in-house.",
    "PRIVACY_PROMPTS": "Written in-house.",
    "SAFETY_REFUSAL_PROMPTS": "Written in-house.",
    # The four below are named after benchmarks but contain none of their data.
    # Checked entry by entry: they are hand-written items imitating each
    # benchmark's *format*, e.g. StereoSetItem("Men are typically", "strong",
    # "emotional", "hungry", "gender"), which is not a StereoSet row. The names
    # assert a provenance the content does not have — see docs/datasets.md.
    # Currently referenced by no code path. Rename or delete (POS-4).
    # VOS-6. Not prompts or dataset rows at all: reproduction targets — published
    # figures with their citations, our tolerances, and our own prose explaining
    # each deviation. The published *numbers* are facts quoted with attribution,
    # not redistributed dataset content.
    "VALIDITY_TARGETS": "Written in-house; published figures quoted with citation.",
    "CROWS_PAIRS_CURATED": "Hand-written in CrowS-Pairs format. NOT CrowS-Pairs data.",
    "WINOBIAS_CURATED": "Hand-written in WinoBias format. NOT WinoBias data.",
    "STEREOSET_CURATED": "Hand-written in StereoSet format. NOT StereoSet data.",
    "BOLD_PROMPTS_CURATED": "Hand-written in BOLD format. NOT BOLD data.",
}


def get_license(dataset_id: str) -> DatasetLicense:
    """Look up the licence record for a dataset, by loader identifier."""
    try:
        return DATASET_LICENSES[dataset_id]
    except KeyError:
        raise KeyError(
            f"{dataset_id!r} is not in the licence registry. Every dataset this "
            f"package loads must be registered in dataset_licenses.py with its "
            f"licence and citation before it can be used."
        ) from None


def non_redistributable() -> Dict[str, DatasetLicense]:
    """Datasets we may not ship copies of. Useful as a vendoring guard."""
    return {k: v for k, v in DATASET_LICENSES.items() if not v.redistributable}
