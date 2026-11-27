"""Feature version 4: features.encode's columns (version 3, unchanged), then features_ext.encode_ext's block.

The encoder module valuenet.ENCODERS[4] plays: `encode`, `CATEGORY_OF_SLOT`, and the v3 names hp_lead reads
(`PLAYER_NUMS`, `PLAYER_WIDTH`, which are positions inside the v3 part, so they are the same here). A model's
meta `ext_parts` (default all) picks which ext parts carry values; the others encode as zeros (`encoder_for`).
"""

from __future__ import annotations

import functools
from typing import Any

from yomibench import features, features_ext
from yomibench.features import PLAYER_NUMS, PLAYER_WIDTH, Vocab  # noqa: F401  (hp_lead reads these)

N_V3_NUMERIC = features.N_NUMERIC
N_V3_CATEGORICAL = features.N_CATEGORICAL
N_NUMERIC = N_V3_NUMERIC + features_ext.N_EXT_NUMERIC
CATEGORY_OF_SLOT = list(features.CATEGORY_OF_SLOT) + list(features_ext.EXT_CATEGORY_OF_SLOT)
N_CATEGORICAL = len(CATEGORY_OF_SLOT)
NUMERIC_NAMES = tuple(features.NUMERIC_NAMES) + tuple(features_ext.EXT_NUMERIC_NAMES)


def encode(state: dict[str, Any], seat: int, vocab: Vocab,
           parts: tuple[int, ...] = features_ext.PARTS) -> tuple[list[float], list[int]]:
    nums, cats = features.encode(state, seat, vocab)
    en, ec = features_ext.encode_ext(state, seat, vocab, parts)
    return nums + en, cats + ec


class _Encoder:
    """A module-like encoder with the ext parts bound (what ValueNet calls .encode on)."""

    def __init__(self, parts: tuple[int, ...]):
        self.parts = tuple(parts)
        self.encode = functools.partial(encode, parts=self.parts)
        self.CATEGORY_OF_SLOT = CATEGORY_OF_SLOT
        self.PLAYER_NUMS = PLAYER_NUMS
        self.PLAYER_WIDTH = PLAYER_WIDTH


def encoder_for(meta: dict[str, Any]):
    parts = tuple(meta.get("ext_parts", features_ext.PARTS))
    return _Encoder(parts)
