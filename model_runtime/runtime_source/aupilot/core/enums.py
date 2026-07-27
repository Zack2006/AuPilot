from __future__ import annotations

from enum import StrEnum


class EvidenceMode(StrEnum):
    PROTOTYPE_ONLY = "PROTOTYPE_ONLY"
    OUTER_OOS = "OUTER_OOS"
    RELEASE_CANDIDATE = "RELEASE_CANDIDATE"
    FINAL_EVAL = "FINAL_EVAL"
    SHADOW = "SHADOW"


class Action(StrEnum):
    NO_ACTION = "NO_ACTION"
    HOLD = "HOLD"
    REDUCE_TACTICAL = "REDUCE_TACTICAL"
    REBUY_TACTICAL = "REBUY_TACTICAL"


class MacroRiskLevel(StrEnum):
    """Independent external-news risk intensity; never a trade permission."""

    APPROVED = "Approved"
    CLEARED = "Cleared"
    CAUTION = "Caution"
    HOLD = "Hold"
    CANCEL = "Cancel"

    @property
    def score(self) -> int:
        return {
            type(self).APPROVED: 1,
            type(self).CLEARED: 2,
            type(self).CAUTION: 3,
            type(self).HOLD: 4,
            type(self).CANCEL: 5,
        }[self]


class QualityState(StrEnum):
    VALID = "VALID"
    STALE = "STALE"
    INVALID = "INVALID"


class TurningPointLabel(StrEnum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    NORMAL = "NORMAL"
    AMBIGUOUS = "AMBIGUOUS"
