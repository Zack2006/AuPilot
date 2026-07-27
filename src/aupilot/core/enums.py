from __future__ import annotations

from enum import StrEnum


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
