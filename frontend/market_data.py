"""Shared read-through cache for the two Databento gold-chart consumers."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import api


@st.cache_data(show_spinner=False, max_entries=4)
def load_gold_daily(content_sha256: str) -> dict:
    """Load one full local-cache snapshot and key it by the backend content hash."""
    payload = api.get("market/gold/daily")
    if payload["metadata"]["content_sha256"] != content_sha256:
        raise ValueError("Databento cache changed while the chart payload was loading")
    return payload


@st.cache_data(show_spinner=False, max_entries=4)
def load_gold_latest(content_sha256: str) -> dict:
    payload = api.get("market/latest")
    if payload.get("content_sha256") != content_sha256:
        raise ValueError("Databento cache changed while the latest bar was loading")
    return payload


def load_gold_cache(*, include_latest: bool = False) -> tuple[dict, dict | None]:
    """Read local data only; this function never invokes the paid refresh endpoint."""
    for attempt in range(2):
        status = api.get("market/gold/status")
        content_sha256 = status["content_sha256"]
        try:
            history = load_gold_daily(content_sha256)
            latest = load_gold_latest(content_sha256) if include_latest else None
            return history, latest
        except ValueError:
            if attempt:
                raise
            # The backend atomically published a newer manifest between the
            # status and payload reads. Retry against the new content hash.
            load_gold_daily.clear()
            load_gold_latest.clear()
    raise ValueError("Databento cache identity could not be stabilized")


def refresh_gold_cache() -> dict:
    """Run the explicit paid refresh boundary and invalidate both page caches."""
    result = api.post("market/refresh", timeout=180)
    load_gold_daily.clear()
    load_gold_latest.clear()
    return result
