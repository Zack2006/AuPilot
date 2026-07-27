"""Deterministic technical indicators from canonical Databento daily OHLCV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.repositories.databento_market_repository import DatabentoMarketRepository


class FeatureService:
    def calculate(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = DatabentoMarketRepository.validate(frame).copy()
        close = data["close"]
        data["change_amount_1d"] = close.diff()
        data["return_1d"] = close.pct_change()
        data["return_5d"] = close.pct_change(5)
        for window in (5, 10, 20, 60, 120, 250):
            data[f"sma_{window}"] = close.rolling(window, min_periods=window).mean()
        data["ma_distance"] = close / data["sma_20"] - 1

        boll_std = close.rolling(20, min_periods=20).std(ddof=1)
        data["boll_mid"] = data["sma_20"]
        data["boll_upper"] = data["boll_mid"] + 2 * boll_std
        data["boll_lower"] = data["boll_mid"] - 2 * boll_std

        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        data["macd_dif"] = ema12 - ema26
        data["macd_dea"] = data["macd_dif"].ewm(span=9, adjust=False, min_periods=9).mean()
        data["macd_hist"] = 2 * (data["macd_dif"] - data["macd_dea"])

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        average_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        average_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        relative_strength = average_gain / average_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + relative_strength)
        rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
        rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)
        data["rsi_14"] = rsi

        low9 = data["low"].rolling(9, min_periods=9).min()
        high9 = data["high"].rolling(9, min_periods=9).max()
        spread = (high9 - low9).replace(0, np.nan)
        rsv = ((close - low9) / spread * 100).fillna(50).where(low9.notna())
        data["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
        data["kdj_d"] = data["kdj_k"].ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
        data["kdj_j"] = 3 * data["kdj_k"] - 2 * data["kdj_d"]

        prior_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"], (data["high"] - prior_close).abs(), (data["low"] - prior_close).abs()],
            axis=1,
        ).max(axis=1)
        data["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        data["atr_percentage"] = data["atr_14"] / close
        data["volatility_20"] = data["return_1d"].rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
        data["drawdown_from_high"] = close / close.cummax() - 1
        return data.reset_index(drop=True)
