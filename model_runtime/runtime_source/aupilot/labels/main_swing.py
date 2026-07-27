from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MainSwingPolicy:
    """Define an unforced alternating price swing with explicit confirmation."""

    reversal_threshold: float = 0.05
    threshold_mode: str = "log_symmetric"
    plateau_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.reversal_threshold < 1.0:
            raise ValueError("reversal_threshold must be between zero and one")
        if self.threshold_mode not in {"arithmetic", "log_symmetric"}:
            raise ValueError("threshold_mode must be arithmetic or log_symmetric")
        if self.plateau_tolerance < 0.0:
            raise ValueError("plateau_tolerance must be non-negative")

    @property
    def upward_factor(self) -> float:
        return 1.0 + self.reversal_threshold

    @property
    def downward_factor(self) -> float:
        if self.threshold_mode == "log_symmetric":
            return 1.0 / self.upward_factor
        return 1.0 - self.reversal_threshold


@dataclass(frozen=True)
class MainSwingPathResult:
    events: pd.DataFrame
    state_tape: pd.DataFrame
    active_trend: str
    open_candidate_type: str | None
    open_candidate_at: object | None
    open_candidate_price: float | None
    open_candidate_age_sessions: int | None


_EVENT_COLUMNS = (
    "sequence_id",
    "event_type",
    "event_at",
    "event_price",
    "confirmed_at",
    "confirmation_price",
    "confirmation_delay_sessions",
    "label_available_at",
    "previous_event_at",
    "previous_event_price",
    "leg_log_amplitude",
    "reversal_threshold",
    "threshold_mode",
)

_STATE_COLUMNS = (
    "trade_date",
    "observed_price",
    "active_trend",
    "causal_episode_id",
    "candidate_type",
    "candidate_at",
    "candidate_price",
    "confirmed_event_type_at_t",
)


def _same_extreme(left: float, right: float, tolerance: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=tolerance))


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=_EVENT_COLUMNS)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    frame = prices.loc[:, ["trade_date", "close"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.sort_values("trade_date", kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Main-swing input is empty")
    if frame["trade_date"].duplicated().any():
        raise ValueError("Duplicate trade_date")
    values = frame["close"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("close must be finite and positive")
    return frame


def scan_main_swing_path(
    prices: pd.DataFrame,
    policy: MainSwingPolicy | None = None,
) -> MainSwingPathResult:
    """Scan a price path without ever forcing an event at a time boundary.

    A candidate extreme is silently replaced by a more extreme observation in
    the same direction. It becomes a confirmed event only after the opposite
    fixed barrier is crossed. Consequently, confirmed events alternate and an
    arbitrarily long one-sided path remains one open episode.
    """

    policy = MainSwingPolicy() if policy is None else policy
    frame = _prepare_prices(prices)
    dates = frame["trade_date"].tolist()
    values = frame["close"].to_numpy(dtype=float)

    state = "UNKNOWN"
    running_high = float(values[0])
    running_high_index = 0
    running_low = float(values[0])
    running_low_index = 0
    sequence_id = 0
    records: list[dict[str, object]] = []
    state_records: list[dict[str, object]] = []

    def state_record(index: int, confirmed: str | None) -> dict[str, object]:
        if state == "UP":
            candidate_type = "TOP"
            candidate_index = running_high_index
            candidate_price = running_high
        elif state == "DOWN":
            candidate_type = "BOTTOM"
            candidate_index = running_low_index
            candidate_price = running_low
        else:
            candidate_type = None
            candidate_index = None
            candidate_price = None
        return {
            "trade_date": dates[index],
            "observed_price": float(values[index]),
            "active_trend": state,
            "causal_episode_id": sequence_id,
            "candidate_type": candidate_type,
            "candidate_at": (
                dates[candidate_index] if candidate_index is not None else pd.NaT
            ),
            "candidate_price": candidate_price,
            "confirmed_event_type_at_t": confirmed,
        }

    state_records.append(state_record(0, None))
    for index in range(1, len(values)):
        current = float(values[index])
        confirmed_event: str | None = None
        if state == "UNKNOWN":
            if current > running_high or _same_extreme(
                current, running_high, policy.plateau_tolerance
            ):
                running_high = current
                running_high_index = index
            if current < running_low or _same_extreme(
                current, running_low, policy.plateau_tolerance
            ):
                running_low = current
                running_low_index = index
            top_trigger = (
                running_high_index < index
                and current <= running_high * policy.downward_factor
            )
            bottom_trigger = (
                running_low_index < index
                and current >= running_low * policy.upward_factor
            )
            if top_trigger and bottom_trigger:
                raise RuntimeError("Ambiguous initial main-swing direction")
            if top_trigger:
                state = "DOWN"
                running_low = current
                running_low_index = index
            elif bottom_trigger:
                state = "UP"
                running_high = current
                running_high_index = index
            state_records.append(state_record(index, None))
            continue

        if state == "UP":
            if current > running_high or _same_extreme(
                current, running_high, policy.plateau_tolerance
            ):
                running_high = current
                running_high_index = index
            if (
                running_high_index < index
                and current <= running_high * policy.downward_factor
            ):
                sequence_id += 1
                previous = records[-1] if records else None
                previous_price = (
                    float(previous["event_price"]) if previous is not None else None
                )
                records.append(
                    {
                        "sequence_id": sequence_id,
                        "event_type": "TOP",
                        "event_at": dates[running_high_index],
                        "event_price": running_high,
                        "confirmed_at": dates[index],
                        "confirmation_price": current,
                        "confirmation_delay_sessions": index - running_high_index,
                        "label_available_at": dates[index],
                        "previous_event_at": (
                            previous["event_at"] if previous is not None else pd.NaT
                        ),
                        "previous_event_price": previous_price,
                        "leg_log_amplitude": (
                            float(np.log(running_high / previous_price))
                            if previous_price is not None
                            else np.nan
                        ),
                        "reversal_threshold": policy.reversal_threshold,
                        "threshold_mode": policy.threshold_mode,
                    }
                )
                confirmed_event = "TOP"
                state = "DOWN"
                running_low = current
                running_low_index = index
            state_records.append(state_record(index, confirmed_event))
            continue

        if current < running_low or _same_extreme(
            current, running_low, policy.plateau_tolerance
        ):
            running_low = current
            running_low_index = index
        if (
            running_low_index < index
            and current >= running_low * policy.upward_factor
        ):
            sequence_id += 1
            previous = records[-1] if records else None
            previous_price = (
                float(previous["event_price"]) if previous is not None else None
            )
            records.append(
                {
                    "sequence_id": sequence_id,
                    "event_type": "BOTTOM",
                    "event_at": dates[running_low_index],
                    "event_price": running_low,
                    "confirmed_at": dates[index],
                    "confirmation_price": current,
                    "confirmation_delay_sessions": index - running_low_index,
                    "label_available_at": dates[index],
                    "previous_event_at": (
                        previous["event_at"] if previous is not None else pd.NaT
                    ),
                    "previous_event_price": previous_price,
                    "leg_log_amplitude": (
                        float(np.log(previous_price / running_low))
                        if previous_price is not None
                        else np.nan
                    ),
                    "reversal_threshold": policy.reversal_threshold,
                    "threshold_mode": policy.threshold_mode,
                }
            )
            confirmed_event = "BOTTOM"
            state = "UP"
            running_high = current
            running_high_index = index
        state_records.append(state_record(index, confirmed_event))

    events = (
        pd.DataFrame.from_records(records, columns=_EVENT_COLUMNS)
        if records
        else _empty_events()
    )
    if not events.empty:
        events["sequence_id"] = events["sequence_id"].astype("Int64")
        events["confirmation_delay_sessions"] = events[
            "confirmation_delay_sessions"
        ].astype("Int64")
    state_tape = pd.DataFrame.from_records(state_records, columns=_STATE_COLUMNS)
    state_tape["causal_episode_id"] = state_tape["causal_episode_id"].astype("Int64")

    final_state = state_tape.iloc[-1]
    candidate_at = final_state["candidate_at"]
    if pd.isna(candidate_at):
        open_type = None
        open_at = None
        open_price = None
        open_age = None
    else:
        open_type = str(final_state["candidate_type"])
        open_at = candidate_at
        open_price = float(final_state["candidate_price"])
        open_age = len(frame) - 1 - dates.index(candidate_at)
    return MainSwingPathResult(
        events=events,
        state_tape=state_tape,
        active_trend=str(final_state["active_trend"]),
        open_candidate_type=open_type,
        open_candidate_at=open_at,
        open_candidate_price=open_price,
        open_candidate_age_sessions=open_age,
    )


def label_main_swing_calendar(
    prices: pd.DataFrame,
    *,
    horizon_sessions: int = 30,
    policy: MainSwingPolicy | None = None,
) -> tuple[pd.DataFrame, MainSwingPathResult]:
    """Build a conservative finite-horizon calendar target from the unbounded path."""

    if horizon_sessions < 1:
        raise ValueError("horizon_sessions must be positive")
    frame = _prepare_prices(prices)
    result = scan_main_swing_path(frame, policy)
    dates = frame["trade_date"].tolist()
    states = result.state_tape.reset_index(drop=True)
    row_count = len(frame)
    labels = pd.DataFrame(
        {
            "trade_date": dates,
            "label": pd.Series([pd.NA] * row_count, dtype="string"),
            "label_status": "RIGHT_CENSORED",
            "label_available_at": pd.Series([pd.NaT] * row_count, dtype="object"),
            "resolution_reason": "RIGHT_CENSORED_OPEN_CANDIDATE",
            "event_group_id": [
                f"main-swing-state:{int(value):05d}"
                for value in states["causal_episode_id"]
            ],
            "event_sequence_id": pd.Series(
                [pd.NA] * row_count, dtype="Int64"
            ),
            "confirmed_at": pd.Series([pd.NaT] * row_count, dtype="object"),
            "confirmation_delay_sessions": pd.Series(
                [pd.NA] * row_count, dtype="Int64"
            ),
            "unbounded_event_type": pd.Series(
                [pd.NA] * row_count, dtype="string"
            ),
            "unbounded_confirmed_at": pd.Series(
                [pd.NaT] * row_count, dtype="object"
            ),
            "turn_within_horizon": False,
            "horizon_sessions": horizon_sessions,
        }
    )
    positions = {value: index for index, value in enumerate(dates)}
    event_by_index: dict[int, object] = {}
    for event in result.events.itertuples(index=False):
        event_index = positions[event.event_at]
        event_by_index[event_index] = event
        labels.loc[event_index, "event_sequence_id"] = int(event.sequence_id)
        labels.loc[event_index, "unbounded_event_type"] = str(event.event_type)
        labels.loc[event_index, "unbounded_confirmed_at"] = event.confirmed_at

    for index in range(row_count):
        event = event_by_index.get(index)
        delay = int(event.confirmation_delay_sessions) if event is not None else None
        if event is not None and delay is not None and delay <= horizon_sessions:
            labels.loc[index, "label"] = str(event.event_type)
            labels.loc[index, "label_available_at"] = event.confirmed_at
            labels.loc[index, "confirmed_at"] = event.confirmed_at
            labels.loc[index, "confirmation_delay_sessions"] = delay
            labels.loc[index, "turn_within_horizon"] = True
            labels.loc[index, "label_status"] = "MATURED"
            labels.loc[index, "resolution_reason"] = "CONFIRMED_TURN"
            continue

        candidate_at = states.loc[index, "candidate_at"]
        if pd.isna(candidate_at) or candidate_at != dates[index]:
            resolution_index = index + 1 if index + 1 < row_count else None
            resolution_reason = "NOT_CAUSAL_CANDIDATE"
        else:
            resolution_index = None
            search_stop = min(row_count - 1, index + horizon_sessions)
            for future_index in range(index + 1, search_stop + 1):
                if states.loc[future_index, "candidate_at"] != dates[index]:
                    resolution_index = future_index
                    break
            if resolution_index is not None:
                resolution_reason = "SUPERSEDED_OR_STATE_CHANGED"
            elif index + horizon_sessions < row_count:
                resolution_index = index + horizon_sessions
                resolution_reason = "NO_CONFIRMATION_WITHIN_HORIZON"
            else:
                resolution_reason = "RIGHT_CENSORED_OPEN_CANDIDATE"

        if resolution_index is not None:
            labels.loc[index, "label"] = "NORMAL"
            labels.loc[index, "label_status"] = "MATURED"
            labels.loc[index, "label_available_at"] = dates[resolution_index]
            labels.loc[index, "resolution_reason"] = resolution_reason
    return labels, result
