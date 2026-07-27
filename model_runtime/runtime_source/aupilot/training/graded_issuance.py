"""Common-risk-set 21-slot issuance tables for graded pivot labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

GRADED_LABELS = (
    "NORMAL",
    "TOP_L1",
    "TOP_L2",
    "TOP_L3",
    "BOTTOM_L1",
    "BOTTOM_L2",
    "BOTTOM_L3",
)


@dataclass(frozen=True)
class GradedLabelSource:
    frame: pd.DataFrame
    label_column: str
    status_column: str
    available_at_column: str


@dataclass(frozen=True)
class CommonGradedIssuanceResult:
    tables: dict[str, pd.DataFrame]
    audit: dict[str, Any]


@dataclass(frozen=True)
class SingleGradedIssuanceResult:
    table: pd.DataFrame
    audit: dict[str, Any]


def _candidate_map(
    source: GradedLabelSource,
    *,
    name: str,
) -> pd.DataFrame:
    required = {
        "trade_date",
        source.label_column,
        source.status_column,
        source.available_at_column,
    }
    missing = required - set(source.frame.columns)
    if missing:
        raise ValueError(f"{name} graded source missing: {sorted(missing)}")
    frame = source.frame.loc[:, list(required)].copy()
    frame["target_date"] = pd.to_datetime(
        frame["trade_date"],
        errors="coerce",
        utc=True,
    ).dt.date
    if (
        frame["target_date"].isna().any()
        or frame["target_date"].duplicated().any()
    ):
        raise ValueError(f"{name} candidate dates must be unique and valid")
    frame["target_label"] = frame[source.label_column].astype("string")
    frame["target_status"] = frame[source.status_column].astype(str)
    frame["target_label_available_at_utc"] = pd.to_datetime(
        frame[source.available_at_column],
        errors="coerce",
        utc=True,
    )
    frame["target_mature"] = frame["target_status"].str.startswith(
        "MATURED"
    )
    mature = frame["target_mature"]
    if frame.loc[mature, "target_label_available_at_utc"].isna().any():
        raise ValueError(f"{name} matured labels lack availability")
    invalid_labels = sorted(
        set(frame.loc[mature, "target_label"].astype(str))
        - set(GRADED_LABELS)
    )
    if invalid_labels:
        raise ValueError(f"{name} unsupported labels: {invalid_labels}")
    return frame.loc[
        :,
        [
            "target_date",
            "target_label",
            "target_status",
            "target_label_available_at_utc",
            "target_mature",
        ],
    ].sort_values("target_date", kind="stable").reset_index(drop=True)


def _validate_skeleton(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "issuance_id",
        "feature_anchor_bucket",
        "feature_anchor_position",
        "horizon_index",
        "target_bucket",
        "target_bucket_end_utc",
        "target_event_group_id",
        "target_label",
        "target_label_available_at_utc",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Graded issuance skeleton missing: {sorted(missing)}")
    frame = rows.copy().reset_index(drop=True)
    frame["_target_date"] = pd.to_datetime(
        frame["target_bucket"],
        errors="coerce",
        utc=True,
    ).dt.date
    frame["target_bucket_end_utc"] = pd.to_datetime(
        frame["target_bucket_end_utc"],
        errors="coerce",
        utc=True,
    )
    if (
        frame["_target_date"].isna().any()
        or frame["target_bucket_end_utc"].isna().any()
    ):
        raise ValueError("Graded issuance skeleton dates are invalid")
    sizes = frame.groupby("issuance_id", sort=False).size()
    if not sizes.eq(21).all():
        raise ValueError("Graded issuance skeleton must have 21-row issuances")
    expected = tuple(range(1, 22))
    horizons = frame.groupby("issuance_id", sort=False)[
        "horizon_index"
    ].apply(tuple)
    if not horizons.map(lambda value: value == expected).all():
        raise ValueError("Graded issuance horizons must be exactly 1...21")
    return frame


def build_common_graded_issuance_tables(
    skeleton_rows: pd.DataFrame,
    sources: dict[str, GradedLabelSource],
) -> CommonGradedIssuanceResult:
    """Map labels onto one issuance skeleton and retain one common risk set."""

    if tuple(sources) != ("G01", "G03", "G04"):
        raise ValueError("Graded sources must be ordered G01, G03, G04")
    skeleton = _validate_skeleton(skeleton_rows)
    candidate_maps = {
        name: _candidate_map(source, name=name)
        for name, source in sources.items()
    }
    candidate_date_sets = {
        name: frozenset(frame["target_date"])
        for name, frame in candidate_maps.items()
    }
    if len(set(candidate_date_sets.values())) != 1:
        raise ValueError("Graded label candidate universes differ")
    candidate_dates = next(iter(candidate_date_sets.values()))
    common_mature_dates = set(candidate_dates)
    for frame in candidate_maps.values():
        common_mature_dates &= set(
            frame.loc[frame["target_mature"], "target_date"]
        )
    common_mature_dates = frozenset(common_mature_dates)
    unresolved_dates = candidate_dates - common_mature_dates

    row_is_common = ~skeleton["_target_date"].isin(unresolved_dates)
    complete_issuance = row_is_common.groupby(
        skeleton["issuance_id"],
        sort=False,
    ).all()
    retained_ids = set(complete_issuance.loc[complete_issuance].index)
    common = skeleton.loc[
        skeleton["issuance_id"].isin(retained_ids)
    ].copy().reset_index(drop=True)
    if not common.groupby("issuance_id", sort=False).size().eq(21).all():
        raise AssertionError("Common maturity filtering split an issuance")

    tables: dict[str, pd.DataFrame] = {}
    label_audits: dict[str, Any] = {}
    for name, candidate_map in candidate_maps.items():
        lookup = candidate_map.set_index("target_date")
        labels = lookup["target_label"].to_dict()
        availability = lookup[
            "target_label_available_at_utc"
        ].to_dict()
        output = common.copy()
        output["target_label"] = [
            str(labels.get(value, "NORMAL"))
            for value in output["_target_date"]
        ]
        output["target_label_available_at_utc"] = [
            availability[value]
            if value in availability
            else bucket_end
            for value, bucket_end in zip(
                output["_target_date"],
                output["target_bucket_end_utc"],
                strict=True,
            )
        ]
        if (
            output["target_label_available_at_utc"].isna().any()
            or not set(output["target_label"]).issubset(GRADED_LABELS)
        ):
            raise AssertionError(f"{name} mapped issuance is invalid")
        output["target_label_maturity_reason"] = (
            "NON_CANDIDATE_BUCKET_COMPLETE"
        )
        output.loc[
            output["_target_date"].isin(candidate_dates),
            "target_label_maturity_reason",
        ] = f"{name}_MATURED_CANDIDATE_LABEL"
        output = output.drop(columns="_target_date").reset_index(drop=True)
        tables[name] = output
        label_audits[name] = {
            "rows": len(output),
            "issuances": int(output["issuance_id"].nunique()),
            "label_counts": {
                str(key): int(value)
                for key, value in output["target_label"].value_counts().items()
            },
            "candidate_maturity_max": (
                output.loc[
                    output["target_label_maturity_reason"].eq(
                        f"{name}_MATURED_CANDIDATE_LABEL"
                    ),
                    "target_label_available_at_utc",
                ]
                .max()
                .isoformat()
            ),
        }

    identity_columns = (
        "issuance_id",
        "feature_anchor_bucket",
        "feature_anchor_position",
        "horizon_index",
        "target_bucket",
        "target_event_group_id",
    )
    identity = tables["G01"].loc[:, identity_columns]
    if not all(
        identity.equals(table.loc[:, identity_columns])
        for table in tables.values()
    ):
        raise AssertionError("Graded issuance identities differ")
    return CommonGradedIssuanceResult(
        tables=tables,
        audit={
            "candidate_dates": len(candidate_dates),
            "common_mature_candidate_dates": len(common_mature_dates),
            "unresolved_candidate_dates": len(unresolved_dates),
            "input_rows": len(skeleton),
            "input_issuances": int(skeleton["issuance_id"].nunique()),
            "retained_rows": len(common),
            "retained_issuances": int(common["issuance_id"].nunique()),
            "removed_issuances": int(
                skeleton["issuance_id"].nunique()
                - common["issuance_id"].nunique()
            ),
            "issuance_all_in_all_out": True,
            "shared_row_identity": True,
            "labels": label_audits,
        },
    )


def build_single_graded_issuance_table(
    skeleton_rows: pd.DataFrame,
    source: GradedLabelSource,
    *,
    name: str,
) -> SingleGradedIssuanceResult:
    """Map one graded label source without importing another label's censoring."""

    if not name:
        raise ValueError("Single graded issuance source name must be nonempty")
    skeleton = _validate_skeleton(skeleton_rows)
    candidate_map = _candidate_map(source, name=name)
    candidate_dates = frozenset(candidate_map["target_date"])
    mature_candidate_dates = frozenset(
        candidate_map.loc[
            candidate_map["target_mature"],
            "target_date",
        ]
    )
    unresolved_dates = candidate_dates - mature_candidate_dates

    row_is_mature = ~skeleton["_target_date"].isin(unresolved_dates)
    complete_issuance = row_is_mature.groupby(
        skeleton["issuance_id"],
        sort=False,
    ).all()
    retained_ids = set(complete_issuance.loc[complete_issuance].index)
    output = skeleton.loc[
        skeleton["issuance_id"].isin(retained_ids)
    ].copy().reset_index(drop=True)
    if not output.groupby("issuance_id", sort=False).size().eq(21).all():
        raise AssertionError("Single-source maturity filtering split an issuance")

    lookup = candidate_map.set_index("target_date")
    labels = lookup["target_label"].to_dict()
    availability = lookup["target_label_available_at_utc"].to_dict()
    output["target_label"] = [
        str(labels.get(value, "NORMAL"))
        for value in output["_target_date"]
    ]
    output["target_label_available_at_utc"] = [
        availability[value]
        if value in availability
        else bucket_end
        for value, bucket_end in zip(
            output["_target_date"],
            output["target_bucket_end_utc"],
            strict=True,
        )
    ]
    if (
        output["target_label_available_at_utc"].isna().any()
        or not set(output["target_label"]).issubset(GRADED_LABELS)
        or output["_target_date"].isin(unresolved_dates).any()
    ):
        raise AssertionError(f"{name} single-source issuance is invalid")
    output["target_label_maturity_reason"] = (
        "NON_CANDIDATE_BUCKET_COMPLETE"
    )
    output.loc[
        output["_target_date"].isin(candidate_dates),
        "target_label_maturity_reason",
    ] = f"{name}_MATURED_CANDIDATE_LABEL"
    output = output.drop(columns="_target_date").reset_index(drop=True)

    maturity_max = output.loc[
        output["target_label_maturity_reason"].eq(
            f"{name}_MATURED_CANDIDATE_LABEL"
        ),
        "target_label_available_at_utc",
    ].max()
    return SingleGradedIssuanceResult(
        table=output,
        audit={
            "source_name": name,
            "candidate_dates": len(candidate_dates),
            "mature_candidate_dates": len(mature_candidate_dates),
            "unresolved_candidate_dates": len(unresolved_dates),
            "unresolved_candidate_date_values": [
                value.isoformat() for value in sorted(unresolved_dates)
            ],
            "input_rows": len(skeleton),
            "input_issuances": int(skeleton["issuance_id"].nunique()),
            "retained_rows": len(output),
            "retained_issuances": int(output["issuance_id"].nunique()),
            "removed_issuances": int(
                skeleton["issuance_id"].nunique()
                - output["issuance_id"].nunique()
            ),
            "label_counts": {
                str(key): int(value)
                for key, value in output["target_label"].value_counts().items()
            },
            "candidate_maturity_max": maturity_max.isoformat(),
            "issuance_all_in_all_out": True,
            "foreign_label_censoring_used": False,
        },
    )
