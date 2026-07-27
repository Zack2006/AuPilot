from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def restore_balanced_class_probabilities(
    weighted_probabilities: pd.DataFrame,
    *,
    class_counts: Mapping[str, int],
    class_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Undo sklearn's balanced class weights at the leaf-probability level.

    A weighted tree reports weighted class proportions. Dividing each column by
    its training class weight and renormalizing recovers the unweighted empirical
    leaf proportions. This is a prior restoration, not a probability calibration.
    """

    classes = tuple(class_columns)
    if not classes:
        raise ValueError("At least one class is required")
    if set(class_counts) != set(classes):
        raise ValueError("class_counts and class_columns must identify the same classes")
    if any(int(class_counts[label]) <= 0 for label in classes):
        raise ValueError("Every class count must be positive")
    columns = [class_columns[label] for label in classes]
    missing = set(columns) - set(weighted_probabilities.columns)
    if missing:
        raise ValueError(f"Missing weighted probability columns: {sorted(missing)}")
    values = weighted_probabilities.loc[:, columns].to_numpy(dtype=float)
    if (
        values.ndim != 2
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or (values > 1.0).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-12)
    ):
        raise ValueError("Weighted probabilities are invalid")

    total = float(sum(int(class_counts[label]) for label in classes))
    class_total = float(len(classes))
    balanced_weights = np.asarray(
        [
            total / (class_total * float(class_counts[label]))
            for label in classes
        ],
        dtype=float,
    )
    restored = values / balanced_weights
    denominators = restored.sum(axis=1, keepdims=True)
    if (denominators <= 0.0).any():
        raise ValueError("Balanced probabilities cannot be restored")
    restored /= denominators
    output = pd.DataFrame(
        restored,
        index=weighted_probabilities.index,
        columns=[f"u_{label.lower()}" for label in classes],
    )
    return output


def event_probability_lifts(
    restored_probabilities: pd.DataFrame,
    *,
    class_counts: Mapping[str, int],
    event_classes: Sequence[str] = ("TOP", "BOTTOM"),
) -> pd.DataFrame:
    """Express each restored event probability as lift over its train prior."""

    if any(int(value) <= 0 for value in class_counts.values()):
        raise ValueError("Every class count must be positive")
    total = float(sum(int(value) for value in class_counts.values()))
    output = pd.DataFrame(index=restored_probabilities.index)
    for label in event_classes:
        if label not in class_counts:
            raise ValueError(f"Missing class count for {label}")
        column = f"u_{label.lower()}"
        if column not in restored_probabilities.columns:
            raise ValueError(f"Missing restored probability column: {column}")
        probability = pd.to_numeric(restored_probabilities[column], errors="coerce")
        if (
            not np.isfinite(probability.to_numpy(dtype=float)).all()
            or (probability < 0.0).any()
            or (probability > 1.0).any()
        ):
            raise ValueError("Restored event probabilities are invalid")
        prior = float(class_counts[label]) / total
        output[f"{label.lower()}_prior"] = prior
        output[f"{label.lower()}_lift"] = probability / prior
    return output


def threshold_event_lift_labels(
    event_lifts: pd.DataFrame,
    *,
    multiplier: float | pd.Series,
) -> pd.Series:
    """Convert event lifts to TOP/BOTTOM/NORMAL without using portfolio state."""

    required = {"top_lift", "bottom_lift"}
    missing = required - set(event_lifts.columns)
    if missing:
        raise ValueError(f"Missing event lift columns: {sorted(missing)}")
    values = event_lifts.loc[:, ["top_lift", "bottom_lift"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0.0).any().any():
        raise ValueError("Event lifts must be finite and non-negative")
    thresholds = (
        pd.Series(float(multiplier), index=values.index)
        if np.isscalar(multiplier)
        else pd.to_numeric(multiplier, errors="coerce").reindex(values.index)
    )
    if (
        not np.isfinite(thresholds.to_numpy(dtype=float)).all()
        or (thresholds < 1.0).any()
    ):
        raise ValueError("Lift multipliers must be finite and at least one")
    top = values["top_lift"].ge(thresholds) & values["top_lift"].gt(
        values["bottom_lift"]
    )
    bottom = values["bottom_lift"].ge(thresholds) & values["bottom_lift"].gt(
        values["top_lift"]
    )
    labels = pd.Series("NORMAL", index=values.index, dtype="string")
    labels.loc[top] = "TOP"
    labels.loc[bottom] = "BOTTOM"
    return labels
