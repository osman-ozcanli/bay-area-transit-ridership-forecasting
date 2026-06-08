"""Smoke tests for feature engineering (`src/features/build_features.py`).

Guards the methodological decisions: self-trip drop, valid distances, and
leakage-safe lag features.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def test_self_trips_dropped(feat_df: pd.DataFrame) -> None:
    """Origin==Destination records are removed (analiz 3.5, policy=drop)."""
    assert int((feat_df["Origin"] == feat_df["Destination"]).sum()) == 0


def test_distance_valid(feat_df: pd.DataFrame) -> None:
    """`dist_km` is present, non-null and strictly positive (self-trips gone)."""
    assert feat_df["dist_km"].notna().all()
    assert (feat_df["dist_km"] > 0).all()


def test_categoricals_are_category_dtype(
    feat_df: pd.DataFrame, cfg: dict[str, Any]
) -> None:
    """Configured categorical columns use the pandas `category` dtype."""
    for col in cfg["features"]["categorical"]:
        assert str(feat_df[col].dtype) == "category", f"{col} category olmali"


def test_lag_is_leakage_safe(feat_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """First observation of each OD pair has a NaN lag (only past is used)."""
    target = cfg["target"]
    lag1 = f"{target}_lag_1"
    assert lag1 in feat_df.columns

    first_per_pair = (
        feat_df.sort_values(cfg["datetime_col"])
        .groupby(["Origin", "Destination"], observed=True)
        .head(1)
    )
    # No past record exists for the first row of each pair -> must be NaN.
    assert first_per_pair[lag1].isna().all()
