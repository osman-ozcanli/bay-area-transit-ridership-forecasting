"""Smoke tests for the EDA module (`src/eda.py`)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.eda import (berkeley_sf_best_hour, busiest_stations, prepare_eda_frame,
                     run_eda)


def test_run_eda_returns_all_results(
    raw_df: pd.DataFrame, cfg: dict[str, Any]
) -> None:
    """`run_eda` answers all six questions and saves figures."""
    res = run_eda(raw_df, cfg)
    for key in ["stations", "least", "popular", "day_load",
                "late", "berkeley_sf", "figures"]:
        assert key in res
    # Every declared figure was actually written to disk.
    for path in res["figures"].values():
        assert path.exists(), f"figur uretilmedi: {path}"


def test_busiest_stations_sorted_desc(
    raw_df: pd.DataFrame, cfg: dict[str, Any]
) -> None:
    """Busiest-station ranking is sorted descending by total load."""
    eda_df = prepare_eda_frame(raw_df, cfg)
    s = busiest_stations(eda_df, cfg["target"], top_n=10)
    assert list(s.values) == sorted(s.values, reverse=True)


def test_berkeley_best_hour_in_range(
    raw_df: pd.DataFrame, cfg: dict[str, Any]
) -> None:
    """Berkeley->SF best hour is a valid hour of day (0-23)."""
    eda_df = prepare_eda_frame(raw_df, cfg)
    r = berkeley_sf_best_hour(eda_df, cfg)
    assert isinstance(r["best_hour"], int)
    assert 0 <= r["best_hour"] <= 23
