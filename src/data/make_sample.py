"""Derive a small, time-aware sample from the full raw ridership data.

Why a sample: the full data is ~13.3M rows (428 MB). For fast local pipeline
development we keep a representative subset that **preserves temporal
continuity** so lag/rolling features and the temporal train/test split stay
meaningful.

Strategy `by_station`: pick the busiest origin stations plus mandatory
edge-case stations (e.g. WSPR) and keep *every* hourly record for them across
the full 2016-2017 timeline. This keeps the hourly sequence unbroken — unlike
random row sampling, which would destroy it.

The real, full-data training is still done on Kaggle (`use_sample: false`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_paths, load_config
from src.data.load import read_ridership


def select_stations(
    df: pd.DataFrame,
    n_stations: int,
    must_include: list[str],
) -> list[str]:
    """Select origin stations: busiest first, with mandatory ones forced in.

    Args:
        df: Ridership DataFrame with `Origin` and `Throughput`.
        n_stations: Total number of origin stations to keep.
        must_include: Station codes that must always be present (edge cases).

    Returns:
        Sorted list of selected origin station codes.
    """
    totals = df.groupby("Origin")["Throughput"].sum().sort_values(ascending=False)
    selected = [s for s in must_include if s in totals.index]
    for code in totals.index:
        if len(selected) >= n_stations:
            break
        if code not in selected:
            selected.append(code)
    return sorted(selected)


def make_sample(config: dict[str, Any] | None = None) -> Path:
    """Build and persist the time-aware sample CSV.

    Args:
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        Path to the written sample CSV.
    """
    cfg = config or load_config()
    paths = get_paths(cfg)
    files = cfg["files"]
    samp = cfg["sampling"]
    dt_col = cfg["datetime_col"]

    # Read full raw ridership (only the 4 base columns we sample on).
    df = read_ridership(
        paths["raw_dir"], files["ridership_2016"], files["ridership_2017"], dt_col
    )

    strategy = samp["strategy"]
    if strategy == "by_station":
        stations = select_stations(df, samp["n_stations"], samp.get("must_include", []))
        # Keep a closed sub-network: both endpoints among the selected stations.
        # This preserves each OD pair's full hourly timeline while keeping the
        # sample compact (~12x12 OD pairs instead of 12x46).
        sample = df[
            df["Origin"].isin(stations) & df["Destination"].isin(stations)
        ].copy()
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    sample = sample.sort_values(dt_col).reset_index(drop=True)

    out_dir = paths["sample_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / files["sample"]
    sample.to_csv(out_path, index=False)

    # Report (rapor) — so the run is verifiable.
    self_trips = int((sample["Origin"] == sample["Destination"]).sum())
    print("=== Sample created ===")
    print(f"strategy        : {strategy}")
    print(f"origin stations : {len(stations)} -> {stations}")
    print(f"rows            : {len(sample):,} (from {len(df):,} = "
          f"{len(sample) / len(df) * 100:.2f}%)")
    print(f"date range      : {sample[dt_col].min()} -> {sample[dt_col].max()}")
    print(f"self-trips      : {self_trips:,}")
    print(f"WSPR present    : {'WSPR' in set(sample['Origin'])}")
    print(f"written to      : {out_path}")
    return out_path


if __name__ == "__main__":
    make_sample()
