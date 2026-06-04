"""Data loading and station merging for the BART project.

Responsibilities:
    * Read the raw hourly ridership CSVs (2016 + 2017) or the local sample.
    * Read the station reference table and inject the missing WSPR station
      (referans butunlugu / referential integrity fix, analiz 2.1).
    * Merge station attributes (Location, Name) onto every origin/destination.

All paths and parameters come from `config.yaml` (no hardcoding).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_paths, load_config


def read_ridership(
    raw_dir: Path,
    file_2016: str,
    file_2017: str,
    datetime_col: str = "DateTime",
) -> pd.DataFrame:
    """Read and concatenate the 2016 and 2017 hourly ridership CSVs.

    Args:
        raw_dir: Directory containing the raw ridership CSV files.
        file_2016: File name for the 2016 ridership data.
        file_2017: File name for the 2017 ridership data.
        datetime_col: Name of the timestamp column to parse.

    Returns:
        Concatenated DataFrame with a parsed datetime column, sorted by time.
    """
    frames = [
        pd.read_csv(raw_dir / file_2016),
        pd.read_csv(raw_dir / file_2017),
    ]
    df = pd.concat(frames, ignore_index=True)
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    return df.sort_values(datetime_col).reset_index(drop=True)


def read_sample(
    sample_dir: Path,
    sample_file: str,
    datetime_col: str = "DateTime",
) -> pd.DataFrame:
    """Read the locally derived sample CSV (single file).

    Args:
        sample_dir: Directory containing the sample CSV.
        sample_file: Sample file name.
        datetime_col: Name of the timestamp column to parse.

    Returns:
        Sample DataFrame with a parsed datetime column, sorted by time.
    """
    df = pd.read_csv(sample_dir / sample_file)
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    return df.sort_values(datetime_col).reset_index(drop=True)


def read_stations(
    raw_dir: Path,
    stations_file: str,
    manual_stations: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Read the station reference table and append manual stations.

    The raw `station_info.csv` is missing the WSPR (Warm Springs) station that
    appears in the ridership data; `manual_stations` from config closes this
    referential-integrity gap.

    Args:
        raw_dir: Directory containing the station CSV.
        stations_file: Station file name.
        manual_stations: List of station dicts (abbreviation/description/
            location/name) to append. Skips ones already present.

    Returns:
        Station DataFrame including any appended manual stations.
    """
    # utf-8-sig: station_info.csv has a BOM on the first header.
    stations = pd.read_csv(raw_dir / stations_file, encoding="utf-8-sig")

    if manual_stations:
        existing = set(stations["Abbreviation"])
        to_add = [
            {
                "Abbreviation": s["abbreviation"],
                "Description": s["description"],
                "Location": s["location"],
                "Name": s["name"],
            }
            for s in manual_stations
            if s["abbreviation"] not in existing
        ]
        if to_add:
            stations = pd.concat(
                [stations, pd.DataFrame(to_add)], ignore_index=True
            )
    return stations


def merge_station_info(
    df: pd.DataFrame, stations: pd.DataFrame
) -> pd.DataFrame:
    """Merge station attributes onto both origin and destination.

    Mirrors the original notebook merge (cell-22): a left join keyed on the
    station abbreviation, with `_dest` suffix for destination-side columns.
    Helper join keys are dropped afterwards.

    Args:
        df: Ridership DataFrame with `Origin` and `Destination` columns.
        stations: Station reference table.

    Returns:
        DataFrame enriched with origin/destination `Location` and `Name`.
    """
    merged = df.merge(
        stations, left_on="Origin", right_on="Abbreviation", how="left"
    )
    merged = merged.merge(
        stations,
        left_on="Destination",
        right_on="Abbreviation",
        how="left",
        suffixes=("", "_dest"),
    )
    drop_cols = [
        c
        for c in ["Abbreviation", "Abbreviation_dest", "Description", "Description_dest"]
        if c in merged.columns
    ]
    return merged.drop(columns=drop_cols)


def load_dataset(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the full ridership dataset with station info merged.

    Honours the `use_sample` flag in config: when True the local sample is
    read, otherwise the full raw CSVs are used (e.g. on Kaggle).

    Args:
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        Merged ridership DataFrame ready for feature engineering.
    """
    cfg = config or load_config()
    paths = get_paths(cfg)
    files = cfg["files"]
    dt_col = cfg["datetime_col"]

    if cfg.get("use_sample", False):
        df = read_sample(paths["sample_dir"], files["sample"], dt_col)
    else:
        df = read_ridership(
            paths["raw_dir"], files["ridership_2016"], files["ridership_2017"], dt_col
        )

    stations = read_stations(
        paths["raw_dir"], files["stations"], cfg.get("manual_stations")
    )
    return merge_station_info(df, stations)


if __name__ == "__main__":
    # Smoke test (hizli kontrol).
    _df = load_dataset()
    print("shape:", _df.shape)
    print("columns:", list(_df.columns))
    print("date range:", _df["DateTime"].min(), "->", _df["DateTime"].max())
