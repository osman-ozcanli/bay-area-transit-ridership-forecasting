"""Feature engineering for the BART ridership project.

Pipeline (config-driven):
    1. Time features        : Hour, DayOfWeek, Month, IsWeekend, Period
    2. Holiday flag         : California (CA) public holidays  -> IsHoliday
    3. Distance             : haversine km between origin & destination
    4. Self-trip handling   : Origin==Destination policy (drop/flag/keep)
    5. Lag / rolling        : per-OD past throughput (time-sorted, leakage-safe)
    6. Categorical dtype     : Origin/Destination/... as `category` for LightGBM
                              (cat.codes ordinal varsayimindan kacinma, analiz 3.3)

All parameters come from `config.yaml` (no hardcoding).
"""
from __future__ import annotations

from typing import Any

import holidays
import numpy as np
import pandas as pd

from src.config import load_config

# Period bins (saat dilimleri) — original notebook (cell-9) ile ayni.
_PERIOD_BINS = [0, 5, 10, 16, 20, 24]
_PERIOD_LABELS = ["LateNight", "Morning", "Midday", "Evening", "Night"]


def add_time_features(df: pd.DataFrame, datetime_col: str = "DateTime") -> pd.DataFrame:
    """Add calendar/time features derived from the timestamp.

    Args:
        df: DataFrame with a parsed datetime column.
        datetime_col: Name of the timestamp column.

    Returns:
        DataFrame with Hour, DayOfWeek, Month, IsWeekend and Period columns.
    """
    df = df.copy()
    ts = df[datetime_col]
    df["Hour"] = ts.dt.hour
    df["DayOfWeek"] = ts.dt.day_name()
    df["Month"] = ts.dt.month
    df["IsWeekend"] = ts.dt.day_name().isin(["Saturday", "Sunday"]).astype(int)
    df["Period"] = pd.cut(
        df["Hour"], bins=_PERIOD_BINS, labels=_PERIOD_LABELS, right=False
    )
    return df


def add_holiday_feature(
    df: pd.DataFrame, datetime_col: str = "DateTime", subdiv: str = "CA"
) -> pd.DataFrame:
    """Add a binary IsHoliday flag based on public holidays.

    Args:
        df: DataFrame with a parsed datetime column.
        datetime_col: Name of the timestamp column.
        subdiv: US subdivision (state) code for the holiday calendar.

    Returns:
        DataFrame with an `IsHoliday` (0/1) column.
    """
    df = df.copy()
    years = sorted(df[datetime_col].dt.year.dropna().unique().tolist())
    us_holidays = holidays.US(subdiv=subdiv, years=years)
    df["IsHoliday"] = df[datetime_col].dt.date.isin(us_holidays).astype(int)
    return df


def _parse_lonlat(location: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse a 'lon,lat[,alt]' string Series into (lon, lat) float Series.

    Robust to entries with or without the trailing altitude component.

    Args:
        location: Series of "lon,lat" or "lon,lat,alt" strings.

    Returns:
        Tuple of (longitude, latitude) float Series.
    """
    parts = location.str.split(",", expand=True)
    lon = parts[0].astype(float)
    lat = parts[1].astype(float)
    return lon, lat


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance (km) between two coordinate arrays.

    Vectorized; mirrors the original notebook formula (cell-80).

    Args:
        lat1, lon1: Origin latitude/longitude (degrees).
        lat2, lon2: Destination latitude/longitude (degrees).

    Returns:
        Distance in kilometres.
    """
    earth_radius = 6371.0  # km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return earth_radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add origin-destination haversine distance (`dist_km`).

    Parses origin/destination coordinates from the merged `Location` columns.

    Args:
        df: DataFrame with `Location` and `Location_dest` columns.

    Returns:
        DataFrame with `dist_km` (and intermediate lon/lat columns).
    """
    df = df.copy()
    df["lon_origin"], df["lat_origin"] = _parse_lonlat(df["Location"])
    df["lon_dest"], df["lat_dest"] = _parse_lonlat(df["Location_dest"])
    df["dist_km"] = haversine_km(
        df["lat_origin"], df["lon_origin"], df["lat_dest"], df["lon_dest"]
    )
    return df


def handle_self_trips(df: pd.DataFrame, policy: str = "drop") -> pd.DataFrame:
    """Apply the self-trip (Origin==Destination) policy (analiz 3.5).

    Same-station records have dist_km=0 and are likely turnstile artefacts.

    Args:
        df: DataFrame with `Origin`/`Destination`.
        policy: "drop" (remove), "flag" (add IsSelfTrip, keep), or "keep".

    Returns:
        DataFrame adjusted per policy.
    """
    df = df.copy()
    is_self = df["Origin"] == df["Destination"]
    if policy == "drop":
        return df.loc[~is_self].reset_index(drop=True)
    if policy == "flag":
        df["IsSelfTrip"] = is_self.astype(int)
        return df
    if policy == "keep":
        return df
    raise ValueError(f"Unknown self_trip_policy: {policy}")


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    lags: list[int],
    rolling_window: int,
    datetime_col: str = "DateTime",
) -> pd.DataFrame:
    """Add per-OD lag and rolling-mean features (leakage-safe).

    Features are computed within each (Origin, Destination) pair on
    time-sorted data using only **past** records (shifted), so no future
    information leaks. NaNs at the start of each group are left for LightGBM
    to handle natively.

    Note: lags index *observed* records per OD pair, not strict clock hours
    (the hourly grid can be sparse for low-traffic pairs).

    Args:
        df: DataFrame with Origin/Destination/target/datetime.
        target: Target column to lag (e.g. Throughput).
        lags: List of lag offsets (in observed records).
        rolling_window: Window size for the shifted rolling mean.
        datetime_col: Timestamp column to sort by.

    Returns:
        DataFrame with `<target>_lag_<k>` and `<target>_roll_mean_<w>` columns.
    """
    df = df.sort_values(datetime_col).reset_index(drop=True)
    grp = df.groupby(["Origin", "Destination"], observed=True)[target]

    for lag in lags:
        df[f"{target}_lag_{lag}"] = grp.shift(lag)

    df[f"{target}_roll_mean_{rolling_window}"] = grp.transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean()
    )
    return df


def prepare_categoricals(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Cast categorical columns to pandas `category` dtype for LightGBM.

    LightGBM consumes the category dtype directly, avoiding the false ordinal
    ordering that `cat.codes` would impose on stations (analiz 3.3).

    Args:
        df: DataFrame to adjust.
        categorical: Column names to cast.

    Returns:
        DataFrame with the given columns as `category` dtype.
    """
    df = df.copy()
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def build_features(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline.

    Args:
        df: Merged ridership DataFrame (output of `load_dataset`).
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        Feature-rich DataFrame ready for model training.
    """
    cfg = config or load_config()
    fcfg = cfg["features"]
    dt_col = cfg["datetime_col"]
    target = cfg["target"]

    df = add_time_features(df, dt_col)
    df = add_holiday_feature(df, dt_col, fcfg["holiday_subdiv"])
    df = add_distance_feature(df)
    df = handle_self_trips(df, fcfg["self_trip_policy"])
    df = add_lag_features(df, target, fcfg["lags"], fcfg["rolling_window"], dt_col)
    df = prepare_categoricals(df, fcfg["categorical"])
    return df


if __name__ == "__main__":
    # Smoke test (hizli kontrol) — yereldeki ornek veri uzerinde.
    from src.data.load import load_dataset

    _df = build_features(load_dataset())
    print("shape:", _df.shape)
    print("dtypes (sample):")
    print(_df[["Origin", "Period", "dist_km", "IsHoliday"]].dtypes.to_dict())
    print("new columns:", [c for c in _df.columns if "lag" in c or "roll" in c])
