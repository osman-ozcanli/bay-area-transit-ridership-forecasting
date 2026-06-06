"""Exploratory Data Analysis (EDA / kesifsel veri analizi) for the BART project.

Answers the six business questions from the original brief as modular,
type-hinted, documented functions (config-driven, no hardcoding). Each
question function returns its computed result; companion `plot_*` helpers save a
figure under reports/figures/, and `interpret_*` helpers produce a written
interpretation (YORUM) for the narrative notebook.

Business questions (is sorulari):
    1. Busiest station        (en yogun istasyon)
    2. Least popular route     (en az populer rota)
    3. Busiest day of week     (en yogun gun)
    4. Late-night ridership    (gece yolculari) + hour/period distribution
    5. Most popular routes     (en populer rotalar)
    6. Best time Berkeley->SF  (koltuk bulmak icin en iyi saat)

The analysis frame is prepared by reusing the feature-engineering primitives
(`add_time_features`, `add_distance_feature`) so no logic is duplicated; EDA is
run on the *raw* ridership (self-trips kept — descriptive), while modelling later
drops them. All parameters come from `config.yaml`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figurleri dosyaya yaz, ekran acma
import matplotlib.pyplot as plt
import pandas as pd

from src.config import get_paths, load_config
from src.features.build_features import add_distance_feature, add_time_features

# Day order (gun sirasi) for weekly aggregation plots.
_WEEK_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


# --------------------------------------------------------------------------- #
# Analysis frame
# --------------------------------------------------------------------------- #
def prepare_eda_frame(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Build an analysis-ready frame for EDA (time + coordinate features).

    Reuses the feature-engineering primitives to add Hour/DayOfWeek/Period and
    parsed lon/lat coordinates, **without** dropping self-trips or adding lag
    features (EDA describes raw ridership; modelling handles those later).

    Args:
        df: Merged ridership DataFrame (output of `load_dataset`).
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        A copy of the data with Hour/DayOfWeek/Month/IsWeekend/Period and
        lon_*/lat_* coordinate columns added.
    """
    cfg = config or load_config()
    dt_col = cfg["datetime_col"]
    out = add_time_features(df, dt_col)
    out = add_distance_feature(out)
    return out


# --------------------------------------------------------------------------- #
# Q1 — Busiest station
# --------------------------------------------------------------------------- #
def busiest_stations(
    df: pd.DataFrame, target: str = "Throughput", top_n: int = 10
) -> pd.Series:
    """Rank stations by total passenger load (boardings + alightings).

    A station's load is the sum of its throughput as an origin plus as a
    destination (`fill_value=0` keeps stations that appear on only one side).

    Args:
        df: Ridership frame with Origin/Destination/`target`.
        target: Throughput column name.
        top_n: Number of top stations to return.

    Returns:
        Series indexed by station, sorted descending by total load.
    """
    as_origin = df.groupby("Origin", observed=True)[target].sum()
    as_dest = df.groupby("Destination", observed=True)[target].sum()
    loads = as_origin.add(as_dest, fill_value=0)
    return loads.sort_values(ascending=False).head(top_n)


# --------------------------------------------------------------------------- #
# Q2 / Q5 — Routes (least popular / most popular)
# --------------------------------------------------------------------------- #
def route_totals(
    df: pd.DataFrame,
    target: str = "Throughput",
    top_n: int = 10,
    ascending: bool = False,
    drop_self_trips: bool = True,
) -> pd.Series:
    """Aggregate total throughput per (Origin, Destination) route.

    Args:
        df: Ridership frame with Origin/Destination/`target`.
        target: Throughput column name.
        top_n: Number of routes to return.
        ascending: True for least-popular routes, False for most-popular.
        drop_self_trips: Exclude Origin==Destination (turnstile artefacts) so
            "least popular route" is a real inter-station route.

    Returns:
        Series indexed by (Origin, Destination), sliced to `top_n`.
    """
    work = df
    if drop_self_trips:
        work = df[df["Origin"] != df["Destination"]]
    totals = work.groupby(["Origin", "Destination"], observed=True)[target].sum()
    return totals.sort_values(ascending=ascending).head(top_n)


def least_popular_routes(
    df: pd.DataFrame, target: str = "Throughput", top_n: int = 10
) -> pd.Series:
    """Least-popular inter-station routes (smallest total throughput).

    Args:
        df: Ridership frame.
        target: Throughput column name.
        top_n: Number of routes to return.

    Returns:
        Series indexed by (Origin, Destination), ascending by total.
    """
    return route_totals(df, target, top_n, ascending=True, drop_self_trips=True)


def popular_routes(
    df: pd.DataFrame, target: str = "Throughput", top_n: int = 10
) -> pd.Series:
    """Most-popular inter-station routes (largest total throughput).

    Args:
        df: Ridership frame.
        target: Throughput column name.
        top_n: Number of routes to return.

    Returns:
        Series indexed by (Origin, Destination), descending by total.
    """
    return route_totals(df, target, top_n, ascending=False, drop_self_trips=True)


# --------------------------------------------------------------------------- #
# Q3 — Busiest day of week
# --------------------------------------------------------------------------- #
def busiest_day(df: pd.DataFrame, target: str = "Throughput") -> pd.Series:
    """Total ridership per weekday, ordered Monday -> Sunday.

    Args:
        df: Frame with a `DayOfWeek` column.
        target: Throughput column name.

    Returns:
        Series indexed by weekday (calendar order) of total throughput.
    """
    day_load = df.groupby("DayOfWeek", observed=True)[target].sum()
    return day_load.reindex(_WEEK_ORDER)


# --------------------------------------------------------------------------- #
# Q4 — Late-night ridership + hour/period distribution
# --------------------------------------------------------------------------- #
def ridership_by_period(df: pd.DataFrame, target: str = "Throughput") -> pd.Series:
    """Total throughput per time-of-day period (Morning/Midday/.../LateNight).

    Args:
        df: Frame with a `Period` column.
        target: Throughput column name.

    Returns:
        Series indexed by Period of total throughput.
    """
    return df.groupby("Period", observed=True)[target].sum()


def ridership_by_hour(df: pd.DataFrame, target: str = "Throughput") -> pd.Series:
    """Total throughput per hour of day (0-23).

    Uses throughput-weighted totals (not a raw row count), correcting the
    original notebook's `histplot(df['Hour'])` which counted records rather
    than passengers (analiz 3.6).

    Args:
        df: Frame with an `Hour` column.
        target: Throughput column name.

    Returns:
        Series indexed by Hour of total throughput.
    """
    return df.groupby("Hour", observed=True)[target].sum().sort_index()


def hour_day_pivot(df: pd.DataFrame, target: str = "Throughput") -> pd.DataFrame:
    """Day-of-week x hour pivot of total throughput (for a heatmap).

    Args:
        df: Frame with DayOfWeek/Hour columns.
        target: Throughput column name.

    Returns:
        Pivot table (rows = weekday in calendar order, cols = hour).
    """
    pivot = df.pivot_table(
        values=target, index="DayOfWeek", columns="Hour", aggfunc="sum"
    )
    return pivot.reindex(_WEEK_ORDER)


def late_night_summary(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Summarise late-night ridership and its share of the daily total.

    Args:
        df: Frame with Period/`target`.
        config: Optional config (for target + late-night label).

    Returns:
        Dict with late_night total, grand total, share %, and per-period Series.
    """
    cfg = config or load_config()
    target = cfg["target"]
    label = cfg["eda"]["late_night_period"]
    by_period = ridership_by_period(df, target)
    late = float(by_period.get(label, 0.0))
    total = float(by_period.sum())
    return {
        "late_night": late,
        "total": total,
        "share_pct": 100.0 * late / total if total else 0.0,
        "by_period": by_period,
    }


# --------------------------------------------------------------------------- #
# Q6 — Best time Berkeley -> SF (find a seat)
# --------------------------------------------------------------------------- #
def _stations_in_box(
    df: pd.DataFrame, lat_col: str, lon_col: str, station_col: str, box: dict[str, list]
) -> list[str]:
    """Return unique stations whose coordinates fall inside a lat/lon box.

    Args:
        df: Frame with coordinate + station columns.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        station_col: Station abbreviation column to collect.
        box: Dict with `lat` [min,max] and `lon` [min,max] bounds.

    Returns:
        Sorted list of unique station abbreviations inside the box.
    """
    lat_lo, lat_hi = box["lat"]
    lon_lo, lon_hi = box["lon"]
    mask = (
        df[lat_col].between(lat_lo, lat_hi)
        & df[lon_col].between(lon_lo, lon_hi)
    )
    return sorted(df.loc[mask, station_col].unique().tolist())


def berkeley_sf_best_hour(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Best hour to travel Berkeley -> SF for seat availability.

    Identifies Berkeley-origin and SF-destination stations via spatial
    bounding-boxes (no manual station picking), then computes average hourly
    throughput on that OD subset. The lowest-average hour is the most
    comfortable; an evening-period low is also reported for the commute home.

    Args:
        df: EDA frame with lon_*/lat_*/Hour/Period columns.
        config: Optional config (boxes + period labels).

    Returns:
        Dict with berkeley/sf station lists, hourly_mean Series, best_hour
        (global min), and best_evening_hour (min within the evening period).
    """
    cfg = config or load_config()
    ecfg = cfg["eda"]
    target = cfg["target"]

    berkeley = _stations_in_box(
        df, "lat_origin", "lon_origin", "Origin", ecfg["berkeley_box"]
    )
    sf = _stations_in_box(
        df, "lat_dest", "lon_dest", "Destination", ecfg["sf_box"]
    )

    subset = df[df["Origin"].isin(berkeley) & df["Destination"].isin(sf)]
    hourly_mean = subset.groupby("Hour", observed=True)[target].mean().sort_index()

    evening = subset[subset["Period"] == ecfg["evening_period"]]
    evening_mean = evening.groupby("Hour", observed=True)[target].mean()

    return {
        "berkeley_stations": berkeley,
        "sf_stations": sf,
        "hourly_mean": hourly_mean,
        "best_hour": int(hourly_mean.idxmin()) if not hourly_mean.empty else None,
        "best_evening_hour": (
            int(evening_mean.idxmin()) if not evening_mean.empty else None
        ),
        "n_rows": int(len(subset)),
    }


# --------------------------------------------------------------------------- #
# Plot helpers (figures -> reports/figures/)
# --------------------------------------------------------------------------- #
def _barh(series: pd.Series, title: str, xlabel: str, out: Path, color: str) -> Path:
    """Save a horizontal bar chart of a Series (highest at top)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(i) for i in series.index]
    plt.figure(figsize=(9, 5))
    plt.barh(labels[::-1], series.values[::-1], color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_busiest_stations(series: pd.Series, figures_dir: Path) -> Path:
    """Bar chart of busiest stations."""
    return _barh(
        series, "Top Busiest BART Stations (boardings + alightings)",
        "Total passenger load", figures_dir / "eda_busiest_stations.png", "#2b8cbe",
    )


def plot_routes(series: pd.Series, figures_dir: Path, *, least: bool) -> Path:
    """Bar chart of routes (least or most popular)."""
    labels = pd.Series(
        [f"{o} -> {d}" for o, d in series.index], index=series.index
    )
    s = pd.Series(series.values, index=labels.values)
    name = "least" if least else "popular"
    title = ("Least Popular" if least else "Most Popular") + " BART Routes"
    color = "#cb4b16" if least else "#238b45"
    return _barh(
        s, title, "Total passenger count",
        figures_dir / f"eda_{name}_routes.png", color,
    )


def plot_busiest_day(series: pd.Series, figures_dir: Path) -> Path:
    """Bar chart of total ridership by weekday."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "eda_busiest_day.png"
    plt.figure(figsize=(9, 5))
    plt.bar(series.index, series.values, color="#6baed6")
    plt.title("Total Ridership by Day of Week")
    plt.ylabel("Total passengers")
    plt.xlabel("Day")
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_period_and_hour(
    by_period: pd.Series, by_hour: pd.Series, figures_dir: Path, late_label: str
) -> Path:
    """Side-by-side: ridership by period (late-night highlighted) and by hour."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "eda_period_hour.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["crimson" if p == late_label else "steelblue" for p in by_period.index]
    axes[0].bar([str(p) for p in by_period.index], by_period.values, color=colors)
    axes[0].set_title("Ridership by Time Period")
    axes[0].set_ylabel("Total passengers")
    axes[0].ticklabel_format(style="plain", axis="y")

    axes[1].bar(by_hour.index, by_hour.values, color="#6baed6")
    axes[1].set_title("Ridership by Hour of Day (throughput-weighted)")
    axes[1].set_xlabel("Hour")
    axes[1].ticklabel_format(style="plain", axis="y")

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_hour_day_heatmap(pivot: pd.DataFrame, figures_dir: Path) -> Path:
    """Heatmap of ridership intensity by hour (x) and weekday (y)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "eda_hour_day_heatmap.png"
    plt.figure(figsize=(13, 5))
    plt.imshow(pivot.values, aspect="auto", cmap="magma")
    plt.colorbar(label="Total throughput")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(0, len(pivot.columns), 2), pivot.columns[::2])
    plt.title("BART Ridership Intensity by Hour and Day")
    plt.xlabel("Hour of day")
    plt.ylabel("Day of week")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_berkeley_sf(hourly_mean: pd.Series, figures_dir: Path) -> Path:
    """Line chart of average Berkeley -> SF throughput by hour."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "eda_berkeley_sf_hourly.png"
    plt.figure(figsize=(10, 4))
    plt.plot(hourly_mean.index, hourly_mean.values, marker="o", color="#2b8cbe")
    plt.title("Average Ridership Berkeley -> SF by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Average throughput")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


# --------------------------------------------------------------------------- #
# Interpretations (YORUM)
# --------------------------------------------------------------------------- #
def interpret_eda(
    stations: pd.Series,
    least: pd.Series,
    popular: pd.Series,
    day_load: pd.Series,
    late: dict[str, Any],
    bsf: dict[str, Any],
) -> str:
    """Produce a written, data-driven interpretation of all six questions.

    Args:
        stations: Busiest-stations Series.
        least: Least-popular-routes Series.
        popular: Most-popular-routes Series.
        day_load: Weekday total Series.
        late: `late_night_summary` result.
        bsf: `berkeley_sf_best_hour` result.

    Returns:
        Multi-line interpretation string for the narrative notebook.
    """
    top_station = stations.index[0]
    lo_o, lo_d = least.index[0]
    pop_o, pop_d = popular.index[0]
    busy_day = day_load.idxmax()
    lines = [
        f"1) En yogun istasyon (busiest): {top_station} "
        f"(toplam yuk {stations.iloc[0]:,.0f}).",
        f"2) En az populer rota (least popular route): {lo_o} -> {lo_d}.",
        f"3) En yogun gun (busiest day): {busy_day}.",
        f"4) Gece (LateNight) yolcu payi: %{late['share_pct']:.1f} "
        f"-> talep gunduz zirve saatlerinde yogun, gece cok dusuk.",
        f"5) En populer rota (most popular route): {pop_o} -> {pop_d}.",
        f"6) Berkeley -> SF koltuk icin en iyi saat: {bsf['best_hour']}:00 "
        f"(en dusuk ortalama doluluk); aksam (Evening) donusunde en rahat saat "
        f"{bsf['best_evening_hour']}:00. Zirve sikisiklik 08-09 civari.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_eda(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run all six EDA questions: compute -> plot -> interpret.

    Args:
        df: Merged ridership DataFrame (output of `load_dataset`); time and
            coordinate features are derived internally.
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        Dict with every computed result and the saved figure paths.
    """
    cfg = config or load_config()
    target = cfg["target"]
    ecfg = cfg["eda"]
    top_n = ecfg["top_n"]
    figs = get_paths(cfg)["figures_dir"]

    eda_df = prepare_eda_frame(df, cfg)

    stations = busiest_stations(eda_df, target, top_n)
    least = least_popular_routes(eda_df, target, top_n)
    popular = popular_routes(eda_df, target, top_n)
    day_load = busiest_day(eda_df, target)
    late = late_night_summary(eda_df, cfg)
    by_hour = ridership_by_hour(eda_df, target)
    pivot = hour_day_pivot(eda_df, target)
    bsf = berkeley_sf_best_hour(eda_df, cfg)

    paths = {
        "stations": plot_busiest_stations(stations, figs),
        "least": plot_routes(least, figs, least=True),
        "popular": plot_routes(popular, figs, least=False),
        "day": plot_busiest_day(day_load, figs),
        "period_hour": plot_period_and_hour(
            late["by_period"], by_hour, figs, ecfg["late_night_period"]
        ),
        "heatmap": plot_hour_day_heatmap(pivot, figs),
        "berkeley_sf": plot_berkeley_sf(bsf["hourly_mean"], figs),
    }

    interpretation = interpret_eda(stations, least, popular, day_load, late, bsf)

    print("=== EDA — 6 is sorusu ===")
    print("\n[Q1] Busiest stations:\n", stations.to_string())
    print("\n[Q2] Least popular routes:\n", least.to_string())
    print("\n[Q3] Ridership by day:\n", day_load.to_string())
    print(
        f"\n[Q4] Late-night: {late['late_night']:,.0f} "
        f"(%{late['share_pct']:.1f} of total)"
    )
    print("\n[Q5] Popular routes:\n", popular.to_string())
    print(
        f"\n[Q6] Berkeley {bsf['berkeley_stations']} -> SF {bsf['sf_stations']} | "
        f"best hour {bsf['best_hour']}:00, evening {bsf['best_evening_hour']}:00"
    )
    print("\n=== YORUM ===\n" + interpretation)
    print("\nfigures:", *[str(p) for p in paths.values()], sep="\n  ")

    return {
        "stations": stations, "least": least, "popular": popular,
        "day_load": day_load, "late": late, "by_hour": by_hour,
        "pivot": pivot, "berkeley_sf": bsf,
        "interpretation": interpretation, "figures": paths,
    }


if __name__ == "__main__":
    # Smoke test (hizli kontrol) — yereldeki ornek veri uzerinde.
    from src.data.load import load_dataset

    run_eda(load_dataset())
