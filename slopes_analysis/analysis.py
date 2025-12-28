"""Analysis helpers for Slopes datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from . import ingest


def load_datasets(
    processed_dir: Path = Path("data/processed"), raw_dir: Path | None = Path("data/GPSLogs")
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed datasets, generating them from raw exports if needed."""
    summary_path = processed_dir / "run_summary.parquet"
    points_path = processed_dir / "points.parquet"

    if not summary_path.exists() or not points_path.exists():
        if raw_dir is None or not raw_dir.exists():
            raise FileNotFoundError(
                f"Processed data missing at {processed_dir}, and raw exports not found at {raw_dir}."
            )
        ingest.build_datasets(raw_dir=raw_dir, output_dir=processed_dir)

    summary_df = pd.read_parquet(summary_path)
    points_df = pd.read_parquet(points_path)
    return summary_df, points_df


def overview_cards(summary_df: pd.DataFrame) -> dict:
    if summary_df.empty:
        return {
            "total_runs": 0,
            "total_distance_km": 0.0,
            "total_vertical_m": 0.0,
            "avg_speed_kmh": 0.0,
            "years_active": "",
        }

    total_distance_km = summary_df["distance_m"].sum() / 1000
    total_vertical_m = summary_df["vertical_drop_m"].sum()
    avg_speed_kmh = summary_df["avg_speed_kmh"].mean()

    years = summary_df["year"].dropna().astype(int)
    years_active = f"{years.min()}–{years.max()}" if not years.empty else ""

    return {
        "total_runs": int(len(summary_df)),
        "total_distance_km": float(total_distance_km),
        "total_vertical_m": float(total_vertical_m),
        "avg_speed_kmh": float(avg_speed_kmh),
        "years_active": years_active,
    }


def yearly_trends(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "year" not in summary_df:
        return pd.DataFrame(columns=["year", "run_count", "distance_km", "vertical_m", "avg_speed_kmh"])

    grouped = (
        summary_df.dropna(subset=["year"])
        .groupby("year")
        .agg(
            run_count=("run_id", "count"),
            distance_km=("distance_m", lambda s: s.sum() / 1000),
            vertical_m=("vertical_drop_m", "sum"),
            avg_speed_kmh=("avg_speed_kmh", "mean"),
        )
        .reset_index()
        .sort_values("year")
    )
    return grouped


def location_leaderboard(summary_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["location_name", "run_count", "distance_km"])

    return (
        summary_df.fillna({"location_name": "Unknown"})
        .groupby("location_name")
        .agg(run_count=("run_id", "count"), distance_km=("distance_m", lambda s: s.sum() / 1000))
        .reset_index()
        .sort_values(["distance_km", "run_count"], ascending=[False, False])
        .head(top_n)
    )


def top_runs(summary_df: pd.DataFrame, metric: str, n: int = 5) -> pd.DataFrame:
    if metric not in summary_df.columns or summary_df.empty:
        return pd.DataFrame(columns=summary_df.columns)
    return summary_df.sort_values(metric, ascending=False).head(n)


def run_profile(points_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    run_points = points_df[points_df["run_id"] == run_id].copy()
    if run_points.empty:
        return run_points

    run_points = run_points.sort_values("time").reset_index(drop=True)
    run_points["elapsed_s"] = (run_points["time"] - run_points["time"].min()).dt.total_seconds()
    run_points["distance_km"] = run_points["cumulative_distance_m"] / 1000
    run_points["altitude_m"] = run_points["altitude_m"]
    return run_points


def compare_runs(points_df: pd.DataFrame, run_ids: Iterable[str]) -> pd.DataFrame:
    frames = []
    for rid in run_ids:
        profile = run_profile(points_df, rid)
        if profile.empty:
            continue
        profile = profile[["elapsed_s", "distance_km", "altitude_m"]].copy()
        profile["run_id"] = rid
        frames.append(profile)
    if not frames:
        return pd.DataFrame(columns=["run_id", "elapsed_s", "distance_km", "altitude_m"])
    return pd.concat(frames, ignore_index=True)


def seasonal_density(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "start_time" not in summary_df:
        return pd.DataFrame(columns=["month", "run_count"])

    df = summary_df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])
    df["month"] = df["start_time"].dt.month
    return df.groupby("month").agg(run_count=("run_id", "count")).reset_index().sort_values("month")


# --- Downhill segmentation helpers ---

def _smoothed_speed_kmh(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Compute a rolling-median smoothed speed in km/h to reduce spikes."""
    return df["segment_speed_kmh"].rolling(window=window, center=True, min_periods=1).median()


def segment_downhill_runs(
    points_df: pd.DataFrame,
    speed_kmh_min: float = 5.0,
    gap_seconds: float = 90.0,
    min_drop_m: float = 30.0,
    min_distance_m: float = 200.0,
    max_top_speed_kmh: float = 120.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each day (run_id) into downhill segments and compute per-segment stats.

    Heuristic:
      - Smooth speed with rolling median to suppress spikes
      - Mark points downhill when speed > speed_kmh_min and altitude is decreasing
      - Split segments when time gaps exceed gap_seconds or uphill/idle stretches
      - Discard tiny segments (distance, drop)
      - Cap top speed to max_top_speed_kmh to avoid GPS glitches
    """
    if points_df.empty:
        return pd.DataFrame(), points_df

    labeled_points = []
    segments = []

    for run_id, day_points in points_df.groupby("run_id"):
        df = day_points.sort_values("time").copy()
        if df.empty:
            continue

        df["speed_smooth"] = _smoothed_speed_kmh(df)
        df["alt_diff"] = df["altitude_m"].diff().fillna(0)
        df["time_diff_s"] = df["time"].diff().dt.total_seconds().fillna(0)

        is_downhill = (df["speed_smooth"] > speed_kmh_min) & (df["alt_diff"] < 0)

        segment_id = []
        current_id = 0
        for i in range(len(df)):
            if i == 0:
                segment_id.append(current_id)
                continue
            gap = df.iloc[i]["time_diff_s"]
            if gap > gap_seconds or not is_downhill.iloc[i]:
                current_id += 1
            segment_id.append(current_id)
        df["segment_id"] = segment_id

        # Keep only downhill stretches
        downhill_ids = df[is_downhill]["segment_id"].unique().tolist()
        df = df[df["segment_id"].isin(downhill_ids)]
        if df.empty:
            continue

        for seg_id, seg in df.groupby("segment_id"):
            seg = seg.copy()
            seg_distance = float(seg["segment_distance_m"].sum())
            seg_drop = float(-(seg["alt_diff"][seg["alt_diff"] < 0].sum()))
            seg_duration = float(seg["time_diff_s"].sum())
            if seg_distance < min_distance_m or seg_drop < min_drop_m or seg_duration <= 0:
                continue

            top_speed = min(float(seg["speed_smooth"].max()), max_top_speed_kmh)
            avg_speed = (seg_distance / 1000) / (seg_duration / 3600) if seg_duration > 0 else 0.0

            segments.append(
                {
                    "day_run_id": run_id,
                    "segment_global_id": f"{run_id}__seg{seg_id}",
                    "start_time": seg["time"].min(),
                    "end_time": seg["time"].max(),
                    "distance_m": seg_distance,
                    "vertical_drop_m": seg_drop,
                    "duration_s": seg_duration,
                    "avg_speed_kmh": avg_speed,
                    "top_speed_kmh": top_speed,
                    "location_name": seg["location_name"].iloc[0] if "location_name" in seg else None,
                    "year": seg["year"].iloc[0] if "year" in seg else None,
                }
            )
            seg["segment_global_id"] = f"{run_id}__seg{seg_id}"
            labeled_points.append(seg)

    segments_df = pd.DataFrame(segments)
    labeled_points_df = pd.concat(labeled_points, ignore_index=True) if labeled_points else points_df
    return segments_df, labeled_points_df


def get_runs_by_location(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Group runs by location and year for cross-year comparison."""
    if summary_df.empty:
        return pd.DataFrame()
    
    df = summary_df.copy()
    df["location_name"] = df["location_name"].fillna("Unknown")
    df = df.dropna(subset=["year", "location_name"])
    
    # Group by location and year, compute stats
    location_year_stats = (
        df.groupby(["location_name", "year"])
        .agg(
            run_count=("run_id", "count"),
            avg_speed_kmh_mean=("avg_speed_kmh", "mean"),
            avg_speed_kmh_median=("avg_speed_kmh", "median"),
            top_speed_kmh_mean=("top_speed_kmh", "mean"),
            top_speed_kmh_max=("top_speed_kmh", "max"),
            distance_m_mean=("distance_m", "mean"),
            vertical_drop_m_mean=("vertical_drop_m", "mean"),
        )
        .reset_index()
        .sort_values(["location_name", "year"])
    )
    
    return location_year_stats


def compute_year_over_year_trends(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute year-over-year changes for locations with multiple years of data."""
    location_year_stats = get_runs_by_location(summary_df)
    
    if location_year_stats.empty:
        return pd.DataFrame()
    
    trends = []
    
    for location in location_year_stats["location_name"].unique():
        location_data = location_year_stats[location_year_stats["location_name"] == location].sort_values("year")
        
        if len(location_data) < 2:
            continue
        
        for i in range(1, len(location_data)):
            prev_year = location_data.iloc[i - 1]
            curr_year = location_data.iloc[i]
            
            year_diff = curr_year["year"] - prev_year["year"]
            
            trends.append({
                "location_name": location,
                "year": curr_year["year"],
                "prev_year": prev_year["year"],
                "year_diff": year_diff,
                "avg_speed_change": curr_year["avg_speed_kmh_mean"] - prev_year["avg_speed_kmh_mean"],
                "avg_speed_change_pct": (
                    (curr_year["avg_speed_kmh_mean"] - prev_year["avg_speed_kmh_mean"]) / prev_year["avg_speed_kmh_mean"] * 100
                    if prev_year["avg_speed_kmh_mean"] > 0 else 0
                ),
                "top_speed_change": curr_year["top_speed_kmh_max"] - prev_year["top_speed_kmh_max"],
                "top_speed_change_pct": (
                    (curr_year["top_speed_kmh_max"] - prev_year["top_speed_kmh_max"]) / prev_year["top_speed_kmh_max"] * 100
                    if prev_year["top_speed_kmh_max"] > 0 else 0
                ),
                "avg_speed_prev": prev_year["avg_speed_kmh_mean"],
                "avg_speed_curr": curr_year["avg_speed_kmh_mean"],
                "top_speed_prev": prev_year["top_speed_kmh_max"],
                "top_speed_curr": curr_year["top_speed_kmh_max"],
            })
    
    return pd.DataFrame(trends)


def get_location_performance_trend(summary_df: pd.DataFrame, location_name: str) -> dict:
    """Determine if performance at a location is improving, degrading, or steady."""
    location_year_stats = get_runs_by_location(summary_df)
    location_data = location_year_stats[location_year_stats["location_name"] == location_name].sort_values("year")
    
    if len(location_data) < 2:
        return {
            "trend": "insufficient_data",
            "avg_speed_trend": "insufficient_data",
            "top_speed_trend": "insufficient_data",
            "years": len(location_data),
        }
    
    # Compute linear regression slope for avg speed
    years = location_data["year"].values
    avg_speeds = location_data["avg_speed_kmh_mean"].values
    top_speeds = location_data["top_speed_kmh_max"].values
    
    # Simple trend: positive slope = improving, negative = degrading
    avg_speed_slope = (avg_speeds[-1] - avg_speeds[0]) / (years[-1] - years[0]) if len(years) > 1 else 0
    top_speed_slope = (top_speeds[-1] - top_speeds[0]) / (years[-1] - years[0]) if len(years) > 1 else 0
    
    # Determine trend (using 2% threshold to account for noise)
    threshold = 0.5  # km/h per year
    
    def classify_trend(slope):
        if abs(slope) < threshold:
            return "steady"
        return "improving" if slope > 0 else "degrading"
    
    avg_trend = classify_trend(avg_speed_slope)
    top_trend = classify_trend(top_speed_slope)
    
    # Overall trend (if both agree, use that; otherwise "mixed")
    if avg_trend == top_trend:
        overall_trend = avg_trend
    elif avg_trend == "steady" or top_trend == "steady":
        overall_trend = avg_trend if top_trend == "steady" else top_trend
    else:
        overall_trend = "mixed"
    
    return {
        "trend": overall_trend,
        "avg_speed_trend": avg_trend,
        "top_speed_trend": top_trend,
        "avg_speed_slope": avg_speed_slope,
        "top_speed_slope": top_speed_slope,
        "years": len(location_data),
        "first_year": int(years[0]),
        "last_year": int(years[-1]),
    }


def get_location_coordinates(points_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Get centroid coordinates for each location."""
    if points_df.empty or summary_df.empty:
        return pd.DataFrame()
    
    # Merge points with summary to get location names
    location_coords = (
        points_df.groupby("location_name")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            min_lat=("latitude", "min"),
            max_lat=("latitude", "max"),
            min_lon=("longitude", "min"),
            max_lon=("longitude", "max"),
        )
        .reset_index()
    )
    
    # Add summary statistics
    location_stats = (
        summary_df.groupby("location_name")
        .agg(
            run_count=("run_id", "count"),
            total_distance_km=("distance_m", lambda s: s.sum() / 1000),
            avg_speed_kmh=("avg_speed_kmh", "mean"),
            top_speed_kmh=("top_speed_kmh", "max"),
            years_visited=("year", lambda s: sorted(s.dropna().unique().astype(int).tolist())),
            first_visit=("start_time", "min"),
            last_visit=("start_time", "max"),
        )
        .reset_index()
    )
    
    result = location_coords.merge(location_stats, on="location_name", how="inner")
    return result


def get_detailed_location_stats(summary_df: pd.DataFrame, points_df: pd.DataFrame) -> pd.DataFrame:
    """Get comprehensive statistics for each location."""
    if summary_df.empty:
        return pd.DataFrame()
    
    df = summary_df.copy()
    df["location_name"] = df["location_name"].fillna("Unknown")
    
    location_stats = (
        df.groupby("location_name")
        .agg(
            run_count=("run_id", "count"),
            total_distance_km=("distance_m", lambda s: s.sum() / 1000),
            avg_distance_km=("distance_m", lambda s: s.mean() / 1000),
            total_vertical_drop_m=("vertical_drop_m", "sum"),
            avg_vertical_drop_m=("vertical_drop_m", "mean"),
            avg_speed_kmh_mean=("avg_speed_kmh", "mean"),
            avg_speed_kmh_median=("avg_speed_kmh", "median"),
            top_speed_kmh_max=("top_speed_kmh", "max"),
            top_speed_kmh_mean=("top_speed_kmh", "mean"),
            years_visited=("year", lambda s: sorted(s.dropna().unique().astype(int).tolist())),
            year_count=("year", "nunique"),
            first_visit=("start_time", "min"),
            last_visit=("start_time", "max"),
        )
        .reset_index()
    )
    
    # Add visit frequency
    location_stats["visits_per_year"] = (
        location_stats["run_count"] / location_stats["year_count"]
    ).round(1)
    
    # Add date range string
    location_stats["visit_years"] = location_stats["years_visited"].apply(
        lambda x: f"{min(x)}-{max(x)}" if len(x) > 1 else str(x[0]) if x else "Unknown"
    )
    
    return location_stats.sort_values("run_count", ascending=False)


def get_overall_performance_summary(summary_df: pd.DataFrame) -> dict:
    """Get overall performance summary with year-over-year changes."""
    if summary_df.empty:
        return {}
    
    df = summary_df.copy()
    df = df.dropna(subset=["year"])
    
    yearly_stats = (
        df.groupby("year")
        .agg(
            run_count=("run_id", "count"),
            avg_speed_mean=("avg_speed_kmh", "mean"),
            avg_speed_median=("avg_speed_kmh", "median"),
            top_speed_mean=("top_speed_kmh", "mean"),
            top_speed_max=("top_speed_kmh", "max"),
        )
        .reset_index()
        .sort_values("year")
    )
    
    if len(yearly_stats) < 2:
        return {
            "years": len(yearly_stats),
            "trend": "insufficient_data",
        }
    
    # Compare first and last year
    first_year = yearly_stats.iloc[0]
    last_year = yearly_stats.iloc[-1]
    
    avg_speed_change = last_year["avg_speed_mean"] - first_year["avg_speed_mean"]
    top_speed_change = last_year["top_speed_max"] - first_year["top_speed_max"]
    
    # Determine overall trend
    threshold = 0.5
    if abs(avg_speed_change) < threshold and abs(top_speed_change) < threshold:
        trend = "steady"
    elif avg_speed_change > 0 and top_speed_change > 0:
        trend = "improving"
    elif avg_speed_change < 0 and top_speed_change < 0:
        trend = "degrading"
    else:
        trend = "mixed"
    
    return {
        "years": len(yearly_stats),
        "first_year": int(first_year["year"]),
        "last_year": int(last_year["year"]),
        "trend": trend,
        "avg_speed_first": float(first_year["avg_speed_mean"]),
        "avg_speed_last": float(last_year["avg_speed_mean"]),
        "avg_speed_change": float(avg_speed_change),
        "top_speed_first": float(first_year["top_speed_max"]),
        "top_speed_last": float(last_year["top_speed_max"]),
        "top_speed_change": float(top_speed_change),
        "yearly_stats": yearly_stats,
    }
