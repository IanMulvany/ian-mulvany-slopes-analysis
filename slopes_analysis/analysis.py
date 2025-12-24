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
