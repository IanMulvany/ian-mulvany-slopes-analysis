"""Utilities for converting Slopes ``.slopes`` archives to analysis-ready data.

This module focuses on:
- unpacking the ``.slopes`` export format
- normalizing the raw GPS points
- computing base metrics such as distance, duration, and elevation change
- exporting derived GPX traces and Parquet datasets

The output produced by :func:`build_datasets` can be re-used by visualisation
or modeling layers without having to touch the raw archives again.
"""
from __future__ import annotations

import io
import math
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import gpxpy.gpx as gpx

CSV_COLUMNS = [
    "timestamp_s",
    "latitude",
    "longitude",
    "altitude_m",
    "course_deg",
    "speed_mps",
    "horizontal_accuracy_m",
    "vertical_accuracy_m",
]


@dataclass
class RunSummary:
    """A light-weight container for per-run summary statistics."""

    run_id: str
    file_name: str
    location_name: Optional[str]
    start_time: Optional[pd.Timestamp]
    end_time: Optional[pd.Timestamp]
    distance_m: float
    duration_s: float
    avg_speed_kmh: float
    top_speed_kmh: float
    elevation_gain_m: float
    vertical_drop_m: float
    vertical_range_m: float
    year: Optional[int]
    metadata: dict

    def to_dict(self) -> dict:
        base = {
            "run_id": self.run_id,
            "file_name": self.file_name,
            "location_name": self.location_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
            "avg_speed_kmh": self.avg_speed_kmh,
            "top_speed_kmh": self.top_speed_kmh,
            "elevation_gain_m": self.elevation_gain_m,
            "vertical_drop_m": self.vertical_drop_m,
            "vertical_range_m": self.vertical_range_m,
            "year": self.year,
        }
        return {**base, **{f"meta_{k}": v for k, v in self.metadata.items()}}


def _safe_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_metadata(xml_bytes: bytes) -> Tuple[dict, dict]:
    """Parse ``Metadata.xml`` into dictionaries for activity and run-level data."""
    root = ET.fromstring(xml_bytes)
    activity_attrs = dict(root.attrib)
    action_attrs: dict = {}
    actions_node = root.find("actions")
    if actions_node is not None:
        action_node = actions_node.find("Action")
        if action_node is not None:
            action_attrs = dict(action_node.attrib)
    return activity_attrs, action_attrs


def _haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Compute haversine distance between two coordinate arrays in meters."""
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_m = 6_371_000
    return earth_radius_m * c


def load_points(csv_bytes: bytes) -> pd.DataFrame:
    """Load the raw point CSV from a ``.slopes`` archive and enrich it."""
    df = pd.read_csv(io.BytesIO(csv_bytes), names=CSV_COLUMNS, header=None)
    df["time"] = pd.to_datetime(df["timestamp_s"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    lat_shifted = df["latitude"].shift()
    lon_shifted = df["longitude"].shift()
    df["segment_distance_m"] = _haversine_np(
        lat_shifted.fillna(df["latitude"]),
        lon_shifted.fillna(df["longitude"]),
        df["latitude"],
        df["longitude"],
    )
    df.loc[0, "segment_distance_m"] = 0.0

    df["segment_seconds"] = df["time"].diff().dt.total_seconds().fillna(0).clip(lower=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["segment_speed_kmh"] = np.where(
            df["segment_seconds"] > 0,
            (df["segment_distance_m"] / df["segment_seconds"]) * 3.6,
            np.nan,
        )

    df["cumulative_distance_m"] = df["segment_distance_m"].cumsum()
    return df


def _run_id_for(path: Path, metadata: dict, action: dict) -> str:
    identifier = metadata.get("identifier") or action.get("identifier")
    if identifier:
        return identifier
    return path.stem.replace(" ", "_")


def _build_summary(
    path: Path, df: pd.DataFrame, metadata: dict, action: dict
) -> RunSummary:
    distance_m = float(df["segment_distance_m"].sum())
    duration_s = float(df["segment_seconds"].sum())

    alt_diff = df["altitude_m"].diff().fillna(0)
    elevation_gain_m = float(alt_diff[alt_diff > 0].sum())
    vertical_drop_m = float(-alt_diff[alt_diff < 0].sum())
    vertical_range_m = float(df["altitude_m"].max() - df["altitude_m"].min())

    avg_speed_kmh = (distance_m / 1000) / (duration_s / 3600) if duration_s > 0 else 0.0
    top_speed_kmh = float(np.nanmax(df["segment_speed_kmh"])) if not df.empty else 0.0

    meta_top_speed = _safe_float(metadata.get("topSpeed"))
    if meta_top_speed and meta_top_speed * 3.6 > top_speed_kmh:
        # Slopes reports speed in m/s in Metadata.xml.
        top_speed_kmh = meta_top_speed * 3.6

    start_time = pd.to_datetime(metadata.get("start"), utc=True, errors="coerce")
    end_time = pd.to_datetime(metadata.get("end"), utc=True, errors="coerce")

    run_id = _run_id_for(path, metadata, action)

    return RunSummary(
        run_id=run_id,
        file_name=path.name,
        location_name=metadata.get("locationName"),
        start_time=start_time,
        end_time=end_time,
        distance_m=distance_m,
        duration_s=duration_s,
        avg_speed_kmh=avg_speed_kmh,
        top_speed_kmh=top_speed_kmh,
        elevation_gain_m=elevation_gain_m,
        vertical_drop_m=vertical_drop_m,
        vertical_range_m=vertical_range_m,
        year=start_time.year if pd.notnull(start_time) else None,
        metadata={**metadata, **{f"action_{k}": v for k, v in action.items()}},
    )


def _export_gpx(df: pd.DataFrame, summary: RunSummary, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    gpx_doc = gpx.GPX()
    track = gpx.GPXTrack(name=summary.file_name)
    gpx_doc.tracks.append(track)
    segment = gpx.GPXTrackSegment()
    track.segments.append(segment)

    for _, row in df.iterrows():
        segment.points.append(
            gpx.GPXTrackPoint(
                row["latitude"],
                row["longitude"],
                elevation=row.get("altitude_m"),
                time=row.get("time").to_pydatetime() if pd.notnull(row.get("time")) else None,
            )
        )

    destination.write_text(gpx_doc.to_xml())


def process_slopes_file(path: Path, write_gpx_to: Optional[Path] = None) -> Tuple[RunSummary, pd.DataFrame]:
    """Convert a single ``.slopes`` archive into summary and point data."""
    with zipfile.ZipFile(path) as zf:
        metadata_bytes = zf.read("Metadata.xml")
        gps_bytes = zf.read("RawGPS.csv")
    activity_meta, action_meta = parse_metadata(metadata_bytes)
    df = load_points(gps_bytes)
    summary = _build_summary(path, df, activity_meta, action_meta)

    df = df.copy()
    df["run_id"] = summary.run_id
    df["file_name"] = summary.file_name
    df["location_name"] = summary.location_name
    df["year"] = summary.year

    if write_gpx_to is not None:
        _export_gpx(df, summary, write_gpx_to / f"{summary.run_id}.gpx")

    return summary, df


def build_datasets(
    raw_dir: Path,
    output_dir: Path,
    write_gpx: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process all ``.slopes`` files under ``raw_dir``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Summary dataframe and raw point dataframe respectively.
    """
    summaries: List[RunSummary] = []
    points: List[pd.DataFrame] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    gpx_dir = output_dir / "gpx"
    for path in sorted(raw_dir.glob("*.slopes")):
        summary, df = process_slopes_file(path, gpx_dir if write_gpx else None)
        summaries.append(summary)
        points.append(df)

    summary_df = pd.DataFrame([s.to_dict() for s in summaries])
    points_df = pd.concat(points, ignore_index=True) if points else pd.DataFrame()

    summary_path = output_dir / "run_summary.parquet"
    points_path = output_dir / "points.parquet"
    summary_df.to_parquet(summary_path, index=False)
    points_df.to_parquet(points_path, index=False)

    return summary_df, points_df


def main(argv: Optional[Iterable[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert Slopes .slopes exports to analysis datasets.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/GPSLogs"), help="Directory containing .slopes files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Where to write derived data")
    parser.add_argument("--write-gpx", action="store_true", help="Also emit GPX files for each run")

    args = parser.parse_args(list(argv) if argv is not None else None)

    summaries, points = build_datasets(args.raw_dir, args.output_dir, write_gpx=args.write_gpx)
    print(f"Processed {len(summaries)} runs from {args.raw_dir}")
    print(f"Summary table -> {args.output_dir / 'run_summary.parquet'} ({len(summaries)} rows)")
    print(f"Point table   -> {args.output_dir / 'points.parquet'} ({len(points)} points)")
    if args.write_gpx:
        print(f"GPX exports   -> {args.output_dir / 'gpx'}")


if __name__ == "__main__":
    main()
