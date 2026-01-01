"""Persistence layer for segment gluing decisions."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class GlueDecision:
    """A decision to glue segments together."""
    glue_id: str
    segment_ids: List[str]  # Ordered list of segments to merge
    status: str  # "approved", "rejected", "pending"
    notes: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


@dataclass
class GlueDecisionFile:
    """Container for all glue decisions."""
    version: int = 1
    created: str = ""
    decisions: List[GlueDecision] = None

    def __post_init__(self):
        if not self.created:
            self.created = datetime.utcnow().isoformat()
        if self.decisions is None:
            self.decisions = []


def save_glue_decisions(decisions: List[GlueDecision], path: Path) -> None:
    """Save glue decisions to a JSON file."""
    container = GlueDecisionFile(
        decisions=decisions,
        created=datetime.utcnow().isoformat()
    )

    data = {
        "version": container.version,
        "created": container.created,
        "decisions": [asdict(d) for d in container.decisions]
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_glue_decisions(path: Path) -> List[GlueDecision]:
    """Load glue decisions from a JSON file."""
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    decisions = []
    for d in data.get("decisions", []):
        decisions.append(GlueDecision(
            glue_id=d["glue_id"],
            segment_ids=d["segment_ids"],
            status=d["status"],
            notes=d.get("notes", ""),
            created_at=d.get("created_at", "")
        ))

    return decisions


def merge_segment_points(
    segment_ids: List[str],
    labeled_points_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge GPS points from multiple segments into one.

    Args:
        segment_ids: Ordered list of segment IDs to merge
        labeled_points_df: DataFrame with all labeled GPS points

    Returns:
        DataFrame with merged points, sorted by time
    """
    # Collect points from all segments
    merged = labeled_points_df[
        labeled_points_df["segment_global_id"].isin(segment_ids)
    ].copy()

    # Sort by time
    merged = merged.sort_values("time").reset_index(drop=True)

    return merged


def compute_merged_stats(
    merged_points: pd.DataFrame,
    new_segment_id: str,
    max_top_speed_kmh: float = 120.0
) -> Dict:
    """
    Compute statistics for a merged segment.

    Returns a dictionary with the same fields as segments_df rows.
    """
    if merged_points.empty:
        return {}

    # Compute distance (sum of segment distances)
    total_distance = float(merged_points["segment_distance_m"].sum())

    # Compute vertical drop (sum of negative altitude diffs)
    alt_diff = merged_points["altitude_m"].diff().fillna(0)
    total_drop = float(-(alt_diff[alt_diff < 0].sum()))

    # Compute duration
    time_diff = merged_points["time"].diff().dt.total_seconds().fillna(0)
    total_duration = float(time_diff.sum())

    # Compute speeds
    if total_duration > 0:
        avg_speed = (total_distance / 1000) / (total_duration / 3600)
    else:
        avg_speed = 0.0

    top_speed = min(
        float(merged_points["segment_speed_kmh"].max()),
        max_top_speed_kmh
    )

    return {
        "day_run_id": merged_points.iloc[0]["run_id"],
        "segment_global_id": new_segment_id,
        "start_time": merged_points["time"].min(),
        "end_time": merged_points["time"].max(),
        "distance_m": total_distance,
        "vertical_drop_m": total_drop,
        "duration_s": total_duration,
        "avg_speed_kmh": avg_speed,
        "top_speed_kmh": top_speed,
        "location_name": merged_points.iloc[0].get("location_name"),
        "year": merged_points.iloc[0].get("year"),
    }


def apply_glue_decisions(
    segments_df: pd.DataFrame,
    labeled_points_df: pd.DataFrame,
    decisions: List[GlueDecision],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply approved glue decisions to create merged segment data.

    Args:
        segments_df: Original segments DataFrame
        labeled_points_df: Original labeled points DataFrame
        decisions: List of glue decisions (only "approved" are applied)

    Returns:
        Tuple of (merged_segments_df, merged_points_df)
    """
    approved = [d for d in decisions if d.status == "approved"]

    if not approved:
        return segments_df.copy(), labeled_points_df.copy()

    # Track which original segments get merged
    merged_segment_ids = set()
    new_segments = []
    new_points_updates = {}  # segment_id -> new_segment_id mapping

    for decision in approved:
        segment_ids = decision.segment_ids

        # Skip if any segment in the chain was already merged
        if any(sid in merged_segment_ids for sid in segment_ids):
            continue

        # Mark these segments as merged
        merged_segment_ids.update(segment_ids)

        # Create new merged segment ID
        new_segment_id = f"merged__{decision.glue_id}"

        # Merge points
        merged_points = merge_segment_points(segment_ids, labeled_points_df)

        # Compute stats for new segment
        new_stats = compute_merged_stats(merged_points, new_segment_id)
        new_segments.append(new_stats)

        # Track point updates
        for sid in segment_ids:
            new_points_updates[sid] = new_segment_id

    # Build new segments DataFrame
    # Keep segments that weren't merged
    remaining_segments = segments_df[
        ~segments_df["segment_global_id"].isin(merged_segment_ids)
    ].copy()

    # Add new merged segments
    if new_segments:
        new_segments_df = pd.DataFrame(new_segments)
        merged_segments_df = pd.concat([remaining_segments, new_segments_df], ignore_index=True)
    else:
        merged_segments_df = remaining_segments

    # Build new points DataFrame with updated segment IDs
    merged_points_df = labeled_points_df.copy()
    for old_id, new_id in new_points_updates.items():
        mask = merged_points_df["segment_global_id"] == old_id
        merged_points_df.loc[mask, "segment_global_id"] = new_id

    return merged_segments_df, merged_points_df


def save_merged_data(
    segments_df: pd.DataFrame,
    points_df: pd.DataFrame,
    output_dir: Path,
    suffix: str = "_merged"
) -> tuple[Path, Path]:
    """
    Save merged data to new parquet files.

    Args:
        segments_df: Merged segments DataFrame
        points_df: Merged points DataFrame
        output_dir: Directory to save to
        suffix: Suffix to add to filenames

    Returns:
        Tuple of (segments_path, points_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    segments_path = output_dir / f"segments{suffix}.parquet"
    points_path = output_dir / f"labeled_points{suffix}.parquet"

    segments_df.to_parquet(segments_path, index=False)
    points_df.to_parquet(points_path, index=False)

    return segments_path, points_path


# CLI interface
def main():
    """CLI for applying glue decisions."""
    import argparse
    from . import analysis

    parser = argparse.ArgumentParser(description="Apply segment glue decisions")
    parser.add_argument("command", choices=["apply", "status"],
                       help="Command to run")
    parser.add_argument("--decisions", type=Path,
                       default=Path("data/glue_decisions.json"),
                       help="Path to decisions JSON file")
    parser.add_argument("--output", type=Path,
                       default=Path("data/processed"),
                       help="Output directory for merged parquet files")

    args = parser.parse_args()

    if args.command == "status":
        decisions = load_glue_decisions(args.decisions)
        approved = [d for d in decisions if d.status == "approved"]
        rejected = [d for d in decisions if d.status == "rejected"]
        pending = [d for d in decisions if d.status == "pending"]

        print(f"Glue decisions in {args.decisions}:")
        print(f"  Approved: {len(approved)}")
        print(f"  Rejected: {len(rejected)}")
        print(f"  Pending:  {len(pending)}")

        if approved:
            print("\nApproved merges:")
            for d in approved:
                print(f"  {d.glue_id}: {' + '.join(d.segment_ids)}")

    elif args.command == "apply":
        print("Loading datasets...")
        summary_df, points_df = analysis.load_datasets()

        print("Segmenting runs...")
        segments_df, labeled_points_df = analysis.segment_downhill_runs(points_df)

        print(f"Original segments: {len(segments_df)}")

        decisions = load_glue_decisions(args.decisions)
        approved = [d for d in decisions if d.status == "approved"]

        if not approved:
            print("No approved decisions to apply.")
            return

        print(f"Applying {len(approved)} approved glue decisions...")
        merged_segments, merged_points = apply_glue_decisions(
            segments_df, labeled_points_df, decisions
        )

        print(f"Merged segments: {len(merged_segments)}")

        seg_path, pts_path = save_merged_data(
            merged_segments, merged_points, args.output
        )

        print(f"Saved merged data to:")
        print(f"  {seg_path}")
        print(f"  {pts_path}")


if __name__ == "__main__":
    main()
