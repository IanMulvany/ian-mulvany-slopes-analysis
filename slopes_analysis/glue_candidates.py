"""Detect candidate segment pairs for gluing together."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass
class GlueCandidate:
    """A pair of segments that are candidates for gluing."""
    pair_id: str
    segment_a_id: str
    segment_b_id: str
    time_gap_s: float
    spatial_gap_m: float
    confidence: float  # 0-1, higher = more likely should be glued
    # Metadata for display
    location: str
    date: str
    segment_a_end_time: str
    segment_b_start_time: str


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in meters between two points."""
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(R * c)


def compute_confidence(time_gap_s: float, spatial_gap_m: float,
                       max_time_s: float = 1800, max_dist_m: float = 200) -> float:
    """
    Compute confidence score for a glue candidate.

    Score is higher when:
    - Time gap is smaller (closer to 90s = likely just a pause)
    - Spatial gap is smaller (end of A near start of B)

    Returns value between 0 and 1.
    """
    # Normalize to 0-1 range (inverted - smaller gaps = higher score)
    time_score = max(0, 1 - (time_gap_s / max_time_s))
    dist_score = max(0, 1 - (spatial_gap_m / max_dist_m))

    # Weight spatial proximity slightly more than time
    return 0.4 * time_score + 0.6 * dist_score


def get_segment_endpoints(
    segment_id: str,
    labeled_points_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    """Get first and last points of a segment."""
    seg_points = labeled_points_df[
        labeled_points_df["segment_global_id"] == segment_id
    ].sort_values("time")

    if seg_points.empty:
        raise ValueError(f"No points found for segment {segment_id}")

    return seg_points.iloc[0], seg_points.iloc[-1]


def find_glue_candidates(
    segments_df: pd.DataFrame,
    labeled_points_df: pd.DataFrame,
    time_threshold_min: float = 30.0,
    spatial_threshold_m: float = 200.0,
) -> List[GlueCandidate]:
    """
    Find pairs of segments that are candidates for gluing.

    A candidate pair is two consecutive segments from the same run where:
    - The time gap between them is less than time_threshold_min
    - The spatial gap (end of A to start of B) is less than spatial_threshold_m

    Args:
        segments_df: DataFrame with segment summary data
        labeled_points_df: DataFrame with GPS points labeled by segment
        time_threshold_min: Maximum time gap in minutes to consider
        spatial_threshold_m: Maximum spatial gap in meters to consider

    Returns:
        List of GlueCandidate objects sorted by confidence (highest first)
    """
    if segments_df.empty:
        return []

    candidates = []
    time_threshold_s = time_threshold_min * 60

    # Group segments by day_run_id and process each run
    for run_id, run_segments in segments_df.groupby("day_run_id"):
        # Sort by start_time
        run_segments = run_segments.sort_values("start_time")

        if len(run_segments) < 2:
            continue

        # Check consecutive pairs
        for i in range(len(run_segments) - 1):
            seg_a = run_segments.iloc[i]
            seg_b = run_segments.iloc[i + 1]

            # Calculate time gap
            time_gap_s = (seg_b["start_time"] - seg_a["end_time"]).total_seconds()

            # Skip if time gap too large or negative
            if time_gap_s < 0 or time_gap_s > time_threshold_s:
                continue

            # Get endpoints for spatial check
            try:
                _, end_a = get_segment_endpoints(seg_a["segment_global_id"], labeled_points_df)
                start_b, _ = get_segment_endpoints(seg_b["segment_global_id"], labeled_points_df)
            except ValueError:
                continue

            # Calculate spatial gap
            spatial_gap_m = haversine_m(
                end_a["latitude"], end_a["longitude"],
                start_b["latitude"], start_b["longitude"]
            )

            # Skip if too far apart
            if spatial_gap_m > spatial_threshold_m:
                continue

            # Compute confidence
            confidence = compute_confidence(time_gap_s, spatial_gap_m,
                                           max_time_s=time_threshold_s,
                                           max_dist_m=spatial_threshold_m)

            # Create candidate
            pair_id = f"glue_{seg_a['segment_global_id']}__{seg_b['segment_global_id']}"

            candidates.append(GlueCandidate(
                pair_id=pair_id,
                segment_a_id=seg_a["segment_global_id"],
                segment_b_id=seg_b["segment_global_id"],
                time_gap_s=time_gap_s,
                spatial_gap_m=spatial_gap_m,
                confidence=confidence,
                location=seg_a.get("location_name") or "Unknown",
                date=seg_a["start_time"].strftime("%Y-%m-%d") if pd.notna(seg_a["start_time"]) else "",
                segment_a_end_time=seg_a["end_time"].strftime("%H:%M:%S") if pd.notna(seg_a["end_time"]) else "",
                segment_b_start_time=seg_b["start_time"].strftime("%H:%M:%S") if pd.notna(seg_b["start_time"]) else "",
            ))

    # Sort by confidence (highest first)
    candidates.sort(key=lambda c: c.confidence, reverse=True)

    return candidates


def find_chain_candidates(
    segments_df: pd.DataFrame,
    labeled_points_df: pd.DataFrame,
    time_threshold_min: float = 30.0,
    spatial_threshold_m: float = 200.0,
) -> List[List[str]]:
    """
    Find chains of segments that could all be glued together.

    This extends find_glue_candidates to identify cases where
    segments A, B, C are all candidates to form one long segment.

    Returns:
        List of segment ID chains, e.g., [["seg1", "seg2", "seg3"], ["seg4", "seg5"]]
    """
    pairs = find_glue_candidates(segments_df, labeled_points_df,
                                  time_threshold_min, spatial_threshold_m)

    if not pairs:
        return []

    # Build adjacency: segment_id -> next segment it can glue to
    next_seg = {}
    prev_seg = {}
    for pair in pairs:
        next_seg[pair.segment_a_id] = pair.segment_b_id
        prev_seg[pair.segment_b_id] = pair.segment_a_id

    # Find chain starts (segments with no predecessor in glue pairs)
    chain_starts = [seg for seg in next_seg if seg not in prev_seg]

    chains = []
    for start in chain_starts:
        chain = [start]
        current = start
        while current in next_seg:
            current = next_seg[current]
            chain.append(current)
        if len(chain) > 1:
            chains.append(chain)

    return chains
