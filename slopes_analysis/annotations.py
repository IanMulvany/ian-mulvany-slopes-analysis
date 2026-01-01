"""Unified module for loading, merging and deduplicating annotations.

This module reads all annotation files from data/annotated/ and provides
a unified interface to access glue decisions and outlier markings.

Multiple annotation files may exist over time (e.g., glue_decisions.json,
glue_decisions-2.json, etc.). This module merges them and keeps the latest
decision for each segment/glue_id based on timestamps.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


# Default annotation directory
DEFAULT_ANNOTATED_DIR = Path("data/annotated")


@dataclass
class GlueDecision:
    """A decision to glue segments together."""
    glue_id: str
    segment_ids: List[str]
    status: str  # "approved", "rejected", "pending"
    notes: str = ""
    created_at: str = ""


@dataclass
class OutlierMarking:
    """A segment marked as an outlier."""
    id: str  # segment_global_id
    location: str
    date: str
    distance_m: float
    avg_speed: float
    top_speed: float
    marked_at: str


def parse_timestamp(ts: str) -> datetime:
    """Parse various timestamp formats to datetime."""
    if not ts:
        return datetime.min

    # Try various formats
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue

    return datetime.min


def load_glue_decisions_from_file(path: Path) -> List[GlueDecision]:
    """Load glue decisions from a single JSON file."""
    if not path.exists():
        return []

    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    decisions = []
    for d in data.get("decisions", []):
        decisions.append(GlueDecision(
            glue_id=d.get("glue_id", ""),
            segment_ids=d.get("segment_ids", []),
            status=d.get("status", "pending"),
            notes=d.get("notes", ""),
            created_at=d.get("created_at", data.get("created", ""))
        ))

    return decisions


def load_outliers_from_file(path: Path) -> List[OutlierMarking]:
    """Load outlier markings from a single JSON file."""
    if not path.exists():
        return []

    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    # Handle both array format and object format
    if isinstance(data, list):
        items = data
    else:
        items = data.get("outliers", [])

    outliers = []
    for item in items:
        outliers.append(OutlierMarking(
            id=item.get("id", ""),
            location=item.get("location", "Unknown"),
            date=item.get("date", ""),
            distance_m=item.get("distance_m", 0),
            avg_speed=item.get("avg_speed", 0),
            top_speed=item.get("top_speed", 0),
            marked_at=item.get("markedAt", item.get("marked_at", ""))
        ))

    return outliers


def load_all_glue_decisions(
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR
) -> List[GlueDecision]:
    """
    Load and merge all glue decisions from the annotated directory.

    Files matching glue_decisions*.json are loaded and merged.
    When the same glue_id appears multiple times, the latest version
    (by created_at timestamp) is kept.

    Returns:
        List of deduplicated GlueDecision objects
    """
    if not annotated_dir.exists():
        return []

    # Find all glue decision files
    files = sorted(annotated_dir.glob("glue_decisions*.json"))

    # Collect all decisions
    all_decisions: Dict[str, GlueDecision] = {}
    all_timestamps: Dict[str, datetime] = {}

    for file_path in files:
        decisions = load_glue_decisions_from_file(file_path)
        for decision in decisions:
            ts = parse_timestamp(decision.created_at)
            existing_ts = all_timestamps.get(decision.glue_id, datetime.min)

            if ts >= existing_ts:
                all_decisions[decision.glue_id] = decision
                all_timestamps[decision.glue_id] = ts

    return list(all_decisions.values())


def load_all_outliers(
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR
) -> List[OutlierMarking]:
    """
    Load and merge all outlier markings from the annotated directory.

    Files matching ski_outliers*.json are loaded and merged.
    When the same segment ID appears multiple times, the latest version
    (by marked_at timestamp) is kept.

    Returns:
        List of deduplicated OutlierMarking objects
    """
    if not annotated_dir.exists():
        return []

    # Find all outlier files
    files = sorted(annotated_dir.glob("ski_outliers*.json"))

    # Collect all outliers
    all_outliers: Dict[str, OutlierMarking] = {}
    all_timestamps: Dict[str, datetime] = {}

    for file_path in files:
        outliers = load_outliers_from_file(file_path)
        for outlier in outliers:
            ts = parse_timestamp(outlier.marked_at)
            existing_ts = all_timestamps.get(outlier.id, datetime.min)

            if ts >= existing_ts:
                all_outliers[outlier.id] = outlier
                all_timestamps[outlier.id] = ts

    return list(all_outliers.values())


def get_approved_glues(
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR
) -> List[GlueDecision]:
    """
    Get only approved glue decisions.

    Returns:
        List of GlueDecision objects with status == "approved"
    """
    all_decisions = load_all_glue_decisions(annotated_dir)
    return [d for d in all_decisions if d.status == "approved"]


def get_outlier_segment_ids(
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR
) -> Set[str]:
    """
    Get set of segment IDs that are marked as outliers.

    Returns:
        Set of segment_global_id strings
    """
    outliers = load_all_outliers(annotated_dir)
    return {o.id for o in outliers}


def get_outliers_dict(
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR
) -> Dict[str, OutlierMarking]:
    """
    Get outliers as a dictionary keyed by segment ID.

    Returns:
        Dict mapping segment_global_id to OutlierMarking
    """
    outliers = load_all_outliers(annotated_dir)
    return {o.id: o for o in outliers}


def get_segments_to_merge(
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR
) -> List[List[str]]:
    """
    Get list of segment ID groups that should be merged.

    This deduplicates overlapping merge requests by building
    connected components from all approved glue decisions.

    Returns:
        List of segment ID lists, each representing a merge group
    """
    approved = get_approved_glues(annotated_dir)

    if not approved:
        return []

    # Build adjacency from all approved decisions
    # Use union-find to group connected segments
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union all segments that appear together in approved decisions
    for decision in approved:
        segment_ids = decision.segment_ids
        for i in range(len(segment_ids) - 1):
            union(segment_ids[i], segment_ids[i + 1])

    # Group by root
    groups: Dict[str, List[str]] = {}
    all_segments = set()
    for decision in approved:
        all_segments.update(decision.segment_ids)

    for seg_id in all_segments:
        root = find(seg_id)
        if root not in groups:
            groups[root] = []
        if seg_id not in groups[root]:
            groups[root].append(seg_id)

    # Sort each group and filter to groups with 2+ segments
    result = []
    for group in groups.values():
        if len(group) >= 2:
            # Sort by segment number
            sorted_group = sorted(group, key=lambda x: (
                x.rsplit("__seg", 1)[0],  # run_id
                int(x.rsplit("__seg", 1)[1]) if "__seg" in x else 0  # seg number
            ))
            result.append(sorted_group)

    return result


def get_annotation_summary(
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR
) -> Dict:
    """
    Get summary statistics about annotations.

    Returns:
        Dict with counts and summaries
    """
    all_glue = load_all_glue_decisions(annotated_dir)
    all_outliers = load_all_outliers(annotated_dir)

    glue_by_status = {}
    for d in all_glue:
        glue_by_status[d.status] = glue_by_status.get(d.status, 0) + 1

    merge_groups = get_segments_to_merge(annotated_dir)

    return {
        "total_glue_decisions": len(all_glue),
        "glue_by_status": glue_by_status,
        "total_outliers": len(all_outliers),
        "merge_groups": len(merge_groups),
        "segments_to_merge": sum(len(g) for g in merge_groups),
    }


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View annotation summary")
    parser.add_argument("--dir", type=Path, default=DEFAULT_ANNOTATED_DIR,
                       help="Annotations directory")
    args = parser.parse_args()

    print(f"Loading annotations from {args.dir}...")
    summary = get_annotation_summary(args.dir)

    print(f"\nGlue Decisions: {summary['total_glue_decisions']}")
    for status, count in summary['glue_by_status'].items():
        print(f"  - {status}: {count}")

    print(f"\nOutliers: {summary['total_outliers']}")
    print(f"\nMerge Groups: {summary['merge_groups']}")
    print(f"Total Segments to Merge: {summary['segments_to_merge']}")
