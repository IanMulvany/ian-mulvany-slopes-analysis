"""
Streamlit app to cluster downhill runs (segments) across days based on start/end proximity.

Workflow:
- Loads processed data (or instructs to rebuild via setup_data.py).
- Derives downhill segments using analysis.segment_downhill_runs.
- User adjusts a radius slider (10–400 m); segments whose starts AND ends are within that radius are clustered.
- Clusters are visualized on a map; user can add annotations per cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from slopes_analysis import analysis

RAW_DIR = Path("data/GPSLogs")
PROCESSED_DIR = Path("data/processed")


# --- Helpers -----------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in meters."""
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(R * c)


@dataclass
class SegmentEndpoints:
    segment_id: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    location_name: str | None
    year: int | None


def compute_segment_endpoints(seg_points: pd.DataFrame) -> List[SegmentEndpoints]:
    """Compute start/end coords for each segment_global_id."""
    endpoints: List[SegmentEndpoints] = []
    for seg_id, df in seg_points.groupby("segment_global_id"):
        df_sorted = df.sort_values("time")
        first = df_sorted.iloc[0]
        last = df_sorted.iloc[-1]
        endpoints.append(
            SegmentEndpoints(
                segment_id=seg_id,
                start_lat=float(first["latitude"]),
                start_lon=float(first["longitude"]),
                end_lat=float(last["latitude"]),
                end_lon=float(last["longitude"]),
                location_name=first.get("location_name"),
                year=int(first["year"]) if pd.notnull(first.get("year")) else None,
            )
        )
    return endpoints


class UnionFind:
    def __init__(self, items: List[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def cluster_segments(endpoints: List[SegmentEndpoints], radius_m: float) -> Dict[str, List[str]]:
    """Cluster segments whose start AND end are within radius_m."""
    if not endpoints:
        return {}
    uf = UnionFind([e.segment_id for e in endpoints])
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1 :]:
            start_d = haversine_m(a.start_lat, a.start_lon, b.start_lat, b.start_lon)
            end_d = haversine_m(a.end_lat, a.end_lon, b.end_lat, b.end_lon)
            if start_d <= radius_m and end_d <= radius_m:
                uf.union(a.segment_id, b.segment_id)
    clusters: Dict[str, List[str]] = {}
    for e in endpoints:
        root = uf.find(e.segment_id)
        clusters.setdefault(root, []).append(e.segment_id)
    return clusters


def color_palette(n: int) -> List[List[int]]:
    base = [
        [52, 152, 219],
        [231, 76, 60],
        [46, 204, 113],
        [155, 89, 182],
        [241, 196, 15],
        [230, 126, 34],
        [26, 188, 156],
        [127, 140, 141],
        [52, 73, 94],
        [149, 165, 166],
    ]
    if n <= len(base):
        return base[:n]
    # Repeat if more clusters than base colors
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def render_gallery(labeled_points: pd.DataFrame, clusters: Dict[str, List[str]], cluster_color: Dict[str, List[int]]) -> None:
    """Render a grid of small maps, one per segment, colored by cluster."""
    st.subheader("Runs gallery")
    if labeled_points.empty:
        st.info("No point data to display.")
        return

    # Build lookup for quick subset
    seg_groups = {seg_id: df for seg_id, df in labeled_points.groupby("segment_global_id")}
    seg_list = sorted(seg_groups.keys())

    cols_per_row = 4
    rows = (len(seg_list) + cols_per_row - 1) // cols_per_row

    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx >= len(seg_list):
                break
            seg_id = seg_list[idx]
            df = seg_groups[seg_id]
            cid = None
            for root, members in clusters.items():
                if seg_id in members:
                    cid = root
                    break
            color = cluster_color.get(cid, [52, 73, 94])
            with cols[c]:
                st.caption(f"{seg_id} | cluster: {cid}")
                path = df[["longitude", "latitude"]].values.tolist()
                center_lat = df["latitude"].mean()
                center_lon = df["longitude"].mean()
                view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=40)
                layer = pdk.Layer(
                    "PathLayer",
                    data=[{"path": path, "name": seg_id, "color": color}],
                    get_path="path",
                    get_color="color",
                    width_scale=2,
                    width_min_pixels=2,
                )
                deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}"})
                st.pydeck_chart(deck)


# --- App ---------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Run clustering", layout="wide")
    st.title("Downhill run clustering")
    st.write(
        "Cluster downhill runs (segments) across days by proximity of start/end points. "
        "Adjust the radius to merge nearby routes and optionally add annotations per cluster."
    )

    # Load data
    try:
        summary_df, points_df = analysis.load_datasets(PROCESSED_DIR, RAW_DIR)
    except FileNotFoundError:
        st.error("Processed data not found. Run `uv run python setup_data.py --force-unzip` first.")
        return

    # Derive downhill segments
    seg_df, labeled_points = analysis.segment_downhill_runs(points_df)
    if seg_df.empty:
        st.info("No downhill segments detected. Add more data or rebuild processed datasets.")
        return

    # Radius slider
    radius_m = st.slider("Clustering radius (meters)", 10, 400, 120, 10)

    # Compute endpoints and clusters
    endpoints = compute_segment_endpoints(labeled_points)
    clusters = cluster_segments(endpoints, radius_m)

    # Prepare data for display
    cluster_ids = sorted(clusters.keys())
    palette = color_palette(len(cluster_ids))
    cluster_color = {cid: palette[i] for i, cid in enumerate(cluster_ids)}

    # Annotations stored in session state
    if "cluster_notes" not in st.session_state:
        st.session_state["cluster_notes"] = {}

    # Cluster selection and annotation
    st.subheader("Clusters")
    sel_cluster = st.selectbox(
        "Select a cluster", options=cluster_ids, format_func=lambda cid: f"{cid} ({len(clusters[cid])} runs)"
    )
    if sel_cluster:
        note = st.text_area(
            "Annotation for this cluster",
            value=st.session_state["cluster_notes"].get(sel_cluster, ""),
            placeholder="Notes, conditions, trail name, etc.",
        )
        if st.button("Save annotation"):
            st.session_state["cluster_notes"][sel_cluster] = note
            st.success("Saved annotation")

    # Build map data
    path_records = []
    for seg_id, color in cluster_color.items():
        if seg_id not in clusters:
            continue
        for member in clusters[seg_id]:
            seg_points = labeled_points[labeled_points["segment_global_id"] == member]
            if seg_points.empty:
                continue
            path_records.append(
                {
                    "path": seg_points[["longitude", "latitude"]].values.tolist(),
                    "name": member,
                    "cluster": seg_id,
                    "color": color,
                }
            )

    if not path_records:
        st.info("No paths to display for current clustering.")
        return

    # Map
    mean_lat = labeled_points["latitude"].mean()
    mean_lon = labeled_points["longitude"].mean()
    view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=11, pitch=40)
    layer = pdk.Layer(
        "PathLayer",
        data=path_records,
        get_path="path",
        get_color="color",
        width_scale=2,
        width_min_pixels=3,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {"text": "{cluster}\n{name}"}
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(deck)

    # Table of clusters
    st.subheader("Cluster summary")
    records = []
    for cid, members in clusters.items():
        sub = seg_df[seg_df["segment_global_id"].isin(members)]
        records.append(
            {
                "cluster": cid,
                "runs": len(members),
                "locations": ", ".join(sorted(set(sub["location_name"].fillna("Unknown")))),
                "years": ", ".join(sorted({str(y) for y in sub["year"].dropna().unique()})),
                "note": st.session_state["cluster_notes"].get(cid, ""),
            }
        )
    st.dataframe(pd.DataFrame(records), use_container_width=True)

    # Gallery view
    render_gallery(labeled_points, clusters, cluster_color)


if __name__ == "__main__":
    main()

