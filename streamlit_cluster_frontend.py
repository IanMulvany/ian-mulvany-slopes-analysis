"""
Cluster viewer (front-end optimized):
- Single PyDeck map for interactive view.
- Gallery of runs rendered as small SVG charts (no WebGL per-tile).
- Radius slider (10–400m) clusters segments whose start AND end are within the radius.

This app computes clusters locally to avoid many network round-trips, but uses a single
WebGL context and lightweight SVG for the gallery to reduce browser load.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from slopes_analysis import analysis
from streamlit_cluster import (
    SegmentEndpoints,
    cluster_segments,
    compute_segment_endpoints,
    color_palette,
    haversine_m,
)

RAW_DIR = Path("data/GPSLogs")
PROCESSED_DIR = Path("data/processed")


@lru_cache(maxsize=1)
def load_segments():
    summary_df, points_df = analysis.load_datasets(PROCESSED_DIR, RAW_DIR)
    seg_df, labeled_points = analysis.segment_downhill_runs(points_df)
    endpoints = compute_segment_endpoints(labeled_points)
    return seg_df, labeled_points, endpoints


@lru_cache(maxsize=64)
def cluster_for_radius(radius_m: float):
    _, labeled_points, endpoints = load_segments()
    clusters = cluster_segments(endpoints, radius_m)
    return clusters


def render_main_map(labeled_points: pd.DataFrame, clusters: Dict[str, List[str]]) -> None:
    cluster_ids = sorted(clusters.keys())
    palette = color_palette(len(cluster_ids))
    cluster_color = {cid: palette[i] for i, cid in enumerate(cluster_ids)}

    path_records = []
    for cid, members in clusters.items():
        color = cluster_color.get(cid, [52, 73, 94])
        for member in members:
            seg_points = labeled_points[labeled_points["segment_global_id"] == member]
            if seg_points.empty:
                continue
            path_records.append(
                {
                    "path": seg_points[["longitude", "latitude"]].values.tolist(),
                    "name": member,
                    "cluster": cid,
                    "color": color,
                }
            )

    if not path_records:
        st.info("No paths to display.")
        return

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
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{cluster}\n{name}"})
    st.pydeck_chart(deck)


def render_gallery(labeled_points: pd.DataFrame, clusters: Dict[str, List[str]]) -> None:
    st.subheader("Runs gallery (SVG thumbnails)")
    if labeled_points.empty:
        st.info("No point data to display.")
        return

    # Mapping segment -> cluster
    seg_to_cluster = {}
    for cid, members in clusters.items():
        for m in members:
            seg_to_cluster[m] = cid

    seg_groups = {seg_id: df for seg_id, df in labeled_points.groupby("segment_global_id")}
    seg_list = sorted(seg_groups.keys())
    cols_per_row = 4
    palette = color_palette(len(clusters))

    def cluster_color(cid):
        if cid is None:
            return "rgba(80,80,80,0.6)"
        idx = list(clusters.keys()).index(cid)
        r, g, b = palette[idx % len(palette)]
        return f"rgba({r},{g},{b},0.9)"

    rows = (len(seg_list) + cols_per_row - 1) // cols_per_row
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx >= len(seg_list):
                break
            seg_id = seg_list[idx]
            df = seg_groups[seg_id]
            cid = seg_to_cluster.get(seg_id)
            color = cluster_color(cid)
            with cols[c]:
                st.caption(f"{seg_id} | cluster: {cid}")
                df_plot = df.copy()
                df_plot["distance_km"] = df_plot["cumulative_distance_m"] / 1000
                fig = px.line(
                    df_plot,
                    x="distance_km",
                    y="altitude_m",
                    render_mode="svg",
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    height=180,
                )
                fig.update_traces(line=dict(color=color, width=2))
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Run clustering (frontend)", layout="wide")
    st.title("Downhill run clustering (frontend optimized)")
    st.write("Single WebGL map + SVG thumbnails to keep the browser light.")

    try:
        seg_df, labeled_points, _ = load_segments()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    radius_m = st.slider("Clustering radius (meters)", 10, 400, 120, 10)
    clusters = cluster_for_radius(radius_m)

    col_top1, col_top2 = st.columns([3, 2])
    with col_top1:
        render_main_map(labeled_points, clusters)
    with col_top2:
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
                }
            )
        st.dataframe(pd.DataFrame(records), use_container_width=True)

    render_gallery(labeled_points, clusters)


if __name__ == "__main__":
    main()

