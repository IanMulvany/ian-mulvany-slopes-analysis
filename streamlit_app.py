"""Interactive explorer for Slopes exports.

Steps to use:
1. Unzip your ``.slopes`` files into ``data/GPSLogs``
2. Install dependencies: ``python -m pip install -r requirements.txt``
3. Run ``streamlit run streamlit_app.py``
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from slopes_analysis import analysis, ingest

RAW_DIR = Path("data/GPSLogs")
PROCESSED_DIR = Path("data/processed")

REQUIRED_COLUMNS_DEFAULTS = {
    "run_id": "",
    "start_time": pd.NaT,
    "location_name": "Unknown",
    "distance_m": 0.0,
    "vertical_drop_m": 0.0,
    "avg_speed_kmh": 0.0,
    "top_speed_kmh": 0.0,
}


def ensure_columns(df: pd.DataFrame, defaults: dict[str, object]) -> pd.DataFrame:
    """Ensure required columns exist with default values."""
    missing = [c for c in defaults if c not in df.columns]
    if not missing:
        return df
    df = df.copy()
    for col in missing:
        df[col] = defaults[col]
    return df


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure critical columns exist for downstream UI sections."""
    return ensure_columns(df, REQUIRED_COLUMNS_DEFAULTS)


def ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the year column exists, deriving it from start_time if needed."""
    if "year" not in df.columns and "start_time" in df.columns:
        df = df.copy()
        df["year"] = pd.to_datetime(df["start_time"], errors="coerce").dt.year
    return df


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df, points_df = analysis.load_datasets(PROCESSED_DIR, RAW_DIR)
    # Ensure year column exists
    summary_df = ensure_year_column(summary_df)
    summary_df = ensure_required_columns(summary_df)
    if not points_df.empty:
        points_df = ensure_year_column(points_df)
        points_df = ensure_required_columns(points_df)
    return summary_df, points_df


@st.cache_data(show_spinner=True)
def rebuild_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return ingest.build_datasets(raw_dir=RAW_DIR, output_dir=PROCESSED_DIR, write_gpx=True)


def friendly_label(row: pd.Series) -> str:
    date_part = "" if pd.isna(row.get("start_time")) else pd.to_datetime(row.get("start_time")).strftime("%Y-%m-%d")
    loc = row.get("location_name") or "Unknown location"
    return f"{date_part} – {loc} ({row.get('file_name')})"


def render_overview(summary_df: pd.DataFrame) -> None:
    cards = analysis.overview_cards(summary_df)
    cols = st.columns(4)
    cols[0].metric("Total runs", cards["total_runs"])
    cols[1].metric("Distance", f"{cards['total_distance_km']:.1f} km")
    cols[2].metric("Vertical skied", f"{cards['total_vertical_m']:.0f} m")
    cols[3].metric("Avg speed", f"{cards['avg_speed_kmh']:.1f} km/h")
    if cards["years_active"]:
        st.caption(f"Seasons covered: {cards['years_active']}")


def render_yearly_trends(summary_df: pd.DataFrame) -> None:
    st.subheader("Yearly trends")
    yearly = analysis.yearly_trends(summary_df)
    if yearly.empty:
        st.info("No yearly data yet. Add more runs to see trends.")
        return
    fig = px.bar(yearly, x="year", y="run_count", title="Run counts per year", labels={"run_count": "Runs"})
    fig.add_scatter(
        x=yearly["year"],
        y=yearly["distance_km"],
        mode="lines+markers",
        name="Distance (km)",
    )
    # Render distance on a secondary axis for readability
    fig.update_layout(
        yaxis_title="Runs",
        yaxis2=dict(overlaying="y", side="right", title="Distance (km)"),
    )
    fig.data[-1].update(yaxis="y2")
    st.plotly_chart(fig, use_container_width=True)


def render_locations_map(summary_df: pd.DataFrame, points_df: pd.DataFrame) -> None:
    """Render a map showing all locations where the user has skied."""
    st.subheader("Locations Map")
    
    location_coords = analysis.get_location_coordinates(points_df, summary_df)
    
    if location_coords.empty:
        st.info("No location coordinate data available.")
        return
    
    # Prepare data for map
    map_data = location_coords.copy()
    map_data["size"] = map_data["run_count"] * 1000  # Scale marker size by run count
    # Format numbers for tooltip display
    map_data["total_distance_km_str"] = map_data["total_distance_km"].round(1).astype(str)
    map_data["avg_speed_kmh_str"] = map_data["avg_speed_kmh"].round(1).astype(str)
    map_data["top_speed_kmh_str"] = map_data["top_speed_kmh"].round(1).astype(str)
    
    # Create scatter plot layer
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data.to_dict("records"),
        get_position=["longitude", "latitude"],
        get_color=[255, 140, 0, 200],  # Orange color with transparency
        get_radius="size",
        radius_min_pixels=5,
        radius_max_pixels=50,
        pickable=True,
        auto_highlight=True,
    )
    
    # Calculate center of map
    center_lat = map_data["latitude"].mean()
    center_lon = map_data["longitude"].mean()
    
    # Create view state
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=4,
        pitch=0,
    )
    
    # Create tooltip
    tooltip = {
        "html": "<b>{location_name}</b><br/>"
                "Runs: {run_count}<br/>"
                "Total Distance: {total_distance_km_str} km<br/>"
                "Avg Speed: {avg_speed_kmh_str} km/h<br/>"
                "Top Speed: {top_speed_kmh_str} km/h",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }
    
    # Create deck (using default style - no API key needed)
    deck = pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
    )
    
    st.pydeck_chart(deck)
    
    # Show location count
    st.caption(f"Showing {len(map_data)} locations")


def render_locations(summary_df: pd.DataFrame, points_df: pd.DataFrame) -> None:
    st.subheader("Location Summary Statistics")
    
    detailed_stats = analysis.get_detailed_location_stats(summary_df, points_df)
    
    if detailed_stats.empty:
        st.info("No location data found yet.")
        return
    
    # Display summary cards
    cols = st.columns(4)
    cols[0].metric("Total Locations", len(detailed_stats))
    cols[1].metric("Total Runs", int(detailed_stats["run_count"].sum()))
    cols[2].metric("Total Distance", f"{detailed_stats['total_distance_km'].sum():.1f} km")
    cols[3].metric("Avg Runs/Location", f"{detailed_stats['run_count'].mean():.1f}")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        min_runs = st.slider("Minimum runs", 1, int(detailed_stats["run_count"].max()), 1)
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            options=["run_count", "total_distance_km", "avg_speed_kmh_mean", "top_speed_kmh_max"],
            format_func=lambda x: {
                "run_count": "Number of Runs",
                "total_distance_km": "Total Distance",
                "avg_speed_kmh_mean": "Average Speed",
                "top_speed_kmh_max": "Top Speed",
            }[x],
        )
    
    # Filter and sort
    filtered_stats = detailed_stats[detailed_stats["run_count"] >= min_runs].sort_values(
        sort_by, ascending=False
    )
    
    # Display table
    display_cols = [
        "location_name",
        "run_count",
        "year_count",
        "visit_years",
        "total_distance_km",
        "avg_distance_km",
        "avg_speed_kmh_mean",
        "top_speed_kmh_max",
        "avg_vertical_drop_m",
    ]
    
    display_df = filtered_stats[display_cols].copy()
    display_df = display_df.rename(
        columns={
            "location_name": "Location",
            "run_count": "Runs",
            "year_count": "Years",
            "visit_years": "Years Visited",
            "total_distance_km": "Total Distance (km)",
            "avg_distance_km": "Avg Distance (km)",
            "avg_speed_kmh_mean": "Avg Speed (km/h)",
            "top_speed_kmh_max": "Top Speed (km/h)",
            "avg_vertical_drop_m": "Avg Vertical Drop (m)",
        }
    )
    
    st.dataframe(display_df, use_container_width=True)
    
    # Show top locations chart
    st.markdown("**Top Locations by Run Count**")
    top_locations = filtered_stats.head(10)
    fig = px.bar(
        top_locations,
        x="location_name",
        y="run_count",
        labels={"location_name": "Location", "run_count": "Number of Runs"},
        title="Top 10 Locations",
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def render_run_filters(summary_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Run explorer")
    
    # Handle missing year column - derive from start_time if needed
    if "year" not in summary_df.columns and "start_time" in summary_df.columns:
        summary_df = summary_df.copy()
        summary_df["year"] = pd.to_datetime(summary_df["start_time"], errors="coerce").dt.year
    summary_df = ensure_required_columns(summary_df)
    
    filtered = summary_df.copy()
    
    # Year filter (only show if year column exists)
    if "year" in filtered.columns:
        years = sorted(filtered["year"].dropna().astype(int).unique())
        if len(years) > 0:
            year_filter = st.multiselect("Filter by year", options=years, default=years, key="run_filters_year")
            if year_filter:
                filtered = filtered[filtered["year"].isin(year_filter)]
        else:
            st.caption("No year data available")
    else:
        st.caption("Year filtering not available (no year or start_time data)")
    
    # Location filter
    locations = sorted(filtered["location_name"].fillna("Unknown").unique())
    location_filter = st.multiselect("Filter by location", options=locations, default=locations, key="run_filters_location")
    if location_filter:
        filtered = filtered[filtered["location_name"].fillna("Unknown").isin(location_filter)]

    display_cols = [
        "run_id",
        "start_time",
        "location_name",
        "distance_m",
        "vertical_drop_m",
        "avg_speed_kmh",
        "top_speed_kmh",
    ]
    st.dataframe(
        filtered[display_cols]
        .assign(distance_km=lambda df: df["distance_m"] / 1000)
        .rename(
            columns={
                "distance_km": "Distance (km)",
                "vertical_drop_m": "Vertical drop (m)",
                "avg_speed_kmh": "Avg speed (km/h)",
                "top_speed_kmh": "Top speed (km/h)",
            }
        )
        .drop(columns=["distance_m"]),
        use_container_width=True,
    )
    return filtered


def render_run_comparison(summary_df: pd.DataFrame, points_df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    st.subheader("Compare runs")
    filtered = filtered.copy()
    filtered["label"] = filtered.apply(friendly_label, axis=1)
    options = dict(zip(filtered["label"], filtered["run_id"]))
    default_selection = list(options.values())[:2]
    selected_labels = st.multiselect(
        "Choose runs to compare",
        options=list(options.keys()),
        default=list(options.keys())[:2],
        max_selections=3,
        key="run_compare_multiselect",
    )
    selected_runs: List[str] = [options[label] for label in selected_labels]

    if not selected_runs:
        st.info("Pick at least one run to compare.")
        return

    summary_view = summary_df[summary_df["run_id"].isin(selected_runs)][
        [
            "run_id",
            "location_name",
            "start_time",
            "distance_m",
            "vertical_drop_m",
            "avg_speed_kmh",
            "top_speed_kmh",
        ]
    ].copy()
    summary_view["distance_km"] = summary_view["distance_m"] / 1000
    summary_view = summary_view.drop(columns=["distance_m"])
    st.dataframe(summary_view, use_container_width=True)

    comparison = analysis.compare_runs(points_df, selected_runs)
    if comparison.empty:
        st.warning("No point data available for the selected runs.")
        return

    fig = px.line(
        comparison,
        x="distance_km",
        y="altitude_m",
        color="run_id",
        labels={"distance_km": "Distance (km)", "altitude_m": "Altitude (m)"},
        title="Altitude profile",
    )
    st.plotly_chart(fig, use_container_width=True)

    first_run_id = selected_runs[0]
    run_points = points_df[points_df["run_id"] == first_run_id]
    render_map(run_points, label=friendly_label(summary_df[summary_df["run_id"] == first_run_id].iloc[0]))


def render_map(points: pd.DataFrame, label: str) -> None:
    if points.empty:
        st.info("No coordinates available for the selected run.")
        return

    path_data = [{"path": points[["longitude", "latitude"]].values.tolist(), "name": label, "color": [0, 122, 255]}]
    view_state = pdk.ViewState(
        latitude=points["latitude"].mean(),
        longitude=points["longitude"].mean(),
        zoom=12,
        pitch=45,
    )
    layer = pdk.Layer(
        "PathLayer",
        path_data,
        get_path="path",
        get_color="color",
        width_scale=2,
        width_min_pixels=3,
    )
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}"})
    st.pydeck_chart(deck)


def render_seasonality(summary_df: pd.DataFrame) -> None:
    st.subheader("Seasonality")
    monthly = analysis.seasonal_density(summary_df)
    if monthly.empty:
        st.info("Not enough data to show monthly distribution.")
        return
    fig = px.bar(monthly, x="month", y="run_count", labels={"run_count": "Runs", "month": "Month"})
    st.plotly_chart(fig, use_container_width=True)


def render_overall_performance(summary_df: pd.DataFrame) -> None:
    st.subheader("Overall Performance Trends")
    perf_summary = analysis.get_overall_performance_summary(summary_df)
    
    if not perf_summary or perf_summary.get("trend") == "insufficient_data":
        st.info("Need data from at least 2 years to show performance trends.")
        return
    
    # Display trend indicator
    trend = perf_summary["trend"]
    trend_colors = {
        "improving": "🟢",
        "degrading": "🔴",
        "steady": "🟡",
        "mixed": "🟠",
    }
    trend_labels = {
        "improving": "Improving",
        "degrading": "Degrading",
        "steady": "Steady",
        "mixed": "Mixed",
    }
    
    st.markdown(f"**Overall Trend:** {trend_colors.get(trend, '')} {trend_labels.get(trend, trend)}")
    
    # Display metrics
    cols = st.columns(4)
    cols[0].metric(
        "Average Speed",
        f"{perf_summary['avg_speed_last']:.1f} km/h",
        delta=f"{perf_summary['avg_speed_change']:+.1f} km/h",
        delta_color="normal" if abs(perf_summary['avg_speed_change']) < 0.5 else ("normal" if perf_summary['avg_speed_change'] > 0 else "inverse"),
    )
    cols[1].metric(
        "Top Speed",
        f"{perf_summary['top_speed_last']:.1f} km/h",
        delta=f"{perf_summary['top_speed_change']:+.1f} km/h",
        delta_color="normal" if abs(perf_summary['top_speed_change']) < 0.5 else ("normal" if perf_summary['top_speed_change'] > 0 else "inverse"),
    )
    cols[2].metric("First Year", perf_summary["first_year"])
    cols[3].metric("Last Year", perf_summary["last_year"])
    
    # Yearly trends chart
    yearly_stats = perf_summary["yearly_stats"].copy()
    name_map = {
        "avg_speed_mean": "Average speed",
        "top_speed_max": "Top speed",
    }
    yearly_stats = yearly_stats.rename(columns=name_map)
    fig = px.line(
        yearly_stats,
        x="year",
        y=list(name_map.values()),
        labels={"value": "Speed (km/h)", "year": "Year", "variable": "Metric"},
        title="Speed Trends Over Years",
    )
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)


def render_run_list(summary_df: pd.DataFrame, points_df: pd.DataFrame) -> None:
    """List runs and show details for a selected run."""
    st.subheader("Run list and details")

    summary_df = ensure_year_column(summary_df)
    summary_df = ensure_required_columns(summary_df)

    # Year filter (optional)
    years = sorted(summary_df["year"].dropna().astype(int).unique()) if "year" in summary_df.columns else []
    if years:
        year_sel = st.multiselect("Filter by year", options=years, default=years, key="run_list_years")
        if year_sel:
            summary_df = summary_df[summary_df["year"].isin(year_sel)]

    # Data cleaning controls
    colc1, colc2, colc3 = st.columns(3)
    with colc1:
        max_top = st.slider("Top speed cap (km/h)", 50.0, 300.0, 180.0, 5.0, key="clean_top_speed_cap")
    with colc2:
        max_avg = st.slider("Avg speed cap (km/h)", 20.0, 200.0, 120.0, 5.0, key="clean_avg_speed_cap")
    with colc3:
        sort_metric = st.selectbox(
            "Sort by",
            options=[
                "start_time",
                "top_speed_kmh",
                "avg_speed_kmh",
                "distance_m",
                "vertical_drop_m",
            ],
            format_func=lambda x: {
                "start_time": "Date",
                "top_speed_kmh": "Top speed",
                "avg_speed_kmh": "Avg speed",
                "distance_m": "Distance",
                "vertical_drop_m": "Vertical drop",
            }[x],
            key="run_list_sort_metric",
        )

    # Apply cleaning filters
    summary_df = summary_df[
        (summary_df["top_speed_kmh"] <= max_top) & (summary_df["avg_speed_kmh"] <= max_avg)
    ]

    if summary_df.empty:
        st.info("No runs available.")
        return

    # Table view
    table_cols = [
        c
        for c in [
            "run_id",
            "start_time",
            "location_name",
            "year",
            "distance_m",
            "vertical_drop_m",
            "avg_speed_kmh",
            "top_speed_kmh",
        ]
        if c in summary_df.columns
    ]
    table = summary_df[table_cols].copy()
    if "distance_m" in table.columns:
        table["distance_km"] = table["distance_m"] / 1000
        table = table.drop(columns=["distance_m"])
    table = table.rename(
        columns={
            "location_name": "Location",
            "start_time": "Start",
            "year": "Year",
            "distance_km": "Distance (km)",
            "vertical_drop_m": "Vertical drop (m)",
            "avg_speed_kmh": "Avg speed (km/h)",
            "top_speed_kmh": "Top speed (km/h)",
        }
    )
    sort_col = {
        "start_time": "Start",
        "top_speed_kmh": "Top speed (km/h)",
        "avg_speed_kmh": "Avg speed (km/h)",
        "distance_m": "Distance (km)" if "Distance (km)" in table.columns else "Start",
        "vertical_drop_m": "Vertical drop (m)",
    }.get(sort_metric, "Start")
    ascending = sort_metric == "start_time"
    st.dataframe(table.sort_values(sort_col, ascending=ascending), use_container_width=True)

    # Run selector
    summary_df["label"] = summary_df.apply(friendly_label, axis=1)
    options = dict(zip(summary_df["label"], summary_df["run_id"]))
    default_label = summary_df.iloc[0]["label"]
    selected_label = st.selectbox("View run details", options=list(options.keys()), index=0 if options else None)
    if not selected_label:
        return
    run_id = options[selected_label]

    detail = summary_df[summary_df["run_id"] == run_id].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Location", detail.get("location_name", "Unknown"))
    cols[1].metric("Date", str(detail.get("start_time", ""))[:10])
    if "distance_m" in detail:
        cols[2].metric("Distance", f"{detail.get('distance_m', 0)/1000:.2f} km")
    cols[3].metric("Avg speed", f"{detail.get('avg_speed_kmh', 0):.1f} km/h")

    cols2 = st.columns(3)
    cols2[0].metric("Top speed", f"{detail.get('top_speed_kmh', 0):.1f} km/h")
    if "vertical_drop_m" in detail:
        cols2[1].metric("Vertical drop", f"{detail.get('vertical_drop_m', 0):.0f} m")
    if "duration_s" in detail:
        cols2[2].metric("Duration", f"{detail.get('duration_s', 0)/60:.1f} min")

    profile = analysis.run_profile(points_df, run_id)
    if not profile.empty:
        fig = px.line(
            profile,
            x="distance_km",
            y="altitude_m",
            labels={"distance_km": "Distance (km)", "altitude_m": "Altitude (m)"},
            title="Altitude profile",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No point data available for this run.")


def render_downhill_runs(summary_df: pd.DataFrame, points_df: pd.DataFrame) -> None:
    """Segment downhill runs within each day and enable cleaning/outlier control."""
    st.subheader("Downhill runs (segmented)")

    # Segment downhill runs
    seg_df, labeled_points = analysis.segment_downhill_runs(points_df)
    if seg_df.empty:
        st.info("No downhill segments detected. Add more data or adjust thresholds.")
        return

    # Filters
    seg_df = analysis.ensure_required_columns(seg_df) if hasattr(analysis, "ensure_required_columns") else seg_df
    years = sorted(seg_df["year"].dropna().astype(int).unique()) if "year" in seg_df.columns else []
    locations = sorted(seg_df["location_name"].fillna("Unknown").unique()) if "location_name" in seg_df.columns else []

    colf1, colf2 = st.columns(2)
    with colf1:
        year_sel = st.multiselect("Year", options=years, default=years, key="seg_years") if years else []
    with colf2:
        loc_sel = st.multiselect("Location", options=locations, default=locations, key="seg_locations") if locations else []

    filtered = seg_df.copy()
    if year_sel:
        filtered = filtered[filtered["year"].isin(year_sel)]
    if loc_sel:
        filtered = filtered[filtered["location_name"].fillna("Unknown").isin(loc_sel)]

    # Speed caps for cleaning
    colc1, colc2 = st.columns(2)
    with colc1:
        top_cap = st.slider("Cap top speed (km/h)", 40.0, 150.0, 110.0, 5.0, key="seg_top_cap")
    with colc2:
        avg_cap = st.slider("Cap avg speed (km/h)", 20.0, 120.0, 80.0, 5.0, key="seg_avg_cap")

    filtered = filtered[(filtered["top_speed_kmh"] <= top_cap) & (filtered["avg_speed_kmh"] <= avg_cap)]

    if filtered.empty:
        st.info("No segments after filters/caps.")
        return

    # Table
    display = filtered.copy()
    display["distance_km"] = display["distance_m"] / 1000
    display = display.rename(
        columns={
            "segment_global_id": "Segment ID",
            "day_run_id": "Day run",
            "location_name": "Location",
            "distance_km": "Distance (km)",
            "vertical_drop_m": "Vertical drop (m)",
            "avg_speed_kmh": "Avg speed (km/h)",
            "top_speed_kmh": "Top speed (km/h)",
            "duration_s": "Duration (s)",
            "year": "Year",
        }
    )
    st.dataframe(
        display[
            [
                "Segment ID",
                "Day run",
                "Location",
                "Year",
                "Distance (km)",
                "Vertical drop (m)",
                "Avg speed (km/h)",
                "Top speed (km/h)",
                "Duration (s)",
            ]
        ].sort_values("Top speed (km/h)", ascending=False),
        use_container_width=True,
    )

    # Detail view
    seg_options = dict(zip(filtered["segment_global_id"], filtered["segment_global_id"]))
    seg_selected = st.selectbox("Select a downhill run", options=list(seg_options.keys()), key="seg_select")
    if not seg_selected:
        return

    seg_detail = filtered[filtered["segment_global_id"] == seg_selected].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Location", seg_detail.get("location_name", "Unknown"))
    c2.metric("Distance", f"{seg_detail.get('distance_m', 0)/1000:.2f} km")
    c3.metric("Drop", f"{seg_detail.get('vertical_drop_m', 0):.0f} m")
    c4.metric("Top speed", f"{seg_detail.get('top_speed_kmh', 0):.1f} km/h")

    # Plot profile for this segment
    seg_points = labeled_points[labeled_points["segment_global_id"] == seg_selected].copy()
    if not seg_points.empty:
        seg_points["distance_km"] = seg_points["cumulative_distance_m"] / 1000
        fig_seg = px.line(
            seg_points,
            x="distance_km",
            y="altitude_m",
            labels={"distance_km": "Distance (km)", "altitude_m": "Altitude (m)"},
            title="Downhill profile",
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.info("No point data for this segment.")


def render_location_comparison(summary_df: pd.DataFrame, points_df: pd.DataFrame) -> None:
    st.subheader("Compare Same Run Across Years")
    
    location_year_stats = analysis.get_runs_by_location(summary_df)
    
    if location_year_stats.empty:
        st.info("No location data available for comparison.")
        return
    
    # Get locations with multiple years
    location_counts = location_year_stats.groupby("location_name")["year"].nunique()
    multi_year_locations = location_counts[location_counts >= 2].index.tolist()
    
    if not multi_year_locations:
        st.info("No locations with data from multiple years found.")
        return
    
    selected_location = st.selectbox(
        "Select a location to compare across years",
        options=sorted(multi_year_locations),
        help="Select a location where you have runs from multiple years",
    )
    
    if not selected_location:
        return
    
    # Get data for selected location
    location_data = location_year_stats[location_year_stats["location_name"] == selected_location].sort_values("year")
    
    # Display summary table
    st.markdown(f"**Performance at {selected_location}**")
    display_df = location_data[
        ["year", "run_count", "avg_speed_kmh_mean", "top_speed_kmh_max", "distance_m_mean", "vertical_drop_m_mean"]
    ].copy()
    display_df = display_df.rename(
        columns={
            "year": "Year",
            "run_count": "Runs",
            "avg_speed_kmh_mean": "Avg Speed (km/h)",
            "top_speed_kmh_max": "Top Speed (km/h)",
            "distance_m_mean": "Avg Distance (m)",
            "vertical_drop_m_mean": "Avg Vertical Drop (m)",
        }
    )
    display_df["Avg Distance (km)"] = display_df["Avg Distance (m)"] / 1000
    display_df = display_df.drop(columns=["Avg Distance (m)"])
    st.dataframe(display_df, use_container_width=True)
    
    # Get trend analysis
    trend_info = analysis.get_location_performance_trend(summary_df, selected_location)
    
    if trend_info["trend"] != "insufficient_data":
        trend_colors = {
            "improving": "🟢",
            "degrading": "🔴",
            "steady": "🟡",
            "mixed": "🟠",
        }
        st.markdown(
            f"**Trend:** {trend_colors.get(trend_info['trend'], '')} {trend_info['trend'].title()} "
            f"(Average Speed: {trend_info['avg_speed_trend']}, Top Speed: {trend_info['top_speed_trend']})"
        )
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_avg = px.bar(
            location_data,
            x="year",
            y="avg_speed_kmh_mean",
            labels={"year": "Year", "avg_speed_kmh_mean": "Average Speed (km/h)"},
            title="Average Speed by Year",
        )
        fig_avg.update_traces(marker_color="steelblue")
        st.plotly_chart(fig_avg, use_container_width=True)
    
    with col2:
        fig_top = px.bar(
            location_data,
            x="year",
            y="top_speed_kmh_max",
            labels={"year": "Year", "top_speed_kmh_max": "Top Speed (km/h)"},
            title="Top Speed by Year",
        )
        fig_top.update_traces(marker_color="crimson")
        st.plotly_chart(fig_top, use_container_width=True)
    
    # Year-over-year changes
    yoy_trends = analysis.compute_year_over_year_trends(summary_df)
    location_yoy = yoy_trends[yoy_trends["location_name"] == selected_location].sort_values("year")
    
    if not location_yoy.empty:
        st.markdown("**Year-over-Year Changes**")
        yoy_display = location_yoy[
            ["year", "prev_year", "avg_speed_change", "avg_speed_change_pct", "top_speed_change", "top_speed_change_pct"]
        ].copy()
        yoy_display = yoy_display.rename(
            columns={
                "year": "Year",
                "prev_year": "Previous Year",
                "avg_speed_change": "Avg Speed Change (km/h)",
                "avg_speed_change_pct": "Avg Speed Change (%)",
                "top_speed_change": "Top Speed Change (km/h)",
                "top_speed_change_pct": "Top Speed Change (%)",
            }
        )
        yoy_display["Avg Speed Change (%)"] = yoy_display["Avg Speed Change (%)"].round(1)
        yoy_display["Top Speed Change (%)"] = yoy_display["Top Speed Change (%)"].round(1)
        st.dataframe(yoy_display, use_container_width=True)
        
        # Visualize year-over-year changes
        yoy_plot = location_yoy.rename(
            columns={
                "avg_speed_change": "Avg speed change",
                "top_speed_change": "Top speed change",
            }
        )
        fig_yoy = px.bar(
            yoy_plot,
            x="year",
            y=["Avg speed change", "Top speed change"],
            barmode="group",
            labels={"value": "Change (km/h)", "year": "Year", "variable": "Metric"},
            title="Year-over-Year Speed Changes",
        )
        fig_yoy.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_yoy, use_container_width=True)
    
    # Compare actual runs from different years
    st.markdown("**Compare Individual Runs**")
    location_runs = summary_df[summary_df["location_name"] == selected_location].copy()
    
    # Ensure year column exists
    location_runs = ensure_year_column(location_runs)
    
    if "year" in location_runs.columns:
        location_runs = location_runs.sort_values("year")
    
    if len(location_runs) > 0:
        if "year" in location_runs.columns:
            years_with_runs = sorted(location_runs["year"].dropna().astype(int).unique())
            if len(years_with_runs) > 0:
                selected_years = st.multiselect(
                    "Select years to compare individual runs",
                    options=years_with_runs,
                    default=years_with_runs[:min(3, len(years_with_runs))],
                    key="location_compare_years",
                )
                
                if selected_years:
                    filtered_runs = location_runs[location_runs["year"].isin(selected_years)]
                else:
                    filtered_runs = location_runs.head(3)  # Show first 3 runs if no year selection
            else:
                filtered_runs = location_runs.head(3)  # Show first 3 runs if no year data
        else:
            filtered_runs = location_runs.head(3)  # Show first 3 runs if no year column
        
        filtered_runs["label"] = filtered_runs.apply(friendly_label, axis=1)
        
        if len(filtered_runs) > 0:
            comparison = analysis.compare_runs(points_df, filtered_runs["run_id"].tolist())
            if not comparison.empty:
                fig_profile = px.line(
                    comparison,
                    x="distance_km",
                    y="altitude_m",
                    color="run_id",
                    labels={"distance_km": "Distance (km)", "altitude_m": "Altitude (m)"},
                    title="Altitude Profile Comparison",
                )
                st.plotly_chart(fig_profile, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Slopes history explorer", layout="wide")
    st.title("Slopes history explorer")
    st.write(
        "Compare your days on snow across seasons, and export normalized GPX traces for other tools. "
        "Use the button below to rebuild datasets if you've added new `.slopes` files."
    )

    if st.sidebar.button("Rebuild processed data", help="Run the converter and refresh cached datasets"):
        with st.spinner("Converting .slopes archives..."):
            rebuild_data()
        load_data.clear()
        st.rerun()

    try:
        summary_df, points_df = load_data()
    except FileNotFoundError:
        st.error(
            "Processed data not found. Unzip your `GPSLogs.zip` into `data/GPSLogs` "
            "and click 'Rebuild processed data'."
        )
        return

    render_overview(summary_df)
    render_overall_performance(summary_df)
    render_run_list(summary_df, points_df)
    render_yearly_trends(summary_df)
    render_location_comparison(summary_df, points_df)
    render_locations_map(summary_df, points_df)
    render_locations(summary_df, points_df)
    render_downhill_runs(summary_df, points_df)
    filtered = render_run_filters(summary_df)
    render_seasonality(summary_df)
    render_run_comparison(summary_df, points_df, filtered)

    st.caption(
        "GPX exports are written to data/processed/gpx whenever you rebuild. "
        "Processed tables live in data/processed/run_summary.parquet and data/processed/points.parquet."
    )


if __name__ == "__main__":
    main()
