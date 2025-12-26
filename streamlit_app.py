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


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return analysis.load_datasets(PROCESSED_DIR, RAW_DIR)


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


def render_locations(summary_df: pd.DataFrame) -> None:
    st.subheader("Where you've skied")
    leaderboard = analysis.location_leaderboard(summary_df, top_n=15)
    if leaderboard.empty:
        st.info("No location data found yet.")
        return
    st.dataframe(
        leaderboard.rename(columns={"run_count": "Runs", "distance_km": "Distance (km)"}),
        use_container_width=True,
    )


def render_run_filters(summary_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Run explorer")
    years = sorted(summary_df["year"].dropna().astype(int).unique())
    year_filter = st.multiselect("Filter by year", options=years, default=years)
    locations = sorted(summary_df["location_name"].fillna("Unknown").unique())
    location_filter = st.multiselect("Filter by location", options=locations, default=locations)

    filtered = summary_df.copy()
    if year_filter:
        filtered = filtered[filtered["year"].isin(year_filter)]
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
        "Choose runs to compare", options=list(options.keys()), default=list(options.keys())[:2], max_selections=3
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
        st.experimental_rerun()

    try:
        summary_df, points_df = load_data()
    except FileNotFoundError:
        st.error(
            "Processed data not found. Unzip your `GPSLogs.zip` into `data/GPSLogs` "
            "and click 'Rebuild processed data'."
        )
        return

    render_overview(summary_df)
    render_yearly_trends(summary_df)
    render_locations(summary_df)
    filtered = render_run_filters(summary_df)
    render_seasonality(summary_df)
    render_run_comparison(summary_df, points_df, filtered)

    st.caption(
        "GPX exports are written to data/processed/gpx whenever you rebuild. "
        "Processed tables live in data/processed/run_summary.parquet and data/processed/points.parquet."
    )


if __name__ == "__main__":
    main()
