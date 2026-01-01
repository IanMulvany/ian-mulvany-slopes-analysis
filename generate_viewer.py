#!/usr/bin/env python3
"""Generate a standalone HTML viewer for ski segment data with clustering."""

import json
from pathlib import Path
from slopes_analysis.analysis import load_datasets, segment_downhill_runs
from slopes_analysis.annotations import (
    get_outliers_dict,
    DEFAULT_ANNOTATED_DIR,
)
import pandas as pd
import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two points."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def generate_html_viewer(
    output_path: Path = Path("ski_segments_viewer.html"),
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR,
):
    """Generate a self-contained HTML viewer with embedded segment data."""

    print("Loading datasets...")
    summary_df, points_df = load_datasets()

    print("Segmenting downhill runs...")
    segments_df, labeled_points_df = segment_downhill_runs(points_df)

    print(f"Found {len(segments_df)} downhill segments")

    # Load outliers from annotations
    print(f"Loading annotations from {annotated_dir}...")
    outliers_dict = get_outliers_dict(annotated_dir)
    print(f"Loaded {len(outliers_dict)} outlier markings")

    # Prepare segment data with GPS tracks
    segments_data = []
    for _, seg in segments_df.iterrows():
        seg_id = seg["segment_global_id"]
        seg_points = labeled_points_df[labeled_points_df["segment_global_id"] == seg_id]

        if len(seg_points) < 2:
            continue

        # Sample points to reduce file size (every 2nd point for tracks > 50 points)
        full_points = seg_points.copy()
        if len(seg_points) > 50:
            seg_points = seg_points.iloc[::2]

        track = [
            {"lat": row["latitude"], "lng": row["longitude"], "alt": row["altitude_m"]}
            for _, row in seg_points.iterrows()
        ]

        # Get start and end points for clustering
        start_pt = full_points.iloc[0]
        end_pt = full_points.iloc[-1]

        segments_data.append({
            "id": seg_id,
            "day_id": seg["day_run_id"],
            "location": seg["location_name"] or "Unknown",
            "year": int(seg["year"]) if pd.notna(seg["year"]) else 0,
            "date": seg["start_time"].strftime("%Y-%m-%d") if pd.notna(seg["start_time"]) else "",
            "time": seg["start_time"].strftime("%H:%M") if pd.notna(seg["start_time"]) else "",
            "distance_m": round(seg["distance_m"], 1),
            "vertical_m": round(seg["vertical_drop_m"], 1),
            "duration_s": round(seg["duration_s"], 1),
            "avg_speed": round(seg["avg_speed_kmh"], 1),
            "top_speed": round(seg["top_speed_kmh"], 1),
            "track": track,
            "start": {"lat": float(start_pt["latitude"]), "lng": float(start_pt["longitude"])},
            "end": {"lat": float(end_pt["latitude"]), "lng": float(end_pt["longitude"])},
        })

    # Compute summary stats
    total_segments = len(segments_data)
    total_distance = sum(s["distance_m"] for s in segments_data) / 1000
    total_vertical = sum(s["vertical_m"] for s in segments_data)
    avg_speed = sum(s["avg_speed"] for s in segments_data) / total_segments if total_segments else 0
    max_speed = max((s["top_speed"] for s in segments_data), default=0)
    locations = sorted(set(s["location"] for s in segments_data))
    years = sorted(set(s["year"] for s in segments_data if s["year"]))

    stats = {
        "total_segments": total_segments,
        "total_distance_km": round(total_distance, 1),
        "total_vertical_m": round(total_vertical, 0),
        "avg_speed_kmh": round(avg_speed, 1),
        "max_speed_kmh": round(max_speed, 1),
        "locations": locations,
        "years": years,
    }

    # Convert outliers to serializable format
    outliers_data = {
        seg_id: {
            "id": o.id,
            "location": o.location,
            "date": o.date,
            "distance_m": o.distance_m,
            "avg_speed": o.avg_speed,
            "top_speed": o.top_speed,
            "markedAt": o.marked_at,
        }
        for seg_id, o in outliers_dict.items()
    }

    html_content = generate_html(segments_data, stats, outliers_data)

    output_path.write_text(html_content)
    print(f"Viewer generated: {output_path.absolute()}")
    return output_path


def generate_html(segments: list, stats: dict, outliers: dict = None) -> str:
    """Generate the complete HTML document."""

    segments_json = json.dumps(segments)
    stats_json = json.dumps(stats)
    outliers_json = json.dumps(outliers or {})

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ski Segments Viewer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #7dd3fc;
            --success: #4ade80;
            --warning: #fbbf24;
            --danger: #f87171;
            --border: #475569;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}

        .app {{
            display: grid;
            grid-template-columns: 400px 1fr;
            grid-template-rows: auto 1fr;
            height: 100vh;
        }}

        header {{
            grid-column: 1 / -1;
            background: var(--bg-secondary);
            padding: 0.75rem 1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        header h1 {{
            font-size: 1.25rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        header h1::before {{ content: "\\26F7"; font-size: 1.5rem; }}

        .stats-bar {{ display: flex; gap: 2rem; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 1.1rem; font-weight: 700; color: var(--accent); }}
        .stat-label {{ font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }}

        .view-tabs {{
            display: flex;
            gap: 0.5rem;
        }}

        .view-tab {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 0.4rem 1rem;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.15s;
        }}

        .view-tab:hover {{ border-color: var(--accent); color: var(--accent); }}
        .view-tab.active {{ background: var(--accent); border-color: var(--accent); color: var(--bg-primary); }}

        .map-toggle {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-left: 1rem;
        }}

        .map-toggle-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}

        .toggle-btn {{
            display: flex;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
        }}

        .toggle-btn button {{
            background: none;
            border: none;
            color: var(--text-secondary);
            padding: 0.3rem 0.6rem;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.15s;
        }}

        .toggle-btn button:hover {{ color: var(--text-primary); }}
        .toggle-btn button.active {{ background: var(--accent); color: var(--bg-primary); }}

        .settings-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 0.4rem 0.75rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-left: 1rem;
        }}

        .settings-btn:hover {{ border-color: var(--accent); color: var(--text-primary); }}

        .settings-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.6);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }}

        .settings-overlay.visible {{ display: flex; }}

        .settings-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            min-width: 320px;
            max-width: 90vw;
        }}

        .settings-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
        }}

        .settings-header h2 {{
            font-size: 1.1rem;
            font-weight: 600;
        }}

        .settings-close {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5rem;
            cursor: pointer;
            line-height: 1;
        }}

        .settings-close:hover {{ color: var(--text-primary); }}

        .setting-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
        }}

        .setting-row:last-child {{ border-bottom: none; }}

        .setting-label {{
            font-size: 0.9rem;
        }}

        .setting-description {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.2rem;
        }}

        .toggle-switch {{
            position: relative;
            width: 44px;
            height: 24px;
            background: var(--bg-tertiary);
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.2s;
        }}

        .toggle-switch.active {{
            background: var(--accent);
        }}

        .toggle-switch::after {{
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: transform 0.2s;
        }}

        .toggle-switch.active::after {{
            transform: translateX(20px);
        }}

        .setting-status {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}

        .setting-status.hidden {{ color: var(--danger); }}
        .setting-status.shown {{ color: var(--success); }}

        .sidebar {{
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .filters {{
            padding: 0.75rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .filter-row {{ display: flex; gap: 0.5rem; }}
        .filter-group {{ flex: 1; }}
        .filter-group label {{
            display: block;
            font-size: 0.7rem;
            color: var(--text-secondary);
            margin-bottom: 0.2rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        select, input[type="text"], input[type="number"], input[type="range"] {{
            width: 100%;
            padding: 0.4rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.85rem;
        }}

        select:focus, input:focus {{ outline: none; border-color: var(--accent); }}

        .cluster-controls {{
            padding: 0.75rem;
            border-bottom: 1px solid var(--border);
            background: var(--bg-tertiary);
        }}

        .radius-control {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .radius-control label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            white-space: nowrap;
        }}

        .radius-control input[type="range"] {{
            flex: 1;
            -webkit-appearance: none;
            background: var(--border);
            height: 6px;
            border-radius: 3px;
            border: none;
            padding: 0;
        }}

        .radius-control input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
        }}

        .radius-value {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent);
            min-width: 50px;
            text-align: right;
        }}

        .cluster-stats {{
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .filter-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            padding: 0.4rem 0.75rem;
            border-bottom: 1px solid var(--border);
            min-height: 2rem;
        }}

        .chip {{
            background: var(--accent);
            color: var(--bg-primary);
            padding: 0.15rem 0.5rem;
            border-radius: 100px;
            font-size: 0.7rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            cursor: pointer;
        }}

        .chip:hover {{ opacity: 0.8; }}

        .sort-controls {{
            display: flex;
            gap: 0.4rem;
            padding: 0.4rem 0.75rem;
            border-bottom: 1px solid var(--border);
            align-items: center;
        }}

        .sort-btn {{
            background: none;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            cursor: pointer;
        }}

        .sort-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
        .sort-btn.active {{ background: var(--accent); border-color: var(--accent); color: var(--bg-primary); }}

        .segment-count {{
            padding: 0.4rem 0.75rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
        }}

        .segment-list {{
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem;
        }}

        .segment-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 0.6rem;
            margin-bottom: 0.4rem;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.15s;
        }}

        .segment-card:hover {{ background: #3d4f66; }}
        .segment-card.selected {{ border-color: var(--accent); background: rgba(56, 189, 248, 0.1); }}

        .segment-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.4rem;
        }}

        .segment-location {{ font-weight: 600; font-size: 0.85rem; }}
        .segment-date {{ font-size: 0.7rem; color: var(--text-secondary); }}

        .cluster-badge {{
            display: inline-block;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            font-size: 0.65rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }}

        .segment-metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.4rem;
        }}

        .metric {{ text-align: center; }}
        .metric-value {{ font-size: 0.85rem; font-weight: 600; }}
        .metric-value.speed {{ color: var(--warning); }}
        .metric-value.vertical {{ color: var(--success); }}
        .metric-value.distance {{ color: var(--accent); }}
        .metric-label {{ font-size: 0.6rem; color: var(--text-secondary); }}

        .main-content {{
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        #map {{ flex: 1; background: var(--bg-primary); }}

        .details-panel {{
            height: 180px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            padding: 0.75rem;
            display: none;
            flex-direction: column;
        }}

        .details-panel.visible {{ display: flex; }}

        .details-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }}

        .details-title {{ font-size: 0.9rem; font-weight: 600; }}

        .close-details {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 1.25rem;
            cursor: pointer;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }}

        .close-details:hover {{ background: var(--bg-tertiary); color: var(--text-primary); }}

        .elevation-chart {{ flex: 1; position: relative; }}
        .elevation-chart svg {{ width: 100%; height: 100%; }}
        .chart-path {{ fill: none; stroke: var(--accent); stroke-width: 2; }}
        .chart-area {{ fill: rgba(56, 189, 248, 0.2); }}
        .chart-axis {{ stroke: var(--border); stroke-width: 1; }}
        .chart-label {{ fill: var(--text-secondary); font-size: 10px; }}

        /* Cluster view specific */
        .cluster-list {{
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem;
        }}

        .cluster-group {{
            margin-bottom: 1rem;
        }}

        .cluster-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin-bottom: 0.4rem;
            cursor: pointer;
        }}

        .cluster-header:hover {{ background: #3d4f66; }}

        .cluster-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }}

        .cluster-info {{ flex: 1; }}
        .cluster-name {{ font-weight: 600; font-size: 0.85rem; }}
        .cluster-meta {{ font-size: 0.7rem; color: var(--text-secondary); }}

        .cluster-segments {{
            padding-left: 1.5rem;
        }}

        .cluster-segments .segment-card {{
            border-left: 3px solid;
        }}

        @media (max-width: 900px) {{
            .app {{ grid-template-columns: 1fr; grid-template-rows: auto auto 1fr; }}
            .sidebar {{ max-height: 45vh; border-right: none; border-bottom: 1px solid var(--border); }}
            .stats-bar {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="app">
        <header>
            <h1>Ski Segments</h1>
            <div class="view-tabs">
                <button class="view-tab active" data-view="segments">All Segments</button>
                <button class="view-tab" data-view="clusters">Clustered Runs</button>
            </div>
            <div class="map-toggle">
                <span class="map-toggle-label">Map:</span>
                <div class="toggle-btn">
                    <button id="map-dark" class="active" onclick="setMapTheme('dark')">Dark</button>
                    <button id="map-light" onclick="setMapTheme('light')">Light</button>
                    <button id="map-satellite" onclick="setMapTheme('satellite')">Satellite</button>
                </div>
            </div>
            <button class="settings-btn" onclick="openSettings()">
                <span>⚙</span> Settings
            </button>
            <div class="stats-bar">
                <div class="stat">
                    <div class="stat-value" id="stat-segments">-</div>
                    <div class="stat-label">Segments</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-distance">-</div>
                    <div class="stat-label">Total km</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-vertical">-</div>
                    <div class="stat-label">Vertical m</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-speed">-</div>
                    <div class="stat-label">Avg km/h</div>
                </div>
            </div>
        </header>

        <aside class="sidebar">
            <div class="filters">
                <div class="filter-row">
                    <div class="filter-group">
                        <label>Location</label>
                        <select id="filter-location"><option value="">All Locations</option></select>
                    </div>
                    <div class="filter-group">
                        <label>Year</label>
                        <select id="filter-year"><option value="">All Years</option></select>
                    </div>
                </div>
                <div class="filter-row">
                    <div class="filter-group">
                        <label>Min Speed (km/h)</label>
                        <input type="number" id="filter-min-speed" placeholder="0" min="0">
                    </div>
                    <div class="filter-group">
                        <label>Min Vertical (m)</label>
                        <input type="number" id="filter-min-vertical" placeholder="0" min="0">
                    </div>
                </div>
            </div>

            <div class="cluster-controls" id="cluster-controls" style="display: none;">
                <div class="radius-control">
                    <label>Cluster Radius:</label>
                    <input type="range" id="cluster-radius" min="20" max="500" value="100" step="10">
                    <span class="radius-value" id="radius-value">100m</span>
                </div>
                <div class="cluster-stats" id="cluster-stats"></div>
            </div>

            <div class="filter-chips" id="active-filters"></div>

            <div class="sort-controls" id="sort-controls">
                <span style="color: var(--text-secondary); font-size: 0.7rem;">Sort:</span>
                <button class="sort-btn active" data-sort="date">Date</button>
                <button class="sort-btn" data-sort="speed">Speed</button>
                <button class="sort-btn" data-sort="vertical">Vertical</button>
                <button class="sort-btn" data-sort="distance">Distance</button>
            </div>

            <div class="segment-count" id="segment-count">Loading...</div>

            <div class="segment-list" id="segment-list"></div>
            <div class="cluster-list" id="cluster-list" style="display: none;"></div>
        </aside>

        <main class="main-content">
            <div id="map"></div>
            <div class="details-panel" id="details-panel">
                <div class="details-header">
                    <div class="details-title" id="details-title">Segment Details</div>
                    <button class="close-details" id="close-details">&times;</button>
                </div>
                <div class="elevation-chart" id="elevation-chart"></div>
            </div>
        </main>
    </div>

    <div class="settings-overlay" id="settings-overlay" onclick="closeSettings(event)">
        <div class="settings-panel" onclick="event.stopPropagation()">
            <div class="settings-header">
                <h2>Settings</h2>
                <button class="settings-close" onclick="closeSettings()">&times;</button>
            </div>
            <div class="setting-row">
                <div>
                    <div class="setting-label">Outlier Segments</div>
                    <div class="setting-description">Hide segments marked as outliers in Data Cleaning app</div>
                    <div class="setting-status" id="outlier-status">Showing all segments</div>
                </div>
                <div class="toggle-switch" id="outlier-toggle" onclick="toggleHideOutliers()"></div>
            </div>
        </div>
    </div>

    <script>
        const allSegments = {segments_json};
        const stats = {stats_json};
        const embeddedOutliers = {outliers_json};

        // Cluster colors
        const COLORS = [
            '#38bdf8', '#4ade80', '#fbbf24', '#f87171', '#a78bfa',
            '#fb923c', '#2dd4bf', '#f472b6', '#60a5fa', '#84cc16'
        ];

        // State
        let filteredSegments = [...allSegments];
        let selectedSegment = null;
        let currentView = 'segments';
        let currentSort = 'date';
        let sortAsc = false;
        let clusterRadius = 100;
        let clusters = [];
        let map, trackLayer, tileLayer;
        let mapTheme = localStorage.getItem('mapTheme') || 'dark';
        // Merge embedded outliers with any localStorage outliers (embedded takes precedence as it's from annotated dir)
        let outliers = {{...JSON.parse(localStorage.getItem('ski_outliers') || '{{}}'), ...embeddedOutliers}};
        let hideOutliers = localStorage.getItem('hideOutliers') === 'true';

        const TILE_URLS = {{
            dark: 'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
            light: 'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
            satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}'
        }};

        document.addEventListener('DOMContentLoaded', init);

        function init() {{
            initMap();
            populateFilters();
            setupEventListeners();
            initOutlierToggle();
            applyFilters();
        }}

        function initOutlierToggle() {{
            updateOutlierUI();
        }}

        function updateOutlierUI() {{
            const count = Object.keys(outliers).length;
            const toggle = document.getElementById('outlier-toggle');
            const status = document.getElementById('outlier-status');
            toggle.classList.toggle('active', hideOutliers);
            if (hideOutliers) {{
                status.textContent = `${{count}} outlier${{count !== 1 ? 's' : ''}} hidden`;
                status.className = 'setting-status hidden';
            }} else {{
                status.textContent = `Showing all segments (${{count}} marked as outliers)`;
                status.className = 'setting-status shown';
            }}
        }}

        function toggleHideOutliers() {{
            hideOutliers = !hideOutliers;
            localStorage.setItem('hideOutliers', hideOutliers);
            updateOutlierUI();
            applyFilters();
        }}

        function openSettings() {{
            document.getElementById('settings-overlay').classList.add('visible');
        }}

        function closeSettings(e) {{
            if (!e || e.target === e.currentTarget) {{
                document.getElementById('settings-overlay').classList.remove('visible');
            }}
        }}

        function initMap() {{
            map = L.map('map', {{ zoomControl: true, attributionControl: false }}).setView([46.31, 7.39], 12);
            tileLayer = L.tileLayer(TILE_URLS[mapTheme], {{ maxZoom: 19 }}).addTo(map);
            trackLayer = L.layerGroup().addTo(map);

            // Set initial button state
            document.getElementById('map-dark').classList.toggle('active', mapTheme === 'dark');
            document.getElementById('map-light').classList.toggle('active', mapTheme === 'light');
            document.getElementById('map-satellite').classList.toggle('active', mapTheme === 'satellite');
        }}

        function setMapTheme(theme) {{
            mapTheme = theme;
            localStorage.setItem('mapTheme', theme);
            map.removeLayer(tileLayer);
            tileLayer = L.tileLayer(TILE_URLS[theme], {{ maxZoom: 19 }}).addTo(map);
            tileLayer.bringToBack();

            document.getElementById('map-dark').classList.toggle('active', theme === 'dark');
            document.getElementById('map-light').classList.toggle('active', theme === 'light');
            document.getElementById('map-satellite').classList.toggle('active', theme === 'satellite');
        }}

        function populateFilters() {{
            const locationSelect = document.getElementById('filter-location');
            stats.locations.forEach(loc => {{
                const opt = document.createElement('option');
                opt.value = loc;
                opt.textContent = loc;
                locationSelect.appendChild(opt);
            }});

            const yearSelect = document.getElementById('filter-year');
            stats.years.forEach(year => {{
                const opt = document.createElement('option');
                opt.value = year;
                opt.textContent = year;
                yearSelect.appendChild(opt);
            }});
        }}

        function setupEventListeners() {{
            document.getElementById('filter-location').addEventListener('change', applyFilters);
            document.getElementById('filter-year').addEventListener('change', applyFilters);
            document.getElementById('filter-min-speed').addEventListener('input', debounce(applyFilters, 300));
            document.getElementById('filter-min-vertical').addEventListener('input', debounce(applyFilters, 300));

            document.querySelectorAll('.sort-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    const sort = btn.dataset.sort;
                    if (currentSort === sort) sortAsc = !sortAsc;
                    else {{ currentSort = sort; sortAsc = false; }}
                    document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    if (currentView === 'segments') renderSegments();
                    else renderClusters();
                }});
            }});

            document.querySelectorAll('.view-tab').forEach(tab => {{
                tab.addEventListener('click', () => {{
                    currentView = tab.dataset.view;
                    document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    toggleView();
                }});
            }});

            document.getElementById('cluster-radius').addEventListener('input', (e) => {{
                clusterRadius = parseInt(e.target.value);
                document.getElementById('radius-value').textContent = clusterRadius + 'm';
                if (currentView === 'clusters') {{
                    computeClusters();
                    renderClusters();
                    updateMapClusters();
                }}
            }});

            document.getElementById('close-details').addEventListener('click', () => {{
                document.getElementById('details-panel').classList.remove('visible');
            }});
        }}

        function toggleView() {{
            const segList = document.getElementById('segment-list');
            const clustList = document.getElementById('cluster-list');
            const clusterControls = document.getElementById('cluster-controls');
            const sortControls = document.getElementById('sort-controls');

            if (currentView === 'clusters') {{
                segList.style.display = 'none';
                clustList.style.display = 'block';
                clusterControls.style.display = 'block';
                computeClusters();
                renderClusters();
                updateMapClusters();
            }} else {{
                segList.style.display = 'block';
                clustList.style.display = 'none';
                clusterControls.style.display = 'none';
                renderSegments();
                updateMapTracks();
            }}
        }}

        function debounce(fn, delay) {{
            let timeout;
            return (...args) => {{ clearTimeout(timeout); timeout = setTimeout(() => fn(...args), delay); }};
        }}

        function applyFilters() {{
            const location = document.getElementById('filter-location').value;
            const year = document.getElementById('filter-year').value;
            const minSpeed = parseFloat(document.getElementById('filter-min-speed').value) || 0;
            const minVertical = parseFloat(document.getElementById('filter-min-vertical').value) || 0;

            filteredSegments = allSegments.filter(seg => {{
                if (hideOutliers && outliers[seg.id]) return false;
                if (location && seg.location !== location) return false;
                if (year && seg.year !== parseInt(year)) return false;
                if (seg.avg_speed < minSpeed) return false;
                if (seg.vertical_m < minVertical) return false;
                return true;
            }});

            updateActiveFilters(location, year, minSpeed, minVertical);
            updateStats();

            if (currentView === 'clusters') {{
                computeClusters();
                renderClusters();
                updateMapClusters();
            }} else {{
                renderSegments();
                updateMapTracks();
            }}
            updateMapBounds();
        }}

        function updateActiveFilters(location, year, minSpeed, minVertical) {{
            const container = document.getElementById('active-filters');
            container.innerHTML = '';
            if (location) addChip(container, 'Location: ' + location, () => {{ document.getElementById('filter-location').value = ''; applyFilters(); }});
            if (year) addChip(container, 'Year: ' + year, () => {{ document.getElementById('filter-year').value = ''; applyFilters(); }});
            if (minSpeed > 0) addChip(container, 'Speed ≥ ' + minSpeed, () => {{ document.getElementById('filter-min-speed').value = ''; applyFilters(); }});
            if (minVertical > 0) addChip(container, 'Vertical ≥ ' + minVertical, () => {{ document.getElementById('filter-min-vertical').value = ''; applyFilters(); }});
        }}

        function addChip(container, text, onRemove) {{
            const chip = document.createElement('div');
            chip.className = 'chip';
            chip.innerHTML = text + ' <span>×</span>';
            chip.addEventListener('click', onRemove);
            container.appendChild(chip);
        }}

        function updateStats() {{
            const count = filteredSegments.length;
            const totalDist = filteredSegments.reduce((sum, s) => sum + s.distance_m, 0) / 1000;
            const totalVert = filteredSegments.reduce((sum, s) => sum + s.vertical_m, 0);
            const avgSpeed = count > 0 ? filteredSegments.reduce((sum, s) => sum + s.avg_speed, 0) / count : 0;

            document.getElementById('stat-segments').textContent = count;
            document.getElementById('stat-distance').textContent = totalDist.toFixed(1);
            document.getElementById('stat-vertical').textContent = Math.round(totalVert).toLocaleString();
            document.getElementById('stat-speed').textContent = avgSpeed.toFixed(1);
        }}

        // ========== CLUSTERING ==========

        function haversine(lat1, lon1, lat2, lon2) {{
            const R = 6371000;
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
            return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        }}

        function computeClusters() {{
            // Union-Find for clustering
            const parent = {{}};
            const rank = {{}};

            function find(x) {{
                if (parent[x] === undefined) parent[x] = x;
                if (parent[x] !== x) parent[x] = find(parent[x]);
                return parent[x];
            }}

            function union(x, y) {{
                const px = find(x), py = find(y);
                if (px === py) return;
                if ((rank[px] || 0) < (rank[py] || 0)) parent[px] = py;
                else if ((rank[px] || 0) > (rank[py] || 0)) parent[py] = px;
                else {{ parent[py] = px; rank[px] = (rank[px] || 0) + 1; }}
            }}

            // Compare all pairs - cluster if BOTH start AND end within radius
            for (let i = 0; i < filteredSegments.length; i++) {{
                for (let j = i + 1; j < filteredSegments.length; j++) {{
                    const a = filteredSegments[i], b = filteredSegments[j];
                    const startDist = haversine(a.start.lat, a.start.lng, b.start.lat, b.start.lng);
                    const endDist = haversine(a.end.lat, a.end.lng, b.end.lat, b.end.lng);
                    if (startDist <= clusterRadius && endDist <= clusterRadius) {{
                        union(a.id, b.id);
                    }}
                }}
            }}

            // Group by cluster
            const clusterMap = {{}};
            filteredSegments.forEach(seg => {{
                const root = find(seg.id);
                if (!clusterMap[root]) clusterMap[root] = [];
                clusterMap[root].push(seg);
            }});

            // Sort clusters by size desc, assign colors
            clusters = Object.values(clusterMap)
                .sort((a, b) => b.length - a.length)
                .map((segs, i) => ({{
                    id: i,
                    color: COLORS[i % COLORS.length],
                    segments: segs.sort((a, b) => a.date.localeCompare(b.date))
                }}));

            // Update cluster stats
            const multiClusters = clusters.filter(c => c.segments.length > 1);
            document.getElementById('cluster-stats').innerHTML = `
                <span>${{clusters.length}} clusters</span>
                <span>${{multiClusters.length}} with multiple runs</span>
                <span>${{filteredSegments.length - multiClusters.reduce((s, c) => s + c.segments.length, 0)}} unique</span>
            `;
        }}

        function renderClusters() {{
            document.getElementById('segment-count').textContent = `${{clusters.length}} cluster${{clusters.length !== 1 ? 's' : ''}} found`;

            const list = document.getElementById('cluster-list');
            list.innerHTML = clusters.map(cluster => `
                <div class="cluster-group" data-cluster-id="${{cluster.id}}">
                    <div class="cluster-header" onclick="toggleCluster(${{cluster.id}})">
                        <div class="cluster-color" style="background: ${{cluster.color}}"></div>
                        <div class="cluster-info">
                            <div class="cluster-name">${{cluster.segments[0].location}} Run #${{cluster.id + 1}}</div>
                            <div class="cluster-meta">
                                ${{cluster.segments.length}} run${{cluster.segments.length > 1 ? 's' : ''}} ·
                                Avg ${{(cluster.segments.reduce((s, seg) => s + seg.avg_speed, 0) / cluster.segments.length).toFixed(1)}} km/h ·
                                ${{Math.round(cluster.segments.reduce((s, seg) => s + seg.vertical_m, 0))}}m total drop
                            </div>
                        </div>
                    </div>
                    <div class="cluster-segments" id="cluster-segs-${{cluster.id}}" style="display: none;">
                        ${{cluster.segments.map(seg => `
                            <div class="segment-card ${{selectedSegment === seg.id ? 'selected' : ''}}"
                                 data-id="${{seg.id}}" style="border-left-color: ${{cluster.color}}"
                                 onclick="selectSegment('${{seg.id}}')">
                                <div class="segment-header">
                                    <div class="segment-date">${{seg.date}} ${{seg.time}}</div>
                                </div>
                                <div class="segment-metrics">
                                    <div class="metric">
                                        <div class="metric-value distance">${{(seg.distance_m / 1000).toFixed(2)}}</div>
                                        <div class="metric-label">km</div>
                                    </div>
                                    <div class="metric">
                                        <div class="metric-value vertical">${{Math.round(seg.vertical_m)}}</div>
                                        <div class="metric-label">m drop</div>
                                    </div>
                                    <div class="metric">
                                        <div class="metric-value speed">${{seg.avg_speed.toFixed(1)}}</div>
                                        <div class="metric-label">avg km/h</div>
                                    </div>
                                    <div class="metric">
                                        <div class="metric-value speed">${{seg.top_speed.toFixed(1)}}</div>
                                        <div class="metric-label">top km/h</div>
                                    </div>
                                </div>
                            </div>
                        `).join('')}}
                    </div>
                </div>
            `).join('');
        }}

        function toggleCluster(id) {{
            const el = document.getElementById('cluster-segs-' + id);
            el.style.display = el.style.display === 'none' ? 'block' : 'none';

            // Highlight cluster on map
            const cluster = clusters.find(c => c.id === id);
            if (cluster && cluster.segments.length > 0) {{
                const allCoords = cluster.segments.flatMap(s => s.track.map(p => [p.lat, p.lng]));
                if (allCoords.length > 0) map.fitBounds(L.latLngBounds(allCoords), {{ padding: [50, 50] }});
            }}
        }}

        function updateMapClusters() {{
            trackLayer.clearLayers();

            clusters.forEach(cluster => {{
                cluster.segments.forEach(seg => {{
                    if (seg.track.length < 2) return;
                    const coords = seg.track.map(p => [p.lat, p.lng]);
                    const isSelected = seg.id === selectedSegment;

                    const polyline = L.polyline(coords, {{
                        color: cluster.color,
                        weight: isSelected ? 4 : 2,
                        opacity: isSelected ? 1 : 0.7
                    }});

                    polyline.on('click', () => selectSegment(seg.id));
                    trackLayer.addLayer(polyline);

                    if (isSelected) {{
                        polyline.bringToFront();
                        addMarkers(coords);
                    }}
                }});
            }});
        }}

        // ========== SEGMENT VIEW ==========

        function renderSegments() {{
            const sorted = [...filteredSegments].sort((a, b) => {{
                let cmp = 0;
                switch (currentSort) {{
                    case 'date': cmp = a.date.localeCompare(b.date) || a.time.localeCompare(b.time); break;
                    case 'speed': cmp = a.avg_speed - b.avg_speed; break;
                    case 'vertical': cmp = a.vertical_m - b.vertical_m; break;
                    case 'distance': cmp = a.distance_m - b.distance_m; break;
                }}
                return sortAsc ? cmp : -cmp;
            }});

            document.getElementById('segment-count').textContent = `${{sorted.length}} segment${{sorted.length !== 1 ? 's' : ''}} found`;

            const list = document.getElementById('segment-list');
            list.innerHTML = sorted.map(seg => `
                <div class="segment-card ${{selectedSegment === seg.id ? 'selected' : ''}}" data-id="${{seg.id}}">
                    <div class="segment-header">
                        <div class="segment-location">${{seg.location}}</div>
                        <div class="segment-date">${{seg.date}} ${{seg.time}}</div>
                    </div>
                    <div class="segment-metrics">
                        <div class="metric">
                            <div class="metric-value distance">${{(seg.distance_m / 1000).toFixed(2)}}</div>
                            <div class="metric-label">km</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value vertical">${{Math.round(seg.vertical_m)}}</div>
                            <div class="metric-label">m drop</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value speed">${{seg.avg_speed.toFixed(1)}}</div>
                            <div class="metric-label">avg km/h</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value speed">${{seg.top_speed.toFixed(1)}}</div>
                            <div class="metric-label">top km/h</div>
                        </div>
                    </div>
                </div>
            `).join('');

            list.querySelectorAll('.segment-card').forEach(card => {{
                card.addEventListener('click', () => selectSegment(card.dataset.id));
            }});

            updateMapTracks();
        }}

        function selectSegment(id) {{
            selectedSegment = id;
            const seg = allSegments.find(s => s.id === id);

            document.querySelectorAll('.segment-card').forEach(card => {{
                card.classList.toggle('selected', card.dataset.id === id);
            }});

            if (seg && seg.track.length > 0) {{
                const bounds = L.latLngBounds(seg.track.map(p => [p.lat, p.lng]));
                map.fitBounds(bounds, {{ padding: [50, 50] }});
                if (currentView === 'clusters') updateMapClusters();
                else updateMapTracks();
                showDetailsPanel(seg);
            }}
        }}

        function updateMapTracks() {{
            trackLayer.clearLayers();

            filteredSegments.forEach(seg => {{
                if (seg.track.length < 2) return;
                const isSelected = seg.id === selectedSegment;
                const coords = seg.track.map(p => [p.lat, p.lng]);

                const polyline = L.polyline(coords, {{
                    color: isSelected ? '#38bdf8' : '#64748b',
                    weight: isSelected ? 4 : 2,
                    opacity: isSelected ? 1 : 0.6
                }});

                polyline.on('click', () => selectSegment(seg.id));
                trackLayer.addLayer(polyline);

                if (isSelected) {{
                    polyline.bringToFront();
                    addMarkers(coords);
                }}
            }});
        }}

        function addMarkers(coords) {{
            const startIcon = L.divIcon({{
                className: 'marker',
                html: '<div style="width:12px;height:12px;background:#4ade80;border-radius:50%;border:2px solid #fff;"></div>',
                iconSize: [12, 12]
            }});
            const endIcon = L.divIcon({{
                className: 'marker',
                html: '<div style="width:12px;height:12px;background:#f87171;border-radius:50%;border:2px solid #fff;"></div>',
                iconSize: [12, 12]
            }});
            L.marker(coords[0], {{ icon: startIcon }}).addTo(trackLayer);
            L.marker(coords[coords.length - 1], {{ icon: endIcon }}).addTo(trackLayer);
        }}

        function updateMapBounds() {{
            if (filteredSegments.length === 0) return;
            const allCoords = filteredSegments.flatMap(seg => seg.track.map(p => [p.lat, p.lng]));
            if (allCoords.length > 0) map.fitBounds(L.latLngBounds(allCoords), {{ padding: [30, 30] }});
        }}

        function showDetailsPanel(seg) {{
            document.getElementById('details-panel').classList.add('visible');
            document.getElementById('details-title').textContent = `${{seg.location}} · ${{seg.date}} ${{seg.time}}`;
            renderElevationChart(seg.track);
        }}

        function renderElevationChart(track) {{
            const container = document.getElementById('elevation-chart');
            const width = container.clientWidth;
            const height = container.clientHeight - 10;
            const margin = {{ top: 10, right: 10, bottom: 25, left: 45 }};

            if (track.length < 2) {{
                container.innerHTML = '<div style="color: var(--text-secondary); padding: 1rem;">Insufficient data</div>';
                return;
            }}

            let cumDist = 0;
            const points = track.map((p, i) => {{
                if (i > 0) cumDist += haversine(track[i-1].lat, track[i-1].lng, p.lat, p.lng);
                return {{ x: cumDist, y: p.alt }};
            }});

            const xMax = Math.max(...points.map(p => p.x));
            const yMin = Math.min(...points.map(p => p.y));
            const yMax = Math.max(...points.map(p => p.y));
            const yPad = (yMax - yMin) * 0.1 || 10;

            const xScale = (x) => margin.left + (x / xMax) * (width - margin.left - margin.right);
            const yScale = (y) => height - margin.bottom - ((y - yMin + yPad) / (yMax - yMin + 2 * yPad)) * (height - margin.top - margin.bottom);

            const linePath = points.map((p, i) => `${{i === 0 ? 'M' : 'L'}} ${{xScale(p.x)}} ${{yScale(p.y)}}`).join(' ');
            const areaPath = linePath + ` L ${{xScale(xMax)}} ${{height - margin.bottom}} L ${{margin.left}} ${{height - margin.bottom}} Z`;

            container.innerHTML = `
                <svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
                    <path class="chart-area" d="${{areaPath}}"/>
                    <path class="chart-path" d="${{linePath}}"/>
                    <line class="chart-axis" x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}"/>
                    <line class="chart-axis" x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}"/>
                    <text class="chart-label" x="${{margin.left - 5}}" y="${{yScale(yMax)}}" text-anchor="end" dominant-baseline="middle">${{Math.round(yMax)}}m</text>
                    <text class="chart-label" x="${{margin.left - 5}}" y="${{yScale(yMin)}}" text-anchor="end" dominant-baseline="middle">${{Math.round(yMin)}}m</text>
                    <text class="chart-label" x="${{margin.left}}" y="${{height - 5}}" text-anchor="start">0</text>
                    <text class="chart-label" x="${{width - margin.right}}" y="${{height - 5}}" text-anchor="end">${{(xMax / 1000).toFixed(2)}} km</text>
                </svg>
            `;
        }}
    </script>
</body>
</html>
'''


if __name__ == "__main__":
    generate_html_viewer()
