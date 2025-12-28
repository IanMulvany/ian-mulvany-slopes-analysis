#!/usr/bin/env python3
"""Generate standalone analysis apps for ski segment data."""

import json
from pathlib import Path
from slopes_analysis.analysis import load_datasets, segment_downhill_runs
import pandas as pd
import numpy as np


def generate_all_apps():
    """Generate all analysis apps."""
    print("Loading datasets...")
    summary_df, points_df = load_datasets()

    print("Segmenting downhill runs...")
    segments_df, labeled_points_df = segment_downhill_runs(points_df)

    print(f"Found {len(segments_df)} downhill segments")

    # Prepare segment data with detailed GPS tracks for rest analysis
    segments_data = []
    for _, seg in segments_df.iterrows():
        seg_id = seg["segment_global_id"]
        seg_points = labeled_points_df[labeled_points_df["segment_global_id"] == seg_id].copy()

        if len(seg_points) < 2:
            continue

        seg_points = seg_points.sort_values("time").reset_index(drop=True)

        # Calculate time deltas for rest detection
        seg_points["time_delta_s"] = seg_points["time"].diff().dt.total_seconds().fillna(0)

        # Full track with timing for rest detection
        track_full = []
        cumulative_time = 0
        for _, row in seg_points.iterrows():
            cumulative_time += row["time_delta_s"]
            track_full.append({
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
                "alt": float(row["altitude_m"]),
                "t": round(cumulative_time, 1),
                "spd": round(float(row.get("speed_smooth", row.get("segment_speed_kmh", 0))), 1)
            })

        # Sampled track for display
        display_points = seg_points.iloc[::2] if len(seg_points) > 50 else seg_points
        track_display = [
            {"lat": row["latitude"], "lng": row["longitude"], "alt": row["altitude_m"]}
            for _, row in display_points.iterrows()
        ]

        # Get start and end points
        start_pt = seg_points.iloc[0]
        end_pt = seg_points.iloc[-1]

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
            "track": track_display,
            "track_full": track_full,
            "start": {"lat": float(start_pt["latitude"]), "lng": float(start_pt["longitude"])},
            "end": {"lat": float(end_pt["latitude"]), "lng": float(end_pt["longitude"])},
        })

    # Compute stats
    locations = sorted(set(s["location"] for s in segments_data))
    years = sorted(set(s["year"] for s in segments_data if s["year"]))

    stats = {
        "total_segments": len(segments_data),
        "locations": locations,
        "years": years,
        "max_distance": max(s["distance_m"] for s in segments_data),
        "max_speed": max(s["top_speed"] for s in segments_data),
        "max_avg_speed": max(s["avg_speed"] for s in segments_data),
    }

    # Generate apps
    print("Generating Data Cleaning App...")
    generate_data_cleaning_app(segments_data, stats)

    print("Generating Cluster Analysis App...")
    generate_cluster_analysis_app(segments_data, stats)

    print("Done!")


def generate_data_cleaning_app(segments: list, stats: dict):
    """Generate the data cleaning/outlier detection app."""

    segments_json = json.dumps(segments)
    stats_json = json.dumps(stats)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ski Data Cleaning</title>
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
            --success: #4ade80;
            --warning: #fbbf24;
            --danger: #f87171;
            --border: #475569;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            min-height: 100vh;
        }}

        .app {{
            display: grid;
            grid-template-columns: 420px 1fr;
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
        }}

        .header-stats {{
            display: flex;
            gap: 1.5rem;
            font-size: 0.85rem;
        }}

        .header-stat {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .header-stat .value {{
            font-weight: 700;
            color: var(--accent);
        }}

        .outlier-count {{
            background: var(--danger);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 100px;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        .sidebar {{
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .controls {{
            padding: 1rem;
            border-bottom: 1px solid var(--border);
        }}

        .control-section {{
            margin-bottom: 1rem;
        }}

        .control-section:last-child {{
            margin-bottom: 0;
        }}

        .control-label {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .control-value {{
            color: var(--accent);
            font-weight: 600;
            font-size: 0.85rem;
        }}

        .slider-container {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }}

        .slider-min, .slider-max {{
            font-size: 0.7rem;
            color: var(--text-secondary);
            min-width: 45px;
        }}

        .slider-max {{ text-align: right; }}

        input[type="range"] {{
            flex: 1;
            -webkit-appearance: none;
            background: var(--bg-tertiary);
            height: 6px;
            border-radius: 3px;
            border: none;
        }}

        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
        }}

        .dual-slider {{
            position: relative;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            margin: 8px 0;
        }}

        .dual-slider .track {{
            position: absolute;
            height: 100%;
            background: var(--accent);
            border-radius: 3px;
        }}

        .dual-slider input {{
            position: absolute;
            width: 100%;
            height: 6px;
            background: transparent;
            pointer-events: none;
            -webkit-appearance: none;
        }}

        .dual-slider input::-webkit-slider-thumb {{
            pointer-events: auto;
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid var(--bg-secondary);
        }}

        .filter-row {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }}

        .filter-group {{
            flex: 1;
        }}

        .filter-group label {{
            display: block;
            font-size: 0.7rem;
            color: var(--text-secondary);
            margin-bottom: 0.2rem;
            text-transform: uppercase;
        }}

        select {{
            width: 100%;
            padding: 0.4rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.85rem;
        }}

        .view-options {{
            display: flex;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }}

        .view-btn {{
            flex: 1;
            padding: 0.4rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-secondary);
            font-size: 0.75rem;
            cursor: pointer;
        }}

        .view-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
        .view-btn.active {{ background: var(--accent); border-color: var(--accent); color: var(--bg-primary); }}

        .sort-row {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
            padding: 0.5rem 1rem;
            border-bottom: 1px solid var(--border);
        }}

        .sort-row span {{
            font-size: 0.7rem;
            color: var(--text-secondary);
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
            padding: 0.5rem 1rem;
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
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.15s;
            position: relative;
        }}

        .segment-card:hover {{ background: #3d4f66; }}
        .segment-card.selected {{ border-color: var(--accent); }}
        .segment-card.outlier {{ border-color: var(--danger); background: rgba(248, 113, 113, 0.1); }}
        .segment-card.outlier.selected {{ border-color: var(--danger); box-shadow: 0 0 0 2px var(--accent); }}

        .outlier-badge {{
            position: absolute;
            top: -6px;
            right: -6px;
            background: var(--danger);
            color: white;
            font-size: 0.65rem;
            font-weight: 700;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
        }}

        .segment-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }}

        .segment-location {{ font-weight: 600; font-size: 0.85rem; }}
        .segment-date {{ font-size: 0.7rem; color: var(--text-secondary); }}

        .segment-metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }}

        .metric {{
            background: var(--bg-secondary);
            padding: 0.5rem;
            border-radius: 6px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 1rem;
            font-weight: 700;
        }}

        .metric-value.distance {{ color: var(--accent); }}
        .metric-value.avg-speed {{ color: var(--success); }}
        .metric-value.top-speed {{ color: var(--warning); }}
        .metric-value.outlier {{ color: var(--danger); }}

        .metric-label {{
            font-size: 0.65rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}

        .metric-bar {{
            height: 4px;
            background: var(--bg-primary);
            border-radius: 2px;
            margin-top: 0.25rem;
            overflow: hidden;
        }}

        .metric-bar-fill {{
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s;
        }}

        .segment-actions {{
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }}

        .action-btn {{
            flex: 1;
            padding: 0.4rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            border: none;
            transition: all 0.15s;
        }}

        .action-btn.mark-outlier {{
            background: var(--danger);
            color: white;
        }}

        .action-btn.mark-outlier:hover {{ opacity: 0.9; }}

        .action-btn.unmark-outlier {{
            background: var(--success);
            color: white;
        }}

        .main-content {{
            display: flex;
            flex-direction: column;
        }}

        #map {{ flex: 1; background: var(--bg-primary); }}

        .details-panel {{
            height: 200px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            padding: 1rem;
            display: none;
        }}

        .details-panel.visible {{ display: block; }}

        .details-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.75rem;
        }}

        .details-title {{ font-weight: 600; }}

        .close-btn {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 1.25rem;
            cursor: pointer;
        }}

        .elevation-chart {{ height: 120px; }}
        .elevation-chart svg {{ width: 100%; height: 100%; }}
        .chart-path {{ fill: none; stroke: var(--accent); stroke-width: 2; }}
        .chart-area {{ fill: rgba(56, 189, 248, 0.2); }}
        .chart-label {{ fill: var(--text-secondary); font-size: 10px; }}

        .export-btn {{
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            font-size: 0.85rem;
        }}

        .export-btn:hover {{ opacity: 0.9; }}

        .map-toggle {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
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
        }}

        .toggle-btn button:hover {{ color: var(--text-primary); }}
        .toggle-btn button.active {{ background: var(--accent); color: var(--bg-primary); }}
    </style>
</head>
<body>
    <div class="app">
        <header>
            <h1>🔍 Data Cleaning & Outlier Detection</h1>
            <div class="header-stats">
                <div class="header-stat">
                    <span>Showing:</span>
                    <span class="value" id="showing-count">-</span>
                </div>
                <div class="header-stat">
                    <span>Outliers:</span>
                    <span class="outlier-count" id="outlier-count">0</span>
                </div>
                <div class="map-toggle">
                    <span class="map-toggle-label">Map:</span>
                    <div class="toggle-btn">
                        <button id="map-dark" class="active" onclick="setMapTheme('dark')">Dark</button>
                        <button id="map-light" onclick="setMapTheme('light')">Light</button>
                    </div>
                </div>
                <button class="export-btn" onclick="exportOutliers()">Export Outliers JSON</button>
            </div>
        </header>

        <aside class="sidebar">
            <div class="controls">
                <div class="filter-row">
                    <div class="filter-group">
                        <label>Location</label>
                        <select id="filter-location"><option value="">All</option></select>
                    </div>
                    <div class="filter-group">
                        <label>Year</label>
                        <select id="filter-year"><option value="">All</option></select>
                    </div>
                </div>

                <div class="control-section">
                    <div class="control-label">
                        <span>Distance Range</span>
                        <span class="control-value" id="distance-range-value">0 - 5000m</span>
                    </div>
                    <div class="slider-container">
                        <span class="slider-min">0m</span>
                        <div class="dual-slider" id="distance-slider">
                            <div class="track" id="distance-track"></div>
                            <input type="range" id="distance-min" min="0" max="5000" value="0" step="50">
                            <input type="range" id="distance-max" min="0" max="5000" value="5000" step="50">
                        </div>
                        <span class="slider-max" id="distance-max-label">5000m</span>
                    </div>
                </div>

                <div class="control-section">
                    <div class="control-label">
                        <span>Avg Speed Range</span>
                        <span class="control-value" id="avg-speed-range-value">0 - 60 km/h</span>
                    </div>
                    <div class="slider-container">
                        <span class="slider-min">0</span>
                        <div class="dual-slider" id="avg-speed-slider">
                            <div class="track" id="avg-speed-track"></div>
                            <input type="range" id="avg-speed-min" min="0" max="60" value="0" step="1">
                            <input type="range" id="avg-speed-max" min="0" max="60" value="60" step="1">
                        </div>
                        <span class="slider-max">60</span>
                    </div>
                </div>

                <div class="control-section">
                    <div class="control-label">
                        <span>Top Speed Range</span>
                        <span class="control-value" id="top-speed-range-value">0 - 120 km/h</span>
                    </div>
                    <div class="slider-container">
                        <span class="slider-min">0</span>
                        <div class="dual-slider" id="top-speed-slider">
                            <div class="track" id="top-speed-track"></div>
                            <input type="range" id="top-speed-min" min="0" max="120" value="0" step="1">
                            <input type="range" id="top-speed-max" min="0" max="120" value="120" step="1">
                        </div>
                        <span class="slider-max">120</span>
                    </div>
                </div>

                <div class="view-options">
                    <button class="view-btn active" data-view="all">All Segments</button>
                    <button class="view-btn" data-view="outliers">Outliers Only</button>
                    <button class="view-btn" data-view="clean">Clean Only</button>
                </div>
            </div>

            <div class="sort-row">
                <span>Sort:</span>
                <button class="sort-btn active" data-sort="date">Date</button>
                <button class="sort-btn" data-sort="distance">Distance</button>
                <button class="sort-btn" data-sort="avg_speed">Avg Speed</button>
                <button class="sort-btn" data-sort="top_speed">Top Speed</button>
            </div>

            <div class="segment-count" id="segment-count">Loading...</div>

            <div class="segment-list" id="segment-list"></div>
        </aside>

        <main class="main-content">
            <div id="map"></div>
            <div class="details-panel" id="details-panel">
                <div class="details-header">
                    <div class="details-title" id="details-title">Segment Details</div>
                    <button class="close-btn" onclick="closeDetails()">×</button>
                </div>
                <div class="elevation-chart" id="elevation-chart"></div>
            </div>
        </main>
    </div>

    <script>
        const allSegments = {segments_json};
        const stats = {stats_json};

        // State
        let outliers = JSON.parse(localStorage.getItem('ski_outliers') || '{{}}');
        let filteredSegments = [];
        let selectedSegment = null;
        let currentSort = 'date';
        let sortAsc = false;
        let currentView = 'all';
        let map, trackLayer, tileLayer;
        let mapTheme = localStorage.getItem('mapTheme') || 'dark';

        const TILE_URLS = {{
            dark: 'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
            light: 'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png'
        }};

        // Filter state
        let filters = {{
            location: '',
            year: '',
            distanceMin: 0,
            distanceMax: stats.max_distance,
            avgSpeedMin: 0,
            avgSpeedMax: Math.ceil(stats.max_avg_speed),
            topSpeedMin: 0,
            topSpeedMax: Math.ceil(stats.max_speed)
        }};

        document.addEventListener('DOMContentLoaded', init);

        function init() {{
            initMap();
            initSliders();
            populateFilters();
            setupEventListeners();
            applyFilters();
        }}

        function initMap() {{
            map = L.map('map', {{ zoomControl: true, attributionControl: false }}).setView([46.31, 7.39], 12);
            tileLayer = L.tileLayer(TILE_URLS[mapTheme], {{ maxZoom: 19 }}).addTo(map);
            trackLayer = L.layerGroup().addTo(map);

            document.getElementById('map-dark').classList.toggle('active', mapTheme === 'dark');
            document.getElementById('map-light').classList.toggle('active', mapTheme === 'light');
        }}

        function setMapTheme(theme) {{
            mapTheme = theme;
            localStorage.setItem('mapTheme', theme);
            map.removeLayer(tileLayer);
            tileLayer = L.tileLayer(TILE_URLS[theme], {{ maxZoom: 19 }}).addTo(map);
            tileLayer.bringToBack();

            document.getElementById('map-dark').classList.toggle('active', theme === 'dark');
            document.getElementById('map-light').classList.toggle('active', theme === 'light');
        }}

        function initSliders() {{
            // Distance slider
            const distMax = Math.ceil(stats.max_distance / 100) * 100;
            document.getElementById('distance-min').max = distMax;
            document.getElementById('distance-max').max = distMax;
            document.getElementById('distance-max').value = distMax;
            document.getElementById('distance-max-label').textContent = distMax + 'm';
            filters.distanceMax = distMax;

            // Avg speed slider
            const avgMax = Math.ceil(stats.max_avg_speed / 5) * 5;
            document.getElementById('avg-speed-min').max = avgMax;
            document.getElementById('avg-speed-max').max = avgMax;
            document.getElementById('avg-speed-max').value = avgMax;
            filters.avgSpeedMax = avgMax;

            // Top speed slider
            const topMax = Math.ceil(stats.max_speed / 10) * 10;
            document.getElementById('top-speed-min').max = topMax;
            document.getElementById('top-speed-max').max = topMax;
            document.getElementById('top-speed-max').value = topMax;
            filters.topSpeedMax = topMax;

            updateSliderTracks();
        }}

        function updateSliderTracks() {{
            updateTrack('distance', filters.distanceMin, filters.distanceMax);
            updateTrack('avg-speed', filters.avgSpeedMin, filters.avgSpeedMax);
            updateTrack('top-speed', filters.topSpeedMin, filters.topSpeedMax);

            document.getElementById('distance-range-value').textContent =
                `${{filters.distanceMin}} - ${{filters.distanceMax}}m`;
            document.getElementById('avg-speed-range-value').textContent =
                `${{filters.avgSpeedMin}} - ${{filters.avgSpeedMax}} km/h`;
            document.getElementById('top-speed-range-value').textContent =
                `${{filters.topSpeedMin}} - ${{filters.topSpeedMax}} km/h`;
        }}

        function updateTrack(id, min, max) {{
            const slider = document.getElementById(id + '-slider');
            const track = document.getElementById(id + '-track');
            const minInput = document.getElementById(id + '-min');
            const maxVal = parseFloat(minInput.max);
            const left = (min / maxVal) * 100;
            const right = (max / maxVal) * 100;
            track.style.left = left + '%';
            track.style.width = (right - left) + '%';
        }}

        function populateFilters() {{
            const locSelect = document.getElementById('filter-location');
            stats.locations.forEach(loc => {{
                const opt = document.createElement('option');
                opt.value = loc;
                opt.textContent = loc;
                locSelect.appendChild(opt);
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
            document.getElementById('filter-location').addEventListener('change', e => {{
                filters.location = e.target.value;
                applyFilters();
            }});

            document.getElementById('filter-year').addEventListener('change', e => {{
                filters.year = e.target.value;
                applyFilters();
            }});

            // Dual range sliders
            setupDualSlider('distance', 'distanceMin', 'distanceMax');
            setupDualSlider('avg-speed', 'avgSpeedMin', 'avgSpeedMax');
            setupDualSlider('top-speed', 'topSpeedMin', 'topSpeedMax');

            // Sort buttons
            document.querySelectorAll('.sort-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    const sort = btn.dataset.sort;
                    if (currentSort === sort) sortAsc = !sortAsc;
                    else {{ currentSort = sort; sortAsc = false; }}
                    document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    renderSegments();
                }});
            }});

            // View buttons
            document.querySelectorAll('.view-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    currentView = btn.dataset.view;
                    document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    applyFilters();
                }});
            }});
        }}

        function setupDualSlider(id, minKey, maxKey) {{
            const minInput = document.getElementById(id + '-min');
            const maxInput = document.getElementById(id + '-max');

            minInput.addEventListener('input', () => {{
                let minVal = parseFloat(minInput.value);
                let maxVal = parseFloat(maxInput.value);
                if (minVal > maxVal) minInput.value = maxVal;
                filters[minKey] = parseFloat(minInput.value);
                updateSliderTracks();
                applyFilters();
            }});

            maxInput.addEventListener('input', () => {{
                let minVal = parseFloat(minInput.value);
                let maxVal = parseFloat(maxInput.value);
                if (maxVal < minVal) maxInput.value = minVal;
                filters[maxKey] = parseFloat(maxInput.value);
                updateSliderTracks();
                applyFilters();
            }});
        }}

        function applyFilters() {{
            filteredSegments = allSegments.filter(seg => {{
                if (filters.location && seg.location !== filters.location) return false;
                if (filters.year && seg.year !== parseInt(filters.year)) return false;
                if (seg.distance_m < filters.distanceMin || seg.distance_m > filters.distanceMax) return false;
                if (seg.avg_speed < filters.avgSpeedMin || seg.avg_speed > filters.avgSpeedMax) return false;
                if (seg.top_speed < filters.topSpeedMin || seg.top_speed > filters.topSpeedMax) return false;

                if (currentView === 'outliers' && !outliers[seg.id]) return false;
                if (currentView === 'clean' && outliers[seg.id]) return false;

                return true;
            }});

            updateStats();
            renderSegments();
            updateMap();
        }}

        function updateStats() {{
            document.getElementById('showing-count').textContent = filteredSegments.length;
            document.getElementById('outlier-count').textContent = Object.keys(outliers).length;
        }}

        function renderSegments() {{
            const sorted = [...filteredSegments].sort((a, b) => {{
                let cmp = 0;
                switch (currentSort) {{
                    case 'date': cmp = a.date.localeCompare(b.date) || a.time.localeCompare(b.time); break;
                    case 'distance': cmp = a.distance_m - b.distance_m; break;
                    case 'avg_speed': cmp = a.avg_speed - b.avg_speed; break;
                    case 'top_speed': cmp = a.top_speed - b.top_speed; break;
                }}
                return sortAsc ? cmp : -cmp;
            }});

            document.getElementById('segment-count').textContent = `${{sorted.length}} segments`;

            const maxDist = stats.max_distance;
            const maxAvg = stats.max_avg_speed;
            const maxTop = stats.max_speed;

            const list = document.getElementById('segment-list');
            list.innerHTML = sorted.map(seg => {{
                const isOutlier = outliers[seg.id];
                const isSelected = selectedSegment === seg.id;
                return `
                <div class="segment-card ${{isOutlier ? 'outlier' : ''}} ${{isSelected ? 'selected' : ''}}"
                     data-id="${{seg.id}}" onclick="selectSegment('${{seg.id}}')">
                    ${{isOutlier ? '<div class="outlier-badge">OUTLIER</div>' : ''}}
                    <div class="segment-header">
                        <div class="segment-location">${{seg.location}}</div>
                        <div class="segment-date">${{seg.date}} ${{seg.time}}</div>
                    </div>
                    <div class="segment-metrics">
                        <div class="metric">
                            <div class="metric-value distance">${{(seg.distance_m / 1000).toFixed(2)}} km</div>
                            <div class="metric-label">Distance</div>
                            <div class="metric-bar">
                                <div class="metric-bar-fill" style="width: ${{(seg.distance_m / maxDist) * 100}}%; background: var(--accent);"></div>
                            </div>
                        </div>
                        <div class="metric">
                            <div class="metric-value avg-speed">${{seg.avg_speed.toFixed(1)}}</div>
                            <div class="metric-label">Avg km/h</div>
                            <div class="metric-bar">
                                <div class="metric-bar-fill" style="width: ${{(seg.avg_speed / maxAvg) * 100}}%; background: var(--success);"></div>
                            </div>
                        </div>
                        <div class="metric">
                            <div class="metric-value top-speed">${{seg.top_speed.toFixed(1)}}</div>
                            <div class="metric-label">Top km/h</div>
                            <div class="metric-bar">
                                <div class="metric-bar-fill" style="width: ${{(seg.top_speed / maxTop) * 100}}%; background: var(--warning);"></div>
                            </div>
                        </div>
                    </div>
                    <div class="segment-actions">
                        ${{isOutlier
                            ? `<button class="action-btn unmark-outlier" onclick="event.stopPropagation(); toggleOutlier('${{seg.id}}')">✓ Remove Outlier Flag</button>`
                            : `<button class="action-btn mark-outlier" onclick="event.stopPropagation(); toggleOutlier('${{seg.id}}')">⚠ Mark as Outlier</button>`
                        }}
                    </div>
                </div>
            `;
            }}).join('');
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
                updateMap();
                showDetails(seg);
            }}
        }}

        function toggleOutlier(id) {{
            if (outliers[id]) {{
                delete outliers[id];
            }} else {{
                outliers[id] = {{ markedAt: new Date().toISOString() }};
            }}
            localStorage.setItem('ski_outliers', JSON.stringify(outliers));
            updateStats();
            renderSegments();
            updateMap();
        }}

        function updateMap() {{
            trackLayer.clearLayers();

            filteredSegments.forEach(seg => {{
                if (seg.track.length < 2) return;
                const isSelected = seg.id === selectedSegment;
                const isOutlier = outliers[seg.id];
                const coords = seg.track.map(p => [p.lat, p.lng]);

                const polyline = L.polyline(coords, {{
                    color: isOutlier ? '#f87171' : (isSelected ? '#38bdf8' : '#64748b'),
                    weight: isSelected ? 4 : 2,
                    opacity: isSelected ? 1 : 0.6
                }});

                polyline.on('click', () => selectSegment(seg.id));
                trackLayer.addLayer(polyline);

                if (isSelected) polyline.bringToFront();
            }});

            if (filteredSegments.length > 0 && !selectedSegment) {{
                const allCoords = filteredSegments.flatMap(s => s.track.map(p => [p.lat, p.lng]));
                if (allCoords.length > 0) map.fitBounds(L.latLngBounds(allCoords), {{ padding: [30, 30] }});
            }}
        }}

        function showDetails(seg) {{
            document.getElementById('details-panel').classList.add('visible');
            document.getElementById('details-title').textContent = `${{seg.location}} · ${{seg.date}} ${{seg.time}}`;
            renderElevationChart(seg.track);
        }}

        function closeDetails() {{
            document.getElementById('details-panel').classList.remove('visible');
        }}

        function renderElevationChart(track) {{
            const container = document.getElementById('elevation-chart');
            const width = container.clientWidth;
            const height = container.clientHeight;
            const margin = {{ top: 10, right: 10, bottom: 20, left: 40 }};

            if (track.length < 2) return;

            let cumDist = 0;
            const points = track.map((p, i) => {{
                if (i > 0) cumDist += haversine(track[i-1].lat, track[i-1].lng, p.lat, p.lng);
                return {{ x: cumDist, y: p.alt }};
            }});

            const xMax = Math.max(...points.map(p => p.x));
            const yMin = Math.min(...points.map(p => p.y));
            const yMax = Math.max(...points.map(p => p.y));
            const yPad = (yMax - yMin) * 0.1 || 10;

            const xScale = x => margin.left + (x / xMax) * (width - margin.left - margin.right);
            const yScale = y => height - margin.bottom - ((y - yMin + yPad) / (yMax - yMin + 2 * yPad)) * (height - margin.top - margin.bottom);

            const linePath = points.map((p, i) => `${{i === 0 ? 'M' : 'L'}} ${{xScale(p.x)}} ${{yScale(p.y)}}`).join(' ');
            const areaPath = linePath + ` L ${{xScale(xMax)}} ${{height - margin.bottom}} L ${{margin.left}} ${{height - margin.bottom}} Z`;

            container.innerHTML = `
                <svg viewBox="0 0 ${{width}} ${{height}}">
                    <path class="chart-area" d="${{areaPath}}"/>
                    <path class="chart-path" d="${{linePath}}"/>
                    <text class="chart-label" x="${{margin.left - 5}}" y="${{yScale(yMax)}}" text-anchor="end">${{Math.round(yMax)}}m</text>
                    <text class="chart-label" x="${{margin.left - 5}}" y="${{yScale(yMin)}}" text-anchor="end">${{Math.round(yMin)}}m</text>
                    <text class="chart-label" x="${{width - margin.right}}" y="${{height - 5}}" text-anchor="end">${{(xMax / 1000).toFixed(2)}} km</text>
                </svg>
            `;
        }}

        function haversine(lat1, lon1, lat2, lon2) {{
            const R = 6371000;
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
            return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        }}

        function exportOutliers() {{
            const outlierData = Object.keys(outliers).map(id => {{
                const seg = allSegments.find(s => s.id === id);
                return {{
                    id: id,
                    location: seg?.location,
                    date: seg?.date,
                    distance_m: seg?.distance_m,
                    avg_speed: seg?.avg_speed,
                    top_speed: seg?.top_speed,
                    markedAt: outliers[id].markedAt
                }};
            }});

            const blob = new Blob([JSON.stringify(outlierData, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ski_outliers.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>
'''

    Path("data_cleaning_app.html").write_text(html)
    print(f"  Created: data_cleaning_app.html")


def generate_cluster_analysis_app(segments: list, stats: dict):
    """Generate the cluster analysis app with performance trends."""

    segments_json = json.dumps(segments)
    stats_json = json.dumps(stats)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ski Cluster Analysis</title>
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
            --success: #4ade80;
            --warning: #fbbf24;
            --danger: #f87171;
            --purple: #a78bfa;
            --border: #475569;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            min-height: 100vh;
        }}

        .app {{
            display: grid;
            grid-template-columns: 380px 1fr;
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
        }}

        .header-controls {{
            display: flex;
            gap: 1.5rem;
            align-items: center;
        }}

        .radius-control {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .radius-control label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .radius-control input {{
            width: 150px;
            -webkit-appearance: none;
            background: var(--bg-tertiary);
            height: 6px;
            border-radius: 3px;
        }}

        .radius-control input::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
        }}

        .radius-value {{
            font-weight: 700;
            color: var(--accent);
            min-width: 50px;
        }}

        .rest-control {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .rest-control label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .rest-control select {{
            padding: 0.3rem 0.5rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.85rem;
        }}

        .sidebar {{
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            min-height: 0;
        }}

        .cluster-stats {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 1rem;
            font-size: 0.85rem;
        }}

        .cluster-stats .stat-value {{
            font-weight: 700;
            color: var(--accent);
        }}

        .cluster-list {{
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem;
            min-height: 0;
        }}

        .cluster-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            overflow: hidden;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.15s;
        }}

        .cluster-card:hover {{ background: #3d4f66; }}
        .cluster-card.selected {{ border-color: var(--accent); }}

        .cluster-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
        }}

        .cluster-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            flex-shrink: 0;
        }}

        .cluster-info {{ flex: 1; }}
        .cluster-name {{ font-weight: 600; font-size: 0.9rem; }}
        .cluster-meta {{ font-size: 0.75rem; color: var(--text-secondary); }}

        .cluster-mini-stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.25rem;
            padding: 0 0.75rem 0.75rem;
        }}

        .mini-stat {{
            text-align: center;
            background: var(--bg-secondary);
            padding: 0.35rem;
            border-radius: 4px;
        }}

        .mini-stat-value {{
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .mini-stat-label {{
            font-size: 0.6rem;
            color: var(--text-secondary);
        }}

        .main-content {{
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .top-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            height: 50%;
            border-bottom: 1px solid var(--border);
        }}

        #map {{
            background: var(--bg-primary);
            border-right: 1px solid var(--border);
        }}

        .segment-timeline {{
            background: var(--bg-secondary);
            padding: 1rem;
            overflow-y: auto;
        }}

        .timeline-header {{
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .timeline-segment {{
            display: flex;
            gap: 0.75rem;
            padding: 0.5rem;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin-bottom: 0.5rem;
            cursor: pointer;
            border: 2px solid transparent;
        }}

        .timeline-segment:hover {{ background: #3d4f66; }}
        .timeline-segment.selected {{ border-color: var(--accent); }}

        .timeline-date {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            min-width: 80px;
        }}

        .timeline-metrics {{
            flex: 1;
            display: flex;
            gap: 1rem;
            font-size: 0.8rem;
        }}

        .timeline-metric {{
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}

        .timeline-metric .value {{ font-weight: 600; }}
        .timeline-metric.distance .value {{ color: var(--accent); }}
        .timeline-metric.avg-speed .value {{ color: var(--success); }}
        .timeline-metric.top-speed .value {{ color: var(--warning); }}
        .timeline-metric.rests .value {{ color: var(--purple); }}

        .charts-section {{
            height: 50%;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background: var(--border);
        }}

        .chart-panel {{
            background: var(--bg-secondary);
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
        }}

        .chart-title {{
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
        }}

        .chart-trend {{
            font-size: 0.75rem;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
        }}

        .chart-trend.up {{ background: rgba(74, 222, 128, 0.2); color: var(--success); }}
        .chart-trend.down {{ background: rgba(248, 113, 113, 0.2); color: var(--danger); }}
        .chart-trend.stable {{ background: rgba(148, 163, 184, 0.2); color: var(--text-secondary); }}

        .chart-container {{
            flex: 1;
            position: relative;
        }}

        .chart-container svg {{
            width: 100%;
            height: 100%;
        }}

        .chart-line {{ fill: none; stroke-width: 2; }}
        .chart-dot {{ cursor: pointer; }}
        .chart-grid {{ stroke: var(--border); stroke-width: 1; }}
        .chart-label {{ fill: var(--text-secondary); font-size: 9px; }}
        .chart-value {{ fill: var(--text-primary); font-size: 10px; font-weight: 600; }}

        .no-cluster-selected {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-secondary);
            font-size: 1rem;
        }}

        .empty-state {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        }}

        .map-toggle {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
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

        .settings-header h2 {{ font-size: 1.1rem; font-weight: 600; }}

        .settings-close {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5rem;
            cursor: pointer;
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

        .setting-label {{ font-size: 0.9rem; }}

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

        .toggle-switch.active {{ background: var(--accent); }}

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

        .toggle-switch.active::after {{ transform: translateX(20px); }}

        .setting-status {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}

        .setting-status.hidden {{ color: var(--danger); }}
        .setting-status.shown {{ color: var(--success); }}
    </style>
</head>
<body>
    <div class="app">
        <header>
            <h1>📊 Cluster Performance Analysis</h1>
            <div class="header-controls">
                <div class="radius-control">
                    <label>Cluster Radius:</label>
                    <input type="range" id="cluster-radius" min="20" max="500" value="100" step="10">
                    <span class="radius-value" id="radius-value">100m</span>
                </div>
                <div class="rest-control">
                    <label>Pause Threshold:</label>
                    <select id="rest-threshold">
                        <option value="5">5s</option>
                        <option value="10" selected>10s</option>
                        <option value="15">15s</option>
                        <option value="20">20s</option>
                        <option value="30">30s</option>
                        <option value="60">60s</option>
                    </select>
                </div>
                <div class="map-toggle">
                    <span class="map-toggle-label">Map:</span>
                    <div class="toggle-btn">
                        <button id="map-dark" class="active" onclick="setMapTheme('dark')">Dark</button>
                        <button id="map-light" onclick="setMapTheme('light')">Light</button>
                    </div>
                </div>
                <button class="settings-btn" onclick="openSettings()">
                    <span>⚙</span> Settings
                </button>
            </div>
        </header>

        <aside class="sidebar">
            <div class="cluster-stats">
                <span><span class="stat-value" id="cluster-count">-</span> clusters</span>
                <span><span class="stat-value" id="multi-run-count">-</span> with multiple runs</span>
            </div>
            <div class="cluster-list" id="cluster-list"></div>
        </aside>

        <main class="main-content">
            <div class="top-section">
                <div id="map"></div>
                <div class="segment-timeline" id="segment-timeline">
                    <div class="no-cluster-selected">Select a cluster to view segments</div>
                </div>
            </div>
            <div class="charts-section" id="charts-section">
                <div class="chart-panel">
                    <div class="chart-title">
                        <span>Distance</span>
                        <span class="chart-trend stable" id="distance-trend">-</span>
                    </div>
                    <div class="chart-container" id="distance-chart">
                        <div class="no-cluster-selected">-</div>
                    </div>
                </div>
                <div class="chart-panel">
                    <div class="chart-title">
                        <span>Avg Speed</span>
                        <span class="chart-trend stable" id="avg-speed-trend">-</span>
                    </div>
                    <div class="chart-container" id="avg-speed-chart">
                        <div class="no-cluster-selected">-</div>
                    </div>
                </div>
                <div class="chart-panel">
                    <div class="chart-title">
                        <span>Top Speed</span>
                        <span class="chart-trend stable" id="top-speed-trend">-</span>
                    </div>
                    <div class="chart-container" id="top-speed-chart">
                        <div class="no-cluster-selected">-</div>
                    </div>
                </div>
                <div class="chart-panel">
                    <div class="chart-title">
                        <span>Pauses</span>
                        <span class="chart-trend stable" id="rests-trend">-</span>
                    </div>
                    <div class="chart-container" id="rests-chart">
                        <div class="no-cluster-selected">-</div>
                    </div>
                </div>
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

        const COLORS = [
            '#38bdf8', '#4ade80', '#fbbf24', '#f87171', '#a78bfa',
            '#fb923c', '#2dd4bf', '#f472b6', '#60a5fa', '#84cc16'
        ];

        let clusters = [];
        let selectedCluster = null;
        let selectedSegment = null;
        let clusterRadius = 100;
        let restThreshold = 10;
        let map, trackLayer, tileLayer;
        let mapTheme = localStorage.getItem('mapTheme') || 'dark';
        let outliers = JSON.parse(localStorage.getItem('ski_outliers') || '{{}}');
        let hideOutliers = localStorage.getItem('hideOutliers') === 'true';
        let filteredSegments = [];

        const TILE_URLS = {{
            dark: 'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
            light: 'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png'
        }};

        document.addEventListener('DOMContentLoaded', init);

        function init() {{
            initMap();
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

        function applyFilters() {{
            filteredSegments = allSegments.filter(seg => {{
                if (hideOutliers && outliers[seg.id]) return false;
                return true;
            }});
            computeClusters();
            renderClusters();
        }}

        function initMap() {{
            map = L.map('map', {{ zoomControl: true, attributionControl: false }}).setView([46.31, 7.39], 12);
            tileLayer = L.tileLayer(TILE_URLS[mapTheme], {{ maxZoom: 19 }}).addTo(map);
            trackLayer = L.layerGroup().addTo(map);

            document.getElementById('map-dark').classList.toggle('active', mapTheme === 'dark');
            document.getElementById('map-light').classList.toggle('active', mapTheme === 'light');
        }}

        function setMapTheme(theme) {{
            mapTheme = theme;
            localStorage.setItem('mapTheme', theme);
            map.removeLayer(tileLayer);
            tileLayer = L.tileLayer(TILE_URLS[theme], {{ maxZoom: 19 }}).addTo(map);
            tileLayer.bringToBack();

            document.getElementById('map-dark').classList.toggle('active', theme === 'dark');
            document.getElementById('map-light').classList.toggle('active', theme === 'light');
        }}

        function setupEventListeners() {{
            document.getElementById('cluster-radius').addEventListener('input', e => {{
                clusterRadius = parseInt(e.target.value);
                document.getElementById('radius-value').textContent = clusterRadius + 'm';
                computeClusters();
                renderClusters();
                if (selectedCluster !== null) {{
                    const cluster = clusters.find(c => c.id === selectedCluster);
                    if (cluster) updateClusterView(cluster);
                    else clearClusterView();
                }}
            }});

            document.getElementById('rest-threshold').addEventListener('change', e => {{
                restThreshold = parseInt(e.target.value);
                // Recompute rests for all segments
                allSegments.forEach(seg => {{
                    seg.rests = countRests(seg.track_full, restThreshold);
                }});
                if (selectedCluster !== null) {{
                    const cluster = clusters.find(c => c.id === selectedCluster);
                    if (cluster) updateClusterView(cluster);
                }}
            }});
        }}

        function haversine(lat1, lon1, lat2, lon2) {{
            const R = 6371000;
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
            return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        }}

        function countRests(trackFull, thresholdS) {{
            // Detect pauses by looking at time gaps between GPS points
            // When you stop mid-run, the GPS continues recording but points are further apart in time
            if (!trackFull || trackFull.length < 2) return 0;

            let rests = 0;

            for (let i = 1; i < trackFull.length; i++) {{
                const timeGap = trackFull[i].t - trackFull[i-1].t;
                if (timeGap >= thresholdS) {{
                    rests++;
                }}
            }}

            return rests;
        }}

        // Compute rests for all segments
        allSegments.forEach(seg => {{
            seg.rests = countRests(seg.track_full, restThreshold);
        }});

        function computeClusters() {{
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

            const clusterMap = {{}};
            filteredSegments.forEach(seg => {{
                const root = find(seg.id);
                if (!clusterMap[root]) clusterMap[root] = [];
                clusterMap[root].push(seg);
            }});

            clusters = Object.values(clusterMap)
                .filter(segs => segs.length > 1) // Only clusters with multiple runs
                .sort((a, b) => b.length - a.length)
                .map((segs, i) => {{
                    const sortedSegs = segs.sort((a, b) => a.date.localeCompare(b.date));
                    return {{
                        id: i,
                        color: COLORS[i % COLORS.length],
                        segments: sortedSegs,
                        avgDistance: sortedSegs.reduce((s, seg) => s + seg.distance_m, 0) / sortedSegs.length,
                        avgSpeed: sortedSegs.reduce((s, seg) => s + seg.avg_speed, 0) / sortedSegs.length,
                        avgTopSpeed: sortedSegs.reduce((s, seg) => s + seg.top_speed, 0) / sortedSegs.length,
                        avgRests: sortedSegs.reduce((s, seg) => s + seg.rests, 0) / sortedSegs.length,
                        location: sortedSegs[0].location,
                        years: [...new Set(sortedSegs.map(s => s.year))].sort()
                    }};
                }});

            document.getElementById('cluster-count').textContent = clusters.length;
            document.getElementById('multi-run-count').textContent = clusters.filter(c => c.segments.length >= 2).length;
        }}

        function renderClusters() {{
            const list = document.getElementById('cluster-list');

            if (clusters.length === 0) {{
                list.innerHTML = '<div class="empty-state">No clusters found. Try increasing the radius.</div>';
                return;
            }}

            list.innerHTML = clusters.map(cluster => `
                <div class="cluster-card ${{selectedCluster === cluster.id ? 'selected' : ''}}"
                     data-id="${{cluster.id}}" onclick="selectCluster(${{cluster.id}})">
                    <div class="cluster-header">
                        <div class="cluster-color" style="background: ${{cluster.color}}"></div>
                        <div class="cluster-info">
                            <div class="cluster-name">${{cluster.location}} Route #${{cluster.id + 1}}</div>
                            <div class="cluster-meta">
                                ${{cluster.segments.length}} runs · ${{cluster.years.join(', ')}}
                            </div>
                        </div>
                    </div>
                    <div class="cluster-mini-stats">
                        <div class="mini-stat">
                            <div class="mini-stat-value" style="color: var(--accent)">${{(cluster.avgDistance / 1000).toFixed(2)}}</div>
                            <div class="mini-stat-label">km avg</div>
                        </div>
                        <div class="mini-stat">
                            <div class="mini-stat-value" style="color: var(--success)">${{cluster.avgSpeed.toFixed(1)}}</div>
                            <div class="mini-stat-label">avg km/h</div>
                        </div>
                        <div class="mini-stat">
                            <div class="mini-stat-value" style="color: var(--warning)">${{cluster.avgTopSpeed.toFixed(1)}}</div>
                            <div class="mini-stat-label">top km/h</div>
                        </div>
                        <div class="mini-stat">
                            <div class="mini-stat-value" style="color: var(--purple)">${{cluster.avgRests.toFixed(1)}}</div>
                            <div class="mini-stat-label">rests avg</div>
                        </div>
                    </div>
                </div>
            `).join('');
        }}

        function selectCluster(id) {{
            selectedCluster = id;
            selectedSegment = null;

            document.querySelectorAll('.cluster-card').forEach(card => {{
                card.classList.toggle('selected', parseInt(card.dataset.id) === id);
            }});

            const cluster = clusters.find(c => c.id === id);
            if (cluster) updateClusterView(cluster);
        }}

        function updateClusterView(cluster) {{
            updateMap(cluster);
            updateTimeline(cluster);
            updateCharts(cluster);
        }}

        function clearClusterView() {{
            selectedCluster = null;
            trackLayer.clearLayers();
            document.getElementById('segment-timeline').innerHTML =
                '<div class="no-cluster-selected">Select a cluster to view segments</div>';
            ['distance', 'avg-speed', 'top-speed', 'rests'].forEach(id => {{
                document.getElementById(id + '-chart').innerHTML = '<div class="no-cluster-selected">-</div>';
                document.getElementById(id + '-trend').textContent = '-';
                document.getElementById(id + '-trend').className = 'chart-trend stable';
            }});
        }}

        function updateMap(cluster) {{
            trackLayer.clearLayers();

            cluster.segments.forEach(seg => {{
                if (seg.track.length < 2) return;
                const isSelected = seg.id === selectedSegment;
                const coords = seg.track.map(p => [p.lat, p.lng]);

                const polyline = L.polyline(coords, {{
                    color: cluster.color,
                    weight: isSelected ? 4 : 2,
                    opacity: isSelected ? 1 : 0.6
                }});

                polyline.on('click', () => selectSegmentInCluster(seg.id, cluster));
                trackLayer.addLayer(polyline);
            }});

            const allCoords = cluster.segments.flatMap(s => s.track.map(p => [p.lat, p.lng]));
            if (allCoords.length > 0) map.fitBounds(L.latLngBounds(allCoords), {{ padding: [30, 30] }});
        }}

        function selectSegmentInCluster(segId, cluster) {{
            selectedSegment = segId;
            updateMap(cluster);
            updateTimeline(cluster);
        }}

        function updateTimeline(cluster) {{
            const timeline = document.getElementById('segment-timeline');

            timeline.innerHTML = `
                <div class="timeline-header">
                    <span>Segments over time (${{cluster.segments.length}} runs)</span>
                </div>
                ${{cluster.segments.map(seg => `
                    <div class="timeline-segment ${{selectedSegment === seg.id ? 'selected' : ''}}"
                         onclick="selectSegmentInCluster('${{seg.id}}', clusters[${{cluster.id}}])">
                        <div class="timeline-date">${{seg.date}}</div>
                        <div class="timeline-metrics">
                            <div class="timeline-metric distance">
                                <span class="value">${{(seg.distance_m / 1000).toFixed(2)}}</span>
                                <span>km</span>
                            </div>
                            <div class="timeline-metric avg-speed">
                                <span class="value">${{seg.avg_speed.toFixed(1)}}</span>
                                <span>avg</span>
                            </div>
                            <div class="timeline-metric top-speed">
                                <span class="value">${{seg.top_speed.toFixed(1)}}</span>
                                <span>top</span>
                            </div>
                            <div class="timeline-metric rests">
                                <span class="value">${{seg.rests}}</span>
                                <span>rests</span>
                            </div>
                        </div>
                    </div>
                `).join('')}}
            `;
        }}

        function updateCharts(cluster) {{
            const segs = cluster.segments;

            renderLineChart('distance-chart', segs, s => s.distance_m / 1000, 'km', '#38bdf8', 'distance-trend');
            renderLineChart('avg-speed-chart', segs, s => s.avg_speed, 'km/h', '#4ade80', 'avg-speed-trend');
            renderLineChart('top-speed-chart', segs, s => s.top_speed, 'km/h', '#fbbf24', 'top-speed-trend');
            renderLineChart('rests-chart', segs, s => s.rests, '', '#a78bfa', 'rests-trend');
        }}

        function renderLineChart(containerId, segments, getValue, unit, color, trendId) {{
            const container = document.getElementById(containerId);
            const width = container.clientWidth;
            const height = container.clientHeight;
            const margin = {{ top: 15, right: 10, bottom: 25, left: 35 }};

            if (segments.length < 1) {{
                container.innerHTML = '<div class="no-cluster-selected">No data</div>';
                return;
            }}

            const data = segments.map((s, i) => ({{ x: i, y: getValue(s), date: s.date, id: s.id }}));
            const yMin = Math.min(...data.map(d => d.y)) * 0.9;
            const yMax = Math.max(...data.map(d => d.y)) * 1.1;

            const xScale = i => margin.left + (i / (data.length - 1 || 1)) * (width - margin.left - margin.right);
            const yScale = y => height - margin.bottom - ((y - yMin) / (yMax - yMin || 1)) * (height - margin.top - margin.bottom);

            // Compute trend
            const first = data[0].y;
            const last = data[data.length - 1].y;
            const change = ((last - first) / first * 100).toFixed(1);
            const trendEl = document.getElementById(trendId);

            if (Math.abs(change) < 5) {{
                trendEl.textContent = '~ stable';
                trendEl.className = 'chart-trend stable';
            }} else if (change > 0) {{
                trendEl.textContent = '↑ ' + change + '%';
                trendEl.className = 'chart-trend up';
            }} else {{
                trendEl.textContent = '↓ ' + Math.abs(change) + '%';
                trendEl.className = 'chart-trend down';
            }}

            const linePath = data.map((d, i) => `${{i === 0 ? 'M' : 'L'}} ${{xScale(d.x)}} ${{yScale(d.y)}}`).join(' ');

            container.innerHTML = `
                <svg viewBox="0 0 ${{width}} ${{height}}">
                    <!-- Grid lines -->
                    <line class="chart-grid" x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}"/>
                    <line class="chart-grid" x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}"/>

                    <!-- Y axis labels -->
                    <text class="chart-label" x="${{margin.left - 5}}" y="${{yScale(yMax)}}" text-anchor="end" dominant-baseline="middle">${{yMax.toFixed(1)}}</text>
                    <text class="chart-label" x="${{margin.left - 5}}" y="${{yScale(yMin)}}" text-anchor="end" dominant-baseline="middle">${{yMin.toFixed(1)}}</text>

                    <!-- Line -->
                    <path class="chart-line" d="${{linePath}}" stroke="${{color}}"/>

                    <!-- Dots -->
                    ${{data.map(d => `
                        <circle class="chart-dot" cx="${{xScale(d.x)}}" cy="${{yScale(d.y)}}" r="4" fill="${{d.id === selectedSegment ? '#fff' : color}}"
                                stroke="${{d.id === selectedSegment ? color : 'none'}}" stroke-width="2"
                                onclick="selectSegmentInCluster('${{d.id}}', clusters[${{selectedCluster}}])">
                            <title>${{d.date}}: ${{d.y.toFixed(1)}} ${{unit}}</title>
                        </circle>
                    `).join('')}}

                    <!-- X axis labels -->
                    <text class="chart-label" x="${{margin.left}}" y="${{height - 5}}" text-anchor="start">${{data[0].date.slice(0, 4)}}</text>
                    <text class="chart-label" x="${{width - margin.right}}" y="${{height - 5}}" text-anchor="end">${{data[data.length - 1].date.slice(0, 4)}}</text>
                </svg>
            `;
        }}
    </script>
</body>
</html>
'''

    Path("cluster_analysis_app.html").write_text(html)
    print(f"  Created: cluster_analysis_app.html")


if __name__ == "__main__":
    generate_all_apps()
