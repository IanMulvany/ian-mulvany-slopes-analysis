#!/usr/bin/env python3
"""Generate a standalone HTML viewer for segment gluing."""

import json
from dataclasses import asdict
from pathlib import Path
from slopes_analysis.analysis import load_datasets, segment_downhill_runs
from slopes_analysis.glue_candidates import find_glue_candidates, find_chain_candidates
from slopes_analysis.glue_persistence import load_glue_decisions
from slopes_analysis.annotations import (
    load_all_glue_decisions,
    load_all_outliers,
    get_outlier_segment_ids,
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


def generate_glue_viewer(
    output_path: Path = Path("segment_glue_viewer.html"),
    annotated_dir: Path = DEFAULT_ANNOTATED_DIR,
    time_threshold_min: float = 30.0,
    spatial_threshold_m: float = 200.0,
):
    """Generate a self-contained HTML viewer for segment gluing."""

    print("Loading datasets...")
    summary_df, points_df = load_datasets()

    print("Segmenting downhill runs...")
    segments_df, labeled_points_df = segment_downhill_runs(points_df)

    print(f"Found {len(segments_df)} downhill segments")

    # Load existing decisions and outliers from annotations directory
    print(f"Loading annotations from {annotated_dir}...")
    existing_decisions = load_all_glue_decisions(annotated_dir)
    existing_outliers = load_all_outliers(annotated_dir)
    outlier_ids = get_outlier_segment_ids(annotated_dir)

    approved_count = len([d for d in existing_decisions if d.status == "approved"])
    rejected_count = len([d for d in existing_decisions if d.status == "rejected"])
    print(f"Loaded {len(existing_decisions)} existing decisions ({approved_count} approved, {rejected_count} rejected)")
    print(f"Loaded {len(existing_outliers)} outlier markings")

    print("Finding glue candidates...")
    candidates = find_glue_candidates(
        segments_df, labeled_points_df,
        time_threshold_min=time_threshold_min,
        spatial_threshold_m=spatial_threshold_m
    )
    print(f"Found {len(candidates)} candidate pairs")

    chains = find_chain_candidates(
        segments_df, labeled_points_df,
        time_threshold_min=time_threshold_min,
        spatial_threshold_m=spatial_threshold_m
    )
    print(f"Found {len(chains)} chain candidates")

    # Prepare segment data with GPS tracks
    segments_data = []
    for _, seg in segments_df.iterrows():
        seg_id = seg["segment_global_id"]
        seg_points = labeled_points_df[labeled_points_df["segment_global_id"] == seg_id]

        if len(seg_points) < 2:
            continue

        # Sample points to reduce file size
        full_points = seg_points.sort_values("time").copy()
        if len(seg_points) > 50:
            seg_points = full_points.iloc[::2]
        else:
            seg_points = full_points

        track = [
            {"lat": row["latitude"], "lng": row["longitude"], "alt": row["altitude_m"]}
            for _, row in seg_points.iterrows()
        ]

        start_pt = full_points.iloc[0]
        end_pt = full_points.iloc[-1]

        segments_data.append({
            "id": seg_id,
            "day_id": seg["day_run_id"],
            "location": seg["location_name"] or "Unknown",
            "year": int(seg["year"]) if pd.notna(seg["year"]) else 0,
            "date": seg["start_time"].strftime("%Y-%m-%d") if pd.notna(seg["start_time"]) else "",
            "time": seg["start_time"].strftime("%H:%M") if pd.notna(seg["start_time"]) else "",
            "end_time": seg["end_time"].strftime("%H:%M:%S") if pd.notna(seg["end_time"]) else "",
            "distance_m": round(seg["distance_m"], 1),
            "vertical_m": round(seg["vertical_drop_m"], 1),
            "duration_s": round(seg["duration_s"], 1),
            "avg_speed": round(seg["avg_speed_kmh"], 1),
            "top_speed": round(seg["top_speed_kmh"], 1),
            "track": track,
            "start": {"lat": float(start_pt["latitude"]), "lng": float(start_pt["longitude"])},
            "end": {"lat": float(end_pt["latitude"]), "lng": float(end_pt["longitude"])},
        })

    # Prepare candidates data
    candidates_data = [
        {
            "pair_id": c.pair_id,
            "segment_a_id": c.segment_a_id,
            "segment_b_id": c.segment_b_id,
            "time_gap_s": round(c.time_gap_s, 1),
            "spatial_gap_m": round(c.spatial_gap_m, 1),
            "confidence": round(c.confidence, 3),
            "location": c.location,
            "date": c.date,
            "segment_a_end_time": c.segment_a_end_time,
            "segment_b_start_time": c.segment_b_start_time,
        }
        for c in candidates
    ]

    # Prepare chains data
    chains_data = chains

    # Prepare existing decisions (convert dataclass to dict)
    decisions_data = [asdict(d) for d in existing_decisions]

    # Prepare outliers data (convert dataclass to dict, match expected format)
    outliers_data = [
        {
            "id": o.id,
            "location": o.location,
            "date": o.date,
            "distance_m": o.distance_m,
            "avg_speed": o.avg_speed,
            "top_speed": o.top_speed,
            "markedAt": o.marked_at,
        }
        for o in existing_outliers
    ]

    # Compute summary stats
    locations = sorted(set(s["location"] for s in segments_data))
    years = sorted(set(s["year"] for s in segments_data if s["year"]))

    stats = {
        "total_segments": len(segments_data),
        "total_candidates": len(candidates_data),
        "total_chains": len(chains_data),
        "total_outliers": len(outliers_data),
        "locations": locations,
        "years": years,
    }

    html_content = generate_html(segments_data, candidates_data, chains_data, decisions_data, outliers_data, stats)

    output_path.write_text(html_content)
    print(f"Viewer generated: {output_path.absolute()}")
    return output_path


def generate_html(segments: list, candidates: list, chains: list, decisions: list, outliers: list, stats: dict) -> str:
    """Generate the complete HTML document."""

    segments_json = json.dumps(segments)
    candidates_json = json.dumps(candidates)
    chains_json = json.dumps(chains)
    decisions_json = json.dumps(decisions)
    outliers_json = json.dumps(outliers)
    stats_json = json.dumps(stats)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Segment Glue Editor</title>
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
            --seg-a: #3b82f6;
            --seg-b: #f97316;
            --merged: #a855f7;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}

        .app {{
            display: grid;
            grid-template-columns: 450px 1fr;
            grid-template-rows: auto 1fr auto;
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

        .stats-bar {{ display: flex; gap: 2rem; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 1.1rem; font-weight: 700; color: var(--accent); }}
        .stat-label {{ font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; }}

        .view-tabs {{ display: flex; gap: 0.5rem; }}

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
        }}

        select, input {{
            width: 100%;
            padding: 0.4rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.85rem;
        }}

        select:focus, input:focus {{ outline: none; border-color: var(--accent); }}

        .candidate-count {{
            padding: 0.5rem 0.75rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .candidate-list {{
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem;
        }}

        .candidate-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.15s;
        }}

        .candidate-card:hover {{ background: #3d4f66; }}
        .candidate-card.selected {{ border-color: var(--accent); background: rgba(56, 189, 248, 0.1); }}
        .candidate-card.approved {{ border-color: var(--success); background: rgba(74, 222, 128, 0.1); }}
        .candidate-card.rejected {{ border-color: var(--danger); background: rgba(248, 113, 113, 0.1); opacity: 0.6; }}

        .candidate-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }}

        .candidate-location {{ font-weight: 600; font-size: 0.9rem; }}
        .candidate-date {{ font-size: 0.75rem; color: var(--text-secondary); }}

        .confidence-badge {{
            padding: 0.15rem 0.5rem;
            border-radius: 100px;
            font-size: 0.7rem;
            font-weight: 600;
        }}

        .confidence-high {{ background: var(--success); color: var(--bg-primary); }}
        .confidence-medium {{ background: var(--warning); color: var(--bg-primary); }}
        .confidence-low {{ background: var(--danger); color: var(--bg-primary); }}

        .segment-pair {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 0.5rem;
            align-items: center;
        }}

        .segment-box {{
            padding: 0.5rem;
            border-radius: 6px;
            text-align: center;
        }}

        .segment-box.seg-a {{ background: rgba(59, 130, 246, 0.2); border: 1px solid var(--seg-a); }}
        .segment-box.seg-b {{ background: rgba(249, 115, 22, 0.2); border: 1px solid var(--seg-b); }}

        .segment-label {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; font-weight: 600; }}
        .segment-label.seg-a {{ color: var(--seg-a); }}
        .segment-label.seg-b {{ color: var(--seg-b); }}
        .segment-date {{ font-size: 0.7rem; color: var(--text-secondary); }}
        .segment-time {{ font-size: 0.8rem; font-weight: 600; }}
        .segment-stats {{ font-size: 0.7rem; color: var(--text-secondary); }}

        .gap-info {{
            text-align: center;
            padding: 0.25rem;
        }}

        .gap-arrow {{ font-size: 1.2rem; color: var(--text-secondary); }}
        .gap-time {{ font-size: 0.75rem; color: var(--warning); font-weight: 600; }}
        .gap-distance {{ font-size: 0.7rem; color: var(--text-secondary); }}

        .action-buttons {{
            display: flex;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }}

        .action-btn {{
            flex: 1;
            padding: 0.4rem;
            border: none;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }}

        .action-btn.approve {{ background: var(--success); color: var(--bg-primary); }}
        .action-btn.approve:hover {{ opacity: 0.9; }}
        .action-btn.reject {{ background: var(--danger); color: var(--bg-primary); }}
        .action-btn.reject:hover {{ opacity: 0.9; }}
        .action-btn.outlier {{ background: var(--warning); color: var(--bg-primary); }}
        .action-btn.outlier:hover {{ opacity: 0.9; }}
        .action-btn.undo {{ background: var(--bg-secondary); color: var(--text-secondary); border: 1px solid var(--border); }}
        .action-btn.undo:hover {{ border-color: var(--accent); color: var(--accent); }}

        .main-content {{
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        #map {{ flex: 1; background: var(--bg-primary); }}

        .details-panel {{
            height: 200px;
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
        }}

        .close-details:hover {{ color: var(--text-primary); }}

        .elevation-charts {{
            flex: 1;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 1rem;
            align-items: center;
        }}

        .elevation-chart {{ height: 100%; position: relative; }}
        .elevation-chart svg {{ width: 100%; height: 100%; }}
        .chart-path {{ fill: none; stroke-width: 2; }}
        .chart-path.seg-a {{ stroke: var(--seg-a); }}
        .chart-path.seg-b {{ stroke: var(--seg-b); }}
        .chart-area {{ opacity: 0.2; }}
        .chart-area.seg-a {{ fill: var(--seg-a); }}
        .chart-area.seg-b {{ fill: var(--seg-b); }}

        .merge-preview {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
        }}

        footer {{
            grid-column: 1 / -1;
            background: var(--bg-secondary);
            padding: 0.75rem 1.5rem;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .decision-summary {{
            display: flex;
            gap: 1.5rem;
            font-size: 0.85rem;
        }}

        .decision-count {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .decision-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}

        .decision-dot.approved {{ background: var(--success); }}
        .decision-dot.rejected {{ background: var(--danger); }}
        .decision-dot.pending {{ background: var(--warning); }}

        .export-btn {{
            background: var(--accent);
            border: none;
            color: var(--bg-primary);
            padding: 0.5rem 1.25rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }}

        .export-btn:hover {{ background: var(--accent-hover); }}

        /* Manual selection mode */
        .manual-list {{ display: none; }}
        .manual-list.active {{ display: block; }}
        .candidate-list.hidden {{ display: none; }}

        .segment-card {{
            background: var(--bg-tertiary);
            border-radius: 6px;
            padding: 0.5rem;
            margin-bottom: 0.3rem;
            cursor: pointer;
            border: 2px solid transparent;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .segment-card:hover {{ background: #3d4f66; }}
        .segment-card.selected {{ border-color: var(--accent); }}
        .segment-card.in-selection {{ border-color: var(--merged); background: rgba(168, 85, 247, 0.1); }}

        .segment-info {{ flex: 1; }}
        .segment-name {{ font-size: 0.8rem; font-weight: 600; }}
        .segment-meta {{ font-size: 0.7rem; color: var(--text-secondary); }}

        .glue-selected-btn {{
            display: none;
            background: var(--merged);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            margin: 0.5rem;
        }}

        .glue-selected-btn.visible {{ display: block; }}

        /* Chain view */
        .chain-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border: 2px solid transparent;
        }}

        .chain-card.selected {{ border-color: var(--merged); }}

        .chain-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }}

        .chain-segments {{
            display: flex;
            flex-direction: column;
            gap: 0;
            margin-top: 0.5rem;
        }}

        .chain-segment {{
            background: var(--bg-secondary);
            padding: 0.4rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 3px solid var(--accent);
        }}

        .chain-segment:nth-child(odd) {{ border-left-color: var(--seg-a); }}
        .chain-segment:nth-child(even) {{ border-left-color: var(--seg-b); }}

        .chain-seg-info {{ display: flex; gap: 0.75rem; align-items: center; }}
        .chain-seg-num {{ font-weight: 700; color: var(--text-secondary); min-width: 1.5rem; }}
        .chain-seg-time {{ font-weight: 600; }}
        .chain-seg-stats {{ color: var(--text-secondary); }}

        .chain-gap {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.2rem 0;
            font-size: 0.65rem;
            color: var(--warning);
        }}

        .chain-gap-arrow {{ color: var(--text-secondary); }}

        @keyframes fadeInOut {{
            0% {{ opacity: 0; transform: translateX(-50%) translateY(20px); }}
            15% {{ opacity: 1; transform: translateX(-50%) translateY(0); }}
            85% {{ opacity: 1; transform: translateX(-50%) translateY(0); }}
            100% {{ opacity: 0; transform: translateX(-50%) translateY(-20px); }}
        }}

        @media (max-width: 900px) {{
            .app {{ grid-template-columns: 1fr; grid-template-rows: auto auto 1fr auto; }}
            .sidebar {{ max-height: 45vh; border-right: none; border-bottom: 1px solid var(--border); }}
            .stats-bar {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="app">
        <header>
            <h1>Segment Glue Editor</h1>
            <div class="view-tabs">
                <button class="view-tab active" data-view="candidates">Candidates</button>
                <button class="view-tab" data-view="chains">Chains</button>
                <button class="view-tab" data-view="manual">Manual</button>
                <button class="view-tab" data-view="approved">Approved</button>
            </div>
            <div class="stats-bar">
                <div class="stat">
                    <div class="stat-value" id="stat-candidates">-</div>
                    <div class="stat-label">Candidates</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-approved">-</div>
                    <div class="stat-label">Approved</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="stat-remaining">-</div>
                    <div class="stat-label">Remaining</div>
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
                        <label>Min Confidence</label>
                        <input type="range" id="filter-confidence" min="0" max="100" value="0">
                    </div>
                    <div class="filter-group">
                        <label id="confidence-label">0%</label>
                    </div>
                </div>
            </div>

            <div class="candidate-count" id="candidate-count">
                <span>Loading...</span>
            </div>

            <div class="candidate-list" id="candidate-list"></div>
            <div class="candidate-list" id="chain-list" style="display:none;"></div>
            <div class="candidate-list manual-list" id="manual-list"></div>
            <button class="glue-selected-btn" id="glue-selected-btn">Glue Selected Segments</button>
            <div class="candidate-list" id="approved-list" style="display:none;"></div>
        </aside>

        <main class="main-content">
            <div id="map"></div>
            <div class="details-panel" id="details-panel">
                <div class="details-header">
                    <div class="details-title" id="details-title">Elevation Comparison</div>
                    <button class="close-details" id="close-details">&times;</button>
                </div>
                <div class="elevation-charts" id="elevation-charts">
                    <div class="elevation-chart" id="chart-a"></div>
                    <div class="merge-preview">
                        <div style="font-size: 2rem;">+</div>
                        <div>Merge</div>
                    </div>
                    <div class="elevation-chart" id="chart-b"></div>
                </div>
            </div>
        </main>

        <footer>
            <div class="decision-summary">
                <div class="decision-count">
                    <div class="decision-dot approved"></div>
                    <span id="count-approved">0 approved</span>
                </div>
                <div class="decision-count">
                    <div class="decision-dot rejected"></div>
                    <span id="count-rejected">0 rejected</span>
                </div>
                <div class="decision-count">
                    <div class="decision-dot pending"></div>
                    <span id="count-pending">0 pending</span>
                </div>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                <button class="export-btn" onclick="exportDecisions()">Download Decisions</button>
                <button class="export-btn" style="background: var(--warning);" onclick="exportOutliers()">Download Outliers</button>
            </div>
        </footer>
    </div>

    <script>
        // Data
        const allSegments = {segments_json};
        const allCandidates = {candidates_json};
        const allChains = {chains_json};
        const existingDecisions = {decisions_json};
        const existingOutliers = {outliers_json};
        const stats = {stats_json};

        // Create segment lookup
        const segmentById = {{}};
        allSegments.forEach(s => segmentById[s.id] = s);

        // Haversine distance calculation
        function haversine(lat1, lon1, lat2, lon2) {{
            const R = 6371000;
            const dLat = (lat2 - lat1) * Math.PI / 180;
            const dLon = (lon2 - lon1) * Math.PI / 180;
            const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
            return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        }}

        // State
        let currentView = 'candidates';
        let filteredCandidates = [...allCandidates];
        let selectedCandidate = null;
        let selectedChain = null;
        let manualSelection = [];
        let decisions = {{}};  // pair_id -> {{status, notes}}
        let outliers = {{}};  // segment_id -> outlier info
        let map, trackLayer;

        // Initialize from existing decisions
        existingDecisions.forEach(d => {{
            const key = d.segment_ids.length === 2
                ? `glue_${{d.segment_ids[0]}}__${{d.segment_ids[1]}}`
                : d.glue_id;
            decisions[key] = {{ status: d.status, notes: d.notes, segment_ids: d.segment_ids }};
        }});

        // Initialize from existing outliers
        existingOutliers.forEach(o => {{
            outliers[o.id] = o;
        }});

        document.addEventListener('DOMContentLoaded', init);

        function init() {{
            initMap();
            populateFilters();
            setupEventListeners();
            applyFilters();
            updateStats();
            updateDecisionCounts();
        }}

        function initMap() {{
            map = L.map('map', {{ zoomControl: true, attributionControl: false }})
                .setView([46.31, 7.39], 12);
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
                {{ maxZoom: 19 }}).addTo(map);
            trackLayer = L.layerGroup().addTo(map);
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
            document.getElementById('filter-confidence').addEventListener('input', (e) => {{
                document.getElementById('confidence-label').textContent = e.target.value + '%';
                applyFilters();
            }});

            document.querySelectorAll('.view-tab').forEach(tab => {{
                tab.addEventListener('click', () => {{
                    currentView = tab.dataset.view;
                    document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    toggleView();
                }});
            }});

            document.getElementById('close-details').addEventListener('click', () => {{
                document.getElementById('details-panel').classList.remove('visible');
            }});

            document.getElementById('glue-selected-btn').addEventListener('click', glueManualSelection);
        }}

        function toggleView() {{
            document.getElementById('candidate-list').style.display = currentView === 'candidates' ? 'block' : 'none';
            document.getElementById('chain-list').style.display = currentView === 'chains' ? 'block' : 'none';
            document.getElementById('manual-list').style.display = currentView === 'manual' ? 'block' : 'none';
            document.getElementById('approved-list').style.display = currentView === 'approved' ? 'block' : 'none';
            document.getElementById('glue-selected-btn').style.display = currentView === 'manual' && manualSelection.length >= 2 ? 'block' : 'none';

            if (currentView === 'candidates') renderCandidates();
            else if (currentView === 'chains') renderChains();
            else if (currentView === 'manual') renderManualList();
            else if (currentView === 'approved') renderApprovedList();
        }}

        function applyFilters() {{
            const location = document.getElementById('filter-location').value;
            const year = document.getElementById('filter-year').value;
            const minConfidence = parseInt(document.getElementById('filter-confidence').value) / 100;

            filteredCandidates = allCandidates.filter(c => {{
                if (location && c.location !== location) return false;
                const segA = segmentById[c.segment_a_id];
                if (year && segA && segA.year !== parseInt(year)) return false;
                if (c.confidence < minConfidence) return false;
                return true;
            }});

            renderCandidates();
            updateStats();
        }}

        function updateStats() {{
            const approved = Object.values(decisions).filter(d => d.status === 'approved').length;
            const pending = filteredCandidates.filter(c => !decisions[c.pair_id]).length;

            document.getElementById('stat-candidates').textContent = filteredCandidates.length;
            document.getElementById('stat-approved').textContent = approved;
            document.getElementById('stat-remaining').textContent = pending;
        }}

        function updateDecisionCounts() {{
            const approved = Object.values(decisions).filter(d => d.status === 'approved').length;
            const rejected = Object.values(decisions).filter(d => d.status === 'rejected').length;
            const pending = allCandidates.filter(c => !decisions[c.pair_id]).length;

            document.getElementById('count-approved').textContent = approved + ' approved';
            document.getElementById('count-rejected').textContent = rejected + ' rejected';
            document.getElementById('count-pending').textContent = pending + ' pending';
        }}

        function getConfidenceClass(confidence) {{
            if (confidence >= 0.7) return 'confidence-high';
            if (confidence >= 0.4) return 'confidence-medium';
            return 'confidence-low';
        }}

        function formatDuration(seconds) {{
            if (seconds < 60) return Math.round(seconds) + 's';
            const mins = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            return mins + 'm ' + secs + 's';
        }}

        function renderCandidates() {{
            const container = document.getElementById('candidate-list');

            // Filter to only show pending candidates (not approved or rejected)
            const pendingCandidates = filteredCandidates.filter(c => {{
                const decision = decisions[c.pair_id];
                return !decision || decision.status === 'pending';
            }});

            document.getElementById('candidate-count').innerHTML = `<span>${{pendingCandidates.length}} pending (of ${{filteredCandidates.length}} filtered)</span>`;

            if (pendingCandidates.length === 0) {{
                container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">All candidates have been reviewed!</div>';
                return;
            }}

            container.innerHTML = pendingCandidates.map(c => {{
                const segA = segmentById[c.segment_a_id];
                const segB = segmentById[c.segment_b_id];
                if (!segA || !segB) return '';

                const decision = decisions[c.pair_id];
                const statusClass = decision ? decision.status : '';

                return `
                    <div class="candidate-card ${{selectedCandidate === c.pair_id ? 'selected' : ''}} ${{statusClass}}"
                         data-id="${{c.pair_id}}" onclick="selectCandidate('${{c.pair_id}}')">
                        <div class="candidate-header">
                            <div>
                                <div class="candidate-location">${{c.location}}</div>
                                <div class="candidate-date">${{c.date}}</div>
                            </div>
                            <span class="confidence-badge ${{getConfidenceClass(c.confidence)}}">
                                ${{Math.round(c.confidence * 100)}}%
                            </span>
                        </div>
                        <div class="segment-pair">
                            <div class="segment-box seg-a">
                                <div class="segment-label seg-a">First Segment</div>
                                <div class="segment-date">${{segA.date}}</div>
                                <div class="segment-time">ends ${{c.segment_a_end_time}}</div>
                                <div class="segment-stats">${{Math.round(segA.distance_m)}}m / ${{Math.round(segA.vertical_m)}}m drop</div>
                            </div>
                            <div class="gap-info">
                                <div class="gap-arrow">→</div>
                                <div class="gap-time">${{formatDuration(c.time_gap_s)}}</div>
                                <div class="gap-distance">${{Math.round(c.spatial_gap_m)}}m gap</div>
                            </div>
                            <div class="segment-box seg-b">
                                <div class="segment-label seg-b">Second Segment</div>
                                <div class="segment-date">${{segB.date}}</div>
                                <div class="segment-time">starts ${{c.segment_b_start_time}}</div>
                                <div class="segment-stats">${{Math.round(segB.distance_m)}}m / ${{Math.round(segB.vertical_m)}}m drop</div>
                            </div>
                        </div>
                        <div class="action-buttons">
                            ${{decision && decision.status === 'approved' ? `
                                <button class="action-btn undo" onclick="event.stopPropagation(); undoDecision('${{c.pair_id}}')">Undo</button>
                            ` : decision && decision.status === 'rejected' ? `
                                <button class="action-btn undo" onclick="event.stopPropagation(); undoDecision('${{c.pair_id}}')">Undo</button>
                            ` : `
                                <button class="action-btn approve" onclick="event.stopPropagation(); approveCandidate('${{c.pair_id}}')">Approve</button>
                                <button class="action-btn reject" onclick="event.stopPropagation(); rejectCandidate('${{c.pair_id}}')">Reject</button>
                            `}}
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        function isChainProcessed(chain, idx) {{
            // Check if this chain was directly approved or rejected (including outliers)
            const chainKeys = Object.keys(decisions).filter(k => k.startsWith(`chain_${{idx}}_`));
            if (chainKeys.some(k => decisions[k].status === 'approved' || decisions[k].status === 'rejected')) return true;

            // Check if all pairs in the chain are processed (approved or rejected)
            let allPairsProcessed = true;
            for (let i = 0; i < chain.length - 1; i++) {{
                const pairId = `glue_${{chain[i]}}__${{chain[i + 1]}}`;
                const decision = decisions[pairId];
                if (!decision || (decision.status !== 'approved' && decision.status !== 'rejected')) {{
                    allPairsProcessed = false;
                    break;
                }}
            }}
            return allPairsProcessed;
        }}

        function renderChains() {{
            const container = document.getElementById('chain-list');

            // Filter to only show pending chains
            const pendingChains = allChains.map((chain, idx) => ({{ chain, idx }}))
                .filter(({{ chain, idx }}) => !isChainProcessed(chain, idx));

            document.getElementById('candidate-count').innerHTML = `<span>${{pendingChains.length}} pending chains (of ${{allChains.length}} total)</span>`;

            if (pendingChains.length === 0) {{
                container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">All chains have been reviewed!</div>';
                return;
            }}

            container.innerHTML = pendingChains.map(({{ chain, idx }}) => {{
                const firstSeg = segmentById[chain[0]];
                const lastSeg = segmentById[chain[chain.length - 1]];
                const location = firstSeg ? firstSeg.location : 'Unknown';
                const date = firstSeg ? firstSeg.date : '';

                // Calculate total stats
                let totalDist = 0, totalDrop = 0;
                chain.forEach(segId => {{
                    const seg = segmentById[segId];
                    if (seg) {{
                        totalDist += seg.distance_m;
                        totalDrop += seg.vertical_m;
                    }}
                }});

                // Build segment rows with gaps
                let segmentRows = '';
                chain.forEach((segId, i) => {{
                    const seg = segmentById[segId];
                    if (!seg) return;

                    segmentRows += `
                        <div class="chain-segment">
                            <div class="chain-seg-info">
                                <span class="chain-seg-num">#${{i + 1}}</span>
                                <span class="chain-seg-time">${{seg.time}} - ${{seg.end_time}}</span>
                            </div>
                            <div class="chain-seg-stats">${{Math.round(seg.distance_m)}}m / ${{Math.round(seg.vertical_m)}}m</div>
                        </div>
                    `;

                    // Add gap info between segments
                    if (i < chain.length - 1) {{
                        const nextSeg = segmentById[chain[i + 1]];
                        if (nextSeg) {{
                            // Calculate time gap (rough estimate from end_time to next start)
                            const gap = calculateGap(seg, nextSeg);
                            segmentRows += `
                                <div class="chain-gap">
                                    <span class="chain-gap-arrow">↓</span>
                                    <span>${{gap.time}} gap</span>
                                    <span>·</span>
                                    <span>${{gap.distance}}m</span>
                                </div>
                            `;
                        }}
                    }}
                }});

                return `
                    <div class="chain-card ${{selectedChain === idx ? 'selected' : ''}}"
                         onclick="selectChain(${{idx}})">
                        <div class="chain-header">
                            <div>
                                <div class="candidate-location">${{location}}</div>
                                <div class="candidate-date">${{date}} · ${{chain.length}} segments · ${{Math.round(totalDist)}}m total · ${{Math.round(totalDrop)}}m drop</div>
                            </div>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="action-btn approve" onclick="event.stopPropagation(); approveChain(${{idx}})">
                                    Approve
                                </button>
                                <button class="action-btn outlier" onclick="event.stopPropagation(); markChainAsOutlier(${{idx}})">
                                    Outlier
                                </button>
                            </div>
                        </div>
                        <div class="chain-segments">
                            ${{segmentRows}}
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        function calculateGap(segA, segB) {{
            // Parse end time of A and start time of B
            const endParts = segA.end_time.split(':').map(Number);
            const startParts = segB.time.split(':').map(Number);

            const endSecs = endParts[0] * 3600 + endParts[1] * 60 + (endParts[2] || 0);
            const startSecs = startParts[0] * 3600 + startParts[1] * 60 + (startParts[2] || 0);

            const gapSecs = startSecs - endSecs;
            const gapTime = gapSecs < 60 ? gapSecs + 's' : Math.floor(gapSecs / 60) + 'm ' + (gapSecs % 60) + 's';

            // Calculate spatial gap
            const dist = Math.round(haversine(segA.end.lat, segA.end.lng, segB.start.lat, segB.start.lng));

            return {{ time: gapTime, distance: dist }};
        }}

        function renderManualList() {{
            const container = document.getElementById('manual-list');
            container.classList.add('active');
            document.getElementById('candidate-count').innerHTML = `<span>Select segments to glue</span>`;

            // Group segments by day_id
            const byDay = {{}};
            allSegments.forEach(s => {{
                if (!byDay[s.day_id]) byDay[s.day_id] = [];
                byDay[s.day_id].push(s);
            }});

            let html = '';
            Object.entries(byDay).forEach(([dayId, segs]) => {{
                segs.sort((a, b) => a.time.localeCompare(b.time));
                const firstSeg = segs[0];

                html += `<div style="padding: 0.5rem; font-weight: 600; color: var(--text-secondary); font-size: 0.75rem;">
                    ${{firstSeg.location}} - ${{firstSeg.date}}
                </div>`;

                segs.forEach(seg => {{
                    const isSelected = manualSelection.includes(seg.id);
                    html += `
                        <div class="segment-card ${{isSelected ? 'in-selection' : ''}}"
                             onclick="toggleManualSelect('${{seg.id}}')">
                            <div class="segment-info">
                                <div class="segment-name">${{seg.time}} - ${{Math.round(seg.distance_m)}}m</div>
                                <div class="segment-meta">${{Math.round(seg.vertical_m)}}m drop, ${{seg.avg_speed}} km/h avg</div>
                            </div>
                        </div>
                    `;
                }});
            }});

            container.innerHTML = html;

            const btn = document.getElementById('glue-selected-btn');
            btn.classList.toggle('visible', manualSelection.length >= 2);
            btn.textContent = `Glue ${{manualSelection.length}} Selected Segments`;
        }}

        function renderApprovedList() {{
            const container = document.getElementById('approved-list');
            const approved = Object.entries(decisions).filter(([_, d]) => d.status === 'approved');

            document.getElementById('candidate-count').innerHTML = `<span>${{approved.length}} approved merges</span>`;

            container.innerHTML = approved.map(([key, decision]) => {{
                const segIds = decision.segment_ids || key.replace('glue_', '').split('__');
                const segments = segIds.map(id => segmentById[id]).filter(Boolean);
                const firstSeg = segments[0];

                return `
                    <div class="candidate-card approved" onclick="showMergeOnMap(${{JSON.stringify(segIds).replace(/"/g, '&quot;')}})">
                        <div class="candidate-header">
                            <div>
                                <div class="candidate-location">${{firstSeg ? firstSeg.location : 'Unknown'}}</div>
                                <div class="candidate-date">${{firstSeg ? firstSeg.date : ''}} - ${{segments.length}} segments</div>
                            </div>
                            <button class="action-btn undo" onclick="event.stopPropagation(); undoDecision('${{key}}')">Undo</button>
                        </div>
                        <div class="chain-segments">
                            ${{segments.map(seg => `<span class="chain-segment">${{seg.time}}</span>`).join('→')}}
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        function selectCandidate(pairId) {{
            selectedCandidate = pairId;
            const candidate = allCandidates.find(c => c.pair_id === pairId);
            if (!candidate) return;

            renderCandidates();
            showCandidateOnMap(candidate);
            showDetailsPanel(candidate);
        }}

        function selectChain(idx) {{
            selectedChain = idx;
            const chain = allChains[idx];
            if (!chain) return;

            renderChains();
            showChainOnMap(chain);
        }}

        function toggleManualSelect(segId) {{
            const idx = manualSelection.indexOf(segId);
            if (idx >= 0) {{
                manualSelection.splice(idx, 1);
            }} else {{
                manualSelection.push(segId);
            }}
            renderManualList();
            showManualSelectionOnMap();
        }}

        function showCandidateOnMap(candidate) {{
            trackLayer.clearLayers();

            const segA = segmentById[candidate.segment_a_id];
            const segB = segmentById[candidate.segment_b_id];
            if (!segA || !segB) return;

            // Draw segment A (blue)
            const coordsA = segA.track.map(p => [p.lat, p.lng]);
            L.polyline(coordsA, {{ color: '#3b82f6', weight: 4 }}).addTo(trackLayer);

            // Draw segment B (orange)
            const coordsB = segB.track.map(p => [p.lat, p.lng]);
            L.polyline(coordsB, {{ color: '#f97316', weight: 4 }}).addTo(trackLayer);

            // Draw gap (dashed gray)
            const gapCoords = [[segA.end.lat, segA.end.lng], [segB.start.lat, segB.start.lng]];
            L.polyline(gapCoords, {{ color: '#94a3b8', weight: 2, dashArray: '5, 10' }}).addTo(trackLayer);

            // Start marker (green)
            L.circleMarker(coordsA[0], {{ radius: 8, fillColor: '#4ade80', fillOpacity: 1, color: '#fff', weight: 2 }}).addTo(trackLayer);

            // End marker (red)
            L.circleMarker(coordsB[coordsB.length - 1], {{ radius: 8, fillColor: '#f87171', fillOpacity: 1, color: '#fff', weight: 2 }}).addTo(trackLayer);

            // Fit bounds
            const allCoords = [...coordsA, ...coordsB];
            map.fitBounds(L.latLngBounds(allCoords), {{ padding: [50, 50] }});
        }}

        function showChainOnMap(chain) {{
            trackLayer.clearLayers();

            const colors = ['#3b82f6', '#f97316', '#a855f7', '#22c55e', '#eab308'];
            let allCoords = [];

            chain.forEach((segId, idx) => {{
                const seg = segmentById[segId];
                if (!seg) return;

                const coords = seg.track.map(p => [p.lat, p.lng]);
                allCoords = allCoords.concat(coords);

                L.polyline(coords, {{ color: colors[idx % colors.length], weight: 4 }}).addTo(trackLayer);

                // Draw gap to next
                if (idx < chain.length - 1) {{
                    const nextSeg = segmentById[chain[idx + 1]];
                    if (nextSeg) {{
                        const gapCoords = [[seg.end.lat, seg.end.lng], [nextSeg.start.lat, nextSeg.start.lng]];
                        L.polyline(gapCoords, {{ color: '#94a3b8', weight: 2, dashArray: '5, 10' }}).addTo(trackLayer);
                    }}
                }}
            }});

            if (allCoords.length > 0) {{
                map.fitBounds(L.latLngBounds(allCoords), {{ padding: [50, 50] }});
            }}
        }}

        function showManualSelectionOnMap() {{
            trackLayer.clearLayers();

            const colors = ['#3b82f6', '#f97316', '#a855f7', '#22c55e', '#eab308'];
            let allCoords = [];

            manualSelection.forEach((segId, idx) => {{
                const seg = segmentById[segId];
                if (!seg) return;

                const coords = seg.track.map(p => [p.lat, p.lng]);
                allCoords = allCoords.concat(coords);

                L.polyline(coords, {{ color: colors[idx % colors.length], weight: 4 }}).addTo(trackLayer);
            }});

            if (allCoords.length > 0) {{
                map.fitBounds(L.latLngBounds(allCoords), {{ padding: [50, 50] }});
            }}
        }}

        function showMergeOnMap(segIds) {{
            trackLayer.clearLayers();

            let allCoords = [];
            segIds.forEach(segId => {{
                const seg = segmentById[segId];
                if (!seg) return;

                const coords = seg.track.map(p => [p.lat, p.lng]);
                allCoords = allCoords.concat(coords);

                L.polyline(coords, {{ color: '#a855f7', weight: 4 }}).addTo(trackLayer);
            }});

            if (allCoords.length > 0) {{
                map.fitBounds(L.latLngBounds(allCoords), {{ padding: [50, 50] }});
            }}
        }}

        function showDetailsPanel(candidate) {{
            const segA = segmentById[candidate.segment_a_id];
            const segB = segmentById[candidate.segment_b_id];
            if (!segA || !segB) return;

            document.getElementById('details-panel').classList.add('visible');
            document.getElementById('details-title').textContent = `${{candidate.location}} - ${{candidate.date}}`;

            renderMiniChart('chart-a', segA.track, 'seg-a');
            renderMiniChart('chart-b', segB.track, 'seg-b');
        }}

        function renderMiniChart(containerId, track, colorClass) {{
            const container = document.getElementById(containerId);
            const width = container.clientWidth || 200;
            const height = container.clientHeight || 100;
            const margin = {{ top: 5, right: 5, bottom: 5, left: 5 }};

            if (track.length < 2) {{
                container.innerHTML = '<div style="color: var(--text-secondary)">No data</div>';
                return;
            }}

            const altitudes = track.map(p => p.alt);
            const yMin = Math.min(...altitudes);
            const yMax = Math.max(...altitudes);
            const yPad = (yMax - yMin) * 0.1 || 10;

            const xScale = (i) => margin.left + (i / (track.length - 1)) * (width - margin.left - margin.right);
            const yScale = (y) => height - margin.bottom - ((y - yMin + yPad) / (yMax - yMin + 2 * yPad)) * (height - margin.top - margin.bottom);

            const linePath = track.map((p, i) => `${{i === 0 ? 'M' : 'L'}} ${{xScale(i)}} ${{yScale(p.alt)}}`).join(' ');
            const areaPath = linePath + ` L ${{xScale(track.length - 1)}} ${{height - margin.bottom}} L ${{margin.left}} ${{height - margin.bottom}} Z`;

            container.innerHTML = `
                <svg viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
                    <path class="chart-area ${{colorClass}}" d="${{areaPath}}"/>
                    <path class="chart-path ${{colorClass}}" d="${{linePath}}"/>
                </svg>
            `;
        }}

        function approveCandidate(pairId) {{
            const candidate = allCandidates.find(c => c.pair_id === pairId);
            if (!candidate) return;

            decisions[pairId] = {{
                status: 'approved',
                notes: '',
                segment_ids: [candidate.segment_a_id, candidate.segment_b_id]
            }};

            saveToLocalStorage();
            renderCandidates();
            updateStats();
            updateDecisionCounts();
            showToast('Segment pair approved');
        }}

        function rejectCandidate(pairId) {{
            const candidate = allCandidates.find(c => c.pair_id === pairId);
            if (!candidate) return;

            decisions[pairId] = {{
                status: 'rejected',
                notes: '',
                segment_ids: [candidate.segment_a_id, candidate.segment_b_id]
            }};

            saveToLocalStorage();
            renderCandidates();
            updateStats();
            updateDecisionCounts();
            showToast('Segment pair rejected');
        }}

        function undoDecision(key) {{
            delete decisions[key];
            saveToLocalStorage();
            if (currentView === 'candidates') renderCandidates();
            else if (currentView === 'approved') renderApprovedList();
            updateStats();
            updateDecisionCounts();
        }}

        function approveChain(idx) {{
            const chain = allChains[idx];
            if (!chain || chain.length < 2) return;

            const glueId = `chain_${{idx}}_${{Date.now()}}`;
            decisions[glueId] = {{
                status: 'approved',
                notes: 'Chain merge',
                segment_ids: chain
            }};

            // Also mark any candidate pairs within this chain as approved
            for (let i = 0; i < chain.length - 1; i++) {{
                const pairId = `glue_${{chain[i]}}__${{chain[i + 1]}}`;
                const candidate = allCandidates.find(c => c.pair_id === pairId);
                if (candidate) {{
                    decisions[pairId] = {{
                        status: 'approved',
                        notes: 'Part of chain merge',
                        segment_ids: [chain[i], chain[i + 1]]
                    }};
                }}
            }}

            saveToLocalStorage();
            renderChains();
            if (currentView === 'candidates') renderCandidates();
            updateStats();
            updateDecisionCounts();

            // Show confirmation
            showToast(`Approved chain with ${{chain.length}} segments`);
        }}

        function markChainAsOutlier(idx) {{
            const chain = allChains[idx];
            if (!chain || chain.length < 1) return;

            // Mark all segments in the chain as outliers
            chain.forEach(segId => {{
                const seg = segmentById[segId];
                if (seg) {{
                    outliers[segId] = {{
                        id: segId,
                        location: seg.location,
                        date: seg.date,
                        distance_m: seg.distance_m,
                        avg_speed: seg.avg_speed,
                        top_speed: seg.top_speed,
                        markedAt: new Date().toISOString()
                    }};
                }}
            }});

            // Also mark the chain as "rejected" so it disappears
            const glueId = `chain_${{idx}}_outlier_${{Date.now()}}`;
            decisions[glueId] = {{
                status: 'rejected',
                notes: 'Marked as outlier (walk/drive)',
                segment_ids: chain
            }};

            // Mark any candidate pairs as rejected too
            for (let i = 0; i < chain.length - 1; i++) {{
                const pairId = `glue_${{chain[i]}}__${{chain[i + 1]}}`;
                decisions[pairId] = {{
                    status: 'rejected',
                    notes: 'Part of outlier chain',
                    segment_ids: [chain[i], chain[i + 1]]
                }};
            }}

            saveToLocalStorage();
            saveOutliersToLocalStorage();
            renderChains();
            if (currentView === 'candidates') renderCandidates();
            updateStats();
            updateDecisionCounts();

            showToast(`Marked ${{chain.length}} segments as outliers`);
        }}

        function saveOutliersToLocalStorage() {{
            localStorage.setItem('glue_outliers', JSON.stringify(outliers));
        }}

        function loadOutliersFromLocalStorage() {{
            const saved = localStorage.getItem('glue_outliers');
            if (saved) {{
                try {{
                    Object.assign(outliers, JSON.parse(saved));
                }} catch (e) {{
                    console.error('Failed to load outliers:', e);
                }}
            }}
        }}

        function showToast(message) {{
            const toast = document.createElement('div');
            toast.style.cssText = `
                position: fixed;
                bottom: 80px;
                left: 50%;
                transform: translateX(-50%);
                background: var(--success);
                color: var(--bg-primary);
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
                z-index: 1000;
                animation: fadeInOut 2s ease-in-out;
            `;
            toast.textContent = message;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 2000);
        }}

        function glueManualSelection() {{
            if (manualSelection.length < 2) return;

            // Sort by time
            const sorted = manualSelection.slice().sort((a, b) => {{
                const segA = segmentById[a];
                const segB = segmentById[b];
                return (segA.date + segA.time).localeCompare(segB.date + segB.time);
            }});

            const glueId = `manual_${{Date.now()}}`;
            decisions[glueId] = {{
                status: 'approved',
                notes: 'Manual merge',
                segment_ids: sorted
            }};

            manualSelection = [];
            saveToLocalStorage();
            renderManualList();
            updateStats();
            updateDecisionCounts();
        }}

        function saveToLocalStorage() {{
            localStorage.setItem('glue_decisions', JSON.stringify(decisions));
        }}

        function loadFromLocalStorage() {{
            const saved = localStorage.getItem('glue_decisions');
            if (saved) {{
                try {{
                    const parsed = JSON.parse(saved);
                    Object.assign(decisions, parsed);
                }} catch (e) {{
                    console.error('Failed to load saved decisions:', e);
                }}
            }}
        }}

        function exportDecisions() {{
            const decisionsList = Object.entries(decisions).map(([key, d]) => ({{
                glue_id: key,
                segment_ids: d.segment_ids,
                status: d.status,
                notes: d.notes || '',
                created_at: new Date().toISOString()
            }}));

            const exportData = {{
                version: 1,
                created: new Date().toISOString(),
                decisions: decisionsList
            }};

            const blob = new Blob([JSON.stringify(exportData, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'glue_decisions.json';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function exportOutliers() {{
            // Export in same format as ski_outliers.json
            const outliersList = Object.values(outliers);

            if (outliersList.length === 0) {{
                showToast('No outliers to export');
                return;
            }}

            const blob = new Blob([JSON.stringify(outliersList, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ski_outliers_from_glue.json';
            a.click();
            URL.revokeObjectURL(url);

            showToast(`Exported ${{outliersList.length}} outliers`);
        }}

        // Load saved decisions and outliers on init
        loadFromLocalStorage();
        loadOutliersFromLocalStorage();
    </script>
</body>
</html>
'''


if __name__ == "__main__":
    generate_glue_viewer()
