"""
FastAPI backend for downhill run segmentation and clustering.

Endpoints:
- GET /health
- GET /segments          -> list of segment endpoints/metadata
- GET /clusters?radius=R -> cluster memberships at radius (meters)

Notes:
- Uses existing processed parquet files (data/processed).
- Segments are derived via analysis.segment_downhill_runs.
- Clustering reuses the same start/end proximity rule used in the Streamlit app.
- Results are cached in-memory per process.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from slopes_analysis import analysis
from streamlit_cluster import compute_segment_endpoints, cluster_segments, SegmentEndpoints

RAW_DIR = Path("data/GPSLogs")
PROCESSED_DIR = Path("data/processed")

app = FastAPI(title="Slopes clustering API", version="0.1.0")


class SegmentEndpointModel(BaseModel):
    segment_id: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    location_name: str | None
    year: int | None


class ClusterResponse(BaseModel):
    radius_m: float
    clusters: Dict[str, List[str]]  # root_id -> [segment_ids]


@lru_cache(maxsize=1)
def _load_segments():
    try:
        summary_df, points_df = analysis.load_datasets(PROCESSED_DIR, RAW_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="Processed data not found. Run setup_data.py first.") from e

    seg_df, labeled_points = analysis.segment_downhill_runs(points_df)
    endpoints = compute_segment_endpoints(labeled_points)
    return seg_df, labeled_points, endpoints


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/segments", response_model=List[SegmentEndpointModel])
def segments():
    _, _, endpoints = _load_segments()
    return [SegmentEndpointModel(**e.__dict__) for e in endpoints]


@lru_cache(maxsize=64)
def _cluster_for_radius(radius_m: float) -> Dict[str, List[str]]:
    _, labeled_points, endpoints = _load_segments()
    return cluster_segments(endpoints, radius_m)


@app.get("/clusters", response_model=ClusterResponse)
def clusters(radius: float = Query(120.0, ge=1.0, le=2000.0)):
    clusters_map = _cluster_for_radius(radius)
    return ClusterResponse(radius_m=radius, clusters=clusters_map)


# For local dev: uvicorn api_server:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

