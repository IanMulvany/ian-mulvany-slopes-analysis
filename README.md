# Slopes history explorer

This repository converts Slopes `.slopes` exports into GPX traces and analysis-ready tables, then serves them through an interactive Streamlit app for comparing days on snow.

## Getting started

1. **Unpack your data**
   - Unzip `GPSLogs.zip` (or any other `.slopes` files) into `data/GPSLogs/`.
2. **Install dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```
3. **Generate processed data**
   ```bash
   python -m slopes_analysis.ingest --raw-dir data/GPSLogs --output-dir data/processed --write-gpx
   ```
   This writes:
   - `data/processed/run_summary.parquet` – one row per run/day
   - `data/processed/points.parquet` – every GPS point with run IDs attached
   - `data/processed/gpx/` – GPX traces you can load into other tools
4. **Launch the explorer**
   ```bash
   streamlit run streamlit_app.py
   ```
   Use the "Rebuild processed data" button to refresh after adding new `.slopes` archives.

## Project structure
- `slopes_analysis/ingest.py` – converts `.slopes` exports to Parquet and GPX, computing core metrics.
- `slopes_analysis/analysis.py` – helper functions for yearly trends, leaderboards, and run comparisons.
- `streamlit_app.py` – the interactive dashboard and run comparison UI.

## Notes
- Raw and processed data under `data/` are ignored by Git so you can keep your personal exports locally.
- The converter prefers `RawGPS.csv` from each archive and supplements speed data with values from `Metadata.xml` when available.
