# Slopes history explorer

This repository converts Slopes `.slopes` exports into GPX traces and analysis-ready tables, then serves them through an interactive Streamlit app for comparing days on snow.

## Getting started

1. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```
   Or if using `uv run`:
   ```bash
   uv run pip install -r requirements.txt
   ```

2. **Set up and process your data** (automatic setup)
   ```bash
   uv run python setup_data.py
   ```
   This script will:
   - Unzip `GPSLogs.zip` into `data/GPSLogs/`
   - Process all `.slopes` files to create analysis datasets
   - Generate GPX files for each run
   
   **Manual setup** (if you prefer):
   ```bash
   # Unzip GPSLogs.zip into data/GPSLogs/
   unzip GPSLogs.zip -d data/GPSLogs/
   
   # Process the files
   uv run python -m slopes_analysis.ingest --raw-dir data/GPSLogs --output-dir data/processed --write-gpx
   ```
   
   This creates:
   - `data/processed/run_summary.parquet` – one row per run/day
   - `data/processed/points.parquet` – every GPS point with run IDs attached
   - `data/processed/gpx/` – GPX traces you can load into other tools

3. **Launch the explorer**
   ```bash
   uv run streamlit run streamlit_app.py
   ```
   Use the "Rebuild processed data" button in the app to refresh after adding new `.slopes` archives.

## Project structure
- `slopes_analysis/ingest.py` – converts `.slopes` exports to Parquet and GPX, computing core metrics.
- `slopes_analysis/analysis.py` – helper functions for yearly trends, leaderboards, and run comparisons.
- `streamlit_app.py` – the interactive dashboard and run comparison UI.

## Notes
- Raw and processed data under `data/` are ignored by Git so you can keep your personal exports locally.
- The converter prefers `RawGPS.csv` from each archive and supplements speed data with values from `Metadata.xml` when available.
