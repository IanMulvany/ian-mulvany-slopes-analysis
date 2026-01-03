"""Data Management UI for Slopes Analysis.

This page provides a WYSIWYG interface for managing slopes data files:
- View processed and unprocessed files
- Add new .slopes files via upload or directory path
- Trigger processing pipeline
- View processing status and logs
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from slopes_analysis import ingest

# Directories
RAW_DIR = Path("data/GPSLogs")
PROCESSED_DIR = Path("data/processed")
SUMMARY_PARQUET = PROCESSED_DIR / "run_summary.parquet"
POINTS_PARQUET = PROCESSED_DIR / "points.parquet"


def get_raw_files() -> list[Path]:
    """Get all .slopes files in the raw directory."""
    if not RAW_DIR.exists():
        return []
    return sorted(RAW_DIR.glob("*.slopes"))


def get_processed_run_ids() -> set[str]:
    """Get the set of run_ids that have been processed."""
    if not SUMMARY_PARQUET.exists():
        return set()
    df = pd.read_parquet(SUMMARY_PARQUET)
    return set(df["file_name"].tolist())


def get_file_status(raw_files: list[Path], processed_files: set[str]) -> pd.DataFrame:
    """Create a DataFrame showing the status of each file."""
    records = []
    for path in raw_files:
        is_processed = path.name in processed_files
        stat = path.stat()
        records.append({
            "File Name": path.name,
            "Status": "Processed" if is_processed else "Pending",
            "Size (KB)": round(stat.st_size / 1024, 1),
            "Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "path": str(path),
        })
    return pd.DataFrame(records)


def render_file_list() -> None:
    """Render the list of all slopes files with their status."""
    st.subheader("Slopes Data Files")

    raw_files = get_raw_files()
    processed_files = get_processed_run_ids()

    if not raw_files:
        st.info(f"No .slopes files found in `{RAW_DIR}`. Add files using the options below.")
        return

    # Summary stats
    total = len(raw_files)
    processed_count = sum(1 for f in raw_files if f.name in processed_files)
    pending_count = total - processed_count

    cols = st.columns(4)
    cols[0].metric("Total Files", total)
    cols[1].metric("Processed", processed_count)
    cols[2].metric("Pending", pending_count)
    cols[3].metric("Directory", str(RAW_DIR))

    # File table
    df = get_file_status(raw_files, processed_files)

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by status",
            options=["All", "Processed", "Pending"],
            key="file_status_filter"
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            options=["File Name", "Status", "Size (KB)", "Modified"],
            key="file_sort"
        )

    # Apply filters
    if status_filter != "All":
        df = df[df["Status"] == status_filter]

    df = df.sort_values(sort_by, ascending=sort_by != "Modified")

    # Display table (without path column)
    st.dataframe(
        df[["File Name", "Status", "Size (KB)", "Modified"]],
        use_container_width=True,
        hide_index=True,
    )

    # Show pending files that need processing
    if pending_count > 0:
        with st.expander(f"View {pending_count} pending files"):
            pending_df = df[df["Status"] == "Pending"]
            for _, row in pending_df.iterrows():
                st.text(f"  - {row['File Name']} ({row['Size (KB)']} KB)")


def render_upload_section() -> None:
    """Render the file upload section."""
    st.subheader("Add New Data")

    tab1, tab2 = st.tabs(["Upload Files", "Add from Directory"])

    with tab1:
        st.markdown("Upload `.slopes` files directly from your computer.")
        uploaded_files = st.file_uploader(
            "Choose .slopes files",
            type=["slopes"],
            accept_multiple_files=True,
            key="slopes_uploader"
        )

        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) selected:**")
            for f in uploaded_files:
                st.text(f"  - {f.name} ({round(f.size / 1024, 1)} KB)")

            if st.button("Upload and Add to Raw Directory", key="upload_btn"):
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                added = 0
                skipped = 0
                for uploaded_file in uploaded_files:
                    dest = RAW_DIR / uploaded_file.name
                    if dest.exists():
                        st.warning(f"Skipped {uploaded_file.name} (already exists)")
                        skipped += 1
                    else:
                        dest.write_bytes(uploaded_file.getvalue())
                        st.success(f"Added {uploaded_file.name}")
                        added += 1

                if added > 0:
                    st.success(f"Added {added} file(s). Click 'Process New Files' to process them.")
                    st.rerun()

    with tab2:
        st.markdown("Point to a directory containing `.slopes` files to copy them.")
        source_dir = st.text_input(
            "Source directory path",
            placeholder="/path/to/your/slopes/exports",
            key="source_dir_input"
        )

        if source_dir:
            source_path = Path(source_dir).expanduser()
            if source_path.exists() and source_path.is_dir():
                slopes_files = list(source_path.glob("*.slopes"))
                if slopes_files:
                    st.success(f"Found {len(slopes_files)} .slopes file(s)")

                    with st.expander("Preview files"):
                        for f in slopes_files[:20]:
                            st.text(f"  - {f.name}")
                        if len(slopes_files) > 20:
                            st.text(f"  ... and {len(slopes_files) - 20} more")

                    if st.button("Copy Files to Raw Directory", key="copy_btn"):
                        RAW_DIR.mkdir(parents=True, exist_ok=True)
                        added = 0
                        skipped = 0
                        for src in slopes_files:
                            dest = RAW_DIR / src.name
                            if dest.exists():
                                skipped += 1
                            else:
                                shutil.copy2(src, dest)
                                added += 1

                        if added > 0:
                            st.success(f"Copied {added} new file(s). Skipped {skipped} existing.")
                            st.rerun()
                        else:
                            st.info("All files already exist in raw directory.")
                else:
                    st.warning("No .slopes files found in this directory.")
            elif source_dir.strip():
                st.error("Directory not found or is not a valid directory.")


def render_processing_section() -> None:
    """Render the processing pipeline controls."""
    st.subheader("Processing Pipeline")

    raw_files = get_raw_files()
    processed_files = get_processed_run_ids()
    pending_count = sum(1 for f in raw_files if f.name not in processed_files)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Process New Files**")
        st.caption(f"{pending_count} file(s) pending processing")

        write_gpx = st.checkbox("Generate GPX files", value=True, key="write_gpx")

        if st.button(
            "Process New Files",
            disabled=pending_count == 0,
            key="process_new_btn",
            type="primary"
        ):
            with st.spinner("Processing .slopes files..."):
                try:
                    summary_df, points_df = ingest.build_datasets(
                        raw_dir=RAW_DIR,
                        output_dir=PROCESSED_DIR,
                        write_gpx=write_gpx
                    )
                    st.success(f"Processed {len(summary_df)} runs ({len(points_df)} GPS points)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing failed: {e}")

    with col2:
        st.markdown("**Rebuild All Data**")
        st.caption("Re-process all files from scratch")

        if st.button("Rebuild All", key="rebuild_all_btn"):
            with st.spinner("Rebuilding all datasets..."):
                try:
                    # Clear existing processed data
                    if SUMMARY_PARQUET.exists():
                        SUMMARY_PARQUET.unlink()
                    if POINTS_PARQUET.exists():
                        POINTS_PARQUET.unlink()

                    summary_df, points_df = ingest.build_datasets(
                        raw_dir=RAW_DIR,
                        output_dir=PROCESSED_DIR,
                        write_gpx=True
                    )
                    st.success(f"Rebuilt {len(summary_df)} runs ({len(points_df)} GPS points)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")


def render_processed_data_info() -> None:
    """Show info about the processed datasets."""
    st.subheader("Processed Data Status")

    cols = st.columns(2)

    with cols[0]:
        st.markdown("**Run Summary**")
        if SUMMARY_PARQUET.exists():
            df = pd.read_parquet(SUMMARY_PARQUET)
            stat = SUMMARY_PARQUET.stat()
            st.metric("Runs", len(df))
            st.caption(f"Size: {round(stat.st_size / 1024, 1)} KB")
            st.caption(f"Updated: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}")

            # Year distribution
            if "year" in df.columns:
                years = df["year"].dropna().astype(int).value_counts().sort_index()
                st.markdown("**Runs by Year:**")
                for year, count in years.items():
                    st.text(f"  {year}: {count} runs")
        else:
            st.info("Not yet created. Process files to generate.")

    with cols[1]:
        st.markdown("**GPS Points**")
        if POINTS_PARQUET.exists():
            stat = POINTS_PARQUET.stat()
            # Don't load the full points file, just show metadata
            st.metric("File Size", f"{round(stat.st_size / (1024 * 1024), 2)} MB")
            st.caption(f"Updated: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}")

            # GPX files
            gpx_dir = PROCESSED_DIR / "gpx"
            if gpx_dir.exists():
                gpx_files = list(gpx_dir.glob("*.gpx"))
                st.metric("GPX Files", len(gpx_files))
        else:
            st.info("Not yet created. Process files to generate.")


def render_pipeline_diagram() -> None:
    """Show a visual representation of the data pipeline."""
    st.subheader("Data Pipeline")

    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │                        DATA PIPELINE                              │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                   │
    │   ┌─────────────┐      ┌─────────────┐      ┌─────────────────┐  │
    │   │   Upload    │      │  .slopes    │      │   Processed     │  │
    │   │   Files     │ ──▶  │  Files      │ ──▶  │   Parquet       │  │
    │   │             │      │  (Raw)      │      │   + GPX         │  │
    │   └─────────────┘      └─────────────┘      └─────────────────┘  │
    │         │                    │                     │             │
    │         ▼                    ▼                     ▼             │
    │   File Upload OR       data/GPSLogs/         data/processed/     │
    │   Directory Copy       *.slopes              run_summary.parquet │
    │                                              points.parquet      │
    │                                              gpx/*.gpx           │
    │                                                                   │
    └──────────────────────────────────────────────────────────────────┘
    ```

    **Pipeline Steps:**
    1. **Add Data**: Upload `.slopes` files or copy from a directory
    2. **Store Raw**: Files are stored in `data/GPSLogs/`
    3. **Process**: Extract GPS data, metadata, compute metrics
    4. **Output**:
       - `run_summary.parquet` - One row per ski day with stats
       - `points.parquet` - All GPS points with computed metrics
       - `gpx/*.gpx` - Standard GPX files for each run
    """)


def main() -> None:
    st.set_page_config(
        page_title="Slopes Data Manager",
        layout="wide",
        page_icon="🎿"
    )

    st.title("Slopes Data Manager")
    st.write("Manage your slopes data files, add new data, and control the processing pipeline.")

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["File Status", "Add Data", "Process", "Pipeline Info"],
        key="nav"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Stats**")
    raw_files = get_raw_files()
    processed_files = get_processed_run_ids()
    st.sidebar.metric("Raw Files", len(raw_files))
    st.sidebar.metric("Processed", len(processed_files))

    # Links
    st.sidebar.markdown("---")
    if st.sidebar.button("Open Main Dashboard"):
        st.switch_page("streamlit_app.py")

    # Main content based on selection
    if page == "File Status":
        render_file_list()
        st.markdown("---")
        render_processed_data_info()

    elif page == "Add Data":
        render_upload_section()

    elif page == "Process":
        render_processing_section()
        st.markdown("---")
        render_processed_data_info()

    elif page == "Pipeline Info":
        render_pipeline_diagram()

    # Footer
    st.markdown("---")
    st.caption(
        f"Raw directory: `{RAW_DIR}` | "
        f"Processed directory: `{PROCESSED_DIR}` | "
        f"Run `streamlit run streamlit_data_manager.py` to use this tool"
    )


if __name__ == "__main__":
    main()
