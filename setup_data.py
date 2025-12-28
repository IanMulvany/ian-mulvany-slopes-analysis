#!/usr/bin/env python
"""Setup script to unzip GPSLogs.zip and process .slopes files.

This script:
1. Creates data/GPSLogs directory if needed
2. Unzips GPSLogs.zip into data/GPSLogs
3. Processes all .slopes files to create analysis datasets
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from slopes_analysis import ingest


def setup_data(
    zip_path: Path = Path("GPSLogs.zip"),
    raw_dir: Path = Path("data/GPSLogs"),
    output_dir: Path = Path("data/processed"),
    write_gpx: bool = True,
    force_unzip: bool = False,
) -> None:
    """Set up data directories and process .slopes files.
    
    Parameters
    ----------
    zip_path : Path
        Path to the GPSLogs.zip file
    raw_dir : Path
        Directory where .slopes files will be extracted
    output_dir : Path
        Directory where processed data will be written
    write_gpx : bool
        Whether to also write GPX files
    force_unzip : bool
        If True, unzip even if raw_dir already contains files
    """
    # Create raw directory if it doesn't exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    def _collect_slopes(directory: Path) -> list[Path]:
        # Collect .slopes files, including nested "GPSLogs/" folder
        candidates = list(directory.glob("*.slopes"))
        nested = directory / "GPSLogs"
        if nested.exists() and nested.is_dir():
            candidates.extend(nested.glob("*.slopes"))
        # Also search one level deeper to be safe
        candidates.extend(directory.rglob("*.slopes"))
        # Deduplicate
        return list({p.resolve() for p in candidates})

    def _flatten_nested(directory: Path) -> None:
        """Move nested GPSLogs/*.slopes up to raw_dir and clean __MACOSX."""
        nested = directory / "GPSLogs"
        if nested.exists() and nested.is_dir():
            for f in nested.glob("*.slopes"):
                dest = directory / f.name
                if not dest.exists():
                    f.rename(dest)
        # Remove __MACOSX artifacts (recursively)
        macosx = directory / "__MACOSX"
        if macosx.exists():
            import shutil

            shutil.rmtree(macosx, ignore_errors=True)

    # Check if we need to unzip
    existing_slopes_files = _collect_slopes(raw_dir)
    
    if not existing_slopes_files or force_unzip:
        if not zip_path.exists():
            raise FileNotFoundError(
                f"GPSLogs.zip not found at {zip_path.absolute()}. "
                "Please ensure GPSLogs.zip is in the project root directory."
            )
        
        print(f"Unzipping {zip_path} to {raw_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        print(f"✓ Extracted files to {raw_dir}")
        _flatten_nested(raw_dir)
        existing_slopes_files = _collect_slopes(raw_dir)
    else:
        print(f"✓ Found {len(existing_slopes_files)} existing .slopes files in {raw_dir}")
        print("  (Skipping unzip. Use force_unzip=True to re-extract)")
    
    # Count .slopes files
    slopes_files = _collect_slopes(raw_dir)
    if not slopes_files:
        raise ValueError(
            f"No .slopes files found in {raw_dir}. "
            "Please check that GPSLogs.zip contains .slopes files."
        )
    
    print(f"\nProcessing {len(slopes_files)} .slopes files...")
    print(f"Output directory: {output_dir}")
    
    # Process the files
    try:
        summary_df, points_df = ingest.build_datasets(
            raw_dir=raw_dir,
            output_dir=output_dir,
            write_gpx=write_gpx,
        )
        
        print(f"\n✓ Successfully processed {len(summary_df)} runs")
        print(f"  Summary table: {output_dir / 'run_summary.parquet'} ({len(summary_df)} rows)")
        print(f"  Points table:  {output_dir / 'points.parquet'} ({len(points_df)} points)")
        if write_gpx:
            gpx_dir = output_dir / "gpx"
            gpx_count = len(list(gpx_dir.glob("*.gpx"))) if gpx_dir.exists() else 0
            print(f"  GPX files:    {gpx_dir} ({gpx_count} files)")
        
    except Exception as e:
        print(f"\n✗ Error processing files: {e}")
        raise


def main() -> None:
    """Main entry point for the setup script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Set up Slopes data by unzipping GPSLogs.zip and processing .slopes files"
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("GPSLogs.zip"),
        help="Path to GPSLogs.zip file (default: GPSLogs.zip)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/GPSLogs"),
        help="Directory to extract .slopes files to (default: data/GPSLogs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed data (default: data/processed)",
    )
    parser.add_argument(
        "--no-gpx",
        action="store_true",
        help="Don't write GPX files",
    )
    parser.add_argument(
        "--force-unzip",
        action="store_true",
        help="Force re-extraction even if files already exist",
    )
    
    args = parser.parse_args()
    
    setup_data(
        zip_path=args.zip_path,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        write_gpx=not args.no_gpx,
        force_unzip=args.force_unzip,
    )


if __name__ == "__main__":
    main()

