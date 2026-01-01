# Segment Gluing Workflow

This document explains how to use the segment gluing tools to combine ski run segments that were incorrectly split by the automatic segmentation algorithm.

## Why Gluing?

The automatic segmentation algorithm splits runs when:
- There's a time gap > 90 seconds between GPS points
- Speed drops below 5 km/h or altitude stops decreasing

Sometimes these splits are incorrect - you may have taken a brief break mid-run, waited at a lift line, or the GPS had a brief dropout. The gluing workflow lets you review these splits and merge segments that should be one continuous run.

## Annotations Directory

All annotation files (glue decisions and outlier markings) are stored in `data/annotated/`. The system automatically reads all JSON files from this directory and merges them, keeping the latest decision for each segment based on timestamps.

**Supported files:**
- `glue_decisions*.json` - Segment merge decisions (approved/rejected)
- `ski_outliers*.json` - Segments marked as outliers (walks, drives, etc.)

You can download multiple versions over time (e.g., `glue_decisions.json`, `glue_decisions-2.json`) and the system will deduplicate them automatically.

## Quick Start

```bash
# 1. Generate the glue viewer (loads existing annotations from data/annotated/)
python generate_glue_viewer.py

# 2. Open in your browser
open segment_glue_viewer.html

# 3. Review candidates, approve/reject merges, or mark chains as outliers
# 4. Download decisions/outliers JSON from the viewer

# 5. Save downloaded files to data/annotated/
mv ~/Downloads/glue_decisions.json data/annotated/

# 6. Regenerate all viewers to include new annotations
python generate_viewer.py
python generate_apps.py
python generate_glue_viewer.py
```

## Workflow Steps

### Step 1: Generate the Viewer

Run the generator script to create a standalone HTML file with your segment data:

```bash
python generate_glue_viewer.py
```

Options:
- `--output`: Output path (default: `segment_glue_viewer.html`)
- `--annotated-dir`: Path to annotations directory (default: `data/annotated`)
- `--time-threshold`: Max time gap in minutes to consider (default: 30)
- `--spatial-threshold`: Max spatial gap in meters (default: 200)

The viewer automatically loads all glue decisions and outliers from the annotations directory.

### Step 2: Review Candidates

Open `segment_glue_viewer.html` in your browser. The viewer has four modes:

#### Candidates View
Auto-detected pairs of segments that might need gluing. Each card shows:
- **Location and date** of the segments
- **Confidence score** (higher = more likely to need gluing)
- **Time gap** between segments
- **Spatial gap** (distance from end of first to start of second)
- **Segment stats** (distance, vertical drop)

The map shows both segments with the gap highlighted as a dashed line.

Actions:
- **Approve**: Mark this pair to be merged
- **Reject**: Mark this pair to NOT be merged (won't show in future)

#### Chains View
When multiple consecutive segments can be merged (A→B→C), they appear as chains. You can approve the entire chain at once.

#### Manual View
Select any segments from the same day to manually glue together, even if they weren't detected as candidates.

1. Click segments to select them (they'll be highlighted in purple)
2. Click "Glue Selected Segments" when you've selected 2+ segments
3. The merge is added to your approved list

#### Approved View
Review all your approved merges. You can undo any decision from here.

### Step 3: Export Decisions

Click "Download Decisions JSON" at the bottom of the viewer. This saves a file containing all your approve/reject decisions.

You can also download outlier markings using "Download Outliers JSON".

```json
{
  "version": 1,
  "created": "2024-01-15T10:30:00Z",
  "decisions": [
    {
      "glue_id": "glue_abc123__seg5__abc123__seg6",
      "segment_ids": ["abc123__seg5", "abc123__seg6"],
      "status": "approved",
      "notes": ""
    }
  ]
}
```

**Important:** Save these files to `data/annotated/`. You can use versioned filenames:
- `data/annotated/glue_decisions.json` (first export)
- `data/annotated/glue_decisions-2.json` (second export)
- `data/annotated/ski_outliers_from_glue.json` (outliers from glue viewer)

The system will automatically merge and deduplicate all files in this directory.

### Step 4: Apply Decisions

Run the persistence module to create merged segment data:

```bash
python -m slopes_analysis.glue_persistence apply --decisions data/glue_decisions.json
```

This will:
1. Load your original segments and GPS points
2. Apply all "approved" merge decisions
3. Recalculate statistics for merged segments
4. Save new parquet files with merged data

Output files:
- `data/processed/segments_merged.parquet` - Merged segment summaries
- `data/processed/labeled_points_merged.parquet` - GPS points with updated segment IDs

### Step 5: Use Merged Data

To use the merged data in analysis, load the `_merged` parquet files:

```python
import pandas as pd

segments_df = pd.read_parquet("data/processed/segments_merged.parquet")
points_df = pd.read_parquet("data/processed/labeled_points_merged.parquet")
```

## Understanding Candidates

### How Candidates Are Detected

Two segments are candidates for gluing if:

1. **Same run**: They belong to the same day's skiing session (same `day_run_id`)
2. **Consecutive**: They have adjacent segment numbers (seg5 → seg6)
3. **Close in time**: Gap between end of A and start of B is < 30 minutes
4. **Close in space**: Distance from end of A to start of B is < 200 meters

### Confidence Score

The confidence score (0-100%) indicates how likely the segments should be merged:

- **High (70-100%)**: Small time and spatial gaps - very likely should be merged
- **Medium (40-70%)**: Moderate gaps - review the map to decide
- **Low (0-40%)**: Large gaps - less likely to need merging

Confidence formula: `40% * (1 - time_gap/max_time) + 60% * (1 - distance_gap/max_distance)`

### Edge Cases

- **Chair lifts**: If you stopped for a chair lift, the segments shouldn't be merged (different runs)
- **GPS dropouts**: Brief GPS dropouts might create artificial splits - these should be merged
- **Trail crossings**: If you stopped at a trail intersection, consider whether it's one run or two

## File Locations

| File | Purpose |
|------|---------|
| `generate_glue_viewer.py` | Script to generate the HTML viewer |
| `segment_glue_viewer.html` | Generated interactive viewer |
| `slopes_analysis/glue_candidates.py` | Candidate detection algorithm |
| `slopes_analysis/glue_persistence.py` | Save/load decisions, apply merges |
| `slopes_analysis/annotations.py` | Load/merge/dedup all annotations |
| `data/annotated/` | Directory for all annotation JSON files |
| `data/annotated/glue_decisions*.json` | Glue decisions (downloaded from viewer) |
| `data/annotated/ski_outliers*.json` | Outlier markings (downloaded from viewer) |
| `data/processed/segments_merged.parquet` | Output: merged segment summaries |
| `data/processed/labeled_points_merged.parquet` | Output: merged GPS points |

## API Reference

### annotations.py

```python
from slopes_analysis.annotations import (
    load_all_glue_decisions,
    load_all_outliers,
    get_outlier_segment_ids,
    get_outliers_dict,
    get_approved_glues,
    get_segments_to_merge,
    get_annotation_summary,
)

# Load all glue decisions (deduped, latest version of each)
decisions = load_all_glue_decisions(Path("data/annotated"))

# Load all outlier markings
outliers = load_all_outliers(Path("data/annotated"))

# Get set of segment IDs marked as outliers
outlier_ids = get_outlier_segment_ids()

# Get outliers as a dict keyed by segment ID
outliers_dict = get_outliers_dict()

# Get only approved glue decisions
approved = get_approved_glues()

# Get segment groups to merge (deduplicated chains)
merge_groups = get_segments_to_merge()

# Get summary statistics
summary = get_annotation_summary()
# Returns: {"total_glue_decisions": 272, "glue_by_status": {"approved": 256, ...}, ...}
```

### glue_candidates.py

```python
from slopes_analysis.glue_candidates import find_glue_candidates, find_chain_candidates

# Find candidate pairs
candidates = find_glue_candidates(
    segments_df,
    labeled_points_df,
    time_threshold_min=30.0,  # Max time gap in minutes
    spatial_threshold_m=200.0  # Max spatial gap in meters
)

# Find chains of 3+ segments
chains = find_chain_candidates(segments_df, labeled_points_df)
```

### glue_persistence.py

```python
from slopes_analysis.glue_persistence import (
    load_glue_decisions,
    save_glue_decisions,
    apply_glue_decisions,
    save_merged_data
)

# Load decisions from JSON
decisions = load_glue_decisions(Path("data/glue_decisions.json"))

# Apply to segment data
merged_segments, merged_points = apply_glue_decisions(
    segments_df, labeled_points_df, decisions
)

# Save merged data
save_merged_data(merged_segments, merged_points, Path("data/processed"))
```

## Troubleshooting

### "No candidates found"
- Your segments may be too far apart in time (> 30 min) or space (> 200m)
- Adjust thresholds when generating: `--time-threshold 60 --spatial-threshold 500`

### "Viewer shows no data"
- Ensure you have processed parquet files in `data/processed/`
- Run `python -c "from slopes_analysis import analysis; analysis.load_datasets()"` to regenerate

### "Apply command does nothing"
- Check that your JSON file has "approved" decisions (not all "rejected")
- Run `python -m slopes_analysis.glue_persistence status` to see decision counts

### Decisions not persisting in viewer
- The viewer uses localStorage - check that JavaScript is enabled
- Decisions are also saved when you click "Download Decisions JSON"
