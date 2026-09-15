# Microscopy Image Analysis Pipeline

Pipeline for processing confocal microscopy images (ND2 format) through Z-projection,
Cellpose-based cell/nucleus segmentation, per-cell phenotyping, and downstream QC/plotting.
Designed to run on a SLURM cluster, with one array task per image.

## Pipeline Overview

```
 .nd2 raw images
        |
        v
[1] nd2_processing.py  ───────────────►  Z-projected OME-TIFFs + JSON metadata
        |
        v
[2] segmentation_pipeline_CP4.py  ────►  Nucleus & cell masks (.tif, per well/tile)
        |                                (uses image_functions.py)
        v
[3] Segmentation_QC.ipynb  ───────────►  Visual QC of masks overlaid on images
        |
        v
[4] phenotyping.py  ───────────────────►  Per-cell measurements (.csv, per well/tile)
        |                                 (uses image_functions.py)
        v
[5] Phenotyping_cytosub_mean_QC.ipynb ─►  Combined, filtered, normalized cell-level dataset
        |
        v
[6] Plots_cytosub_mean.ipynb  ────────►  Group-level stats & summary plots
        |
[7] Image_figs.ipynb  ────────────────►  Publication figure panels (independent branch,
                                          uses raw images + masks directly)
```

Each numbered `.py` script has a matching `.sh` SLURM submission script. Notebooks are run
interactively (locally or on a cluster Jupyter session) after their corresponding array jobs finish.

---

## File-by-File Description

### 1. Z-stack projection — `nd2_processing.py` / `nd2_processing.sh`
Reads raw `.nd2` confocal files and collapses the Z-stack into a single 2D image per channel
(max/mean/min/median/sum projection, configurable). Extracts acquisition metadata (pixel size,
channel names, stage positions) and saves it alongside the projection.

- **Input:** `.nd2` files in `INPUT_DIR`
- **Output:** OME-TIFF (or TIFF) projections in `OUTPUT_DIR`, plus a `*_metadata.json` sidecar
  per image in `METADATA_DIR`
- **Key config:** `PROJECTION_TYPE`, `Z_SLICE_START/END/STEP`, `OUTPUT_FORMAT`, `CHANNELS_TO_PROCESS`
- **SLURM:** one array task per `.nd2` file found in `INPUT_DIR` (task ID indexes into the sorted file list)

### 2. Segmentation — `segmentation_pipeline_CP4.py` / `segmentation_pipeline.sh`
Runs Cellpose (model `cpsam`) to segment nuclei and/or whole cells from the projected images
(or directly from ND2 if preferred). Cell segmentation can be guided by the nucleus channel or
nucleus masks. Post-processing removes cells with zero or multiple nuclei and any object touching
the image edge, then relabels masks sequentially.

- **Input:** OME-TIFF projections (or `.nd2`) in `INPUT_DIR`
- **Output:** compressed uint16 TIFF masks, organized as:
  ```
  OUTPUT_DIR/
    nuclei/well_{n}/{filename}_nuclei.tif
    cells/well_{n}/{filename}_cells.tif
  ```
  If `SAVE_PRE_CLEAN=True`, each mask file is a 2-frame stack: `[0]` = raw Cellpose output,
  `[1]` = cleaned/relabeled masks.
- **Key config:** `NUCLEUS_CHANNEL`/`CELL_CHANNEL`, diameter/flow/cellprob thresholds per object
  type, `USE_NUCLEUS_FOR_CELL_SEG`, `EDGE_REMOVAL_MODE`
- **Depends on:** `image_functions.py`
- **SLURM:** one array task per (well, tile) combination detected in `INPUT_DIR`; requires GPU partition

### 3. Segmentation QC — `Segmentation_QC.ipynb`
Interactive notebook for spot-checking segmentation quality: loads an image and its masks for a
given well/tile, converts masks to outlines (`cellpose.utils`), and overlays them on the raw
channels to visually confirm nucleus/cell boundaries look correct before running phenotyping.

### 4. Phenotyping — `phenotyping.py` / `phenotyping.sh`
Loads each image and its corresponding masks, then extracts per-cell quantitative features:
intensity (mean/median/sum/std/min/max), morphology (area, perimeter, eccentricity, solidity...),
texture (range, variance, CV), and spatial (centroid, bounding box) — for the nucleus, the whole
cell, and a derived **cytoplasm** region (cell minus nucleus, or a perinuclear ring). Also computes
nuclear:cytoplasmic intensity ratios and assigns an experimental group label based on well number.

- **Input:** raw/projected images (`IMAGE_DIR`, `IMAGE_FORMAT`) + masks from step 2 (`MASK_DIR`)
- **Output:** one CSV per well/tile in `OUTPUT_DIR`, named `Well{n}_Tile{t}_measurements.csv`
- **Key config:** `MEASUREMENT_CHANNELS` (which channels to quantify and their names),
  `CYTOPLASM_METHOD` (`subtract` or `ring`), `EXPERIMENTAL_GROUPS` (well → condition mapping)
- **Depends on:** `image_functions.py`
- **SLURM:** one array task per (well, tile) combination

### 5. Phenotyping QC — `Phenotyping_cytosub_mean_QC.ipynb`
Concatenates all per-tile measurement CSVs from step 4 into one dataframe, then runs a filtering/
QC pass: evaluates mask quality, filters cells by area and other criteria, checks nuclear and cell
staining for correlations/artifacts, filters again to remove nuclear-staining artifacts, and
normalizes intensities to the minimum measured values. Saves the final filtered, per-cell dataset
for use in plotting.

### 6. Group-level plots — `Plots_cytosub_mean.ipynb`
Loads the filtered per-cell dataset from step 5, computes derived ratios and per-well/per-group
summary statistics (`group_stats_mean.csv`), and generates the main summary plots (e.g. dCas9-KRAB
nuclear vs. cytoplasmic intensity, normalized to the `Parental` control group).

### 7. Figure panels — `Image_figs.ipynb`
Builds representative image panels for figures: loads a raw image + masks for a chosen
well/tile per experimental condition, overlays mask outlines, crops the region of interest, and
assembles multi-image panels. Configured for publication-ready output (embeds Arial font, keeps
text editable in exported SVGs). Can be run independently of steps 4–6 as long as steps 1–2 are done.

### Shared utilities — `image_functions.py`
Helper functions imported by `segmentation_pipeline_CP4.py`, `phenotyping.py`, and the notebooks:
- `get_all_well_tile_combinations` / `import_image_by_well_and_tile` — locate and load images by
  well/tile number, for `.nd2`, `.tif`, or `.ome.tif` inputs
- `load_masks` — load nucleus/cell masks for a given well/tile (handles pre/post-clean stacks)
- `combine_phenotype_files` — concatenate all per-tile phenotyping CSVs into one dataframe, with
  optional filtering by well or experimental group
- `filter_by_quality` — filter cells by nucleus/cell area, nucleus:cell ratio, or edge proximity
- `summarize_dataset` / `print_image_details` — print summary stats for QC in notebooks

---

## File Naming Convention

All scripts assume raw filenames follow `Well{n}_Point{tile}_...` (underscore-delimited), e.g.
`Well3_Point12_...nd2`. The position of the well and tile tokens is configurable via
`WELL_INDEX_IN_FILENAME` / `TILE_INDEX_IN_FILENAME` at the top of each script — update these if
your microscope's naming convention differs.

## Environment

All scripts expect a conda environment (referred to as `cellpose` in the `.sh` files) with:
`cellpose` (cpsam model), `nd2`, `tifffile`, `numpy`, `pandas`, `scikit-image`, `scipy`,
`matplotlib`, `tqdm`.

## Running on SLURM

Each `.py` script has a **CONFIGURATION PARAMETERS** block at the top — edit `INPUT_DIR`,
`OUTPUT_DIR`, and processing options there before submitting. Each `.sh` script wraps its script
as a SLURM array job (one task per image or well/tile combination):

```bash
sbatch nd2_processing.sh          # step 1 — set --array to (n_files - 1)
sbatch segmentation_pipeline.sh   # step 2 — set --array to (n_well_tile_combos - 1), needs GPU
sbatch phenotyping.sh             # step 4 — set --array to (n_well_tile_combos - 1)
```

`--array`, `--time`, `--mem`, and `--partition` in each `.sh` file are placeholders — adjust to
match your dataset size and cluster configuration. After steps 1–2 (or 1–4) finish, run the
corresponding notebook(s) interactively.

## Output Directory Structure (end-to-end example)

```
Image_Analysis/
├── MaxIPs/                          # step 1 output: projected OME-TIFFs
├── Metadata/                        # step 1 output: per-image JSON metadata
├── Segmented_CP4_updated2/          # step 2 output
│   ├── nuclei/well_{n}/*_nuclei.tif
│   └── cells/well_{n}/*_cells.tif
└── Phenotyping/
    └── cyto_sub_mean/
        ├── Well{n}_Tile{t}_measurements.csv   # step 4 output (per tile)
        ├── <filtered combined dataset>.csv    # step 5 output
        └── group_stats_mean.csv               # step 6 output
```
