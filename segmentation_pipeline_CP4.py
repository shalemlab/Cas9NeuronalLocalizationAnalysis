#!/usr/bin/env python3
"""
Complete Cell Segmentation Pipeline for Microscopy Images
Processes OME-TIFF projections, TIFF, or ND2 files and saves segmentation masks efficiently
Runs as SLURM array job with each image as a separate task
"""

import os
import sys
import numpy as np
import pandas as pd
import tifffile
import json
from pathlib import Path
from os import listdir
from os.path import isfile, join
from cellpose import models
import skimage.measure as sm
from nd2 import ND2File #Can comment out if not using ND2 files and ND2 is not installed
import image_functions as imf

# ============================================================================
# CONFIGURATION PARAMETERS - SET THESE BEFORE RUNNING
# ============================================================================

# Input/Output directories
INPUT_DIR = "/mnt/isilon/shalemlab/Personal/Greg/dCas9_Optimization/20250331_NtermNLS_Timecourse/Image_Analysis/MaxIPs"  # Directory with OME-TIFF projections or ND2 files
OUTPUT_DIR = "/mnt/isilon/shalemlab/Personal/Greg/dCas9_Optimization/20250331_NtermNLS_Timecourse/Image_Analysis/Segmented_CP4_updated2"  # Base directory for output (will create segmented/ subdirs)

# Input file format
INPUT_FORMAT = "ome-tiff"  # Options: "ome-tiff", "tiff", "nd2"

# Processing parameters
PROCESS_SPECIFIC_WELLS = None  # List of well numbers to process (None = all)
PROCESS_SPECIFIC_TILES = None  # List of tile numbers to process (None = all)

#Segmentation options
SEGMENT_NUCLEI = True   # If True, segment nuclei
SEGMENT_CELLS = True    # If True, segment cells

# Channel selection for segmentation
NUCLEUS_CHANNEL = 4  # Channel index for nucleus segmentation
CELL_CHANNEL = 3     # Channel index for cell segmentation (or use nucleus for cyto inference), pass as [X,Y] if using multiple channels

PROJECTION_TYPE = "max"  # Z-projection method if not using pre-saved Z-stacks; ignored if image is single Z-plane. ("max", "min", "mean"; avoid "sum" or "subtract" since it changes dtype)
CELL_CHANNEL_COMBO_METHOD = "max"   # Method for projecting multiple cell channels into one channel ("max", "min", "mean"; avoid "sum" or "subtract" since it changes dtype)

# Cellpose segmentation parameters - GLOBAL
USE_GPU = True  # Use GPU acceleration if available

# Cellpose segmentation parameters - NUCLEI
NUCLEUS_DIAMETER = 80  # Expected nucleus diameter in pixels (None = auto-detect)
NUCLEUS_MODEL = "cpsam"  # Options: "cpsam" or custom trained model
NUCLEUS_FLOW_THRESHOLD = 0.4  # Higher = more strict (0.0-1.0)
NUCLEUS_CELLPROB_THRESHOLD = 1  # Higher = more strict (-6 to 6)
NUCLEUS_SHARPEN_RADIUS = 10  # Default is 0; if using, recommneded to be set 1/4 to 1/8 object diameter
NUCLEUS_NORMALIZE = True  # Normalize image intensity
NUCLEUS_PERCENTILE = [1,99.9]  # Percentile for normalization (None = auto[0,99], or [low, high])
NUCLEUS_MIN_SIZE = 500   # Min object size, in pixels, to keep (int; Default is 15)
NUCLEUS_TILE_OVERLAP = 0.5  # Fraction of overlap of tiles when computing flows (Default is 0.1)
NUCLEUS_NITER = None # Default is None - sets proportional to diameter; larger values help with longer objects

# Cellpose segmentation parameters - CELLS
CELL_DIAMETER = 120  # Expected cell diameter in pixels (None = auto-detect)
CELL_MODEL = "cpsam"  # Options: "cpsam" or custom model
CELL_FLOW_THRESHOLD = 0.9
CELL_CELLPROB_THRESHOLD = 0.5
CELL_SHARPEN_RADIUS = 30 # Default is 0; if using, recommneded to be set 1/4 to 1/8 object diameter
CELL_NORMALIZE = True
CELL_PERCENTILE = [0,99]
CELL_MIN_SIZE = 2000   # Min object size, in pixels, to keep (int; Default is 15)
CELL_TILE_OVERLAP = 0.2  # Fraction of overlap of tiles when computing flows (Default is 0.1)
CELL_NITER = 1200  # Default is None - sets proportional to diameter; larger values help with longer objects

USE_NUCLEUS_FOR_CELL_SEG = True  # Use nucleus to guide cell segmentation
USE_NUCLEUS_CHANNEL = True  # If True, use nucleus channel; if False, use nucleus masks

# Post-processing parameters - Removes cells with no/multiple nuclei and edge-touching objects
SAVE_PRE_CLEAN = True  # Save both pre-clean and post-clean masks
EDGE_REMOVAL_MODE = "nuclei"  # Options: "any" (removes mask pair if cell or nucleus is touching edge), "nuclei" (removes mask pair only if nucleus touching edge); defaults to appropriate masks if only segmenting nuclei or cells
EDGE_FRAME_SIZE = 5  # Pixels from edge to consider as "touching edge"

# File naming pattern
# Expected format: Well{n}_Point{tile}_...
# Adjust these indices if your naming convention is different
WELL_INDEX_IN_FILENAME = 0  # Position of well number after splitting by '_'
TILE_INDEX_IN_FILENAME = 2  # Position of tile number after splitting by '_'

# ============================================================================
# SEGMENTATION FUNCTIONS
# ============================================================================

def segment(image, diameter, pretrained_model, gpu, flow_threshold, cellprob_threshold,
                   sharpen_radius, normalize, percentile, min_size, tile_overlap, niter):
    """
    Segment objects using Cellpose.
    """
    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    masks, _, _ = model.eval(
        image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"sharpen_radius": sharpen_radius, "normalize": normalize, "percentile": percentile},
        min_size=min_size,
        tile_overlap=tile_overlap,
        niter=niter
    )
    return masks

def clean_and_label_masks(masks_nucs, masks_cells, edge_mode="nuclei", frame_size=3):
    """
    Clean and relabel segmented nuclei and cells.

    Removes:
    - Cells with multiple nuclei
    - Cells without nuclei
    - Objects touching image edges
    """

    def get_edge_labels(masks, frame_size):
        """Get labels of objects touching image edges."""
        frame_1 = frame_size + 1
        frame_2 = -frame_size
        edges = np.concatenate([
            np.unique(masks[:, :frame_1]),
            np.unique(masks[:, frame_2:]),
            np.unique(masks[:frame_1, :]),
            np.unique(masks[frame_2:, :])
        ])
        return np.unique(edges[edges < 0])

    def relabel_masks(masks_new):
        """Relabel mask with positive sequential labels."""
        unique_labels = np.unique(masks_new[mask_new < 0])
        masks_final = np.zeros(masks_new.shape, dtype=int)
        for i, old_label in enumerate(unique_labels):
            new_label = i + 1
            masks_final[masks_new == old_label] = new_label
        return masks_final

    # Handle case where both masks exist
    if masks_nucs is not None and masks_cells is not None:
        # Extract properties
        props_nucs = sm.regionprops(masks_nucs)
        x_n = np.array([p.centroid[0] for p in props_nucs], dtype=int)
        y_n = np.array([p.centroid[1] for p in props_nucs], dtype=int)

        # Match nuclei to cells
        matches = []
        for i in range(len(props_nucs)):
            cell_label = masks_cells[x_n[i], y_n[i]]
            if cell_label != 0:
                matches.append((i + 1, cell_label, x_n[i], y_n[i]))

        df = pd.DataFrame(matches, columns=['nuc', 'cell', 'I', 'J'])

        # Remove cells with multiple nuclei
        multi_nuc_cells = df.groupby('cell').filter(lambda x: len(x) > 1)
        for _, row in multi_nuc_cells.iterrows():
            masks_nucs[masks_nucs == row['nuc']] = 0

        # Keep only cells with exactly one nucleus
        valid_cells = df.groupby('cell').filter(lambda x: len(x) == 1)

        # Create new masks with temporary negative labels
        nuclei_new = np.zeros(masks_nucs.shape, dtype=int)
        cells_new = np.zeros(masks_cells.shape, dtype=int)

        for i, (_, row) in enumerate(valid_cells.iterrows()):
            temp_label = -(i + 1)
            nuclei_new[masks_nucs == row['nuc']] = temp_label
            cells_new[masks_cells == row['cell']] = temp_label

        # Remove edge-touching objects
        if edge_mode == "any":
            edge_labels = np.unique(np.concatenate([
                get_edge_labels(cells_new, frame_size),
                get_edge_labels(nuclei_new, frame_size)
            ]))
        elif edge_mode == "nuclei":
            edge_labels = get_edge_labels(nuclei_new, frame_size)
        else:
            print(f"  edge_removal mode not valid, must be 'any or 'nuclei'...")

        for label in edge_labels:
            cells_new[cells_new == label] = 0
            nuclei_new[nuclei_new == label] = 0

        # Relabel with positive sequential labels
        return relabel_masks(nuclei_new), relabel_masks(cells_new)

    # Handle single mask cases
    elif masks_nucs is not None:
        masks = masks_nucs
        return_nucs = True
    elif masks_cells is not None:
        masks = masks_cells
        return_nucs = False
    else:
        print(f"  No masks found...")
        return None, None

    # Process single mask
    props = sm.regionprops(masks)
    masks_new = np.zeros(masks.shape, dtype=int)

    for i, prop in enumerate(props):
        temp_label = -(i + 1)
        masks_new[masks == prop.label] = temp_label

    # Remove edge-touching objects
    edge_labels = get_edge_labels(masks_new, frame_size)
    for label in edge_labels:
        masks_new[masks_new == label] = 0

    # Relabel with positive sequential labels
    masks_final = relabel_masks(masks_new)

    return (masks_final, None) if return_nucs else (None, masks_final)

# ============================================================================
# MASK SAVING FUNCTIONS
# ============================================================================

def save_masks(nuc_masks, cell_masks, output_dir, well, filename, compress=True):
    """
    Save segmentation masks as compressed uint16 TIFF files in organized subdirectories.
    """
    nuc_path = None
    cell_path = None
    compression = 'zlib' if compress else None

    if nuc_masks is not None:
        nuc_dir = Path(output_dir) / "nuclei" / f"well_{well}"
        nuc_dir.mkdir(parents=True, exist_ok=True)

        max_label_nuc = nuc_masks.max()
        if max_label_nuc > 65535:
            print(f"  Warning: Max nuclei labels exceed uint16 range ({max_label_nuc})")
            print(f"  Using uint32 instead...")
            nuc_uint = nuc_masks.astype(np.uint32)
        else:
            nuc_uint = nuc_masks.astype(np.uint16)

        nuc_path = str(nuc_dir / f"{filename}_nuclei.tif")
        tifffile.imwrite(nuc_path, nuc_uint, compression=compression)

    if cell_masks is not None:
        cell_dir = Path(output_dir) / "cells" / f"well_{well}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        max_label_cell = cell_masks.max()
        if max_label_cell > 65535:
            print(f"  Warning: Max cell labels exceed uint16 range ({max_label_cell})")
            print(f"  Using uint32 instead...")
            cell_uint = cell_masks.astype(np.uint32)
        else:
            cell_uint = cell_masks.astype(np.uint16)

        cell_path = str(cell_dir / f"{filename}_cells.tif")
        tifffile.imwrite(cell_path, cell_uint, compression=compression)

    return nuc_path, cell_path

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def segment_single_image(well, tile, input_dir, output_dir, input_format):
    """
    Process a single image: load, segment, clean, and save masks.
    """
    print(f"\n{'='*70}")
    print(f"Processing Well {well}, Tile {tile}")
    print(f"{'='*70}")
    
    # Load image without metadata and without image intensity details
    print(f"Loading image (format: {input_format})...")
    image, metadata, filename = imf.import_image_by_well_and_tile(
        well, tile, input_dir,
        verbose=False,
        input_format=input_format,
        return_metadata=False
    )
    
    if image.shape[0] == 0:
        return False
    
    # Extract channels - handle different dimensionalities
    if image.ndim == 2:
        # Single channel 2D
        nuc_channel = image
        cell_channel = image
    elif image.ndim == 3:
        # Multi-channel 2D (CYX)
        nuc_channel = image[NUCLEUS_CHANNEL]
        cell_channel = image[CELL_CHANNEL]
    elif image.ndim == 4:
        # 4D data (e.g., ZCYX from ND2)
        print(f"  Detected 4D data (likely ZCYX format)...")
        # Assume format is ZCYX, take max projection if needed
        if image.shape[0] > 1:  # Multiple Z slices
            print(f"  Performing {PROJECTION_TYPE} projection over {image.shape[0]} Z slices...")
            image = getattr(np, PROJECTION_TYPE)(image, axis=0)  # Project Z dimension -> CYX
        else:
            image = image[0]  # Single Z slice -> CYX
        
        nuc_channel = image[NUCLEUS_CHANNEL]
        cell_channel = image[CELL_CHANNEL]
    else:
        print(f"Error: Unexpected image dimensions: {image.ndim}")
        return False

    # Combine multiple images into single channel for cell segmentation
    if cell_channel.ndim == 3:
        cell_channel = getattr(np, CELL_CHANNEL_COMBO_METHOD)(cell_channel, axis=0)
    
    print(f"  Nucleus channel shape: {nuc_channel.shape}")
    print(f"  Cell channel shape: {cell_channel.shape}")
    
    # Segment nuclei
    if SEGMENT_NUCLEI:
        print("\nSegmenting nuclei...")
        masks_nucs = segment(
            nuc_channel,
            NUCLEUS_DIAMETER,
            NUCLEUS_MODEL,
            USE_GPU,
            NUCLEUS_FLOW_THRESHOLD,
            NUCLEUS_CELLPROB_THRESHOLD,
            NUCLEUS_SHARPEN_RADIUS,
            NUCLEUS_NORMALIZE,
            NUCLEUS_PERCENTILE,
            NUCLEUS_MIN_SIZE,
            NUCLEUS_TILE_OVERLAP,
            NUCLEUS_NITER
        )
        print(f"  Found {masks_nucs.max()} nuclei")
    else:
        masks_nucs = None
        print("\nSkipping nuclei segmentation...")

    # Segment cells
    if SEGMENT_CELLS:
        print("\nSegmenting cells...")

        # Determine what to pass as nucleus information
        if USE_NUCLEUS_FOR_CELL_SEG:
            if USE_NUCLEUS_CHANNEL:
                nucleus_info = nuc_channel  # Use the nucleus image channel
                print(f"  Using nucleus channel to guide cell segmentation")
            elif masks_nuc is None:
                nucleus_info = nuc_channel  # Use the nucleus image channel
                print(f"  No nuclei masks found, using nucleus channel to guide cell segmentation")
            else:
                nucleus_info = masks_nucs  # Use the nucleus masks
                print(f"  Using nucleus masks to guide cell segmentation")

            # Make 2-channel image for cellpose input
            seg_data = np.stack([cell_channel, nucleus_info], axis=0)
        else:
            # Pass single-channel image to cellpose
            seg_data = cell_channel
            print(f"  Segmenting cells without nucleus guidance")

        masks_cells = segment(
            seg_data,
            CELL_DIAMETER,
            CELL_MODEL,
            USE_GPU,
            CELL_FLOW_THRESHOLD,
            CELL_CELLPROB_THRESHOLD,
            CELL_SHARPEN_RADIUS,
            CELL_NORMALIZE,
            CELL_PERCENTILE,
            CELL_MIN_SIZE,
            CELL_TILE_OVERLAP,
            CELL_NITER
        )
        print(f"  Found {masks_cells.max()} cells")

    else:
        masks_cells = None
        print("\nSkipping cell segmentation...")

    # Clean and label

    print("\nCleaning and relabeling...")
    nucs_clean, cells_clean = clean_and_label_masks(
        masks_nucs,
        masks_cells,
        EDGE_REMOVAL_MODE,
        EDGE_FRAME_SIZE
    )

    if SEGMENT_NUCLEI:
        print(f"  Before cleaning: {masks_nucs.max()} nuclei, After cleaning: {nucs_clean.max()} nuclei ")
        print(f"  Removed: {masks_nucs.max() - nucs_clean.max()} nuclei")

    if SEGMENT_CELLS:
        print(f"  Before cleaning: {masks_cells.max()} cells, After cleaning {cells_clean.max()} cells")
        print(f"  Removed: {masks_cells.max() - cells_clean.max()} cells")

    # Save masks (with pre-clean if requested)
    if SAVE_PRE_CLEAN:
        # Stack pre-clean and post-clean versions
        if SEGMENT_NUCLEI:
            nucs_out = np.stack([masks_nucs.astype(np.uint16), nucs_clean.astype(np.uint16)])
        else:
            nucs_out = None
        if SEGMENT_CELLS:
            cells_out = np.stack([masks_cells.astype(np.uint16), cells_clean.astype(np.uint16)])
        else:
            cells_out = None

    else:
        nucs_out = nucs_clean
        cells_out = cells_clean

    # Save masks
    print("\nSaving masks...")
    nuc_path, cell_path = save_masks(
        nucs_out, cells_out, output_dir, well, filename, compress=True
    )

    # Get file sizes
    if SEGMENT_NUCLEI:
        nuc_size_mb = os.path.getsize(nuc_path) / (1024 * 1024)
        print(f"  Nuclei mask: {nuc_path} ({nuc_size_mb:.2f} MB)")

    if SEGMENT_CELLS:
        cell_size_mb = os.path.getsize(cell_path) / (1024 * 1024)
        print(f"  Cell mask: {cell_path} ({cell_size_mb:.2f} MB)")

    if SEGMENT_CELLS and SEGMENT_NUCLEI:
        print(f"  Total: {nuc_size_mb + cell_size_mb:.2f} MB")

    print(f"\n✓ Successfully processed Well {well}, Tile {tile}")
    return True

# ============================================================================
# MAIN EXECUTION - SLURM ARRAY JOB
# ============================================================================

def main():
    """
    Main function for SLURM array job execution.
    Each task processes one image based on SLURM_ARRAY_TASK_ID.
    """
    # Get SLURM array task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    print(f"\n{'='*70}")
    print("Cell Segmentation Pipeline")
    print(f"SLURM Array Task ID: {task_id}")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Input directory: {INPUT_DIR}")
    print(f"  Input format: {INPUT_FORMAT}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  GPU enabled: {USE_GPU}")
    if SEGMENT_NUCLEI:
        print(f"\n  Nuclei paramters:")
        print(f"    Nucleus channel: {NUCLEUS_CHANNEL}, nucleus model: {NUCLEUS_MODEL}, diameter: {NUCLEUS_DIAMETER},"
              f"    flow threshold: {NUCLEUS_FLOW_THRESHOLD}, cellprob threshold: {NUCLEUS_CELLPROB_THRESHOLD},"
              f"    sharpen radius: {NUCLEUS_SHARPEN_RADIUS}, normalize: {NUCLEUS_NORMALIZE}, percentile: {NUCLEUS_PERCENTILE},"
              f"    min size: {NUCLEUS_MIN_SIZE}, tile overlap: {NUCLEUS_TILE_OVERLAP}, niter: {NUCLEUS_NITER}")
    if SEGMENT_CELLS:
        print(f"\n  Cell paramters:")
        print(f"    Cell channel: {CELL_CHANNEL}, cell model: {CELL_MODEL}, diameter: {CELL_DIAMETER},"
              f"    flow threshold: {CELL_FLOW_THRESHOLD}, cellprob threshold: {CELL_CELLPROB_THRESHOLD},"
              f"    sharpen radius: {CELL_SHARPEN_RADIUS}, normalize: {CELL_NORMALIZE}, percentile: {CELL_PERCENTILE},"
              f"    min size: {CELL_MIN_SIZE}, tile overlap: {CELL_TILE_OVERLAP}, niter: {CELL_NITER}")
        print(f"    Use nucleus for cell seg: {USE_NUCLEUS_FOR_CELL_SEG}")
    if USE_NUCLEUS_FOR_CELL_SEG:
        print(f"    Use nucleus channel: {USE_NUCLEUS_CHANNEL}")
    print(f"  Edge removal: {EDGE_REMOVAL_MODE}")
    print(f"  Save pre-clean: {SAVE_PRE_CLEAN}")
    print()
    
    # Get all well/tile combinations
    all_combinations = imf.get_all_well_tile_combinations(
        INPUT_DIR,
        INPUT_FORMAT,
        well_index=WELL_INDEX_IN_FILENAME,
        tile_index=TILE_INDEX_IN_FILENAME
    )

    # Filter by specific wells/tiles if specified
    if PROCESS_SPECIFIC_WELLS is not None:
        all_combinations = [(w, t) for w, t in all_combinations if w in PROCESS_SPECIFIC_WELLS]
    if PROCESS_SPECIFIC_TILES is not None:
        all_combinations = [(w, t) for w, t in all_combinations if t in PROCESS_SPECIFIC_TILES]
    
    print(f"Total images found: {len(all_combinations)}")
    
    # Check if task_id is valid
    if task_id >= len(all_combinations):
        print(f"ERROR: Task ID {task_id} exceeds number of images ({len(all_combinations)})")
        sys.exit(1)
    
    # Get the well and tile for this task
    well, tile = all_combinations[task_id]
    
    print(f"\nProcessing image {task_id + 1}/{len(all_combinations)}")
    print(f"Well: {well}, Tile: {tile}\n")
    
    # Process the image
    try:
        success = segment_single_image(
            well, tile, 
            INPUT_DIR, 
            OUTPUT_DIR,
            INPUT_FORMAT
        )
        
        if success:
            print(f"\n{'='*70}")
            print(f"SUCCESS: Task {task_id} completed")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"WARNING: Task {task_id} - image not found or skipped")
            print(f"{'='*70}\n")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Task {task_id} failed")
        print(f"Error message: {str(e)}")
        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
