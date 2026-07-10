#!/usr/bin/env python3
"""
Cell Phenotyping Pipeline for Microscopy Images
Extracts quantitative measurements from segmented cells
Runs as SLURM array job with each image as a separate task
"""

import os
import sys
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from os import listdir
from os.path import isfile, join
import skimage.measure as sm
from scipy import ndimage as ndi
import image_functions as imf

# ============================================================================
# CONFIGURATION PARAMETERS - SET THESE BEFORE RUNNING
# ============================================================================

# Input directories
IMAGE_DIR = "/mnt/isilon/shalemlab/Data/Greg/ShalemLab_Microscope/20250422_dCas9_NtermNLSs/20250422_153817_830"  # Original images
MASK_DIR = "/mnt/isilon/shalemlab/Personal/Greg/dCas9_Optimization/20250331_NtermNLS_Timecourse/Image_Analysis/Segmented_CP4_updated2"  # Segmentation masks
OUTPUT_DIR = "/mnt/isilon/shalemlab/Personal/Greg/dCas9_Optimization/20250331_NtermNLS_Timecourse/Image_Analysis/Phenotyping/cyto_sub_mean"  # Output directory for measurements

# Input file format
IMAGE_FORMAT = "nd2"  # Options: "ome-tiff", "tiff", "nd2"

# Processing parameters
PROCESS_SPECIFIC_WELLS = None  # List of well numbers to process (None = all)
PROCESS_SPECIFIC_TILES = None  # List of tile numbers to process (None = all)

# Image processing
PROJECTION_TYPE = "mean"
NORM_TO_ZERO = True # If True, subtract minimum pixel value for each channel across image

# Mask options
USE_CLEANED_MASKS = True  # If True, use cleaned masks (index 1); if False, use raw masks (index 0)
MEASURE_NUCLEI = True  # Extract nuclear measurements
MEASURE_CELLS = True  # Extract cell measurements

# Channel names and indices for measurement
# Format: [(channel_index, channel_name), ...]
MEASUREMENT_CHANNELS = [
    (1, "dCas9-KRAB"),
    (3, "MAP2"),
    (4, "NucSpot")
]

# Measurements to extract
MEASURE_INTENSITY = True  # Mean, median, sum, std intensity
MEASURE_MORPHOLOGY = True  # Area, perimeter, eccentricity, etc.
MEASURE_TEXTURE = True  # Intensity variance, range
MEASURE_SPATIAL = True  # Centroid, bounding box

# Advanced measurements
MEASURE_CYTOPLASM = True  # Calculate cytoplasm-specific measurements (requires both nuclei and cells)
CYTOPLASM_METHOD = "subtract"  # Options: "subtract" (cell - nucleus), "ring" (perinuclear ring around nucleus)
CYTOPLASM_RING_FACTOR = 1  # For "ring" method: factor to multiply nucleus major axis length (e.g., 0.5 = half the nucleus size, 1.0 = same as nucleus size)

# Experimental group assignment (optional)
# Assign experimental conditions to wells
# Format: [(wells, group_name), ...] where wells can be:
#   - Tuple of individual wells: ((1,6,11), "Control")
#   - Range using slice notation: (range(1,5), "Control") for wells 1-4
# Set to None to skip group assignment
EXPERIMENTAL_GROUPS = [
    ((1, 7, 13), "Parental"),
    ((2, 8, 14), "noNLS"),
    ((3, 9, 15), "NpNLS+2xSV40NLS"),
    ((4, 10, 16), "2xMycNLS"),
    ((5, 11, 17), "2xMeCP2NLS")
]
# Example with ranges:
# EXPERIMENTAL_GROUPS = [
#     (range(1, 5), "Control"),      # Wells 1-4
#     (range(5, 9), "Group1"),       # Wells 5-8
#     (range(9, 13), "Group2"),      # Wells 9-12
# ]

# File naming pattern (must match segmentation pipeline)
WELL_INDEX_IN_FILENAME = 0
TILE_INDEX_IN_FILENAME = 2

# ============================================================================
# MEASUREMENT FUNCTIONS
# ============================================================================

def get_experimental_group(well, experimental_groups):
    """
    Get experimental group name for a given well.

    Parameters:
    -----------
    well : int
        Well number
    experimental_groups : list of tuples or None
        List of (wells, group_name) tuples where wells can be:
        - Tuple of well numbers: ((1,6,11), "Control")
        - Range object: (range(1,5), "Control")

    Returns:
    --------
    str or None
        Group name if well is assigned to a group, None otherwise
    """
    if experimental_groups is None:
        return None

    for wells, group_name in experimental_groups:
        if well in wells:
            return group_name

    return None


def measure_region_intensity(image, mask, label):
    """Extract intensity measurements for a single region."""
    region_pixels = image[mask == label]
    
    measurements = {
        'mean_intensity': np.mean(region_pixels),
        'median_intensity': np.median(region_pixels),
        'sum_intensity': np.sum(region_pixels),
        'std_intensity': np.std(region_pixels),
        'min_intensity': np.min(region_pixels),
        'max_intensity': np.max(region_pixels),
    }
    
    return measurements


def measure_region_morphology(props):
    """Extract morphological measurements from regionprops."""
    measurements = {
        'area': props.area,
        'perimeter': props.perimeter,
        'eccentricity': props.eccentricity,
        'solidity': props.solidity,
        'extent': props.extent,
        'major_axis_length': props.major_axis_length,
        'minor_axis_length': props.minor_axis_length,
        'orientation': props.orientation,
    }
    
    return measurements


def measure_region_spatial(props):
    """Extract spatial measurements from regionprops."""
    centroid = props.centroid
    bbox = props.bbox
    
    measurements = {
        'centroid_y': centroid[0],
        'centroid_x': centroid[1],
        'bbox_min_y': bbox[0],
        'bbox_min_x': bbox[1],
        'bbox_max_y': bbox[2],
        'bbox_max_x': bbox[3],
    }
    
    return measurements


def measure_region_texture(image, mask, label):
    """Extract texture measurements for a single region."""
    region_pixels = image[mask == label]
    
    measurements = {
        'intensity_range': np.ptp(region_pixels),  # Peak-to-peak (max - min)
        'intensity_variance': np.var(region_pixels),
        'intensity_cv': np.std(region_pixels) / np.mean(region_pixels) if np.mean(region_pixels) > 0 else 0,  # Coefficient of variation
    }
    
    return measurements


def create_cytoplasm_mask(cell_mask, nucleus_mask, nucleus_props, method="subtract", ring_factor=0.5):
    """
    Create cytoplasm mask from cell and nucleus masks.
    
    Parameters:
    -----------
    cell_mask : numpy.ndarray
        Binary or labeled mask of the cell
    nucleus_mask : numpy.ndarray
        Binary or labeled mask of the nucleus
    nucleus_props : regionprops object
        Region properties of the nucleus (for accessing major_axis_length)
    method : str
        Method to create cytoplasm mask ("subtract" or "ring")
    ring_factor : float
        For "ring" method: factor to multiply nucleus major axis length to determine ring width
        
    Returns:
    --------
    numpy.ndarray
        Cytoplasm mask
    """
    if method == "subtract":
        # Simple subtraction: cytoplasm = cell - nucleus
        cytoplasm = cell_mask.copy()
        cytoplasm[nucleus_mask > 0] = 0
        
    elif method == "ring":
        # Ring method: create perinuclear ring around nucleus.
        # Dilation is performed on a cropped bounding-box region rather than
        # the full image to keep memory usage low.
        from scipy.ndimage import binary_dilation

        # Calculate ring width based on average of nucleus axes lengths
        ring_pixels = int((nucleus_props.major_axis_length + nucleus_props.minor_axis_length)/2  * ring_factor)
        ring_pixels = max(1, ring_pixels)

        # --- crop to a padded bounding box around the nucleus ---
        min_row, min_col, max_row, max_col = nucleus_props.bbox  # (row_min, col_min, row_max, col_max)
        img_h, img_w = nucleus_mask.shape

        # Pad the bounding box by ring_pixels on every side so the dilation
        # has room to expand without hitting a hard edge
        pad = ring_pixels
        r0 = max(0, min_row - pad)
        r1 = min(img_h, max_row + pad)
        c0 = max(0, min_col - pad)
        c1 = min(img_w, max_col + pad)

        # Extract small crops — these are the only arrays passed to binary_dilation
        nuc_crop  = (nucleus_mask[r0:r1, c0:c1] > 0)
        cell_crop = (cell_mask[r0:r1, c0:c1] > 0)

        # Dilate within the crop
        struct = np.ones((ring_pixels * 2 + 1, ring_pixels * 2 + 1))
        dilated_crop = binary_dilation(nuc_crop, structure=struct)

        # Ring = dilated nucleus minus original nucleus, clipped to cell
        ring_crop = dilated_crop & ~nuc_crop & cell_crop

        # Paste result back into a full-size output mask
        cytoplasm = np.zeros_like(cell_mask)
        cell_crop_labels = cell_mask[r0:r1, c0:c1]
        cytoplasm[r0:r1, c0:c1][ring_crop] = cell_crop_labels[ring_crop]
    
    return cytoplasm

# ============================================================================
# MAIN EXECUTION - SLURM ARRAY JOB
# ============================================================================

def phenotype_single_image(well, tile, image_dir, mask_dir, image_format, 
                          use_cleaned_masks, measurement_channels, projection_type=None, experimental_groups=None):
    """
    Extract phenotypic measurements from a single image.
    
    Returns:
        pandas DataFrame with measurements
    """
    print(f"\n{'='*70}")
    print(f"Phenotyping Well {well}, Tile {tile}")
    print(f"{'='*70}")
    
    # Get experimental group
    experimental_group = get_experimental_group(well, experimental_groups)
    if experimental_group:
        print(f"Experimental group: {experimental_group}")
    
    # Load image without metadata and with image intensity details
    print(f"Loading image (format: {image_format})...")
    image, metadata, filename = imf.import_image_by_well_and_tile(
        well, tile, image_dir,
        verbose=True,
        input_format=image_format,
        return_metadata=False
    )
    
    if image.shape[0] == 0:
        return None
    
    # Project image if Z-stack
    if image.ndim > 3:
        image = imf.project_image(image, projection_type)
    
    # Normalize image channels to 0
    if NORM_TO_ZERO:
        image = imf.norm_to_zero(image)

    # Load masks
    print("Loading masks...")
    masks_nucs, masks_cells = imf.load_masks(well, tile, mask_dir, use_cleaned=use_cleaned_masks, tile_index=TILE_INDEX_IN_FILENAME)
    
    if masks_nucs is None and masks_cells is None:
        print(f"ERROR: No masks found for Well {well}, Tile {tile}")
        return None
    
    if MEASURE_NUCLEI and masks_nucs is not None:
        print(f"  Loaded {masks_nucs.max()} nuclei")
    if MEASURE_CELLS and masks_cells is not None:
        print(f"  Loaded {masks_cells.max()} cells")
    
    # Get unique labels (assuming nuclei and cells have matching labels from segmentation pipeline)
    if masks_nucs is not None and masks_cells is not None:
        # Use labels that exist in both masks
        nuc_labels = set(np.unique(masks_nucs))
        cell_labels = set(np.unique(masks_cells))
        nuc_labels.discard(0)  # Remove background
        cell_labels.discard(0)
        common_labels = sorted(nuc_labels & cell_labels)
        print(f"  Found {len(common_labels)} matched cells with both nucleus and cell masks")
    elif masks_nucs is not None:
        common_labels = sorted([l for l in np.unique(masks_nucs) if l > 0])
        print(f"  Found {len(common_labels)} nuclei")
    elif masks_cells is not None:
        common_labels = sorted([l for l in np.unique(masks_cells) if l > 0])
        print(f"  Found {len(common_labels)} cells")
    else:
        common_labels = []
    
    # Initialize results list
    results = []
    
    # Get region properties once for efficiency
    if MEASURE_NUCLEI and masks_nucs is not None:
        props_nuc_dict = {prop.label: prop for prop in sm.regionprops(masks_nucs)}
    else:
        props_nuc_dict = {}
    
    if MEASURE_CELLS and masks_cells is not None:
        props_cells_dict = {prop.label: prop for prop in sm.regionprops(masks_cells)}
    else:
        props_cells_dict = {}
    
    # Process each cell
    print(f"\nMeasuring {len(common_labels)} cells...")
    for label in common_labels:
        measurement = {
            'well': well,
            'tile': tile,
            'label': label,
            'cell_id': f"{well}_{tile}_{label}",
        }
        
        # Add experimental group if available
        if experimental_group:
            measurement['experimental_group'] = experimental_group
        
        # ====================================================================
        # SECTION 1: NUCLEUS MEASUREMENTS
        # ====================================================================
        if MEASURE_NUCLEI and label in props_nuc_dict:
            measurement['nucleus_label'] = label
            prop_nuc = props_nuc_dict[label]

            # Morphology measurements
            if MEASURE_MORPHOLOGY:
                nuc_morph = measure_region_morphology(prop_nuc)
                for key, value in nuc_morph.items():
                    measurement[f'nucleus_{key}'] = value

            # Spatial measurements
            if MEASURE_SPATIAL:
                nuc_spatial = measure_region_spatial(prop_nuc)
                for key, value in nuc_spatial.items():
                    measurement[f'nucleus_{key}'] = value

            # Measure each channel
            for ch_idx, ch_name in measurement_channels:
                if ch_idx < image.shape[0]:
                    channel_image = image[ch_idx]
                    
                    if MEASURE_INTENSITY:
                        nuc_intensity = measure_region_intensity(channel_image, masks_nucs, label)
                        for key, value in nuc_intensity.items():
                            measurement[f'nucleus_{ch_name}_{key}'] = value
                    
                    if MEASURE_TEXTURE:
                        nuc_texture = measure_region_texture(channel_image, masks_nucs, label)
                        for key, value in nuc_texture.items():
                            measurement[f'nucleus_{ch_name}_{key}'] = value
        
        # ====================================================================
        # SECTION 2: CELL MEASUREMENTS
        # ====================================================================
        if MEASURE_CELLS and label in props_cells_dict:
            measurement['cell_label'] = label
            prop_cell = props_cells_dict[label]
            # Morphology measurements

            if MEASURE_MORPHOLOGY:
                cell_morph = measure_region_morphology(prop_cell)
                for key, value in cell_morph.items():
                    measurement[f'cell_{key}'] = value

            # Nuclear to cell area ratio
            if MEASURE_MORPHOLOGY and label in props_nuc_dict:
                measurement['nucleus_to_cell_area_ratio'] = props_nuc_dict[label].area / prop_cell.area

            # Spatial measurements
            if MEASURE_SPATIAL:
                cell_spatial = measure_region_spatial(prop_cell)
                for key, value in cell_spatial.items():
                    measurement[f'cell_{key}'] = value

            # Measure each channel
            for ch_idx, ch_name in measurement_channels:
                if ch_idx < image.shape[0]:
                    channel_image = image[ch_idx]
                    
                    if MEASURE_INTENSITY:
                        cell_intensity = measure_region_intensity(channel_image, masks_cells, label)
                        for key, value in cell_intensity.items():
                            measurement[f'cell_{ch_name}_{key}'] = value
                    
                    if MEASURE_TEXTURE:
                        cell_texture = measure_region_texture(channel_image, masks_cells, label)
                        for key, value in cell_texture.items():
                            measurement[f'cell_{ch_name}_{key}'] = value

        # ====================================================================
        # SECTION 3: CYTOPLASM MEASUREMENTS
        # ====================================================================
        if MEASURE_CYTOPLASM and label in props_nuc_dict and label in props_cells_dict:
            # Create cytoplasm mask for this cell
            cyto_mask = create_cytoplasm_mask(
                masks_cells == label,
                masks_nucs == label,
                props_nuc_dict[label],  # Pass nucleus properties
                method=CYTOPLASM_METHOD,
                ring_factor=CYTOPLASM_RING_FACTOR
            )
            
            if np.any(cyto_mask):
                # Create labeled mask (label = 1 for this cytoplasm region)
                cyto_mask_labeled = (cyto_mask > 0).astype(int)
                
                # Calculate cytoplasm area
                cytoplasm_area = np.sum(cyto_mask_labeled)
                measurement['cytoplasm_area'] = cytoplasm_area

                # Check for inconsistent cell/nucleus mask overlaps
                if CYTOPLASM_METHOD == "subtract" and 'cell_area' in measurement and 'nucleus_area' in measurement:
                    measurement['cytoplasm_area_mismatch'] = cytoplasm_area - (measurement['cell_area'] - measurement['nucleus_area'])
                
                # Measure each channel
                for ch_idx, ch_name in measurement_channels:
                    if ch_idx < image.shape[0]:
                        channel_image = image[ch_idx]
                        
                        if MEASURE_INTENSITY:
                            cyto_intensity = measure_region_intensity(channel_image, cyto_mask_labeled, 1)
                            for key, value in cyto_intensity.items():
                                measurement[f'cytoplasm_{ch_name}_{key}'] = value
                        
                        if MEASURE_TEXTURE:
                            cyto_texture = measure_region_texture(channel_image, cyto_mask_labeled, 1)
                            for key, value in cyto_texture.items():
                                measurement[f'cytoplasm_{ch_name}_{key}'] = value
                        
                        # Nuclear to cytoplasmic ratio
                        if f'cytoplasm_{ch_name}_mean_intensity' in measurement and f'nucleus_{ch_name}_mean_intensity' in measurement:
                            if measurement[f'cytoplasm_{ch_name}_mean_intensity'] > 0:
                                measurement[f'{ch_name}_nuc_to_cyto_ratio'] = (
                                    measurement[f'nucleus_{ch_name}_mean_intensity'] / 
                                    measurement[f'cytoplasm_{ch_name}_mean_intensity']
                                )
        
        results.append(measurement)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    print(f"\nExtracted measurements for {len(df)} cells")
    print(f"  Total features: {len(df.columns)}")
    
    return df


# ============================================================================
# MAIN EXECUTION - SLURM ARRAY JOB
# ============================================================================

def main():
    """
    Main function for SLURM array job execution.
    Each task processes one well/tile combination.
    """
    # Get SLURM array task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    print(f"\n{'='*70}")
    print("Cell Phenotyping Pipeline")
    print(f"SLURM Array Task ID: {task_id}")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Image directory: {IMAGE_DIR}")
    print(f"  Mask directory: {MASK_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Image format: {IMAGE_FORMAT}")
    print(f"  Norm to zero: {NORM_TO_ZERO}")
    print(f"  Use cleaned masks: {USE_CLEANED_MASKS}")
    print(f"  Measure nuclei: {MEASURE_NUCLEI}")
    print(f"  Measure cells: {MEASURE_CELLS}")
    print(f"  Measure cytoplasm: {MEASURE_CYTOPLASM}")
    if MEASURE_CYTOPLASM:
        print(f"    Cytoplasm method: {CYTOPLASM_METHOD}")
        if CYTOPLASM_METHOD == "ring":
            print(f"    Ring factor: {CYTOPLASM_RING_FACTOR} × nucleus major axis length")
    print(f"  Measurement channels: {MEASUREMENT_CHANNELS}")
    if EXPERIMENTAL_GROUPS:
        print(f"  Experimental groups defined: {len(EXPERIMENTAL_GROUPS)} groups")
    print()
    
    # Get all well/tile combinations
    all_combinations = imf.get_all_well_tile_combinations(
        IMAGE_DIR,
        IMAGE_FORMAT,
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
    
    # Phenotype the image
    try:
        df = phenotype_single_image(
            well, tile,
            IMAGE_DIR,
            MASK_DIR,
            IMAGE_FORMAT,
            USE_CLEANED_MASKS,
            MEASUREMENT_CHANNELS,
            PROJECTION_TYPE,
            EXPERIMENTAL_GROUPS
        )
        
        if df is not None and len(df) > 0:
            # Save measurements
            output_file = output_dir / f"Well{well}_Tile{tile}_measurements.csv"
            df.to_csv(output_file, index=False)
            
            print(f"\nSaved measurements to: {output_file}")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            
            print(f"\n{'='*70}")
            print(f"SUCCESS: Task {task_id} completed")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"WARNING: Task {task_id} - no cells measured")
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
