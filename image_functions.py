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
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
import json
from pathlib import Path
from os import listdir
from os.path import isfile, join
import skimage.measure as sm
from nd2 import ND2File

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_image_details(image):
    """
    Display image details when loading.
    """
    if image.ndim > 2:
        # Get number of channels
        n_channels = image.shape[-3]
        for i in range(n_channels):
            if image.ndim > 3:
                # Multi-stack, multi-channel image
                print(f"Channel[{i}] stacked value range: [{np.min(image[...,i,:,:])}, {np.max(image[...,i,:,:])}]")
            elif image.ndim == 3:
                # Multi-channel image
                print(f"Channel[{i}] value range: [{image[i].min()}, {image[i].max()}]")
    else:
        # Single channel image
        print(f"Image value range: [{image.min()}, {image.max()}]")


def get_all_well_tile_combinations(path_name, input_format, well_index=0, tile_index=2):
    """
    Get all unique well and tile combinations from files in directory.
    """
    files = [f for f in listdir(path_name) if isfile(join(path_name, f))]
    combinations = []

    # Determine file extension to look for
    if input_format.lower() == "nd2":
        extensions = ['.nd2']
    elif input_format.lower() in ["tiff", "ome-tiff"]:
        extensions = ['.tif', '.ome.tif']
    else:
        extensions = ['.tif', '.ome.tif', '.nd2']

    for filename in files:
        if any(filename.endswith(ext) for ext in extensions):
            try:
                parts = filename.split('_')
                n_tile = int(parts[tile_index])
                n_well = int(parts[well_index].split('Well')[-1])
                if (n_well, n_tile) not in combinations:
                    combinations.append((n_well, n_tile))
            except (IndexError, ValueError):
                continue

    return sorted(combinations)


def import_image_by_well_and_tile(well, tile, path_name, input_format="ome-tiff",
                                   verbose=True, return_metadata=False, metadata_dir=None,
                                   well_index=0, tile_index=2):
    """
    Import image files (ND2, TIFF, or OME-TIFF) by tile and well number.
    
    Parameters:
    -----------
    well : int
        Well number
    tile : int
        Tile number
    path_name : str
        Path to directory containing image files
    input_format : str
        Format of input files: "nd2", "tiff", or "ome-tiff"
    verbose : bool
        Print image and/or metadata details (if return_metadata=True)
    return_metadata : bool
        Return metadata from JSON file (only for TIFF/OME-TIFF)
    metadata_dir : str or None
        Path to metadata directory
    well_index : int
        Position of well number in filename split by '_'
    tile_index : int
        Position of tile number in filename split by '_'
        
    Returns:
    --------
    If return_metadata=False: (image, filename)
    If return_metadata=True: (image, metadata, filename)
    """
    image = np.empty([0])
    metadata = None
    filename = None
    files = [f for f in listdir(path_name) if isfile(join(path_name, f))]
    
    # Determine file extensions based on format
    if input_format.lower() == "nd2":
        extensions = ['.nd2']
    elif input_format.lower() in ["tiff", "ome-tiff"]:
        extensions = ['.tif', '.ome.tif']
    else:
        raise ValueError(f"Unknown input format: {input_format}")
    
    # Search for matching file
    for i in range(len(files)):
        if any(files[i].endswith(ext) for ext in extensions):
            try:
                parts = files[i].split('_')
                n_well = int(parts[well_index].split('Well')[-1])
                n_tile = int(parts[tile_index])
            except (IndexError, ValueError):
                continue
                
            if n_well == well and n_tile == tile:
                full_path = join(path_name, files[i])
                filename = files[i].split('.')[0]

                # Load based on format
                if input_format.lower() == "nd2":
                    with ND2File(full_path) as nd2:
                        image = nd2.asarray()
                else:
                    image = tifffile.imread(full_path)

                print(f"\nLoaded: {filename}")
                print(f"Shape: {image.shape}")
                print(f"Image data type: {image.dtype}")
                if verbose:
                    print_image_details(image)
                
                # Load metadata if requested
                if return_metadata:
                    if filename.split('_')[-1] == "projection":
                        base_name = '_'.join(filename.split('_')[0:-2])
                    else:
                        base_name = filename

                    if metadata_dir is not None:
                        json_path = join(metadata_dir, base_name + '_metadata.json')
                    else:
                        json_path = join(path_name, base_name + '_metadata.json')
                    
                    try:
                        with open(json_path, 'r') as f:
                            metadata = json.load(f)
                    except FileNotFoundError:
                        print(f"Metadata file not found: {json_path}")
                        metadata = {}
                    if verbose:
                        print(f"\nMetadata:")
                        if 'stage_positions_um' in metadata:
                            print(f"  Stage position: {metadata['stage_positions_um']}")
                        if 'pixel_size_um' in metadata:
                            print(f"  Pixel size: {metadata['pixel_size_um']}")
                        if 'channels' in metadata:
                            print(f"  Channels: {metadata['channels']}")

                break

    if image.shape[0] == 0:
        print(f"ERROR: Image not found for Well {well}, Tile {tile}")

    return image, metadata, filename



def load_masks(well, tile, mask_dir, use_cleaned=True, tile_index=2):
    """Load nucleus and cell masks for a specific well and tile."""
    nuc_dir = Path(mask_dir) / "nuclei" / f"well_{well}"
    cell_dir = Path(mask_dir) / "cells" / f"well_{well}"

    masks_nucs = None
    masks_cells = None

    # Load nucleus masks
    if nuc_dir.exists():
        for nuc_file in nuc_dir.glob("*.tif"):
            try:
                parts = nuc_file.stem.split('_')
                file_tile = int(parts[tile_index])
                if file_tile == tile:
                    nuc_data = tifffile.imread(str(nuc_file))
                    if nuc_data.ndim == 3:  # Stacked pre/post clean
                        masks_nucs = nuc_data[1 if use_cleaned else 0]
                    else:
                        masks_nucs = nuc_data
                    break
            except (IndexError, ValueError):
                continue

    # Load cell masks
    if cell_dir.exists():
        for cell_file in cell_dir.glob("*.tif"):
            try:
                parts = cell_file.stem.split('_')
                file_tile = int(parts[tile_index])
                if file_tile == tile:
                    cell_data = tifffile.imread(str(cell_file))
                    if cell_data.ndim == 3:  # Stacked pre/post clean
                        masks_cells = cell_data[1 if use_cleaned else 0]
                    else:
                        masks_cells = cell_data
                    break
            except (IndexError, ValueError):
                continue

    return masks_nucs, masks_cells

# ============================================================================
# BASIC IMAGE PROCESSING FUNCTIONS
# ============================================================================

def project_image(image, projection_type):
    projection = getattr(np, projection_type)(image, axis=0)
    if projection.ndim > 3:
        projection = getattr(np, projection_type)(projection, axis=0)
    print("\nProjected image values:")
    print_image_details(projection)
    return projection


def norm_to_zero(image):
    normed_to_zero = image - image.min(axis=(-2,-1), keepdims=True)
    print("\nNormed_to_zero image values:")
    print_image_details(normed_to_zero)
    return normed_to_zero


def norm(image):
    normed = (image - image.min(axis=(-2,-1), keepdims=True))/(image.max(axis=(-2,-1), keepdims=True) - image.min(axis=(-2,-1), keepdims=True))
    print("\nNormed image values:")
    print_image_details(normed)
    return normed


def invert(image):
    inverted = (1 - norm(image))
    print("\nInverted image values:")
    print_image_details(inverted)
    return inverted

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_image(image, title="Image", channel_names=None, cmaps='gray', percentile=[0,100], projection_type="max"):
    """
    Visualize multi-channel or single-channel image.
    
    Parameters:
    -----------
    data : np.ndarray
        Image data (CYX or YX)
    title : str
        Main title
    channels : list
        List of channel names (optional)
    cmap : str or list
        Colormap(s) to use
    """
    
    if image.ndim > 2:
        # Multi-channel image
        n_channels = image.shape[-3]
    else:
        n_channels == 1

    if image.ndim > 3:
        image = project_image(image, projection_type)
        print(f"Displaying {projection_type} projection")
        
    fig, axes = plt.subplots(2, n_channels, figsize=(n_channels*5, 10))
    
    if n_channels == 1:
        axes = [axes] 
    
    for i in range(n_channels):
        p_min=np.percentile(image[i], percentile[0])
        p_max=np.percentile(image[i], percentile[1])
        
        im = axes[0, i].imshow(image[i], cmap=cmaps if isinstance(cmaps, str) else cmaps[i], vmin=p_min, vmax=p_max)
        channel_name = channel_names[i] if channel_names and i < len(channel_names) else f"Channel {i}"
        axes[0, i].set_title(channel_name)
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.04)

        axes[1, i].hist(image[i].ravel(), bins=int(p_max-p_min), range=(p_min, p_max))
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show


def visualize_segmentation(image, NUCLEUS_CHANNEL, CELL_CHANNEL, masks_nuc, masks_cells, figsize=(20, 5)):
    """
    Visualize original image with segmentation masks overlaid.
    
    Parameters:
    -----------
    image : np.ndarray
        Original image (CYX or YX)
    masks_nuc : np.ndarray
        Nuclei masks
    masks_cells : np.ndarray
        Cell masks
    channel : int
        Which channel to display (if multi-channel)
    """
    cell_channel = image[CELL_CHANNEL]
    nuc_channel = image[NUCLEUS_CHANNEL]
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original image
    axes[0].imshow(nuc_channel + cell_channel)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Nuclei masks
    np.random.seed(42)  # For reproducibility
    colors_nuc = np.random.rand(masks_nuc.max() + 1, 3)
    colors_nuc[0] = [0, 0, 0]  # Background is black
    axes[1].imshow(colors_nuc[masks_nuc], alpha=0.8)
    axes[1].imshow(nuc_channel, cmap='gray', alpha=0.6)
    axes[1].set_title(f'Nuclei Masks ({masks_nuc.max()} nuclei)')
    axes[1].axis('off')
    
    # Cell masks
    np.random.seed(42)  # For reproducibility
    colors_cells = np.random.rand(masks_cells.max() + 1, 3)
    colors_cells[0] = [0, 0, 0]  # Background is black
    axes[2].imshow(colors_cells[masks_cells], alpha=0.8)
    axes[2].imshow(cell_channel, cmap='gray', alpha=0.6)
    axes[2].set_title(f'Cell Masks ({masks_cells.max()} cells)')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(colors_nuc[masks_nuc], alpha=1)
    axes[3].imshow(colors_cells[masks_cells], alpha=0.6)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Segmentation Summary:")
    print(f"  Total nuclei detected: {masks_nuc.max()}")
    print(f"  Total cells detected: {masks_cells.max()}")
    
# ============================================================================
# PHENOTYPING POST-PROCESSING FUNCTIONS
# ============================================================================

def concat_phenotyping(phenotype_dir, output_file=None,
                       filter_wells=None, filter_groups=None,
                       verbose=True):
    """
    Load all phenotyping CSV files and concatenate into single DataFrame.

    Parameters:
    -----------
    phenotype_dir : str or Path
        Directory containing individual phenotyping CSV files
    output_file : str or Path, optional
        Path to save concatenated CSV (if None, doesn't save)
    filter_wells : list, optional
        List of well numbers to include (None = all wells)
    filter_groups : list, optional
        List of experimental groups to include (None = all groups)
    verbose : bool
        Print progress messages

    Returns:
    --------
    pandas.DataFrame
        Combined phenotyping data
    """
    phenotype_dir = Path(phenotype_dir)

    # Find all CSV files
    csv_files = sorted(phenotype_dir.glob("Well*_Tile*_measurements.csv"))

    if len(csv_files) == 0:
        print(f"ERROR: No CSV files found in {phenotype_dir}")
        return None

    if verbose:
        print(f"Found {len(csv_files)} CSV files")

    # Load and concatenate all files
    dfs = []
    failed_files = []

    for csv_file in tqdm(csv_files):
        try:
            df = pd.read_csv(csv_file)

            # Apply filters if specified
            if filter_wells is not None:
                df = df[df['well'].isin(filter_wells)]

            if filter_groups is not None and 'experimental_group' in df.columns:
                df = df[df['experimental_group'].isin(filter_groups)]

            if len(df) > 0:  # Only add if rows remain after filtering
                dfs.append(df)

        except Exception as e:
            failed_files.append((csv_file.name, str(e)))
            if verbose:
                print(f"  WARNING: Failed to load {csv_file.name}: {e}")

    if len(dfs) == 0:
        print("ERROR: No data loaded successfully")
        return None

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"\nSuccessfully loaded {len(dfs)}/{len(csv_files)} files")
        if failed_files:
            print(f"Failed files: {len(failed_files)}")
            for fname, error in failed_files:
                print(f"  - {fname}: {error}")

        print(f"\nCombined dataset:")
        print(f"  Total cells: {len(combined_df):,}")
        print(f"  Total features: {len(combined_df.columns)}")
        print(f"  Wells: {sorted([int(w) for w in combined_df['well'].unique()])}")
        print(f"  Tiles per well: {combined_df.groupby('well')['tile'].nunique().to_dict()}")

        if 'experimental_group' in combined_df.columns:
            print(f"\nCells per experimental group:")
            group_counts = combined_df['experimental_group'].value_counts().sort_index()
            for group, count in group_counts.items():
                print(f"  {group}: {count:,} cells")

    # Save if output file specified
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)

        if verbose:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\nSaved combined data to: {output_path}")
            print(f"  File size: {file_size_mb:.2f} MB")

    return combined_df


def filter_by_quality(df, min_nucleus_area=None, max_nucleus_area=None,
                      min_cell_area=None, max_cell_area=None,
                      min_nuc_to_cell_ratio=None, max_nuc_to_cell_ratio=None,
                      remove_edge_cells=False, edge_buffer=50):
    """
    Filter cells based on quality criteria.

    Parameters:
    -----------
    df : pandas.DataFrame
        Combined phenotyping data
    min_nucleus_area : float, optional
        Minimum nucleus area
    max_nucleus_area : float, optional
        Maximum nucleus area
    min_cell_area : float, optional
        Minimum cell area
    max_cell_area : float, optional
        Maximum cell area
    min_nuc_to_cell_ratio : float, optional
        Minimum nucleus/cell area ratio
    max_nuc_to_cell_ratio : float, optional
        Maximum nucleus/cell area ratio
    remove_edge_cells : bool
        Remove cells near image edges
    edge_buffer : int
        Pixels from edge to consider as "edge"

    Returns:
    --------
    pandas.DataFrame
        Filtered dataset
    """
    df_filtered = df.copy()
    n_original = len(df_filtered)

    # Filter by nucleus area
    if min_nucleus_area is not None and 'nucleus_area' in df.columns:
        df_filtered = df_filtered[df_filtered['nucleus_area'] >= min_nucleus_area]
    if max_nucleus_area is not None and 'nucleus_area' in df.columns:
        df_filtered = df_filtered[df_filtered['nucleus_area'] <= max_nucleus_area]

    # Filter by cell area
    if min_cell_area is not None and 'cell_area' in df.columns:
        df_filtered = df_filtered[df_filtered['cell_area'] >= min_cell_area]
    if max_cell_area is not None and 'cell_area' in df.columns:
        df_filtered = df_filtered[df_filtered['cell_area'] <= max_cell_area]

    # Filter by nucleus/cell ratio
    if min_nuc_to_cell_ratio is not None and 'nucleus_to_cell_area_ratio' in df.columns:
        df_filtered = df_filtered[df_filtered['nucleus_to_cell_area_ratio'] >= min_nuc_to_cell_ratio]
    if max_nuc_to_cell_ratio is not None and 'nucleus_to_cell_area_ratio' in df.columns:
        df_filtered = df_filtered[df_filtered['nucleus_to_cell_area_ratio'] <= max_nuc_to_cell_ratio]

    # Remove edge cells
    if remove_edge_cells:
        # Assuming image size - you may need to adjust these values
        if 'nucleus_centroid_x' in df.columns and 'nucleus_centroid_y' in df.columns:
            max_x = df_filtered['nucleus_centroid_x'].max()
            max_y = df_filtered['nucleus_centroid_y'].max()

            df_filtered = df_filtered[
                (df_filtered['nucleus_centroid_x'] > edge_buffer) &
                (df_filtered['nucleus_centroid_x'] < max_x - edge_buffer) &
                (df_filtered['nucleus_centroid_y'] > edge_buffer) &
                (df_filtered['nucleus_centroid_y'] < max_y - edge_buffer)
                ]

    n_filtered = len(df_filtered)
    print(f"Filtered {n_original - n_filtered:,} cells ({100 * (n_original - n_filtered) / n_original:.1f}%)")
    print(f"Remaining: {n_filtered:,} cells")

    return df_filtered


def summarize_dataset(df, verbose=True):
    """
    Print summary statistics of the phenotyping dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        Combined phenotyping data
    verbose : bool
        Print detailed summary
    """
    if verbose:
        print("=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)

        print(f"\nDimensions:")
        print(f"  Rows (cells): {len(df):,}")
        print(f"  Columns (features): {len(df.columns)}")

        print(f"\nWells:")
        print(f"  Number of wells: {df['well'].nunique()}")
        print(f"  Well IDs: {sorted([int(w) for w in df['well'].unique()])}")

        print(f"\nTiles:")
        print(f"  Total unique tiles: {df['tile'].nunique()}")
        print(f"  Tiles per well:")
        for well, tiles in df.groupby('well')['tile'].nunique().items():
            print(f"    Well {well}: {tiles} tiles")

        if 'experimental_group' in df.columns:
            print(f"\nExperimental Groups:")
            for group in sorted(df['experimental_group'].dropna().unique()):
                n_cells = (df['experimental_group'] == group).sum()
                n_wells = df[df['experimental_group'] == group]['well'].nunique()
                print(f"  {group}: {n_cells:,} cells from {n_wells} wells")

        print(f"\nMeasurement Categories:")
        nucleus_cols = [c for c in df.columns if c.startswith('nucleus_')]
        cell_cols = [c for c in df.columns if c.startswith('cell_')]
        cyto_cols = [c for c in df.columns if c.startswith('cytoplasm_')]
        ratio_cols = [c for c in df.columns if 'ratio' in c.lower()]

        print(f"  Nucleus features: {len(nucleus_cols)}")
        print(f"  Cell features: {len(cell_cols)}")
        print(f"  Cytoplasm features: {len(cyto_cols)}")
        print(f"  Ratio features: {len(ratio_cols)}")

        print(f"\nData completeness:")
        print(
            f"  Missing values: {df.isnull().sum().sum():,} ({100 * df.isnull().sum().sum() / (len(df) * len(df.columns)):.2f}%)")

        # Check for common issues
        if 'nucleus_label' in df.columns and 'cell_label' in df.columns:
            matched = (df['nucleus_label'] == df['cell_label']).sum()
            print(f"\nMatching labels:")
            print(f"  Nucleus-Cell matches: {matched}/{len(df)} ({100 * matched / len(df):.1f}%)")