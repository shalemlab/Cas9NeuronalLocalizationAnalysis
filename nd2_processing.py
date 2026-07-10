#!/usr/bin/env python3
"""
Z-Stack Projection Pipeline for Confocal Microscopy Images
Processes ND2 files with SLURM array job parallelization
"""

import os
import sys
import numpy as np
from pathlib import Path
from os import listdir
from os.path import isfile, join
from nd2 import ND2File
import tifffile
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION PARAMETERS - SET THESE BEFORE RUNNING
# ============================================================================

# Input/Output directories
INPUT_DIR = "/mnt/isilon/shalemlab/Data/Greg/ShalemLab_Microscope/20250422_dCas9_NtermNLSs/20250422_153817_830"
OUTPUT_DIR = "/mnt/isilon/shalemlab/Personal/Greg/dCas9_Optimization/20250331_NtermNLS_Timecourse/Image_Analysis/MaxIPs"
METADATA_DIR = "/mnt/isilon/shalemlab/Personal/Greg/dCas9_Optimization/20250331_NtermNLS_Timecourse/Image_Analysis/Metadata" # Directory for JSON metadata files (None = same as OUTPUT_DIR)

# Processing parameters
PROCESS_SPECIFIC_WELLS = None  # List of well numbers to process (None = all)
PROCESS_SPECIFIC_TILES = None  # List of tile numbers to process (None = all)

# Processing mode
METADATA_ONLY = False  # If True, only extract and save metadata (no projection); if False, perform projection and save metadata

# Projection parameters
PROJECTION_TYPE = "max"  # Options: "max", "mean", "min", "median", "sum"
Z_SLICE_START = None     # Start slice index (None = first slice, 0-indexed)
Z_SLICE_END = None       # End slice index (None = last slice, exclusive)
Z_SLICE_STEP = 1         # Step size for slice selection

# Output format
OUTPUT_FORMAT = "ome.tiff"   # Options: "tiff", "ome.tiff"
COMPRESSION = None      # Options: "lzw", "zlib", None
SAVE_METADATA_JSON = True  # Save metadata as JSON sidecar file (recommended for OME-TIFF)

# File naming pattern (for parsing well and tile numbers from filenames)
# Expected format: Well{n}_Point{tile}_...
# Adjust these indices if your naming convention is different
WELL_INDEX_IN_FILENAME = 0  # Position of well number after splitting by '_'
TILE_INDEX_IN_FILENAME = 2  # Position of tile number after splitting by '_'

# Processing options
PRESERVE_DTYPE = True    # Keep original data type or convert to float32
CHANNELS_TO_PROCESS = None  # List of channel indices to process (None = all)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_well_tile_combinations(path_name, well_index=0, tile_index=2):
    """
    Get all unique well and tile combinations from ND2 files in directory.
    
    Parameters:
    -----------
    path_name : str
        Path to directory containing ND2 files
    well_index : int
        Position of well number in filename when split by '_'
    tile_index : int
        Position of tile number in filename when split by '_'
        
    Returns:
    --------
    list : List of (well, tile) tuples
    """
    onlyfiles = [f for f in listdir(path_name) if isfile(join(path_name, f))]
    combinations = []
    
    for filename in onlyfiles:
        if filename.endswith('.nd2'):
            try:
                parts = filename.split('_')
                n_tile = int(parts[tile_index])
                n_well = int(parts[well_index].split('Well')[-1])
                if (n_well, n_tile) not in combinations:
                    combinations.append((n_well, n_tile))
            except (IndexError, ValueError):
                continue
    
    return sorted(combinations)


def get_nd2_file_by_well_tile(path_name, well, tile, well_index=0, tile_index=2):
    """
    Find ND2 file matching specific well and tile numbers.
    
    Parameters:
    -----------
    path_name : str
        Path to directory containing ND2 files
    well : int
        Well number to find
    tile : int
        Tile number to find
    well_index : int
        Position of well number in filename when split by '_'
    tile_index : int
        Position of tile number in filename when split by '_'
        
    Returns:
    --------
    str or None : Full path to ND2 file, or None if not found
    """
    onlyfiles = [f for f in listdir(path_name) if isfile(join(path_name, f))]
    
    for filename in onlyfiles:
        if filename.endswith('.nd2'):
            try:
                parts = filename.split('_')
                n_tile = int(parts[tile_index])
                n_well = int(parts[well_index].split('Well')[-1])
                
                if n_tile == tile and n_well == well:
                    return join(path_name, filename)
            except (IndexError, ValueError):
                continue
    
    return None


# ============================================================================
# PROJECTION FUNCTIONS
# ============================================================================

def project_zstack(image_data, method="max", axis=0):
    """
    Project Z-stack using specified method.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Image data with Z dimension
    method : str
        Projection method: "max", "mean", "min", "median", "sum"
    axis : int
        Axis along which to project (default=0 for Z-axis)
    
    Returns:
    --------
    numpy.ndarray : Projected image
    """
    if method == "max":
        return np.max(image_data, axis=axis)
    elif method == "mean":
        return np.mean(image_data, axis=axis)
    elif method == "min":
        return np.min(image_data, axis=axis)
    elif method == "median":
        return np.median(image_data, axis=axis)
    elif method == "sum":
        return np.sum(image_data, axis=axis)
    else:
        raise ValueError(f"Unknown projection method: {method}")


def extract_metadata(nd2_file):
    """
    Extract metadata from ND2 file.
    
    Parameters:
    -----------
    nd2_file : ND2File
        Opened ND2 file object
    
    Returns:
    --------
    dict : Metadata dictionary
    """
    metadata = {
        'original_shape': nd2_file.shape,
        'dimensions': nd2_file.sizes,
        'pixel_size': None,
        'channels': None,
        'stage_positions': None,
        'acquisition_date': None,
        'objective': None,
        'metadata': {}
    }
    
    # Extract pixel/voxel size
    if hasattr(nd2_file, 'metadata') and nd2_file.metadata:
        meta = nd2_file.metadata
        
        # Pixel size information
        if hasattr(meta, 'channels'):
            metadata['channels'] = [ch.channel.name for ch in meta.channels]
        
        # Try multiple methods to extract voxel size
        pixel_size_extracted = False
        
        # Method 1: Try attributes (calibrationX/Y/Z)
        try:
            if hasattr(nd2_file, 'attributes'):
                attrs = nd2_file.attributes
                x_cal = getattr(attrs, 'calibrationX', None)
                y_cal = getattr(attrs, 'calibrationY', None)
                z_cal = getattr(attrs, 'calibrationZ', None)
                
                if x_cal is not None or y_cal is not None or z_cal is not None:
                    metadata['pixel_size'] = {
                        'x': x_cal,
                        'y': y_cal,
                        'z': z_cal,
                    }
                    pixel_size_extracted = True
        except:
            pass
        
        # Method 2: Try text_info parsing (most reliable for Nikon files)
        if not pixel_size_extracted:
            try:
                if hasattr(nd2_file, 'text_info'):
                    text_info = nd2_file.text_info
                    if 'Voxel size:' in text_info:
                        # Parse line like: "Voxel size: 0.1664x0.1664x0.6 micron^3"
                        for line in text_info.split('\n'):
                            if 'Voxel size:' in line:
                                # Extract numbers from format "X.XXXxY.YYYxZ.ZZZ"
                                voxel_part = line.split('Voxel size:')[1].strip()
                                voxel_values = voxel_part.split()[0]  # Get "0.1664x0.1664x0.6"
                                x_val, y_val, z_val = voxel_values.split('x')
                                
                                metadata['pixel_size'] = {
                                    'x': float(x_val),
                                    'y': float(y_val),
                                    'z': float(z_val),
                                }
                                pixel_size_extracted = True
                                break
            except Exception as e:
                print(f"  Warning: Could not parse voxel size from text_info: {e}")
                pass
        
        # Method 3: Try voxel_size method (if available in nd2 package)
        if not pixel_size_extracted:
            try:
                if hasattr(nd2_file, 'voxel_size'):
                    voxel = nd2_file.voxel_size()
                    if voxel is not None:
                        # voxel_size() typically returns a tuple (x, y, z)
                        if len(voxel) >= 2:
                            metadata['pixel_size'] = {
                                'x': voxel[0] if len(voxel) > 0 else None,
                                'y': voxel[1] if len(voxel) > 1 else None,
                                'z': voxel[2] if len(voxel) > 2 else None,
                            }
                            pixel_size_extracted = True
            except:
                pass
        
        # Extract stage positions
        try:
            # Try to get stage positions from experiment metadata
            if hasattr(nd2_file, 'experiment') and nd2_file.experiment:
                exp = nd2_file.experiment
                if hasattr(exp, 'parameters'):
                    params = exp.parameters
                    if hasattr(params, 'points'):
                        stage_pos = []
                        for point in params.points:
                            pos = {
                                'x': point.stagePositionUm.x if hasattr(point.stagePositionUm, 'x') else None,
                                'y': point.stagePositionUm.y if hasattr(point.stagePositionUm, 'y') else None,
                                'z': point.stagePositionUm.z if hasattr(point.stagePositionUm, 'z') else None,
                            }
                            stage_pos.append(pos)
                        metadata['stage_positions'] = stage_pos
            
            # Also try to get from frame metadata (for single position or per-frame coords)
            if hasattr(nd2_file, 'frame_metadata') and nd2_file.frame_metadata:
                frame_meta = nd2_file.frame_metadata(0)  # Get first frame metadata
                if hasattr(frame_meta, 'channels'):
                    for ch in frame_meta.channels:
                        if hasattr(ch, 'position'):
                            pos = ch.position
                            if metadata['stage_positions'] is None:
                                metadata['stage_positions'] = []
                            stage_pos_single = {
                                'x': pos.stagePositionUm.x if hasattr(pos, 'stagePositionUm') and hasattr(pos.stagePositionUm, 'x') else None,
                                'y': pos.stagePositionUm.y if hasattr(pos, 'stagePositionUm') and hasattr(pos.stagePositionUm, 'y') else None,
                                'z': pos.stagePositionUm.z if hasattr(pos, 'stagePositionUm') and hasattr(pos.stagePositionUm, 'z') else None,
                            }
                            if stage_pos_single not in metadata['stage_positions']:
                                metadata['stage_positions'].append(stage_pos_single)
                            break  # Only need one channel's position
        except Exception as e:
            print(f"  Warning: Could not extract stage positions: {e}")
            pass
        
        # Store raw metadata for later use
        metadata['metadata'] = nd2_file.metadata
    
    return metadata

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_single_image(input_path, output_path, projection_type, 
                        z_start, z_end, z_step, channels, preserve_dtype, metadata_only=False):
    """
    Process a single ND2 file and generate Z-stack projection or extract metadata only.
    
    Parameters:
    -----------
    input_path : str
        Path to input ND2 file
    output_path : str
        Path to output TIFF file (ignored if metadata_only=True)
    projection_type : str
        Type of projection to perform (ignored if metadata_only=True)
    z_start : int or None
        Start slice index (ignored if metadata_only=True)
    z_end : int or None
        End slice index (ignored if metadata_only=True)
    z_step : int
        Step size for slice selection (ignored if metadata_only=True)
    channels : list or None
        Channel indices to process (ignored if metadata_only=True)
    preserve_dtype : bool
        Whether to preserve original data type (ignored if metadata_only=True)
    metadata_only : bool
        If True, only extract and save metadata without performing projection
    """
    print(f"Processing: {input_path}")
    
    with ND2File(input_path) as nd2:
        # Extract metadata
        metadata = extract_metadata(nd2)
        print(f"  Image shape: {nd2.shape}")
        print(f"  Dimensions: {nd2.sizes}")
        
        # Display pixel size if available
        if metadata['pixel_size']:
            print(f"  Pixel size: X={metadata['pixel_size']['x']} µm, Y={metadata['pixel_size']['y']} µm, Z={metadata['pixel_size']['z']} µm")
        
        # Display stage positions if available
        if metadata['stage_positions']:
            print(f"  Stage positions: {metadata['stage_positions']}")
        
        # Display channels if available
        if metadata['channels']:
            print(f"  Channels: {metadata['channels']}")
        
        # If metadata only mode, save metadata and exit
        if metadata_only:
            print(f"  Metadata-only mode: Skipping projection")
            
            # Determine metadata directory and filename
            if METADATA_DIR is not None:
                metadata_dir = Path(METADATA_DIR)
                metadata_dir.mkdir(parents=True, exist_ok=True)
                # Get base filename from input
                base_filename = Path(input_path).stem
                json_filename = base_filename + '_metadata.json'
                json_path = str(metadata_dir / json_filename)
            else:
                # Save in same directory as INPUT_DIR
                output_dir = Path(INPUT_DIR)
                output_dir.mkdir(parents=True, exist_ok=True)
                base_filename = Path(input_path).stem
                json_filename = base_filename + '_metadata.json'
                json_path = str(output_dir / json_filename)
            
            # Prepare JSON metadata
            json_metadata = {
                'original_shape': metadata['original_shape'],
                'dimensions': metadata['dimensions'],
                'pixel_size_um': metadata['pixel_size'],
                'channels': metadata['channels'],
                'stage_positions_um': metadata['stage_positions'],
                'extraction_date': datetime.now().isoformat(),
                'input_file': str(input_path),
            }
            
            with open(json_path, 'w') as f:
                json.dump(json_metadata, f, indent=2, default=str)
            
            print(f"  Metadata saved to: {json_path}")
            print(f"  ✓ Complete")
            return
        
        # Continue with projection if not metadata-only mode
        # Get the image data
        img = nd2.asarray()
        
        # Determine dimension order
        dims = nd2.sizes
        z_axis = None
        c_axis = None
        
        # Find Z and C axes
        for i, (dim_name, dim_size) in enumerate(dims.items()):
            if dim_name == 'Z':
                z_axis = i
            elif dim_name == 'C':
                c_axis = i
        
        if z_axis is None:
            raise ValueError("No Z dimension found in image")
        
        # Select Z slices
        slices = [slice(None)] * img.ndim
        slices[z_axis] = slice(z_start, z_end, z_step)
        img_subset = img[tuple(slices)]
        
        print(f"  Using Z slices: {z_start}:{z_end}:{z_step}")
        print(f"  Subset shape: {img_subset.shape}")
        
        # Select channels if specified
        if channels is not None and c_axis is not None:
            channel_slices = [slice(None)] * img_subset.ndim
            channel_slices[c_axis] = channels
            img_subset = img_subset[tuple(channel_slices)]
            print(f"  Using channels: {channels}")
        
        # Perform projection
        print(f"  Applying {projection_type} projection...")
        projected = project_zstack(img_subset, method=projection_type, axis=z_axis)
        
        # Handle data type
        if not preserve_dtype and projected.dtype != np.float32:
            projected = projected.astype(np.float32)
        
        print(f"  Projected shape: {projected.shape}")
        
        # Prepare metadata for TIFF
        tiff_metadata = {
            'projection_type': projection_type,
            'z_range': f"{z_start}:{z_end}:{z_step}",
            'original_shape': str(metadata['original_shape']),
            'processing_date': datetime.now().isoformat(),
        }
        
        if metadata['pixel_size']:
            tiff_metadata['pixel_size_um'] = str(metadata['pixel_size'])
        
        if metadata['channels']:
            tiff_metadata['channel_names'] = str(metadata['channels'])
        
        if metadata['stage_positions']:
            tiff_metadata['stage_positions_um'] = str(metadata['stage_positions'])
            print(f"  Stage positions: {metadata['stage_positions']}")
        
        # Save metadata as JSON sidecar file if requested
        if SAVE_METADATA_JSON:
            # Determine metadata directory
            if METADATA_DIR is not None:
                metadata_dir = Path(METADATA_DIR)
                metadata_dir.mkdir(parents=True, exist_ok=True)
                base_filename = Path(input_path).stem
                json_filename = base_filename + '_metadata.json'
                json_path = str(metadata_dir / json_filename)
            else:
                # Get base path without any extension
                base_path = output_path.replace('.ome.tif', '').replace('.tif', '')
                json_path = base_path + '_metadata.json'
            
            json_metadata = {
                'projection_type': projection_type,
                'z_range': {'start': z_start, 'end': z_end, 'step': z_step},
                'original_shape': metadata['original_shape'],
                'dimensions': metadata['dimensions'],
                'pixel_size_um': metadata['pixel_size'],
                'channels': metadata['channels'],
                'stage_positions_um': metadata['stage_positions'],
                'processing_date': datetime.now().isoformat(),
                'input_file': str(input_path),
                'output_file': str(output_path),
            }
            with open(json_path, 'w') as f:
                json.dump(json_metadata, f, indent=2, default=str)
            print(f"  Metadata saved to: {json_path}")
        
        # Save as TIFF with metadata
        print(f"  Saving to: {output_path}")
        
        # Determine if OME-TIFF format
        is_ome = OUTPUT_FORMAT.lower() == "ome.tiff" or OUTPUT_FORMAT.lower() == "ome-tiff"
        
        if is_ome:
            print(f"  Saving as OME-TIFF format...")
            # For OME-TIFF, we need to create proper OME-XML metadata
            # Prepare physical pixel sizes
            physical_size_x = metadata['pixel_size']['x'] if metadata['pixel_size'] and metadata['pixel_size']['x'] else None
            physical_size_y = metadata['pixel_size']['y'] if metadata['pixel_size'] and metadata['pixel_size']['y'] else None
            physical_size_z = metadata['pixel_size']['z'] if metadata['pixel_size'] and metadata['pixel_size']['z'] else None
            
            # Create metadata dict for OME
            ome_metadata = {}
            
            # Set physical pixel sizes if available
            if physical_size_x:
                ome_metadata['PhysicalSizeX'] = physical_size_x
                ome_metadata['PhysicalSizeXUnit'] = 'µm'
            if physical_size_y:
                ome_metadata['PhysicalSizeY'] = physical_size_y
                ome_metadata['PhysicalSizeYUnit'] = 'µm'
            
            # Add channel names if available
            if metadata['channels']:
                ome_metadata['Channel'] = {'Name': metadata['channels']}
            
            tifffile.imwrite(
                output_path,
                projected,
                compression=COMPRESSION,
                metadata=ome_metadata,
                ome=True,
                photometric='minisblack'
            )
        else:
            print(f"  Saving as standard TIFF format...")
            # Regular TIFF with ImageJ metadata
            
            # Prepare ImageJ metadata with Info field containing all metadata
            imagej_metadata = {
                'Info': json.dumps(tiff_metadata, indent=2, default=str)
            }
            
            tifffile.imwrite(
                output_path,
                projected,
                compression=COMPRESSION,
                metadata=imagej_metadata,
                imagej=True
            )
        
        print(f"  ✓ Complete")


# ============================================================================
# SLURM ARRAY JOB HANDLER
# ============================================================================

def main():
    """
    Main function for SLURM array job execution.
    """
    # Get SLURM array task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    print(f"\n{'='*70}")
    print(f"Z-Stack Projection Pipeline")
    print(f"SLURM Array Task ID: {task_id}")
    print(f"{'='*70}\n")
    
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of all ND2 files
    input_dir = Path(INPUT_DIR)
    nd2_files = sorted(list(input_dir.glob("*.nd2")))
    
    if not nd2_files:
        print(f"ERROR: No ND2 files found in {INPUT_DIR}")
        sys.exit(1)
    
    print(f"Total ND2 files found: {len(nd2_files)}")
    
    # Check if task_id is valid
    if task_id >= len(nd2_files):
        print(f"ERROR: Task ID {task_id} exceeds number of files ({len(nd2_files)})")
        sys.exit(1)
    
    # Get the file for this task
    input_file = nd2_files[task_id]
    
    # Generate output filename with appropriate extension
    if OUTPUT_FORMAT.lower() == "ome.tiff":
        output_filename = input_file.stem + f"_{PROJECTION_TYPE}_projection.ome.tif"
    else:
        output_filename = input_file.stem + f"_{PROJECTION_TYPE}_projection.tif"
    output_file = output_dir / output_filename
    
    print(f"\nProcessing file {task_id + 1}/{len(nd2_files)}")
    print(f"Configuration:")
    print(f"  Metadata only mode: {METADATA_ONLY}")
    if not METADATA_ONLY:
        print(f"  Projection type: {PROJECTION_TYPE}")
        print(f"  Z slice range: {Z_SLICE_START}:{Z_SLICE_END}:{Z_SLICE_STEP}")
        print(f"  Output format: {OUTPUT_FORMAT}")
        print(f"  Compression: {COMPRESSION}")
        print(f"  Preserve dtype: {PRESERVE_DTYPE}")
        print(f"  Channels: {CHANNELS_TO_PROCESS if CHANNELS_TO_PROCESS else 'all'}")
    print(f"  Save metadata JSON: {SAVE_METADATA_JSON}")
    print()
    
    # Process the image
    try:
        process_single_image(
            str(input_file),
            str(output_file),
            PROJECTION_TYPE,
            Z_SLICE_START,
            Z_SLICE_END,
            Z_SLICE_STEP,
            CHANNELS_TO_PROCESS,
            PRESERVE_DTYPE,
            metadata_only=METADATA_ONLY
        )
        print(f"\n{'='*70}")
        print(f"SUCCESS: Task {task_id} completed")
        print(f"{'='*70}\n")
        
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