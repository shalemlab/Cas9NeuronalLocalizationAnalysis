#!/bin/bash
#SBATCH --job-name=cell_seg
#SBATCH --output=logs/segmentation_CP4_updated2/seg_%A_%a.out
#SBATCH --error=logs/segmentation_CP4_updated2/seg_%A_%a.err
#SBATCH --array=0-74              # Adjust based on your number of images (0 to N-1)
#SBATCH --time=02:00:00           # Adjust based on image size and segmentation complexity
#SBATCH --mem=16G                 # Adjust based on image size (cellpose can be memory intensive)
#SBATCH --cpus-per-task=4         # Cellpose benefits from multiple CPUs
#SBATCH --partition=gpuq       	  # Change to your cluster's partition name (defq if not using gpu, gpuq if using gpu)
#SBATCH --gres=gpu:1              # Set to 0 if not using GPU

# Create logs directory if it doesn't exist
mkdir -p logs/segmentation_CP4_updated2

# Load conda environment
source ~/.bashrc
conda activate cellpose  # Replace with your conda environment name

# Run the Python script
python segmentation_pipeline_CP4.py

echo "Task ${SLURM_ARRAY_TASK_ID} completed"