#!/bin/bash
#SBATCH --job-name=nd2_processing
#SBATCH --output=logs/nd2_processing/nd2_%A_%a.out
#SBATCH --error=logs/nd2_processing/nd2_%A_%a.err
#SBATCH --array=0-74          # Adjust based on your number of images
#SBATCH --time=00:30:00          # Adjust based on image size
#SBATCH --mem=8G                 # Adjust based on image size
#SBATCH --cpus-per-task=1

# Create logs directory if it doesn't exist
mkdir -p logs/nd2_processing

# Load conda environment
source ~/.bashrc
conda activate cellpose  # Replace with your conda environment name

# Run the Python script
python nd2_processing.py

echo "Task ${SLURM_ARRAY_TASK_ID} completed"