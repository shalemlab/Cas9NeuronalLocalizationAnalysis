#!/bin/bash
#SBATCH --job-name=pheno
#SBATCH --output=logs/phenotyping/cyto_sub_mean/pheno_%A_%a.out
#SBATCH --error=logs/phenotyping/cyto_sub_mean/pheno_%A_%a.err
#SBATCH --array=0-74              # Adjust based on your number of images (0 to N-1)
#SBATCH --time=00:30:00           # Adjust based on image size
#SBATCH --mem=8G                  # Adjust based on image size
#SBATCH --cpus-per-task=1         
#SBATCH --partition=defq          # Change to your cluster's partition name (defq if not using gpu, gpuq if using gpu)

# Create logs directory if it doesn't exist
mkdir -p logs/phenotyping/cyto_sub_mean

# Load conda environment
source ~/.bashrc
conda activate cellpose  # Replace with your conda environment name

# Run the Python script
python phenotyping.py

echo "Task ${SLURM_ARRAY_TASK_ID} completed"