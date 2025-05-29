#!/bin/bash
#SBATCH --job-name=py_script
#SBATCH --output=logs_slurm/py_script_%j.out
#SBATCH --error=logs_slurm/py_script_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
##SBATCH --mem-per-cpu=32G
#SBATCH --mem=400G

# Create logs directory if it doesn't exist
mkdir -p logs_slurm

# Activate conda environment
source ~/.bashrc

# Run snakemake
python 

# Deactivate conda environment
conda deactivate 
