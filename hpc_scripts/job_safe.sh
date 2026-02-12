#!/bin/bash
#SBATCH --job-name=GMA_Safe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=result_safe.txt

# --- MAGIC MODULES (From your run.slurm) ---
module purge
module load GCCcore/13.2.0
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.0.0
module load cuDNN/8.8.0.121-CUDA-12.0.0

# Ensure environment is ready
pip install --user tensorflow pandas matplotlib seaborn scikit-learn opencv-python-headless

# Run Benchmark
python benchmark_hpc.py 30