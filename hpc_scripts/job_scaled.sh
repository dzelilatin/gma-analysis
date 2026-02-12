#!/bin/bash
#SBATCH --job-name=GMA_Scale
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=result_scaled.txt

# --- MAGIC MODULES ---
module purge
module load GCCcore/13.2.0
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.0.0
module load cuDNN/8.8.0.121-CUDA-12.0.0

pip install --user tensorflow pandas matplotlib seaborn scikit-learn opencv-python-headless

# Run Benchmark
python benchmark_hpc.py 8