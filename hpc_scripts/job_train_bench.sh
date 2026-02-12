#!/bin/bash
#SBATCH --job-name=TrainBench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=result_train_bench.txt

module purge
module load GCCcore/13.2.0
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.0.0
module load cuDNN/8.8.0.121-CUDA-12.0.0

python benchmark_train.py 32