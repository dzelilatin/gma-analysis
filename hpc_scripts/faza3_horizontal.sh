#!/bin/bash
#SBATCH --job-name=TrueAId_Faza3_Horiz
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.11/site-packages

# Ovdje koristimo poseban kod sa distribuiranom strategijom
python3 Kod_HPC_Dist.py --batch_size 64
