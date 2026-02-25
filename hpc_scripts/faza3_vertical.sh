#!/bin/bash
#SBATCH --job-name=TrueAId_Faza3_Vert
#SBATCH --nodes=1
#SBATCH --mem=240G
#SBATCH --gres=gpu:4

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.11/site-packages

for B in 32 64 128; do python3 Kod_HPC.py --batch_size $B --mode full; done
