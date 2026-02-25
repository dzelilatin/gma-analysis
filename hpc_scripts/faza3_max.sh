#!/bin/bash
#SBATCH --job-name=TrueAId_Faza3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Učitavanje ispravnog modula koji smo pronašli
module load Python/3.11.5-GCCcore-13.2.0

# Putanja do tvojih instaliranih biblioteka (cv2, pandas, itd.)
export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.11/site-packages

# Testovi 10, 11, 12: Vertical Scaling (Batch 32, 64, 128)
# Koristimo cijeli dataset (mode full)
for B in 32 64 128
do
    echo "--------------------------------------------"
    echo "POKREĆEM HPC SCALING: BATCH SIZE $B"
    echo "--------------------------------------------"
    python3 Kod_HPC.py --batch_size $B --mode full
done
