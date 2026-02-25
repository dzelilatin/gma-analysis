#!/bin/bash
#SBATCH --job-name=TrueAId_Faza2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1 # Ili verzija koju tvoj HPC ima
export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.11/site-packages

# Test 9 (Benchmark 200 slika) i Testovi 6,7,8 (Puni trening)
for B in 4 8 16; do 
    python3 Kod_HPC.py --batch_size $B --mode benchmark
    python3 Kod_HPC.py --batch_size $B --mode full
done
