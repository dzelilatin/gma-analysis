# Medical Image Analysis Benchmark (GMA)

This project compares the performance of a VGG16 Deep Learning model for medical image classification across different hardware infrastructures.

## Project Structure
- `benchmarks/`: Python scripts optimized for local Mac execution.
- `hpc_scripts/`: Slurm job scripts and Python code used on the HPC cluster.
- `dataset/data/`: Cropped medical images (Training & Validation).
- `plots/`: Generated performance visualizations.

## Experiment Results (2,147 Images)

| Scenario | Infrastructure | Task | Time |
| :--- | :--- | :--- | :--- |
| **A** | Mac (8GB RAM) | Baseline Inference | 12.9s |
| **B** | Mac (8GB RAM) | Stress Inference | 222.4s |
| **C** | HPC (1 GPU) | Standard Inference | 35.5s |
| **D** | HPC (Scaled) | Scaled Inference | 38.6s |
| **Training**| HPC (GPU) | 1 Epoch Training | 266.7s |

## Key Findings
1. **Infrastructure Bottleneck:** The local machine suffered a 65% efficiency drop under heavy load due to RAM swapping.
2. **Speedup:** The HPC cluster provided a **6.2x speedup** for inference tasks.
3. **Stability:** The Mac environment was unable to complete the full training workload, whereas the HPC remained stable.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Generate plots: `python generate_plots.py`