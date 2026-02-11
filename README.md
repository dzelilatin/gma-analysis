🧠 Automated GMA Ultrasound Analysis Pipeline
This repository contains the technical implementation of my research on automating General Movements Assessment (GMA) for the early detection of neurodevelopmental disorders in infants.

The project bridges Biomedical Engineering and High-Performance Computing (HPC), providing an end-to-end pipeline that processes high-frequency ultrasound video data to predict movement risks.

🚀 Key Features
Video Preprocessing Engine: Python scripts to slice raw ultrasound footage into high-resolution datasets (5,000+ frames per subject) for granular movement tracking.

Deep Learning Inference: Implementation of a CNN-based model for classifying movement patterns (Normal vs. At-Risk).

HPC Scalability: SLURM scripts optimized for the Verlab Institute Supercomputer, enabling massive batch processing (comparing 8 vs. 30 batch configurations).

Visualization Dashboard: A TypeScript/Next.js frontend module that allows clinicians to upload videos and view risk analysis results in real-time.

🛠️ Tech Stack
Core Logic: Python (NumPy, OpenCV, PyTorch/TensorFlow)

Infrastructure: Linux (Ubuntu), SLURM Workload Manager, HPC

Frontend Interface: TypeScript, React, Next.js, Tailwind CSS

Tools: Git, VS Code, SSH (Remote Development)

📊 Performance Benchmarks
This project includes a comparative study of computational efficiency between local environments and HPC clusters.

Environment	Configuration	Dataset Size	Status
Local (Intel Mac)	8 Batches	5,000 Frames	❌ OutOfMemory
Local (Intel Mac)	30 Batches	200 Frames	✅ Success (Slow)
HPC Cluster	8 Batches	5,000 Frames	✅ Success (High RAM)
HPC Cluster	30 Batches	5,000 Frames	✅ Success (Optimized)

📂 Project Structure
Bash
├── analysis-engine/       # Python scripts for video slicing & inference
├── hpc-scripts/           # SLURM job configurations for the supercomputer
├── web-interface/         # TypeScript/React module for the dashboard UI
└── benchmarks/            # Logs and results from efficiency tests