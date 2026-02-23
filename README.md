# TrueAId: KANET Medical Image Analysis Benchmark

This project evaluates the computational efficiency and scalability of the **TrueAId** pipeline, a custom Convolutional Neural Network (CNN) designed for the automated screening of the **Kurjak Antenatal Neurodevelopmental Test (KANET)**. The study identifies the "Hardware Ceiling" of consumer-grade electronics compared to enterprise-grade High-Performance Computing (HPC) environments.

## 📂 Repository Structure
Based on the project's development environment:
* **dataset/**: Contains image data used for training and validation.
    * **cropped/**: Images processed to isolate fetal regions.
    * **data/**: Primary image sets.
* **models/**: Storage for trained models and benchmarking logs.
    * `new_model.csv`: Training history (accuracy and loss).
    * `new_model.h5`: The saved TrueAId CNN model.
    * `report_test200_local.csv`: Results for local workstation latency tests.
    * `report_TrueAId_batch16.csv`: Benchmarking metrics for batch 16.
* **Kod_HPC.py**: Production training script optimized for the Verlab HPC cluster.
* **Kod.py**: Core script containing the TrueAId CNN architecture and local testing logic.
* **test200_local.py**: Specialized script to profile system latency on a local workstation using a 200-image sample.
* **rezultati.py**: Utility script for generating performance visualizations and evaluations.
* **requirements.txt**: List of Python dependencies (TensorFlow, Keras, OpenCV, etc.).

## 📊 Key Experimental Results

| Phase | Infrastructure | Batch Size (B) | Latency / Status | Clinical Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | MacBook Pro (16GB RAM) | 4 | ~1400 ms/step | N/A |
| **Stress Test** | MacBook Pro (16GB RAM) | 8-16 | **Throttled** (Up to 6010s/epoch) | 80% |
| **HPC Match** | Verlab HPC (1 GPU) | 16 | 473 ms/step | 75% |
| **HPC Scaled** | Verlab HPC (4 GPUs) | 128 | **112 ms/step** | **96%** |

## 💡 Key Findings
1. **Infrastructure Bottleneck**: The local Intel-based MacBook Pro suffered extreme performance degradation (latency spikes up to 100 minutes per epoch) due to thermal throttling and RAM saturation.
2. **Computational Speedup**: The HPC cluster achieved a consistent sub-second latency, providing over a **9x speedup** compared to the local baseline.
3. **Diagnostic Accuracy**: The TrueAId CNN achieved a **96% validation accuracy** and a **1.00 recall rate** for fetal "Face" detection, ensuring reliable clinical screening.
4. **On-Premise Security**: Utilizing local HPC infrastructure ensures medical data remains protected within the clinical network, adhering to privacy regulations.

## 🚀 How to Run
1. **Install dependencies**:  
   `pip install -r requirements.txt`
2. **Local Profiling**:  
   `python test200_local.py`
3. **Full Training (HPC)**:  
   `python Kod_HPC.py --batch_size 128`