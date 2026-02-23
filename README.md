# TrueAId: KANET Medical Image Analysis Benchmark

This project evaluates the computational efficiency and scalability of the **TrueAId** pipeline, a custom Convolutional Neural Network (CNN) designed for the automated screening of the **Kurjak Antenatal Neurodevelopmental Test (KANET)**. This benchmark identifies the "Hardware Ceiling" of consumer-grade electronics compared to enterprise-grade High-Performance Computing (HPC) environments.

## 📂 Repository Structure
* **dataset/**: Contains image data used for training and validation.
* **models/**: Storage for trained models and benchmarking logs.
    * `new_model.h5`: The saved TrueAId CNN model (Legacy HDF5 format).
    * `report_test200_local.csv`: Results for local workstation latency tests.
* **Kod_HPC.py**: Production training script optimized for the Verlab HPC cluster.
* **Kod.py**: Core script used for the MacBook Pro stress tests.
* **test200_local.py**: Specialized script for profiling baseline system latency.
* **requirements.txt**: List of Python dependencies (TensorFlow 2.16+, Keras 3, OpenCV).

## 📊 Key Experimental Results

| Phase | Infrastructure | Batch (B) | Latency Behavior | Clinical Accuracy |
| :--- | :--- | :---: | :--- | :---: |
| **Baseline** | MacBook Pro | 4 | Stable Baseline (~1s/step) | N/A (Speed Test) |
| **Stress Test** | MacBook Pro | 4 | **Extreme Jitter** (1s - 19s/step) | **61%** |
| **Stress Test** | MacBook Pro | 8 | Thermal Throttling | **80%** |
| **HPC Match** | Verlab HPC | 16 | Stable (~470ms/step) | 75% |
| **HPC Scaled** | Verlab HPC | 128 | **Ultra-Fast (112ms/step)** | **96%** |

## 💡 Key Findings
1. **The 6000s Epoch**: During the MacBook Batch 4 stress test, the system hit a "Hardware Ceiling" at Epoch 10, where processing time spiked to **6,010 seconds** due to intense thermal throttling and OS power management interference.
2. **Clinical Failure under Stress**: The local stress test resulted in **0.00 sensitivity for the "Face" class**, indicating that hardware instability directly correlates with the model's inability to learn critical anatomical features.
3. **HPC Scalability**: Moving to the Verlab HPC (4 GPUs, 256GB RAM) allowed for a Batch Size of 128, achieving a **96% accuracy** with stable, sub-second latency.
4. **On-Premise Security**: Utilizing local HPC infrastructure ensures medical data remains protected within the hospital network, adhering to global privacy regulations.

## 🚀 How to Run
1. **Install dependencies**:  
   `pip install -r requirements.txt`
2. **Local Stress Test**:  
   `python Kod.py` (Set batch_size to 4 as per mentor requirements)
3. **HPC Production**:  
   `python Kod_HPC.py --batch_size 128`

## 🛠 Future Tasks
* [ ] Convert `.h5` model files to TensorFlow.js (`.json`/`.bin`) for web-based chatbot integration.
* [ ] Implement `MultiWorkerMirroredStrategy` for Phase 3 Horizontal Scaling.