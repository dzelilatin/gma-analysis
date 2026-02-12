import sys
import os
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
# We read the batch size from the command line argument!
try:
    BATCH_SIZE = int(sys.argv[1])
except IndexError:
    BATCH_SIZE = 30 # Default

# POINT TO THE PARENT FOLDER 'dataset' TO FIND ALL IMAGES
BASE_DIR = os.getcwd()
TEST_DIR = os.path.join(BASE_DIR, "dataset") 
MODEL_PATH = 'model_transfer.h5' 
# ---------------------

def run_hpc_benchmark():
    print(f"🚀 STARTING HPC BENCHMARK (Batch: {BATCH_SIZE})")
    
    start_time = time.time()
    
    # 1. Load Model
    # (We disable GPU visibility if needed, but on HPC we usually want it. 
    # For this specific comparison, CPU is fine, but let's just load it standard.)
    model = load_model(MODEL_PATH)
    
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Walk through 'dataset' to find 'Training slike' AND 'Validacija2'
    print(f"Scanning {TEST_DIR}...")
    for root, dirs, files in os.walk(TEST_DIR):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                img = cv2.imread(os.path.join(root, filename))
                if img is not None:
                    img = cv2.resize(img, (512, 512)) 
                    images.append(img)
    
    print(f"✅ Loaded {len(images)} images.")
    
    X_test = np.array(images)
    X_test = X_test / 255.0
    
    print("Running prediction...")
    model.predict(X_test, batch_size=BATCH_SIZE)
    
    end_time = time.time()
    print(f"⏱️ FINAL_TIME: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_hpc_benchmark()