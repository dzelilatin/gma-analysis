import sys
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- CONFIGURATION ---
try:
    BATCH_SIZE = int(sys.argv[1])
except IndexError:
    BATCH_SIZE = 30

# POINT TO THE PARENT FOLDER 'dataset' TO FIND ALL IMAGES
BASE_DIR = os.getcwd()
TEST_DIR = os.path.join(BASE_DIR, "data") 
MODEL_PATH = 'model_transfer.h5' 
# ---------------------

def run_hpc_benchmark():
    print(f"🚀 STARTING HPC BENCHMARK (Batch: {BATCH_SIZE})")
    
    start_time = time.time()
    
    # 1. Load Model
    model = load_model(MODEL_PATH)
    
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # 2. Load Images (Using TensorFlow instead of OpenCV)
    print(f"Scanning {TEST_DIR}...")
    for root, dirs, files in os.walk(TEST_DIR):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                try:
                    img_path = os.path.join(root, filename)
                    # Load and resize in one step (No cv2 needed)
                    img = load_img(img_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    images.append(img_array)
                except Exception as e:
                    pass # Skip broken images
    
    print(f"✅ Loaded {len(images)} images.")
    
    X_test = np.array(images)
    X_test = X_test / 255.0
    
    # 3. Run Prediction
    print("Running prediction...")
    model.predict(X_test, batch_size=BATCH_SIZE)
    
    end_time = time.time()
    print(f"⏱️ FINAL_TIME: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_hpc_benchmark()
