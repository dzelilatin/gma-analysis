import os
# 1. Disable CUDA (just in case)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 2. Import TensorFlow
import tensorflow as tf
import keras

# 3. FORCE DISABLE MAC GPU (Metal)
try:
    # Get all physical devices
    devices = tf.config.list_physical_devices('GPU')
    if devices:
        # Set visible devices to empty list (Hides GPU from TF)
        tf.config.set_visible_devices([], 'GPU')
        print("🚫 GPU disabled for benchmark (Running on CPU).")
except Exception as e:
    print(f"⚠️ Could not disable GPU: {e}")

# 4. Standard Imports
import time
import psutil
import numpy as np
import cv2
from keras.models import load_model

# --- CONFIGURATION ---
BATCH_SIZE = 8   # We start with your "8" setting
MODEL_PATH = 'novimodel_test.h5'
TEST_DIR = '/Users/user/Desktop/gma-analysis/dataset/data' 

# Limit images for the FIRST run to make sure it works (Scenario A)
# Set to 200 for Baseline. Set to 5000 for Stress Test later.
MAX_IMAGES_TO_LOAD = 200
# ---------------------

def run_benchmark():
    print(f"🚀 STARTING LOCAL BENCHMARK")
    print(f"Target Images: {MAX_IMAGES_TO_LOAD}")
    
    start_time = time.time()
    
    # 1. Load Model
    print(f"Loading model: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("❌ ERROR: Model file not found! Put novimodel_test.h5 in this folder.")
        return
    model = load_model(MODEL_PATH)

    # 2. Find Images
    print(f"Scanning {TEST_DIR} for images...")
    images = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for root, dirs, files in os.walk(TEST_DIR):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in valid_exts:
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (512, 512)) # Resize to match VGG16
                    images.append(img)
                
                if MAX_IMAGES_TO_LOAD is not None and len(images) >= MAX_IMAGES_TO_LOAD:
                    break
            if MAX_IMAGES_TO_LOAD is not None and len(images) >= MAX_IMAGES_TO_LOAD:
                break
        
    if len(images) == 0:
        print("❌ ERROR: No images found! Check your path.")
        return

    print(f"✅ Loaded {len(images)} images.")

    # 3. Prepare Data (Normalize)
    print("Converting to numpy array (this eats RAM)...")
    X_test = np.array(images)
    X_test = X_test / 255.0

    # 4. Run Prediction
    print("Running inference...")
    model.predict(X_test, batch_size=BATCH_SIZE)

    # 5. Measure Stats
    end_time = time.time()
    total_time = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)

    print("\n" + "="*30)
    print(f"📊 RESULTS (Scenario A)")
    print(f"Images: {len(X_test)}")
    print(f"Time:   {total_time:.2f} seconds")
    print(f"Memory: {memory_mb:.2f} MB")
    print("="*30)

if __name__ == "__main__":
    run_benchmark()
