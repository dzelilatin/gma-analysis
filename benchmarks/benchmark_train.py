import os
import time
import sys
import numpy as np
import tensorflow as tf

# FORCE CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# CONFIG
BATCH_SIZE = 8
DATA_DIR = "/Users/user/Desktop/gma-analysis/dataset/data"

def run_training_benchmark():
    print(f"🚀 STARTING MANUAL TRAINING SPRINT")
    
    # 1. MANUAL DATA LOADING (No utils needed)
    print("Loading images manually...")
    images = []
    labels = []
    
    # Just grab the first 500 images to get a speed estimate 
    # (Processing 2000+ on a broken CPU driver will take too long)
    count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and count < 500:
                img = tf.io.read_file(os.path.join(root, file))
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [224, 224])
                images.append(img)
                labels.append(0) # Dummy label
                count += 1
    
    X = tf.stack(images) / 255.0
    y = np.array(labels)

    # 2. BUILD MODEL (Manual Layers)
    print("Building Model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # 3. THE SPRINT
    print(f"🏁 Training on {count} images...")
    start_time = time.time()
    
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=1, verbose=1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate what it WOULD take for the full 2147 images
    estimated_full_time = (total_time / count) * 2147
    
    print("\n" + "="*30)
    print(f"⏱️ TIME FOR 500 IMAGES: {total_time:.2f}s")
    print(f"📊 ESTIMATED FULL EPOCH (2147 images): {estimated_full_time:.2f}s")
    print("="*30)

if __name__ == "__main__":
    run_training_benchmark()