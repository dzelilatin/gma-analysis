import os
import time
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, applications

# --- HPC CONFIG ---
# We use Batch 32 to take advantage of the GPU
try:
    BATCH_SIZE = int(sys.argv[1])
except:
    BATCH_SIZE = 32

DATA_DIR = os.path.expanduser("~/trueaid/data")
MODEL_PATH = os.path.expanduser("~/trueaid/model_transfer.h5")

def run_hpc_train_benchmark():
    print(f"🚀 STARTING HPC TRAINING SPRINT (Batch {BATCH_SIZE})")
    print(f"📂 Folder: {DATA_DIR}")

    # 1. Load Data
    # Use the stable HPC way to load images
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    try:
        train_generator = datagen.flow_from_directory(
            DATA_DIR,
            target_size=(224, 224), # Standard VGG16 size
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return

    # 2. Setup Architecture (VGG16)
    print("Building VGG16 architecture...")
    # We build it fresh with the correct number of classes found in your folder
    num_classes = train_generator.num_classes
    model = applications.VGG16(
        weights=None, 
        classes=num_classes, 
        input_shape=(224, 224, 3)
    )
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # 3. The Sprint
    print(f"🏁 Starting 1-Epoch Sprint on {train_generator.samples} images...")
    start_time = time.time()
    
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=1,
        verbose=1
    )
    
    end_time = time.time()
    print("\n" + "="*40)
    print(f"⏱️ HPC TRAINING TIME (1 Epoch): {end_time - start_time:.2f} seconds")
    print("="*40)

if __name__ == "__main__":
    run_hpc_train_benchmark()