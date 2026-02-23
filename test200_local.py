import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- KONFIGURACIJA ---
data_dir = r'/Users/user/Desktop/gma-analysis/dataset/data/Training slike'
img_height, img_width = 512, 512
num_classes = 4
testni_uzorak = 200  # Broj slika za test brzine

# --- IDENTIČNA TRUEAID ARHITEKTURA ---
def create_model(num_classes):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- PROFILIRANJE PO BATCH VELIČINAMA ---
results = []

print(f"--- PROFILIRANJE TRUEAID SISTEMA (Intel MacBook Pro) ---")

for B in [4, 8, 16]:
    # Identičan generator kao u glavnom kodu
    datagen = ImageDataGenerator(rescale=1./255)
    
    generator = datagen.flow_from_directory(
        data_dir, 
        target_size=(img_height, img_width),
        batch_size=B, 
        class_mode='categorical', 
        shuffle=False
    )

    steps = testni_uzorak // B
    model = create_model(num_classes)

    print(f"\n>>> Testiranje brzine za Batch Size: {B}")
    
    start_time = time.time()
    # Pokrećemo samo 1 epohu na 200 slika
    model.fit(generator, steps_per_epoch=steps, epochs=1, verbose=1)
    end_time = time.time()
    
    trajanje = end_time - start_time
    brzina_po_stepu = (trajanje / steps) * 1000  # u milisekundama
    
    results.append((B, trajanje, brzina_po_stepu))

# --- FINALNI PRIKAZ ZA RAD ---
print("\n" + "="*50)
print(f"{'Batch Size':<15} | {'Ukupno (s)':<15} | {'Brzina (ms/step)':<15}")
print("-" * 50)
for B, t, ms in results:
    print(f"{B:<15} | {t:<15.2f} | {ms:<15.2f}")
print("="*50)

# KREIRANJE DATAFRAME-A PRIJE ČUVANJA
report_df = pd.DataFrame(results, columns=['Batch Size', 'Total Time (s)', 'Speed (ms/step)'])

# Čuvanje kao CSV
csv_report_path = r'/Users/user/Desktop/gma-analysis/models/report_test200_local.csv'
report_df.to_csv(csv_report_path, index=False)

print(f"Tabela sa rezultatima je uspješno sačuvana na: {csv_report_path}")