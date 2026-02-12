import sys
import os
# OBAVEZNO ZA HPC: Sprječava rušenje jer nema ekrana
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# KONFIGURACIJA ZA HPC
# ==========================================
# Kod traži podatke u istom folderu gdje se nalazi
BASE_DIR = os.getcwd() 
# Prilagodi ove nazive ako mentorica koristi druge, ali ovo je standard
# Ocekujemo strukturu: TvojFolder -> dataset -> Training slike
DATA_DIR = os.path.join(BASE_DIR, "dataset", "Training slike")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "Validacija2")

IMG_HEIGHT, IMG_WIDTH = 512, 512
BATCH_SIZE = 16
FIRST_LEVEL_CLASSES = ['Face', 'Hand to face', 'Legs', 'Thumb']
NUM_CLASSES = len(FIRST_LEVEL_CLASSES)

EPOCHS = 50           
STEPS_PER_EPOCH = None 
VALIDATION_STEPS = None 

print(f"Radni direktorij: {BASE_DIR}")
print(f"Tražim trening podatke u: {DATA_DIR}")
print("GPU Dostupan:", len(tf.config.list_physical_devices('GPU')))

# ==========================================
# 1. DEFINICIJA MODELA
# ==========================================
def create_model(num_classes):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
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

# ==========================================
# 2. GENERATORI SLIKA
# ==========================================
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2 
)

# Provjera postoji li folder
if not os.path.exists(DATA_DIR):
    print(f"GREŠKA: Ne mogu naći folder: {DATA_DIR}")
    # Ovdje bi kod trebao stati, ali na HPC-u ce samo ispisati gresku u log
else:
    train_generator = datagen.flow_from_directory(
        DATA_DIR, 
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, 
        class_mode='categorical',
        classes=FIRST_LEVEL_CLASSES, 
        subset='training'
    )

    # Pametna validacija: Ako nema Validacija2 foldera, koristi split iz treninga
    if os.path.exists(VAL_DIR):
        print("Koristim poseban folder za validaciju (Validacija2).")
        val_generator = datagen.flow_from_directory(
            VAL_DIR, 
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE, 
            class_mode='categorical',
            classes=FIRST_LEVEL_CLASSES,
            shuffle=False # BITNO: False za validaciju da bi metrike bile tacne kasnije!
        )
    else:
        print("Folder Validacija2 nije nadjen, koristim validation_split iz trening foldera.")
        val_generator = datagen.flow_from_directory(
            DATA_DIR, 
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE, 
            class_mode='categorical',
            classes=FIRST_LEVEL_CLASSES,
            subset='validation',
            shuffle=False # BITNO: False za validaciju!
        )

    # ==========================================
    # 3. TRENING
    # ==========================================
    manual_class_weights = {0: 1.5, 1: 1.0, 2: 1.0, 3: 1.0}
    # Zastita ako generator nije uspio
    if 'train_generator' in locals():
        train_labels = train_generator.classes
        unique_classes = np.unique(train_labels)
        class_weights_dict = {cls: manual_class_weights.get(cls, 1.0) for cls in unique_classes}

        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

        model = create_model(NUM_CLASSES)

        print("Počinjem HPC trening...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            class_weight=class_weights_dict,
            callbacks=[lr_scheduler],
            epochs=EPOCHS 
        )

        # ==========================================
        # 4. CUVANJE REZULTATA
        # ==========================================
        model.save('gma_hpc_model.h5')
        print("Model sacuvan kao 'gma_hpc_model.h5'")

        pd.DataFrame(history.history).to_csv('training_history.csv')

        # Crtanje grafika
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_graph.png') 
        plt.close() # Zatvori plot
        print("Grafik sacuvan kao 'training_graph.png'")

        # ==========================================
        # 5. MATRICA KONFUZIJE (POPRAVLJENO)
        # ==========================================
        print("Generisem matricu konfuzije...")
        
        # Resetiramo generator da krene od pocetka
        val_generator.reset()
        
        # Predikcije
        y_pred_probs = model.predict(val_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Stvarne klase
        y_true = val_generator.classes
        
        # Sijecemo visak ako ga ima
        y_pred = y_pred[:len(y_true)]

        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10,10))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', 
                    xticklabels=FIRST_LEVEL_CLASSES, 
                    yticklabels=FIRST_LEVEL_CLASSES)
        plt.ylabel('Stvarna klasa')
        plt.xlabel('Predvidjena klasa')
        plt.title('Matrica Konfuzije')
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("Matrica konfuzije sacuvana kao 'confusion_matrix.png'")

        # Cuvamo izvjestaj
        report = classification_report(y_true, y_pred, target_names=FIRST_LEVEL_CLASSES)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        print("Izvjestaj sacuvan u 'classification_report.txt'")