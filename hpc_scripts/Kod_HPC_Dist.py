# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg') # OBAVEZNO za HPC
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. DISTRIBUIRANA STRATEGIJA (SRCE TESTA 13) ---
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# --- 2. KONFIGURACIJA ---
data_dir = './data/Training slike' 
test_images_dir = './data/Validacija2'
os.makedirs('./models', exist_ok=True)

img_height, img_width = 512, 512
batch_size = 64 # Fiksirano za Test 13
num_classes = 4
first_level_classes = ['Face', 'Hand to face', 'Legs', 'Thumb']

# --- 3. DEFINICIJA MODELA UNUTAR STRATEGIJE ---
with strategy.scope():
    def create_model():
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

    model = create_model()

# --- 4. IDENTIČNI GENERATORI ---
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir, target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='categorical',
    classes=first_level_classes, subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir, target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='categorical',
    classes=first_level_classes, subset='validation'
)

# --- 5. KLASNE TEŽINE I SCHEDULER (IDU UZ TRENING) ---
manual_class_weights = {0: 1.5, 1: 1.0, 2: 1.0, 3: 1.0}
train_labels = train_generator.classes
unique_classes = np.unique(train_labels)
class_weights_dict = {cls: manual_class_weights.get(cls, 1.0) for cls in unique_classes}

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

# --- 6. TRENING ---
print(">>> ZAPOČINJEM DISTRIBUIRANI TRENING (TEST 13)...")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=[lr_scheduler],
    epochs=30
)

# Čuvanje
model.save('./models/TrueAId_Horizontal_Final.h5')
pd.DataFrame(history1.history).to_csv('./models/history_horizontal.csv')

# --- 7. EVALUACIJA (IDENTIČNA MATEMATIKA ZA SENZITIVNOST, MCC...) ---
print(">>> VRŠIM SUBSEQUENT VALIDACIJU...")
processed_images = []
true_labels = []
class_names = sorted(os.listdir(test_images_dir))

for class_name in class_names:
    class_dir = os.path.join(test_images_dir, class_name)
    if not os.path.isdir(class_dir): continue
    for filename in os.listdir(class_dir):
        if filename.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(class_dir, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
            img = img / 255.0
            processed_images.append(img)
            true_labels.append(class_name)

processed_images = np.array(processed_images)
predictions = model.predict(processed_images)
predicted_labels = np.argmax(predictions, axis=1)
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

confusion_mat = confusion_matrix(true_labels_encoded, predicted_labels)
sensitivity, specificity, f1_score, accuracy, mcc = [], [], [], [], []

for i in range(len(class_names)):
    tp = confusion_mat[i, i]
    fp = confusion_mat[:, i].sum() - tp
    fn = confusion_mat[i, :].sum() - tp
    tn = confusion_mat.sum() - (tp + fp + fn)

    sensitivity.append(tp / (tp + fn))
    specificity.append(tn / (tn + fp))
    precision_i = tp / (tp + fp)
    f1_score.append(2 * (precision_i * (tp / (tp + fn))) / (precision_i + (tp / (tp + fn))))
    accuracy.append((tp + tn) / confusion_mat.sum())
    mcc.append((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

# ISPIS REZULTATA U LOG FAJL
with open('./models/report_horizontal_scaling.txt', 'w') as f:
    f.write(f"HORIZONTAL SCALING REPORT (3 NODES)\n")
    f.write("="*30 + "\n")
    for i, class_name in enumerate(class_names):
        f.write(f'Class: {class_name}\n')
        f.write(f'Sensitivity: {sensitivity[i]:.2f}\nSpecificity: {specificity[i]:.2f}\n')
        f.write(f'F1 Score: {f1_score[i]:.2f}\nMCC: {mcc[i]:.2f}\n' + '-'*20 + '\n')

print(">>> SVI REZULTATI SU SAČUVANI U FOLDERU './models'.")
