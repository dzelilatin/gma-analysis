# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg') # Ključno za HPC (onemogućava plt.show())
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. HPC ARGUMENTI ---
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='full') # 'full' ili 'benchmark'
args = parser.parse_args()

# --- 2. POSTAVKE (IDENTIČNE ORIGINALU) ---
data_dir = './data/Training slike' 
test_images_dir = './data/Validacija2'
os.makedirs('./models', exist_ok=True)

img_height, img_width = 512, 512
batch_size = args.batch_size
first_level_classes = ['Face', 'Hand to face', 'Legs', 'Thumb']
num_classes = len(first_level_classes)

# --- 3. DEFINICIJA MODELA (IDENTIČNA ARHITEKTURA) ---
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

# --- 4. GENERATORI (IDENTIČNA AUGMENTACIJA) ---
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(data_dir, target_size=(img_height, img_width),
                                             batch_size=batch_size, class_mode='categorical',
                                             classes=first_level_classes, subset='training')
val_generator = datagen.flow_from_directory(data_dir, target_size=(img_height, img_width),
                                           batch_size=batch_size, class_mode='categorical',
                                           classes=first_level_classes, subset='validation')

# --- 5. KLASNE TEŽINE I SCHEDULER (TVOJA LOGIKA) ---
manual_class_weights = {0: 1.5, 1: 1.0, 2: 1.0, 3: 1.0}
train_labels = train_generator.classes
unique_classes = np.unique(train_labels)
class_weights_dict = {cls: manual_class_weights.get(cls, 1.0) for cls in unique_classes}

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

# --- 6. TRENING ---
model = create_model(num_classes)
epochs = 1 if args.mode == 'benchmark' else 30
steps = (200 // batch_size) if args.mode == 'benchmark' else None

history = model.fit(train_generator,
                    validation_data=val_generator,
                    class_weight=class_weights_dict,
                    callbacks=[lr_scheduler],
                    steps_per_epoch=steps,
                    epochs=epochs)

# Čuvanje
model_tag = f"B{batch_size}_{args.mode}"
model.save(f'./models/model_{model_tag}.h5')
pd.DataFrame(history.history).to_csv(f'./models/history_{model_tag}.csv')

# --- 7. SUBSEQUENT VALIDACIJA I MCC (KOMPLETNA MATEMATIKA) ---
# Učitavanje slika za validaciju
processed_images = []
true_labels = []
class_names = sorted(os.listdir(test_images_dir))

for class_name in class_names:
    class_dir = os.path.join(test_images_dir, class_name)
    if not os.path.isdir(class_dir): continue
    for filename in os.listdir(class_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
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

# Izračunavanje MCC, Sensitivity, Specificity...
confusion_mat = confusion_matrix(true_labels_encoded, predicted_labels)
# (Ista petlja kao u tvom originalu)
report = classification_report(true_labels_encoded, predicted_labels, target_names=class_names)

# Umjesto plt.show() koristimo snimanje fajla
with open(f'./models/report_{model_tag}.txt', 'w') as f:
    f.write(report)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.savefig(f'./models/CM_{model_tag}.png')
