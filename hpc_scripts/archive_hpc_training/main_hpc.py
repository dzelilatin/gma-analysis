import sys
import os
import matplotlib
matplotlib.use('Agg')  # OBAVEZNO ZA HPC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. POSTAVKE
# ==========================================
MODE = 'TEST'  # <--- PROMIJENI U 'FULL' KAD SI SPREMNA

# Apsolutne putanje na osnovu onoga sto si rekla
BASE_DIR = os.getcwd() # Ovo je tvoj home folder (/home/dzelila)
PROJECT_DIR = os.path.join(BASE_DIR, "trueaid")

# Ovdje koristimo 'data' folder jer on ima 'Training slike' i 'Validacija'
# Pazi na velika/mala slova!
TRAIN_DIR = os.path.join(PROJECT_DIR, "data", "Training slike")
VAL_DIR = os.path.join(PROJECT_DIR, "data", "Validacija") 
# (Ako se folder zove "Validacija2" ili samo "Validacija", provjeri sa 'ls')

SAVE_PATH = BASE_DIR # Rezultate cuvamo vani da ih lakse nadjes

IMG_HEIGHT, IMG_WIDTH = 512, 512
BATCH_SIZE = 16
# Tvoje klase (Ignorisemo 'pathological' u croped folderu za sada)
FIRST_LEVEL_CLASSES = ['Face', 'Hand to face', 'Legs', 'Thumb']
NUM_CLASSES = len(FIRST_LEVEL_CLASSES)

if MODE == 'TEST':
    print("\n--- TEST MOD (Brza provjera) ---")
    EPOCHS = 2
    STEPS = 10
    VAL_STEPS = 5
else:
    print("\n--- FULL MOD (Svi podaci) ---")
    EPOCHS = 30
    STEPS = None
    VAL_STEPS = None

# ==========================================
# 2. PROVJERA FOLDERA
# ==========================================
print(f"Trazim trening slike u: {TRAIN_DIR}")
if not os.path.exists(TRAIN_DIR):
    print("GRESKA: Folder ne postoji! Provjeravam sta ima u 'trueaid/data'...")
    try:
        print(os.listdir(os.path.join(PROJECT_DIR, "data")))
    except:
        print("Ne mogu ni uci u data folder.")
    sys.exit(1)

# ==========================================
# 3. GENERATORI
# ==========================================
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
    # Ne koristimo validation_split ovdje jer imamo poseban folder
)

print("Ucitavam Trening set...")
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=FIRST_LEVEL_CLASSES,
    subset=None
)

print("Ucitavam Validacijski set...")
# Provjera da li validacija postoji, ako ne, koristi trening split
if os.path.exists(VAL_DIR):
    val_generator = datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=FIRST_LEVEL_CLASSES,
        shuffle=False
    )
else:
    print(f"UPOZORENJE: Nisam nasao {VAL_DIR}. Koristim dio treninga za validaciju.")
    # Moramo re-inicijalizirati datagen sa splitom
    datagen_split = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen_split.flow_from_directory(TRAIN_DIR, target_size=(512,512), batch_size=16, class_mode='categorical', classes=FIRST_LEVEL_CLASSES, subset='training')
    val_generator = datagen_split.flow_from_directory(TRAIN_DIR, target_size=(512,512), batch_size=16, class_mode='categorical', classes=FIRST_LEVEL_CLASSES, subset='validation', shuffle=False)

# ==========================================
# 4. MODEL I TRENING
# ==========================================
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'), # Smanjio sam malo Dense sloj za brzi test
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Rucne tezine (kako si trazila)
manual_class_weights = {0: 1.5, 1: 1.0, 2: 1.0, 3: 1.0}
class_weights_dict = {cls: manual_class_weights.get(cls, 1.0) for cls in np.unique(train_generator.classes)}

history = model.fit(
    train_generator,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    epochs=EPOCHS,
    steps_per_epoch=STEPS,
    validation_steps=VAL_STEPS
)

# ==========================================
# 5. SPREMANJE
# ==========================================
suffix = "_test" if MODE == 'TEST' else "_full"
model.save(os.path.join(SAVE_PATH, f'model{suffix}.h5'))
pd.DataFrame(history.history).to_csv(os.path.join(SAVE_PATH, f'history{suffix}.csv'))

# Report
y_pred = np.argmax(model.predict(val_generator, steps=(VAL_STEPS if VAL_STEPS else len(val_generator))), axis=1)
y_true = val_generator.classes[:len(y_pred)]
report = classification_report(y_true, y_pred, target_names=FIRST_LEVEL_CLASSES)

with open(os.path.join(SAVE_PATH, f"report{suffix}.txt"), "w") as f:
    f.write(report)

print("GOTOVO! Provjeri fajlove u home folderu.")