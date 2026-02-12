import sys
import os
import cv2

sys.path.append("/Users/user/Library/Python/3.9/lib/python/site-packages")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


user_home = os.path.expanduser("~")
base_dir = os.path.join(user_home, "Downloads", "DATA")
data_dir = os.path.join(base_dir, "Training slike")
test_images_dir = os.path.join(base_dir, "Validacija2")

save_path = os.path.join(user_home, "Desktop")

# Postavke za brzi test
img_height, img_width = 512, 512
batch_size = 16
first_level_classes = ['Face', 'Hand to face', 'Legs', 'Thumb']
num_classes = len(first_level_classes)

# OGRANIČENJA ZA TEST (Da ne traje satima)
STEPS_PER_EPOCH = 13  # ~200 slika po epohi
VALIDATION_STEPS = 5  
EPOCHS = 3            

print(f"Trening direktorij: {data_dir}")
print(f"Validacijski direktorij: {test_images_dir}")

# ==========================================
# 2. DEFINICIJA MODELA (Original)
# ==========================================
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

# ==========================================
# 3. GENERATORI (Originalna Augmentacija)
# ==========================================
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2 # Zadržano iako koristimo poseban folder za val, za svaki slučaj
)

# Provjera postojanja foldera
if not os.path.exists(data_dir) or not os.path.exists(test_images_dir):
    print("GREŠKA: Folderi nisu pronađeni u Downloads/DATA. Provjeri putanje.")
    exit()

train_generator = datagen.flow_from_directory(
    data_dir, 
    target_size=(img_height, img_width),
    batch_size=batch_size, 
    class_mode='categorical',
    classes=first_level_classes, 
    subset='training'
)

# Za validaciju koristimo folder Validacija2
val_generator = datagen.flow_from_directory(
    test_images_dir, 
    target_size=(img_height, img_width),
    batch_size=batch_size, 
    class_mode='categorical',
    classes=first_level_classes
)

# ==========================================
# 4. TEŽINE I CALLBACKS (Original)
# ==========================================
manual_class_weights = {0: 1.5, 1: 1.0, 2: 1.0, 3: 1.0}
train_labels = train_generator.classes
unique_classes = np.unique(train_labels)
class_weights_dict = {cls: manual_class_weights.get(cls, 1.0) for cls in unique_classes}

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)

# ==========================================
# 5. TRENING (Prilagođen za brzi test)
# ==========================================
model = create_model(num_classes)

print("Počinjem trening...")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=[lr_scheduler],
    epochs=EPOCHS,                     # 3 epohe
    steps_per_epoch=STEPS_PER_EPOCH,   # Limit na ~200 slika
    validation_steps=VALIDATION_STEPS  # Limit validacije
)

# Snimanje modela na Desktop
model_save_path = os.path.join(save_path, 'novimodel_test.h5')
csv_save_path = os.path.join(save_path, 'novimodel_test.csv')

model.save(model_save_path)
history_df = pd.DataFrame(history1.history)
history_df.to_csv(csv_save_path)
print(f"Model sačuvan na: {model_save_path}")

# ==========================================
# 6. PLOTOVI (Original)
# ==========================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'], label='Train Accuracy')
plt.plot(history1.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'], label='Train Loss')
plt.plot(history1.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 7. INTERNA VALIDACIJA (Original)
# ==========================================
# Reset generatora i limitiranje koraka za predikciju da se ne vrti u krug
val_generator.reset()
steps = VALIDATION_STEPS if VALIDATION_STEPS else len(val_generator)

y_pred_probs = model.predict(val_generator, steps=steps)
y_pred = np.argmax(y_pred_probs, axis=1)

# Uzimanje stvarnih labela (moramo ih izvuci rucno jer generator moze biti infinite loop)
y_true = []
for i in range(steps):
    _, labels = next(val_generator)
    y_true.extend(np.argmax(labels, axis=1))
y_true = np.array(y_true)
# Sijecemo visak predikcija ako ih ima zbog batch velicine
y_pred = y_pred[:len(y_true)]

cm = confusion_matrix(y_true, y_pred)
# Dodajemo malu vrijednost da izbjegnemo dijeljenje s nulom ako je matrica prazna u testu
cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

plt.figure(figsize=(10,10))
sns.heatmap(cm_norm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix (Internal)', size=15)
plt.show()

# ==========================================
# 8. DETALJNA VALIDACIJA IZ FOLDERA (Original)
# ==========================================
print("Pokrećem detaljnu validaciju na slikama iz foldera...")

# Ucitavanje modela (ili koristenje postojeceg iz memorije)
model = load_model(model_save_path)

processed_images = []
true_labels = []

# Provjera da li folder postoji prije petlje
if os.path.exists(test_images_dir):
    class_names = sorted(os.listdir(test_images_dir))
    
    # Filtriranje da ne uzima .DS_Store fajlove na Macu
    class_names = [c for c in class_names if not c.startswith('.')]

    for class_name in class_names:
        class_dir = os.path.join(test_images_dir, class_name)
        if not os.path.isdir(class_dir): continue
        
        # Limit slika za test mod (da ne ucitava 1000 slika)
        count = 0 
        for filename in os.listdir(class_dir):
            if count > 20: break # Uzmi samo 20 slika po klasi za brzi test
            
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(class_dir, filename)
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (img_height, img_width))
                    img = img / 255.0
                    processed_images.append(img)
                    true_labels.append(class_name)
                    count += 1
    
    processed_images = np.array(processed_images)

    if len(processed_images) > 0:
        # Predikcije
        predictions = model.predict(processed_images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        label_encoder = LabelEncoder()
        # Forsiramo originalne klase da budemo sigurni u redoslijed
        label_encoder.fit(first_level_classes)
        true_labels_encoded = label_encoder.transform(true_labels)

        # Kalkulacije (Originalni kod)
        confusion_mat = confusion_matrix(true_labels_encoded, predicted_labels)
        
        sensitivity = []
        specificity = []
        f1_score = []
        accuracy = []
        mcc = []
        
        # Ovdje koristimo first_level_classes umjesto class_names da osiguramo tacnu dimenziju
        for i in range(len(first_level_classes)):
            # Zastita ako matrica konfuzije nije puna (ako neke klase falile u testu)
            if i >= confusion_mat.shape[0] or i >= confusion_mat.shape[1]:
                sensitivity.append(0); specificity.append(0); f1_score.append(0); accuracy.append(0); mcc.append(0)
                continue

            true_positive = confusion_mat[i, i]
            false_positive = confusion_mat[:, i].sum() - true_positive
            false_negative = confusion_mat[i, :].sum() - true_positive
            true_negative = confusion_mat.sum() - (true_positive + false_positive + false_negative)

            # Dodato + 1e-10 da se izbjegne dijeljenje sa nulom
            sensitivity_i = true_positive / (true_positive + false_negative + 1e-10)
            specificity_i = true_negative / (true_negative + false_positive + 1e-10)
            precision_i = true_positive / (true_positive + false_positive + 1e-10)
            recall_i = sensitivity_i
            f1_score_i = 2 * (precision_i * recall_i) / (precision_i + recall_i + 1e-10)
            accuracy_i = (true_positive + true_negative) / (confusion_mat.sum() + 1e-10)
            
            mcc_nom = (true_positive * true_negative - false_positive * false_negative)
            mcc_denom = np.sqrt((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative)) + 1e-10
            mcc_i = mcc_nom / mcc_denom

            sensitivity.append(sensitivity_i)
            specificity.append(specificity_i)
            f1_score.append(f1_score_i)
            accuracy.append(accuracy_i)
            mcc.append(mcc_i)

        # Plot Conf Matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title('Confusion Matrix (Detailed)')
        plt.colorbar()
        tick_marks = np.arange(len(first_level_classes))
        plt.xticks(tick_marks, first_level_classes, rotation=45)
        plt.yticks(tick_marks, first_level_classes)

        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                plt.text(j, i, str(confusion_mat[i, j]), horizontalalignment='center', verticalalignment='center')

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        # Ispis metrika
        for i, class_name in enumerate(first_level_classes):
            print(f'Class: {class_name}')
            print(f'Sensitivity: {sensitivity[i]:.2f}')
            print(f'Specificity: {specificity[i]:.2f}')
            print(f'F1 Score: {f1_score[i]:.2f}')
            print(f'Accuracy: {accuracy[i]:.2f}')
            print(f'MCC: {mcc[i]:.2f}')
            print('-' * 20)

        class_report = classification_report(true_labels_encoded, predicted_labels, target_names=first_level_classes)
        print(class_report)
    else:
        print("Nisu pronađene slike u Validacija2 folderu.")
else:
    print(f"Folder {test_images_dir} ne postoji.")