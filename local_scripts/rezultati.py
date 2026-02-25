import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Putanje
model_path = r'/Users/user/Desktop/gma-analysis/models/new_model.h5'
test_images_dir = r'/Users/user/Desktop/gma-analysis/dataset/data/Validacija2'
class_names = sorted(['Face', 'Hand to face', 'Legs', 'Thumb'])

# 2. Učitavanje modela
model = load_model(model_path)

# 3. Učitavanje testnih slika (isto kao u tvom kodu)
processed_images = []
true_labels = []
for class_name in class_names:
    class_dir = os.path.join(test_images_dir, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(class_dir, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
            img = img / 255.0
            processed_images.append(img)
            true_labels.append(class_name)

processed_images = np.array(processed_images)

# 4. Predikcije
predictions = model.predict(processed_images)
predicted_labels = np.argmax(predictions, axis=1)

# 5. FIX ZA GREŠKU (Encoding)
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# 6. Finalni izvještaj (Ispis na ekran)
print("\n--- FINALNI IZVJEŠTAJ ZA TRUEAID MODEL ---")
report = classification_report(true_labels_encoded, predicted_labels, target_names=class_names)
print(report)

# --- OVDJE STAVLJAŠ NOVI KOD ZA SAČUVAVANJE ---

# Pretvaranje izvještaja u rječnik, pa u DataFrame
report_dict = classification_report(true_labels_encoded, predicted_labels, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Čuvanje kao CSV (za tvoju tabelu u radu)
csv_report_path = r'/Users/user/Desktop/gma-analysis/models/report_TrueAId_batch16.csv'
report_df.to_csv(csv_report_path)

print(f"Tabela sa rezultatima je sačuvana na: {csv_report_path}")