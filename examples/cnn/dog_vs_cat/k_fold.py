# Validación K-fold
import tensorflow
import numpy as np
from keras import models, layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

def load_data(data_dir, target_size=(150, 150)):
    images = []
    labels = []
    for label in ['cat', 'dog']:
        class_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            images.append(img_array)
            labels.append(1 if label == 'dog' else 0)
    return np.array(images), np.array(labels)

test_dir = 'C:/Users/izanp/Downloads/cats_and_dogs_data/test'
print("Cargando datos de test...")
X_test, y_test = load_data(test_dir)

# Hacer shuffle de los datos manteniendo la correspondencia
indices = np.arange(len(X_test))
np.random.shuffle(indices)
X_test = X_test[indices]
y_test = y_test[indices]

print("Cargando modelo pre-entrenado...")
model = load_model('cats_and_dogs.h5')

k = 4  # Número de particiones
n_samples = len(X_test)
fold_size = n_samples // k

all_scores = []

for fold in range(k):
    print(f"\nAnalizando fold: #{fold + 1}")
    
    X_val_fold = X_test[fold_size * fold: fold_size * (fold + 1)]
    y_val_fold = y_test[fold_size * fold: fold_size * (fold + 1)]
    
    val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    print(f"Fold #{fold + 1} - Accuracy: {val_acc:.4f}")
    
    all_scores.append(val_acc)

print("\nResultados finales:")
print(f"Media del k-fold: {np.mean(all_scores):.4f}")
print(f"Desviación típica del k-fold: {np.std(all_scores):.4f}")