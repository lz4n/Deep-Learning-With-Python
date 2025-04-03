import os
import numpy as np
from keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt

dataset_dir = 'var/cats_and_dogs_data'
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')

# Generador de entrenamiento (con aumento de datos)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generador de validación (sin aumento de datos)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flujo de datos desde directorios
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Directorio con 2000 imágenes (1000 cats + 1000 dogs)
    target_size=(150, 150),
    batch_size=20,  # 2000 imágenes / 20 = 100 steps por época
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,  # Directorio de validación
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=['acc']
)

steps_per_epoch = train_generator.samples // train_generator.batch_size  # 2000/20 = 100
validation_steps = validation_generator.samples // validation_generator.batch_size

tra_ds = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 150, 150, 3], [None])
).repeat()

val_ds = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 150, 150, 3], [None])
).repeat()

history = model.fit(
    tra_ds,
    steps_per_epoch=steps_per_epoch,  # 100
    epochs=100,
    validation_data=val_ds,
    validation_steps=validation_steps,  # 50
)

model.save('examples/cnn/cats_and_dogs/cats_and_dogs.h5')

# Plot results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()