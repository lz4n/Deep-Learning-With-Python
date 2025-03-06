import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('cats_and_dogs.h5')

img_path = "C:/Users/izanp/Downloads/NationalGeographic_2572187_3x2.png"

img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0


prediction = model.predict(img_array)

if prediction[0] > 0.5:
    print("Es un perro!")
else:
    print("Es un gato!")
