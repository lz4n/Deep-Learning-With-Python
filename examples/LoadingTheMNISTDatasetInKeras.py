from keras.utils import to_categorical
from keras.datasets import mnist
from keras import models
from keras import layers

def build_network_architecture():
    network = models.Sequential()
    network.add(layers.Dense(512,
        activation='relu',
        input_shape=(28 * 28,)
    ))
    network.add(layers.Dense(10,
        activation='softmax'
    ))
    
    return network

def normalize_image_data(train_images, test_images):
    return (train_images.reshape((60000, 28 * 28)).astype('float32') / 255,
        test_images.reshape((10000, 28 * 28)).astype('float32') / 255)


def encode_labels(train_labels, test_labels):
    return to_categorical(train_labels), to_categorical(test_labels)

#Load MNIST dataset step
((train_images, train_labels), (test_images, test_labels)) = mnist.load_data()

#Network architecture step
network = build_network_architecture()

#Loss function & optimizer compiler step
network.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Preparing image data step
train_images, test_images = normalize_image_data(train_images, test_images)

#Preparing labels step
train_labels, test_labels = encode_labels(train_labels, test_labels)

#Training
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#Training result
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(f'test_acc: {int(test_acc *100)}%')