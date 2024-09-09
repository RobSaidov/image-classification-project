# Import modules
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras import datasets, models, layers

# Loading the CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalizing the pixel values between 0-1
training_images = training_images / 255
testing_images = testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

# Build the model (uncomment to train)
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
#
# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss = {loss}")
# print(f'Accuracy = {accuracy}')
#
# model.save('image_classifier.keras')

# Load the trained model
model = keras.models.load_model('image_classifier.keras')

# Preprocess the image to make prediction
img = cv.imread('images/frog.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))  # Resize to 32x32

plt.imshow(img, cmap=plt.cm.binary)

# Normalize and predict
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'It is a {class_names[index]}')
