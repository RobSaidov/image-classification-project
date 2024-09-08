# Import modules

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, labels

# Loading the CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

#Normalizing the pixel values between 0-1
training_images = training_images / 255
testing_images = testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# plt.imshow(training_images[1])
# plt.show()
#
# print(training_images.shape)
# print(training_labels.shape)
