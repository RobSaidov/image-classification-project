
# CIFAR-10 Image Classifier

This project is an image classifier built using **TensorFlow** and **Keras**. It uses the **CIFAR-10** dataset, which consists of 60,000 32x32 color images in 10 different classes. The model is designed to classify images into these categories.

## Dataset

The model uses the **CIFAR-10** dataset, which is a popular dataset for image classification. The dataset includes 10 classes:

- Plane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### CIFAR-10 Dataset Structure

- **Training set**: 50,000 images
- **Testing set**: 10,000 images

The dataset is already included in Keras, so it is directly imported through the following code:

```python
from keras import datasets
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
```

## Model Architecture

The model is built using a **Convolutional Neural Network (CNN)**. Below is a summary of the architecture:

1. **Conv2D Layer** with 32 filters (3x3 kernel, ReLU activation)
2. **MaxPooling2D Layer** with a 2x2 pool size
3. **Conv2D Layer** with 64 filters (3x3 kernel, ReLU activation)
4. **MaxPooling2D Layer** with a 2x2 pool size
5. **Conv2D Layer** with 64 filters (3x3 kernel, ReLU activation)
6. **Flatten** layer to convert 2D matrices into a 1D vector
7. **Dense Layer** with 64 units and ReLU activation
8. **Dense Layer** with 10 units and softmax activation for output

### Model Compilation

The model is compiled with the **Adam** optimizer and uses **Sparse Categorical Crossentropy** as the loss function since it's a multiclass classification problem.

```python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```

### Model Training

The model is trained for 10 epochs on the training dataset and validated on the test dataset.

```python
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
```

## Prediction

Once trained, the model can predict the class of a new image. The following code is used for prediction:

```python
img = cv.imread('ship.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'It is a {class_names[index]}')
```

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Make sure you have **Python 3.x** installed.
2. Install the necessary dependencies from the `requirements.txt` file.
3. Run the `main.py` file:
   ```bash
   python main.py
   ```

## Dependencies

This project uses the following dependencies:

- `TensorFlow`
- `Keras`
- `NumPy`
- `OpenCV`
- `Matplotlib`

## Future Improvements

- Improve accuracy by adding more layers or tuning hyperparameters.
- Explore data augmentation to prevent overfitting.
- Experiment with transfer learning using a pre-trained model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
