# MNIST-Handwritten-Digit-Recognition-with-Deep-Learning-in-Python

Overview
This Python project demonstrates the implementation of a Deep Learning model using TensorFlow for recognizing handwritten digits from the MNIST dataset. The model is a neural network with multiple layers, trained to classify digits from 0 to 9.

Model Architecture
The neural network consists of the following layers:

Flatten Layer:

Input layer to flatten the 28x28 images into a 1D array.
Dense Layers:

Two dense layers with 128 neurons and ReLU activation function.
The final layer with 10 neurons and softmax activation for multi-class classification.
Dataset
The MNIST dataset is used, which contains 28x28 grayscale images of handwritten digits (0-9).

Training and Evaluation
The model is trained using the Adam optimizer and sparse categorical crossentropy loss. After training for three epochs, the model is evaluated on the test dataset, providing accuracy and loss metrics.
