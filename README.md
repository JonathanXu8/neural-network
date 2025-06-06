## Neural Network From Scratch

A fully functional machine learning library implemented **from scratch in NumPy**, complete with core layers, activation functions, and a working web demo for handwritten digit recognition using **Flask** and the **MNIST dataset**.

## Demo

[Demo](google.com)

## Directory Structure

<pre> ```text project-root/ ├── data/ # Datasets and data loaders │ ├── mnist.py │ └── ... ├── ml/ # Core machine learning modules │ ├── layers/ # Layer implementations (Dense, Conv2D, etc.) │ ├── activations/ # Activation functions (ReLU, Softmax, etc.) │ ├── losses/ # Loss functions │ ├── models/ # Model class and training logic │ └── utils.py ├── examples/ # Example training/testing scripts │ └── mnist_test.py ├── README.md └── requirements.txt ``` </pre>

## Features
**Layers**
- Dense
- Convolutional
- flatten
- pooling

**Activations**
- ReLU
- Sigmoid
- Softmax

**Losses**
- Cross Entropy Loss
- Mean Squared Error

## Usage

Example of how to create a network to classify mnist data (data import not included)

 ```python
import numpy as np
from ml.layers.dense import Dense
from ml.layers.flatten import Flatten
from ml.activations.relu import ReLU
from ml.losses.cross_entropy import CrossEntropyLoss
from ml.activations.softmax import Softmax
from ml.models.model import Model

model  =  Model(
	layers = [
		Flatten(),
		Dense(28*28, 64),
		ReLU(),
		Dense(64, 256),
		ReLU(),
		Dense(256, 128),
		ReLU(),
		Dense(128, 64),
		ReLU(),
		Dense(64, 10),
		Softmax()
	],
	loss=CrossEntropyLoss()
	)
  
model.train(5, 20, 0.01, x_train, y_train, x_test, y_test)
model.save('saved_models/mnist.pkl')
```
