## Neural Network From Scratch

A fully functional machine learning library implemented **from scratch in NumPy**, complete with core layers, activation functions, and a working web demo for handwritten digit recognition using **Flask** and the **MNIST dataset**.

## MNIST Demo

[Demo](https://www.render.com)

## Directory Structure

<pre>
project-root/ 
├── data/ # Datasets and data loaders 
│ ├── mnist/ # Mnist data set 
│ └── xor/ # XOR data set
├── examples/ # Example training/testing scripts 
│ └── mnist.py 
├── ml/ # Core machine learning modules 
│ ├── layers/ # Layer implementations (Dense, Conv2D, etc.) 
│ ├── activations/ # Activation functions (ReLU, Softmax, etc.) 
│ ├── losses/ # Loss functions 
│ ├── models/ # Model class and training logic 
│ └── utils/ # Utility functions
├── saved_models/ # Directory to save trained models
├── web_demo/ # React Web Demo
└── README.md 
</pre>

## Features
**Layers**
- Dense
- Convolutional
- Flatten
- Pooling

**Activations**
- ReLU
- Sigmoid
- Softmax

**Losses**
- Cross Entropy Loss
- Mean Squared Error

## Usage

How to create a network to classify mnist data

 ```python
import numpy as np
from ml.layers.dense import Dense
from ml.layers.flatten import Flatten
from ml.activations.relu import ReLU
from ml.losses.cross_entropy import CrossEntropyLoss
from ml.activations.softmax import Softmax
from ml.models.model import Model
from data.mnist.mnist import load_mnist

# load mnist data
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_mnist()

# convert to numpy arrays and normalize pixel values to [0,1]
x_train = np.array(x_train_raw, dtype=np.float32) / 255.0
y_train = np.array(y_train_raw, dtype=np.uint8)
x_test = np.array(x_test_raw,  dtype=np.float32) / 255.0
y_test = np.array(y_test_raw,  dtype=np.uint8)

# one hot encode labels
num_classes = 10
y_train = np.eye(num_classes, dtype=np.uint8)[y_train]
y_test = np.eye(num_classes, dtype=np.uint8)[y_test]

model = Model(
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
