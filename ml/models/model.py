import numpy as np
import pickle

from ml.layers.dense import Dense
from ml.layers.conv import Conv2D
from ml.layers.flatten import Flatten
from ml.layers.pooling import MaxPooling2D

from ml.activations.relu import ReLU
from ml.activations.sigmoid import Sigmoid
from ml.activations.softmax import Softmax

from ml.losses.cross_entropy import CrossEntropyLoss
from ml.losses.mse import MSELoss

class Model:
    def __init__(self, layers=None, loss=None):
        self.layers = layers
        self.loss = loss
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
    
    def train_step(self, x_batch, y_batch, learning_rate):
        logits = self.forward(x_batch)

        loss_val = self.loss.forward(logits, y_batch)
        grad = self.loss.backward()

        self.backward(grad, learning_rate)

        return loss_val

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
    
    def test(self, x_test, y_test):
        accuracy = 0
        for x, y in zip(x_test, y_test):
            x = np.expand_dims(x, axis=0)
            if self.predict(x) == np.argmax(y):
                accuracy += 1
        return accuracy / len(x_test)
    
    def train(self, epochs, batch_size, learning_rate, x_train, y_train, x_test=None, y_test=None):
        for epoch in range(1, epochs+1):
            # permute training data
            perm = np.random.permutation(x_train.shape[0])
            x_train = x_train[perm]
            y_train = y_train[perm]

            epoch_losses = []

            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                loss = self.train_step(x_batch, y_batch, learning_rate)

                epoch_losses.append(loss)

            avg_loss = np.mean(loss)

            if x_test is not None and y_test is not None:
                accuracy = self.test(x_test, y_test)
                print(f"Epoch {epoch:3d}  Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}")
            else:
                print(f"Epoch {epoch:3d}  Loss: {avg_loss:.4f}")
    
    def save(self, path):
        class_to_string = {
            Dense: 'dense',
            Conv2D: 'conv',
            Flatten: 'flatten',
            MaxPooling2D: 'maxpool',
            ReLU: 'relu',
            Sigmoid: 'sigmoid',
            Softmax: 'softmax',
            CrossEntropyLoss: 'celoss',
            MSELoss: 'mseloss'
        }

        layers = []
        weights_and_biases = []
        for layer in self.layers:
            layers.append(class_to_string[type(layer)])
            weights_and_biases.append(layer.save())
        
        loss_fn = class_to_string[type(self.loss)]

        params = (layers, weights_and_biases, loss_fn)

        with open(path, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, path):
        string_to_class = {
            'dense': Dense,
            'conv': Conv2D,
            'flatten': Flatten,
            'maxpool': MaxPooling2D,
            'relu': ReLU,
            'sigmoid': Sigmoid,
            'softmax': Softmax,
            'celoss': CrossEntropyLoss,
            'mseloss': MSELoss
        }

        with open(path, 'rb') as f:
            params = pickle.load(f)
        
        classes, weights_and_biases, loss_fn = params

        layers = []
        for layer in classes:
            layers.append(string_to_class[layer]())


        for layer, weight_and_bias in zip(layers, weights_and_biases):
            layer.load(weight_and_bias)
        
        self.layers = layers
        self.loss = string_to_class[loss_fn]()