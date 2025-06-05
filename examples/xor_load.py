'''
Loads xor model from pkl file
'''

import numpy as np
from data.xor.xor import load_xor
from ml.models.model import Model

# load XOR data
(x_train, y_train), (x_test, y_test) = load_xor()

# convert to NumPy float arrays and one-hot encode labels
x_train = np.array(x_train, dtype=np.float32)
x_test  = np.array(x_test,  dtype=np.float32)

# one-hot encode labels
num_classes = 2
y_train = np.eye(num_classes, dtype=np.float32)[y_train]
y_test  = np.eye(num_classes, dtype=np.float32)[y_test]

model = Model()

model.load('saved_models/xor.pkl')

print(f'Acc: {model.test(x_test, y_test):.4f}')