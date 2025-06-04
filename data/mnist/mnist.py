import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt
import random
import os

def load_images(filename):
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'rb') as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number in image file: {magic}")
        image_data = array("B", f.read())
    images = []
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols], dtype=np.uint8)
        img = img.reshape(rows, cols)
        images.append(img)
    return images

def load_labels(filename):
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number in label file: {magic}")
        labels = array("B", f.read())
    return labels

def load_mnist():
    x_train = load_images('train-images-idx3-ubyte')
    y_train = load_labels('train-labels-idx1-ubyte')
    x_test = load_images('t10k-images-idx3-ubyte')
    y_test = load_labels('t10k-labels-idx1-ubyte')

    return (x_train, y_train), (x_test, y_test) 

def show_random_images(images, labels, n=10):
    plt.figure(figsize=(15, 5))
    indices = random.sample(range(len(images)), n)
    for i, idx in enumerate(indices):
        plt.subplot(2, n//2, i+1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"Label: {labels[idx]}")
        plt.axis('off')
    plt.show()

'''
EXAMPLE USEAGE

# load data
print("Loading data...")
(train_images, train_labels), (test_images, test_labels) = load_mnist()

print(f"Training set: {len(train_images)} images")
print(f"Test set: {len(test_images)} images")

# visualize  samples
show_random_images(train_images, train_labels, n=10)

# convert to numpy arrays
x_train = np.array(train_images, dtype=np.float32)
y_train = np.array(train_labels, dtype=np.uint8)
x_test = np.array(test_images, dtype=np.float32)
y_test = np.array(test_labels, dtype=np.uint8)

# normalize pixel values to [0,1]
x_train /= 255.0
x_test /= 255.0

np.set_printoptions(precision=2, suppress=True, linewidth=200)
print(x_train[0])
'''