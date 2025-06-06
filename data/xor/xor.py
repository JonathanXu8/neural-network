# xor_data.py

import numpy as np
import os
import random
import matplotlib.pyplot as plt

def generate_xor_data(n_per_quadrant=500, noise=0.1, random_seed=42):
    np.random.seed(random_seed)

    # Define the four corner centers
    centers = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])

    # labels: (0,0)=0, (0,1)=1, (1,0)=1, (1,1)=0
    labels = np.array([0, 1, 1, 0])

    X_list = []
    y_list = []

    for center, label in zip(centers, labels):
        pts = np.random.normal(loc=center, scale=noise, size=(n_per_quadrant, 2))
        X_list.append(pts)
        y_list.append(np.full(n_per_quadrant, label, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # shuffle the combined dataset
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def load_xor():
    # generate 2000 points
    X, y = generate_xor_data(n_per_quadrant=500, noise=0.1, random_seed=42)
    total = len(X)
    split = int(0.8 * total)

    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    return (X_train, y_train), (X_test, y_test)


def show_xor_scatter(X, y, n=200):
    N = len(X)
    indices = random.sample(range(N), min(n, N))
    pts = X[indices]
    labels = y[indices]

    plt.figure(figsize=(6, 6))
    plt.scatter(pts[labels == 0, 0], pts[labels == 0, 1], c='tab:blue', edgecolor='k', label='Class 0')
    plt.scatter(pts[labels == 1, 0], pts[labels == 1, 1], c='tab:orange', edgecolor='k', label='Class 1')
    plt.title("XOR Data Scatter (subset of {} points)".format(len(indices)))
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()
    plt.grid(True)
    plt.show()

'''
EXAMPLE USEAGE


print("Generating XOR data...")
(X_train, y_train), (X_test, y_test) = load_xor()
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")


show_xor_scatter(X_train, y_train, n=300)

print(X_test[0])
print(y_test[0])
'''