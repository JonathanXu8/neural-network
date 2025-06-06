import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
'''
np.set_printoptions(
    linewidth=200,   # wider lines so rows don't wrap
    threshold=np.inf,  # print full array (no "..." truncation)
    precision=2,       # round floats to 2 decimal places
    suppress=True      # don't use scientific notation for small numbers
)
'''

from PIL import Image
from scipy.ndimage import center_of_mass, shift

from ml.models.model import Model

# Load your trained model from disk
model = Model()
model.load("saved_models/mnist.pkl")

def predict_digit(image):
    image = np.expand_dims(image, axis=0) # so it works with predict (which is written for batches) - basically makes it a batch of size 1

    # center image
    _, _, y, x = center_of_mass(image)
    delta_y = 14 - y if not np.isnan(14 - y) else 0
    delta_x = 14 - x if not np.isnan(14 - x) else 0
    image = shift(image, shift=[0, 0, round(delta_y), round(delta_x)], mode='constant', cval=0)
    image = np.abs(image) # -0.0 --> 0.0

    # make prediction
    prediction = model.predict(image)
    return prediction
