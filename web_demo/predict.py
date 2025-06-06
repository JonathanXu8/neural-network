import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
np.set_printoptions(
    linewidth=200,   # wider lines so rows don't wrap
    threshold=np.inf,  # print full array (no "..." truncation)
    precision=2,       # round floats to 2 decimal places
    suppress=True      # don't use scientific notation for small numbers
)

from PIL import Image

from ml.models.model import Model

# Load your trained model from disk
model = Model()
model.load("saved_models/mnist.pkl")

def predict_digit(image):
    #print(image)

    prediction = model.predict(image)

    return prediction
