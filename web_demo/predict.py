import numpy as np
import math
#from ml.models.sequential import Sequential
#from ml.utils.data_loader import load_model
from utils import preprocess_image

# Load your trained model from disk
#model = load_model("models/mnist_model.pkl")

def predict_digit(image_bytes):
    return math.random()

    '''
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)
    return int(np.argmax(prediction))
    '''
