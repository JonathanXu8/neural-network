import numpy as np
#from ml.models.sequential import Sequential  # Use your own model class
#from ml.utils.data_loader import load_model  # Your custom model loader
from web_demo.utils import preprocess_image

# Load your trained model from disk
#model = load_model("models/mnist_model.pkl")

def predict_digit(image_bytes):
    return 1

    '''
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)
    return int(np.argmax(prediction))
    '''
