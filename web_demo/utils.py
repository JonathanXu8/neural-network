import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    image = np.array(image).astype(np.float32) / 255.0
    image = 1.0 - image  # Invert if background is black and digit is white
    return image.reshape(1, 28 * 28)