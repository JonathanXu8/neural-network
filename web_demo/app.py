from flask import Flask, render_template, request, jsonify
from predict import predict_digit
import base64
import io
from PIL import Image, ImageOps
import re
import traceback

import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Ensure 'image' key exists
    if 'image' not in data:
        return jsonify({'error': 'Missing image data'}), 400

    # Extract base64 image string
    image_data = data['image']

    # Remove the "data:image/png;base64," prefix
    image_data = re.sub('^data:image/.+;base64,', '', image_data)

    # Decode and convert to image
    try:
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(image_data)

        # Load image and composite onto white background
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
        image = image.convert("L")  # grayscale
        image = ImageOps.invert(image)
        image = image.resize((28, 28), Image.Resampling.NEAREST)

        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        #image_array[image_array >= 0.1] = 1.0

        # Reshape for model
        image_input = image_array.reshape(1, 28, 28)  # or (1, 1, 28, 28) depending on your model

        prediction = predict_digit(image_input)
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print(str(e))
        traceback.print_exc() 
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)