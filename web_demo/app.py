from flask import Flask, render_template, request, jsonify
from predict import predict_digit
import base64
import io
from PIL import Image
import re

import random

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
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')  # grayscale

        #prediction = predict_digit(image)
        prediction = random.randint(1,10)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)