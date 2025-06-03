from flask import Flask, render_template, request, jsonify
from web_demo.predict import predict_digit

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.files['image'].read()
    digit = predict_digit(image_data)
    return jsonify({'prediction': digit})

if __name__ == '__main__':
    app.run(debug=True)