from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
import base64
import io
from PIL import Image
import re
import os

app = Flask(__name__)

# Check if model exists, if not, train it
if not os.path.exists('mnist_model.h5'):
    print("Model file not found. Please run test.py first to train and save the model.")
    exit(1)

try:
    model = load_model('mnist_model.h5')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64 image data from the request
        data = request.get_json()
        image_data = re.sub('^data:image/.+;base64,', '', data['image'])
        
        # Convert base64 to image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Preprocess the image
        image = image.resize((28, 28))
        image = image.convert('L')  # Convert to grayscale
        image_array = np.array(image)
        image_array = image_array.reshape(1, 28, 28, 1)
        image_array = image_array.astype('float32') / 255
        
        # Make prediction
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_digit])
        
        return jsonify({
            'prediction': int(predicted_digit),
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 