from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

print("actuvae")
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model("beauty.h5")  # Make sure this file is in your backend folder

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save the uploaded file temporarily
        upload_path = "temp_upload.jpg"
        file.save(upload_path)
        
        # Preprocess and predict
        processed_img = preprocess_image(upload_path)
        prediction = model.predict(processed_img)
        score = float(prediction[0][0])
        
        # Delete the temporary file
        os.remove(upload_path)
        
        return jsonify({'score': score})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)