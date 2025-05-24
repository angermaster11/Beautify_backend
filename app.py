# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import os

# print("actuvae")
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load the trained model
# model = load_model("beauty.h5")  # Make sure this file is in your backend folder

# # Function to preprocess the image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(128, 128))
#     img_array = image.img_to_array(img)
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     try:
#         # Save the uploaded file temporarily
#         upload_path = "temp_upload.jpg"
#         file.save(upload_path)
        
#         # Preprocess and predict
#         processed_img = preprocess_image(upload_path)
#         prediction = model.predict(processed_img)
#         score = float(prediction[0][0])
        
#         # Delete the temporary file
#         os.remove(upload_path)
        
#         return jsonify({'score': score})
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model("beauty.h5")  # Make sure this file is in your backend folder

# Load face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img_path):
    """Detect faces in an image and return cropped face if found"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # Get the largest face
    (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
    
    # Crop and save the face
    face_img = img[y:y+h, x:x+w]
    face_path = "temp_face.jpg"
    cv2.imwrite(face_path, face_img)
    
    return face_path

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
        
        # Detect faces first
        face_path = detect_faces(upload_path)
        if not face_path:
            os.remove(upload_path)
            return jsonify({'error': 'No face detected in the image'}), 400
        
        # Preprocess and predict
        processed_img = preprocess_image(face_path)
        prediction = model.predict(processed_img)
        score = float(prediction[0][0])
        
        # Delete the temporary files
        os.remove(upload_path)
        os.remove(face_path)
        
        return jsonify({
            'score': score,
            'message': 'Face detected and analyzed successfully'
        })
    
    except Exception as e:
        # Clean up temporary files if they exist
        if 'upload_path' in locals() and os.path.exists(upload_path):
            os.remove(upload_path)
        if 'face_path' in locals() and os.path.exists(face_path):
            os.remove(face_path)
            
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
