import os
import numpy as np
import librosa
import cv2
import pickle
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
MODEL_PATH = "speaker_model.h5"
model = load_model(MODEL_PATH)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = cv2.resize(mfcc, (64, 64))
    return mfcc.reshape(1, 64, 64, 1) / 255.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    mfcc = extract_mfcc(filepath)
    prediction = model.predict(mfcc)
    speaker_id = np.argmax(prediction)
    predicted_speaker = encoder.inverse_transform([speaker_id])[0]

    os.remove(filepath)

    return jsonify({"predicted_speaker": predicted_speaker})

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
