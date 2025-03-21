import os
import numpy as np
import librosa
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = cv2.resize(mfcc, (64, 64))
    return mfcc

def load_data(dataset_path):
    X, y = [], []
    speakers = os.listdir(dataset_path)
    
    for speaker in speakers:
        speaker_path = os.path.join(dataset_path, speaker)
        if os.path.isdir(speaker_path):
            for file in os.listdir(speaker_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(speaker_path, file)
                    mfcc = extract_mfcc(file_path)
                    X.append(mfcc)
                    y.append(speaker)
    
    return np.array(X), np.array(y)

# Set dataset path
dataset_path = r"C:\Users\e0352ax\Desktop\sr\dataset"
X, y = load_data(dataset_path)

# Preprocess Data
X = X / 255.0
X = X.reshape(-1, 64, 64, 1)

# Encode labels and save encoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = tf.keras.utils.to_categorical(y, num_classes=len(np.unique(y)))

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN Model
num_classes = y.shape[1]
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
model.save("speaker_model.h5")
print("Model and Label Encoder saved successfully!")
