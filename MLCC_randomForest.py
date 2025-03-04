import os
import librosa
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Set dataset path
DATASET_PATH = "./dataset" 

# Function to extract MFCC features
def extract_features(file_path, max_pad_len=100):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Pad or truncate to a fixed length
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    return mfcc.flatten()  # Flatten to 1D array

# Load dataset
speakers = []
features = []

for speaker in os.listdir(DATASET_PATH):
    speaker_path = os.path.join(DATASET_PATH, speaker)
    if os.path.isdir(speaker_path):
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                wav_path = os.path.join(speaker_path, file)
                mfcc_features = extract_features(wav_path)
                features.append(mfcc_features)
                speakers.append(speaker)

# Convert labels to numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(speakers)

# Convert features to numpy array
X = np.array(features)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model
with open("speaker_recognition_model.pkl", "wb") as f:
    pickle.dump((clf, label_encoder), f)

print("Model trained and saved!")

# Function to predict speaker from a test file
def predict_speaker(wav_file):
    with open("speaker_recognition_model.pkl", "rb") as f:
        clf, label_encoder = pickle.load(f)

    mfcc_features = extract_features(wav_file)
    mfcc_features = np.array(mfcc_features).reshape(1, -1)
    
    pred = clf.predict(mfcc_features)
    return label_encoder.inverse_transform(pred)[0]

# Example usage
print(predict_speaker("1.wav"))
