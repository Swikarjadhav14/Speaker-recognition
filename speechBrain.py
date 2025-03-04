import torchaudio
import torch
from speechbrain.pretrained import SpeakerRecognition

# Load pre-trained X-vector model
recognizer = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmp_model")

# Function to load and process audio
def load_audio(audio_path):
    """Loads an audio file and converts it to a tensor."""
    signal, fs = torchaudio.load(audio_path)
    return signal

# Function to verify speaker similarity score
def verify_speaker(audio1_path, audio2_path):
    """Compares two audio files and returns a similarity score."""
    audio1 = load_audio(audio1_path)
    audio2 = load_audio(audio2_path)

    score, _ = recognizer.verify_batch(audio1, audio2)
    return score.item()

# Function to classify an unknown speaker
def classify_speaker(test_audio_path, known_speakers):
    """Compare test_audio with multiple known speakers and return the most similar speaker."""
    scores = {}
    for speaker, sample_audio_path in known_speakers.items():
        score = verify_speaker(test_audio_path, sample_audio_path)
        scores[speaker] = score

    return max(scores, key=scores.get), scores

dataset_path = "C:/Users/e0352ax/Desktop/16000_pcm_speeches/dataset/"

known_speakers = {
    "Benjamin_Netanyahu": dataset_path + "Benjamin_Netanyau/1.wav",
    "Jens_Stoltenberg": dataset_path + "Jens_Stoltenberg/1.wav",
    "Julia_Gillard": dataset_path + "Julia_Gillard/1.wav",
    "Margaret_Thatcher": dataset_path + "Magaret_Tarcher/1.wav",
    "Nelson_Mandela": dataset_path + "Nelson_Mandela/1.wav"
}

# Test with an unknown speaker audio
test_audio = r"C:\Users\e0352ax\Desktop\16000_pcm_speeches\dataset\Magaret_Tarcher\4.wav"


# Predict speaker
predicted_speaker, all_scores = classify_speaker(test_audio, known_speakers)

print(f"Predicted Speaker: {predicted_speaker}")
print("All Scores:", all_scores)
