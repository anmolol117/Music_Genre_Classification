import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# === Load Trained Components ===
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model = load_model("neural_net_genre.h5")

# === Feature Extraction Function ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)

    features = []

    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(float(np.mean(chroma)))
    features.append(float(np.var(chroma)))

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))
    features.append(float(np.var(rms)))

    # Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(spec_cent)))
    features.append(float(np.var(spec_cent)))

    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(float(np.mean(spec_bw)))
    features.append(float(np.var(spec_bw)))

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(rolloff)))
    features.append(float(np.var(rolloff)))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.append(float(np.mean(zcr)))
    features.append(float(np.var(zcr)))

    # Harmonic & Percussive
    harm = librosa.effects.harmonic(y=y)
    perc = librosa.effects.percussive(y=y)
    features.append(float(np.mean(harm)))
    features.append(float(np.var(harm)))
    features.append(float(np.mean(perc)))
    features.append(float(np.var(perc)))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for coeff in mfcc:
        features.append(float(np.mean(coeff)))
        features.append(float(np.var(coeff)))

    return np.array(features, dtype=np.float32)


# === Predict Genre ===
def predict_genre_from_file(audio_file_path):
    print(f"\n🔍 Analyzing: {audio_file_path}")
    try:
        features = extract_features(audio_file_path)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        print("\n🎵 Predicted Genre:", predicted_label[0])
    except Exception as e:
        print("❌ Error during prediction:", e)

# === GUI File Picker ===
if __name__ == "__main__":
    print("🎧 Select an audio file (WAV/MP3)...")

    root = Tk()
    root.withdraw()  # Hide tkinter root window
    file_path = askopenfilename(title="Choose an audio file",
                                 filetypes=[("Audio Files", "*.wav *.mp3")])

    if file_path:
        predict_genre_from_file(file_path)
    else:
        print("❌ No file selected.")
