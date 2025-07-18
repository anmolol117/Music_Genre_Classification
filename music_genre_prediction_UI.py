import numpy as np
import librosa
import joblib
import os
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import PhotoImage
from PIL import Image, ImageTk
from pydub import AudioSegment
from pydub.playback import play
import threading

# === Load Artifacts ===
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model = load_model("neural_net_genre.h5")

# === Feature Extraction Function ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)

    features = []

    features.append(float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr))))
    features.append(float(np.var(librosa.feature.chroma_stft(y=y, sr=sr))))

    features.append(float(np.mean(librosa.feature.rms(y=y))))
    features.append(float(np.var(librosa.feature.rms(y=y))))

    features.append(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))))
    features.append(float(np.var(librosa.feature.spectral_centroid(y=y, sr=sr))))

    features.append(float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))))
    features.append(float(np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))))

    features.append(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))))
    features.append(float(np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))))

    features.append(float(np.mean(librosa.feature.zero_crossing_rate(y=y))))
    features.append(float(np.var(librosa.feature.zero_crossing_rate(y=y))))

    harm = librosa.effects.harmonic(y=y)
    perc = librosa.effects.percussive(y=y)
    features.append(float(np.mean(harm)))
    features.append(float(np.var(harm)))
    features.append(float(np.mean(perc)))
    features.append(float(np.var(perc)))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for coeff in mfcc:
        features.append(float(np.mean(coeff)))
        features.append(float(np.var(coeff)))

    return np.array(features, dtype=np.float32)

# === GUI Logic ===
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if not file_path:
        return

    filename = os.path.basename(file_path)
    file_label.config(text=f"Selected File:\n{filename}")

    # Display icon
    audio_icon = Image.open("music_icon.png").resize((50, 50))
    icon_image = ImageTk.PhotoImage(audio_icon)
    icon_label.config(image=icon_image)
    icon_label.image = icon_image  # prevent garbage collection

    try:
        features = extract_features(file_path)
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)
        genre = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        genre_label.config(text=f"🎵 Predicted Genre: {genre}", fg="green")
    except Exception as e:
        genre_label.config(text="❌ Error during prediction", fg="red")
        print("Error:", e)

    threading.Thread(target=play_audio, args=(file_path,), daemon=True).start()


def play_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        play(audio[:10000])  # play first 10 seconds
    except Exception as e:
        print("🔇 Error playing audio:", e)

# === GUI Setup ===
root = tk.Tk()
root.title("Music Genre Classifier")
root.geometry("400x300")
root.resizable(False, False)

title = tk.Label(root, text="🎶 Music Genre Classifier", font=("Helvetica", 16))
title.pack(pady=10)

icon_label = tk.Label(root)
icon_label.pack(pady=5)

file_label = tk.Label(root, text="No file selected", font=("Helvetica", 10))
file_label.pack(pady=5)

select_button = tk.Button(root, text="Select Audio File", command=select_file, font=("Helvetica", 12))
select_button.pack(pady=10)

genre_label = tk.Label(root, text="", font=("Helvetica", 14))
genre_label.pack(pady=15)

root.mainloop()
