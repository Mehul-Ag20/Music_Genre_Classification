import os
import csv
import librosa
import numpy as np


# Function to extract features from a WAV file
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    harmony = librosa.effects.harmonic(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)

    features = [
        np.mean(chroma_stft), np.var(chroma_stft),
        np.mean(rms), np.var(rms),
        np.mean(spectral_centroid), np.var(spectral_centroid),
        np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
        np.mean(rolloff), np.var(rolloff),
        np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
        np.mean(harmony), np.var(harmony)
    ]

    for i in range(20):
        features.extend([np.mean(mfccs[i]), np.var(mfccs[i])])

    return features

# Path to the directory containing the WAV files
directory = '../Data/wav_data'

# List to store features and labels
data = []
c=0
# Iterate over each WAV file in the directory

# Write the extracted features and labels to a CSV file

with open('../Data/features.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "length", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
                     "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean",
                     "spectral_bandwidth_var", "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean",
                     "zero_crossing_rate_var", "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var",
                     "tempo", "mfcc1_mean", "mfcc1_var", "mfcc2_mean", "mfcc2_var", "mfcc3_mean", "mfcc3_var",
                     "mfcc4_mean", "mfcc4_var", "mfcc5_mean", "mfcc5_var", "mfcc6_mean", "mfcc6_var", "mfcc7_mean",
                     "mfcc7_var", "mfcc8_mean", "mfcc8_var", "mfcc9_mean", "mfcc9_var", "mfcc10_mean", "mfcc10_var",
                     "mfcc11_mean", "mfcc11_var", "mfcc12_mean", "mfcc12_var", "mfcc13_mean", "mfcc13_var",
                     "mfcc14_mean", "mfcc14_var", "mfcc15_mean", "mfcc15_var", "mfcc16_mean", "mfcc16_var",
                     "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var", "mfcc19_mean", "mfcc19_var",
                     "mfcc20_mean", "mfcc20_var", "label"])
    
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            features = extract_features(file_path)
            label = filename.split('.')[0]  # Specify your label here
            data = ([filename] + features + [label])
            print(data)
            writer.writerow(data)