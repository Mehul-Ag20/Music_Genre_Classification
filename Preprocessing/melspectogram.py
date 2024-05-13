import librosa 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to extract mel-spectrogram features from audio files and save as image
def extract_melspectrogram_features(filename, save_path, n_mels=128, hop_length=512, n_frames=600):
    y, sr = librosa.load(filename, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)

    if mel_spectrogram.shape[1] < n_frames:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, n_frames - mel_spectrogram.shape[1])), mode='constant')
    elif mel_spectrogram.shape[1] > n_frames:
        mel_spectrogram = mel_spectrogram[:, :n_frames]
    
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Save mel spectrogram as image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.savefig(os.path.join(save_path, os.path.splitext(os.path.basename(filename))[0] + '.png'))
    plt.close()

# Function to load and preprocess data
def load_data(dataset_path, save_path, n_mels=128, hop_length=512, test_size=0.1, random_state=42):
    data = []
    labels = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                label = os.path.basename(root)
                mel_spectrogram = extract_melspectrogram_features(filepath, save_path, n_mels, hop_length)
                data.append(mel_spectrogram)
                labels.append(label)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    return data, labels

# Example usage
dataset_path = 'Data/genre_original'
save_path = 'Data/image_original'
data, labels = load_data(dataset_path, save_path)
