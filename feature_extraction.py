import os
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

DATASET_PATH = "dataset"

label_map = {
    "mutlu": 0,
    "uzgun": 1,
    "ofkeli": 2,
    "notr": 3,
    "saskin": 4
}

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
    rms = np.mean(librosa.feature.rms(y=signal))

    features = np.hstack((mfcc_mean, mfcc_std, zcr, rms))
    return features

X = []
y = []

print("İşlem başlatıldı, lütfen bekleyin...")

for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    current_label = label.lower()

    if current_label not in label_map:
        print(f"UYARI: {label} label_map içinde yok!")
        continue

    for file in os.listdir(folder_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(label_map[current_label])
        except Exception as e:
            print("Hata:", file_path)
            continue

X = np.array(X)
y = np.array(y)

print("---------------------------------")

if len(X) == 0:
    print("HATA: Hiç veri işlenemedi!")
else:
    print(f"BAŞARIYLA TAMAMLANDI!")
    print(f"Toplam işlenen ses sayısı: {X.shape[0]}")
    print(f"Öznitelik sayısı (Sütun): {X.shape[1]}")

    np.save("X.npy", X)
    np.save("y.npy", y)

    print("X.npy ve y.npy dosyaları oluşturuldu.")

print("---------------------------------")