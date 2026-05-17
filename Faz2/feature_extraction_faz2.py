# =========================================================
# BIL216 - EmoChallenge 2026
# FAZ 2 - Feature Engineering ve Görselleştirme
# =========================================================

# Gerekli kütüphaneler
import os
import numpy as np
import librosa
import warnings
import matplotlib.pyplot as plt
import pandas as pd

# Uyarıları gizle
warnings.filterwarnings("ignore")

# =========================================================
# DATASET YOLU
# =========================================================

DATASET_PATH = "dataset"

# =========================================================
# LABEL MAP
# =========================================================

label_map = {
    "mutlu": 0,
    "uzgun": 1,
    "ofkeli": 2,
    "notr": 3,
    "saskin": 4
}

# =========================================================
# GRAFİK KLASÖRÜ OLUŞTUR
# =========================================================

if not os.path.exists("grafikler"):
    os.makedirs("grafikler")

# =========================================================
# FEATURE EXTRACTION FONKSİYONU
# =========================================================

def extract_features(file_path):

    # Ses dosyasını yükle
    signal, sr = librosa.load(file_path, sr=None)

    # =====================================================
    # MFCC
    # =====================================================

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # =====================================================
    # ZCR
    # =====================================================

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal)
    )

    # =====================================================
    # RMS
    # =====================================================

    rms = np.mean(
        librosa.feature.rms(y=signal)
    )

    # =====================================================
    # SPECTRAL CENTROID
    # =====================================================

    spectral_centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    )

    centroid_mean = np.mean(spectral_centroid)
    centroid_std = np.std(spectral_centroid)

    # =====================================================
    # SPECTRAL BANDWIDTH
    # =====================================================

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr
    )

    bandwidth_mean = np.mean(spectral_bandwidth)
    bandwidth_std = np.std(spectral_bandwidth)

    # =====================================================
    # SPECTRAL ROLLOFF
    # =====================================================

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sr
    )

    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_std = np.std(spectral_rolloff)

    # =====================================================
    # CHROMA FEATURES
    # =====================================================

    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr
    )

    chroma_mean = np.mean(chroma, axis=1)

    # =====================================================
    # PITCH
    # =====================================================

    pitches, magnitudes = librosa.piptrack(
        y=signal,
        sr=sr
    )

    pitch_values = pitches[magnitudes > np.median(magnitudes)]

    if len(pitch_values) > 0:
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
    else:
        pitch_mean = 0
        pitch_std = 0

    # =====================================================
    # FEATURELARI BİRLEŞTİR
    # =====================================================

    features = np.hstack((

        mfcc_mean,
        mfcc_std,

        zcr,
        rms,

        centroid_mean,
        centroid_std,

        bandwidth_mean,
        bandwidth_std,

        rolloff_mean,
        rolloff_std,

        chroma_mean,

        pitch_mean,
        pitch_std
    ))

    return features

# =========================================================
# DATASET OLUŞTURMA
# =========================================================

X = []
y = []

print("FAZ 2 işlem başlatıldı...")

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

            # Feature çıkar
            features = extract_features(file_path)

            # Listeye ekle
            X.append(features)
            y.append(label_map[current_label])

        except Exception as e:

            print(f"HATA oluştu: {file_path}")
            continue

# =========================================================
# NUMPY ARRAY DÖNÜŞÜMÜ
# =========================================================

X = np.array(X)
y = np.array(y)

print("---------------------------------")

if len(X) == 0:

    print("HATA: Veri işlenemedi!")

else:

    print("BAŞARIYLA TAMAMLANDI!")
    print(f"Toplam ses sayısı: {X.shape[0]}")
    print(f"Feature sayısı: {X.shape[1]}")

    # =====================================================
    # FAZ 2 DOSYALARI
    # =====================================================

    np.save("X_faz2.npy", X)
    np.save("y_faz2.npy", y)

    print("X_faz2.npy oluşturuldu.")
    print("y_faz2.npy oluşturuldu.")

print("---------------------------------")

# =========================================================
# FEATURE İSİMLERİ
# =========================================================

feature_names = []

for i in range(13):
    feature_names.append(f"mfcc_mean_{i+1}")

for i in range(13):
    feature_names.append(f"mfcc_std_{i+1}")

feature_names += [
    "zcr",
    "rms",

    "centroid_mean",
    "centroid_std",

    "bandwidth_mean",
    "bandwidth_std",

    "rolloff_mean",
    "rolloff_std"
]

for i in range(12):
    feature_names.append(f"chroma_{i+1}")

feature_names += [
    "pitch_mean",
    "pitch_std"
]

# =========================================================
# FEATURE LİSTESİNİ TXT KAYDET
# =========================================================

with open("feature_listesi.txt", "w", encoding="utf-8") as f:

    for feature in feature_names:
        f.write(feature + "\n")

print("feature_listesi.txt oluşturuldu.")

# =========================================================
# İSTATİSTİKSEL ANALİZ
# =========================================================

df = pd.DataFrame(X, columns=feature_names)

print("\nİSTATİSTİKSEL ANALİZ:")
print(df.describe())

# CSV olarak kaydet
df.describe().to_csv("istatistiksel_analiz.csv")

print("\nistatistiksel_analiz.csv oluşturuldu.")

# =========================================================
# GRAFİKLER
# =========================================================

# =====================================================
# MFCC HISTOGRAM
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(df["mfcc_mean_1"], bins=20)

plt.title("MFCC Mean 1 Dağılımı")
plt.xlabel("MFCC Mean 1")
plt.ylabel("Frekans")

plt.savefig("grafikler/mfcc_mean_1_histogram.png")
plt.close()

# =====================================================
# SPECTRAL CENTROID HISTOGRAM
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(df["centroid_mean"], bins=20)

plt.title("Spectral Centroid Dağılımı")
plt.xlabel("Centroid Mean")
plt.ylabel("Frekans")

plt.savefig("grafikler/spectral_centroid_histogram.png")
plt.close()

# =====================================================
# PITCH HISTOGRAM
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(df["pitch_mean"], bins=20)

plt.title("Pitch Dağılımı")
plt.xlabel("Pitch Mean")
plt.ylabel("Frekans")

plt.savefig("grafikler/pitch_histogram.png")
plt.close()

# =====================================================
# RMS BOXPLOT
# =====================================================

plt.figure(figsize=(10,5))

plt.boxplot(df["rms"])

plt.title("RMS Boxplot")

plt.savefig("grafikler/rms_boxplot.png")
plt.close()

print("\nGrafikler oluşturuldu.")
print("Grafikler -> grafikler klasörüne kaydedildi.")

print("\nFAZ 2 Feature Engineering tamamlandı.")

# =====================================================
# DUYGU İSİMLERİ
# =====================================================

emotion_names = {
    0: "Mutlu",
    1: "Uzgun",
    2: "Ofkeli",
    3: "Notr",
    4: "Saskin"
}

# Label sütunu ekle
df["label"] = y

# =====================================================
# DUYGU BAZLI MFCC KARŞILAŞTIRMASI
# =====================================================

mfcc_means = []

for i in range(5):

    emotion_data = df[df["label"] == i]

    mfcc_avg = emotion_data["mfcc_mean_1"].mean()

    mfcc_means.append(mfcc_avg)

plt.figure(figsize=(10,5))

plt.bar(
    list(emotion_names.values()),
    mfcc_means
)

plt.title("Duygu Bazli MFCC Mean 1 Karsilastirmasi")
plt.xlabel("Duygular")
plt.ylabel("MFCC Mean 1")

plt.savefig("grafikler/duygu_bazli_mfcc_karsilastirma.png")
plt.close()

# =====================================================
# SPECTRAL CENTROID KARŞILAŞTIRMASI
# =====================================================

centroid_means = []

for i in range(5):

    emotion_data = df[df["label"] == i]

    centroid_avg = emotion_data["centroid_mean"].mean()

    centroid_means.append(centroid_avg)

plt.figure(figsize=(10,5))

plt.bar(
    list(emotion_names.values()),
    centroid_means
)

plt.title("Duygu Bazli Spectral Centroid Karsilastirmasi")
plt.xlabel("Duygular")
plt.ylabel("Centroid Mean")

plt.savefig("grafikler/spectral_centroid_karsilastirma.png")
plt.close()

print("\nDuygu bazli karsilastirma grafikleri olusturuldu.")
