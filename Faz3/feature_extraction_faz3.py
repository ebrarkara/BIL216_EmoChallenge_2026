# =========================================================
# BIL216 - EmoChallenge 2026
# FAZ 3 - Gelişmiş Feature Engineering, Normalizasyon ve Analiz
# =========================================================

# Projede kullanılacak temel kütüphanelerin içe aktarılması
import os  # Dosya ve klasör dizin işlemlerini yönetmek için
import numpy as np  # Büyük matrisler ve sayısal dizilerle hızlı çalışmak için
import librosa  # Ses işleme, analiz ve öznitelik çıkarımı için ana kütüphane
import warnings  # Sürüm güncellemelerinden kaynaklı log kirliliğini önlemek için
import matplotlib.pyplot as plt  # Verileri görselleştirip grafik olarak kaydetmek için
import pandas as pd  # Öznitelikleri tablo (DataFrame) formatına getirip analiz etmek için
import seaborn as sns  # Seabor kütüphanesi: Gelişmiş korelasyon matrisi (heatmap) görselleştirmesi için
from sklearn.preprocessing import StandardScaler  # Öznitelikleri standartlaştırmak (Mean=0, Std=1 yapmak) için

# Kodun çalışması esnasında terminalde gereksiz kütüphane uyarılarını gizler
warnings.filterwarnings("ignore")

# =========================================================
# DATASET YOLU
# =========================================================
# Ses dosyalarının (WAV formatındaki kayıtların) bulunduğu ana klasörün adı
DATASET_PATH = "dataset"

# =========================================================
# LABEL MAP
# =========================================================
# Sınıflandırma modellerinin anlaması için metinsel duyguların sayısal etiketlere (0-4) eşlenmesi
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
# Çıkarılan grafiklerin kaydedileceği "grafikler" klasörü yoksa otomatik olarak oluşturulur
if not os.path.exists("grafikler"):
    os.makedirs("grafikler")

# =========================================================
# FEATURE EXTRACTION FONKSİYONU (FAZ 3 SÜRÜMÜ)
# =========================================================
# Tek bir ses dosyasının yolunu alıp, içinden 126 boyutlu öznitelik vektörünü çıkaran ana fonksiyon
def extract_features_faz3(file_path):

    # Ses dosyasını orijinal örnekleme frekansıyla (sr=None) genlik dizisi (signal) olarak yükler
    signal, sr = librosa.load(file_path, sr=None)

    # -----------------------------------------------------
    # 1. MFCC & TÜREVLERİ (Delta & Delta-Delta)
    # -----------------------------------------------------
    # İnsan kulağının algısına uygun Mel ölçeğinde 13 adet frekans katsayısı (MFCC) hesaplanır
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    # Zaman ekseni boyunca 13 katsayının ortalamasını alır (13 boyutlu statik öznitelik)
    mfcc_mean = np.mean(mfcc, axis=1)
    # Katsayıların zaman içindeki standart sapmasını (değişkenliğini) hesaplar (13 boyut)
    mfcc_std = np.std(mfcc, axis=1)

    # Delta MFCC: MFCC katsayılarının zamana göre 1. türevidir. Sesin spektral değişim hızını yakalar.
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta_mfcc, axis=1)  # Hız özelliklerinin ortalaması (13 boyut)
    delta_std = np.std(delta_mfcc, axis=1)    # Hız özelliklerinin standart sapması (13 boyut)

    # Delta-Delta MFCC: MFCC'nin zamana göre 2. türevidir. Spektral değişimin ivmesini (sertliğini) yakalar.
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mean = np.mean(delta2_mfcc, axis=1)  # İvme özelliklerinin ortalaması (13 boyut)
    delta2_std = np.std(delta2_mfcc, axis=1)    # İvme özelliklerinin standart sapması (13 boyut)

    # -----------------------------------------------------
    # 2. ZCR & RMS
    # -----------------------------------------------------
    # Zero Crossing Rate: Sinyalin sıfır noktasını kesme sıklığı. Gürültülü/sürtünmeli ses ayrımı için ortalaması alınır.
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
    # Root Mean Square Energy: Ses sinyalinin zamansal enerji/gürültü seviyesinin ortalamasını temsil eder.
    rms = np.mean(librosa.feature.rms(y=signal))

    # -----------------------------------------------------
    # 3. SPECTRAL FEATURES (Centroid, Bandwidth, Rolloff)
    # -----------------------------------------------------
    # Spectral Centroid: Spektrumun "kütle merkezi". Sesin parlaklığını/keskinliğini ölçer (ortalama ve sapması).
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    centroid_mean = np.mean(spectral_centroid)
    centroid_std = np.std(spectral_centroid)

    print("", end="") # Orijinal koddaki akış koruma satırı (değiştirilmedi)

    # Spectral Bandwidth: Ses enerjisinin frekans spektrumuna ne kadar yayıldığını ölçer.
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    bandwidth_mean = np.mean(spectral_bandwidth)
    bandwidth_std = np.std(spectral_bandwidth)

    # Spectral Rolloff: Toplam spektral enerjinin %85'inin altında kaldığı frekans sınırı. Sesin yapısını özetler.
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_std = np.std(spectral_rolloff)

    # -----------------------------------------------------
    # 4. SPECTRAL CONTRAST (Yeni Faz 3)
    # -----------------------------------------------------
    # Spektrumdaki tepe ve vadi enerjileri arasındaki farkı ölçerek ses kalitesini/netliğini 7 farklı bantta analiz eder.
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    contrast_mean = np.mean(spectral_contrast, axis=1)  # 7 bandın ortalaması
    contrast_std = np.std(spectral_contrast, axis=1)    # 7 bandın standart sapması

    # -----------------------------------------------------
    # 5. CHROMA FEATURES
    # -----------------------------------------------------
    # Ses sinyalindeki enerjiyi 12 yarı tonluk (müzikal nota) oktav kartına izdüşürür. Konuşma melodisini yakalar.
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # 12 notanın ortalama enerji dağılımı

    # -----------------------------------------------------
    # 6. TONNETZ (Yeni Faz 3 - Harmonik Bileşenden)
    # -----------------------------------------------------
    try:
        # Önce sesten gürültüyü arındırıp sadece melodik/harmonik yapıyı çekeriz
        y_harmonic = librosa.effects.harmonic(signal)
        # Harmonik alanı 6 boyutlu bir tonal düzlemde hesaplar (Müzikal akor geçiş özellikleri gibi)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)
    except:
        # Eğer ses çok kısa veya gürültülüyse hata almamak için 6'şar boyutlu sıfır dizisi atanır
        tonnetz_mean = np.zeros(6)
        tonnetz_std = np.zeros(6)

    # -----------------------------------------------------
    # 7. PITCH
    # -----------------------------------------------------
    # Sesin temel frekansını (perdesini/inceliğini-kalınlığını) yakalamak için piptrack algoritması kullanılır
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr)
    # Medyan gürültü eşiğinin üzerindeki gerçek konuşma perdeleri filtrelenir
    pitch_values = pitches[magnitudes > np.median(magnitudes)]

    if len(pitch_values) > 0:
        pitch_mean = np.mean(pitch_values)  # Sesin temel incelik ortalaması
        pitch_std = np.std(pitch_values)    # Ses tonu dalgalanmasının sapması
    else:
        pitch_mean = 0
        pitch_std = 0

    # =====================================================
    # FEATURELARI TEK BİR VEKTÖRDE BİRLEŞTİR
    # =====================================================
    # Çıkarılan tüm istatistiksel öznitelikleri yan yana dizerek tek boyutlu yatay bir dizi oluşturur (Toplam: 126 boyut)
    features = np.hstack((
        mfcc_mean,          # 13 boyut (Statik tını)
        mfcc_std,           # 13 boyut
        delta_mean,         # 13 boyut -> Yeni Faz 3 (Hız)
        delta_std,          # 13 boyut -> Yeni Faz 3
        delta2_mean,        # 13 boyut -> Yeni Faz 3 (İvme)
        delta2_std,         # 13 boyut -> Yeni Faz 3
        zcr,                # 1 boyut  (Sürtünme)
        rms,                # 1 boyut  (Enerji)
        centroid_mean,      # 1 boyut  (Parlaklık)
        centroid_std,       # 1 boyut
        bandwidth_mean,     # 1 boyut  (Spektral Genişlik)
        bandwidth_std,      # 1 boyut
        rolloff_mean,       # 1 boyut  (Spektral Sınır)
        rolloff_std,        # 1 boyut
        contrast_mean,      # 7 boyut  -> Yeni Faz 3 (Spektral Kontrast Ortalama)
        contrast_std,       # 7 boyut  -> Yeni Faz 3 (Spektral Kontrast Sapma)
        chroma_mean,        # 12 boyut (Melodi/Nota yapısı)
        tonnetz_mean,       # 6 boyut  -> Yeni Faz 3 (Tonal Merkez Ortalama)
        tonnetz_std,        # 6 boyut  -> Yeni Faz 3 (Tonal Merkez Sapma)
        pitch_mean,         # 1 boyut  (Perde/Ana Frekans)
        pitch_std           # 1 boyut
    ))

    return features

# =========================================================
# DATASET OLUŞTURMA
# =========================================================
# Döngüden elde edilecek ham özellikleri (X_raw) ve hedef etiketleri (y) depolayacak boş listeler
X_raw = []
y = []

print("FAZ 3 işlem başlatıldı...")

# Dataset klasörünün altındaki her bir duygu klasörünü (mutlu, uzgun vb.) tek tek döner
for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)

    # Eğer okunan eleman bir klasör değilse (örn. gizli sistem dosyası) atlar
    if not os.path.isdir(folder_path):
        continue

    current_label = label.lower()

    # Klasör adı label_map sözlüğünde yoksa uyarı verip o klasörü işlemeye almaz
    if current_label not in label_map:
        print(f"UYARI: {label} label_map içinde yok!")
        continue

    # Onaylanan duygu klasörünün içindeki tüm ses dosyalarını tarar
    for file in os.listdir(folder_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            # Ses dosyasını Faz 3 fonksiyonuna gönderip 126 boyutlu vektörünü alır
            features = extract_features_faz3(file_path)
            # Çıkarılan vektörü ana veri matris listesine ekler
            X_raw.append(features)
            # Dosyanın ait olduğu duygunun sayısal kodunu etiket listesine ekler
            y.append(label_map[current_label])
        except Exception as e:
            # Herhangi bir ses dosyasında okuma/bozulma hatası olursa ekrana basar ve çökmeyi önler
            print(f"HATA oluştu: {file_path} -> {e}")
            continue

# Python listelerini makine öğrenmesi modellerinin girdi formatı olan Numpy matrislerine dönüştürür
X_raw = np.array(X_raw)
y = np.array(y)

print("---------------------------------")

# Eğer hiçbir veri toplanamadıysa programı hata vererek sonlandırır
if len(X_raw) == 0:
    print("HATA: Veri işlenemedi!")
    exit()
else:
    print("ÖZNİTELİK ÇIKARIMI BAŞARIYLA TAMAMLANDI!")
    print(f"Toplam ses sayısı: {X_raw.shape[0]}")
    print(f"Ham Feature sayısı: {X_raw.shape[1]}")

# =========================================================
# STANDARD SCALER (NORMALİZASYON) - Yeni Faz 3
# =========================================================
print("StandardScaler normalizasyonu uygulanıyor...")
# Her bir özniteliğin kendi içindeki değer aralıklarını eşitlemek için ölçekleyici nesnesi oluşturulur
scaler = StandardScaler()
# Verideki her sütunun ortalamasını 0, standart sapmasını 1 yapacak şekilde matematiksel dönüşüm uygular
X_scaled = scaler.fit_transform(X_raw)

# Modelleme yapacak olan Ahmet'e gönderilmek üzere nihai Faz 3 dosyalarını diske kaydeder
np.save("X_faz3.npy", X_scaled)
np.save("y_faz3.npy", y)

print("X_faz3.npy (Normalize edilmiş) oluşturuldu.")
print("y_faz3.npy oluşturuldu.")
print("---------------------------------")

# =========================================================
# FEATURE İSİMLERİ LİSTESİ OLUŞTURMA
# =========================================================
# CSV raporlaması ve analizler için 126 özniteliğin ismini sırasıyla tutan liste mimarisi
feature_names = []

# Matristeki sıralamaya sadık kalınarak döngülerle isimler üretilir
for i in range(13): feature_names.append(f"mfcc_mean_{i+1}")
for i in range(13): feature_names.append(f"mfcc_std_{i+1}")
for i in range(13): feature_names.append(f"delta_mfcc_mean_{i+1}")
for i in range(13): feature_names.append(f"delta_mfcc_std_{i+1}")
for i in range(13): feature_names.append(f"delta2_mfcc_mean_{i+1}")
for i in range(13): feature_names.append(f"delta2_mfcc_std_{i+1}")

feature_names += [
    "zcr", "rms",
    "centroid_mean", "centroid_std",
    "bandwidth_mean", "bandwidth_std",
    "rolloff_mean", "rolloff_std"
]

for i in range(7): feature_names.append(f"spectral_contrast_mean_{i+1}")
for i in range(7): feature_names.append(f"spectral_contrast_std_{i+1}")
for i in range(12): feature_names.append(f"chroma_{i+1}")
for i in range(6): feature_names.append(f"tonnetz_mean_{i+1}")
for i in range(6): feature_names.append(f"tonnetz_std_{i+1}")

feature_names += ["pitch_mean", "pitch_std"]

# =========================================================
# FEATURE LİSTESİNİ TXT KAYDET (feature_listesi_faz3.txt)
# =========================================================
# Model girdisinde hangi sütunun hangi özelliğe denk geldiğini dökümante etmek için TXT dosyası yazar
with open("feature_listesi_faz3.txt", "w", encoding="utf-8") as f:
    for feature in feature_names:
        f.write(feature + "\n")

print("feature_listesi_faz3.txt oluşturuldu.")

# =========================================================
# İSTATİSTİKSEL ANALİZ (istatistiksel_analiz_faz3.csv)
# =========================================================
# Normalize edilmiş verileri ve isimlerini birleştirerek Pandas DataFrame yapısı kurar
df = pd.DataFrame(X_scaled, columns=feature_names)

print("\nİSTATİSTİKSEL ANALİZ ÖZETİ (İLK 5 ÖZNİTELİK):")
# İlk 5 özelliğin temel istatistik özetini (count, mean, std, min, max) ekrana basar
print(df.iloc[:, :5].describe())

# Rapor için tüm 126 özelliğin istatistik özetini CSV tablosu olarak kaydeder
df.describe().to_csv("istatistiksel_analiz_faz3.csv")
print("istatistiksel_analiz_faz3.csv oluşturuldu.")

# =========================================================
# KORELASYON MATRİSİ HEATMAP (Yeni Faz 3)
# =========================================================
# Grafik penceresinin boyutunu büyük (16x12 inç) olarak ayarlar
plt.figure(figsize=(16, 12))
# Bütün özelliklerin birbirleriyle olan doğrusal ilişkilerini (-1 ile 1 arasında) hesaplar
corr_matrix = df.corr()
# Seaborn ile hücre ilişkilerini renk yoğunluğuna göre (mavi-kırmızı skalasında) çizer
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, vmin=-1, vmax=1)
plt.title("Faz 3 Genişletilmiş Öznitelik Seti Korelasyon Matrisi", fontsize=16)
plt.tight_layout()  # Grafik elemanlarının dışarı taşmasını önler
plt.savefig("grafikler/korelasyon_matrisi.png", dpi=150)  # Yüksek çözünürlükte kaydeder
plt.close()  # Belleği temizlemek için grafik çizimini kapatır
print("grafikler/korelasyon_matrisi.png oluşturuldu.")

# =========================================================
# GRAFİKLER VE YENİ HISTOGRAMLAR
# =========================================================

# Faz 2 Orijinal Grafikleri (Normalize edilmiş güncel verilerle çizilir)
plt.figure(figsize=(8,5))
plt.hist(df["mfcc_mean_1"], bins=20, color="blue", alpha=0.7)
plt.title("MFCC Mean 1 Dağılımı (Normalize)")
plt.xlabel("MFCC Mean 1")
plt.ylabel("Frekans")
plt.savefig("grafikler/mfcc_mean_1_histogram.png")
plt.close()

plt.figure(figsize=(8,5))
plt.hist(df["centroid_mean"], bins=20, color="green", alpha=0.7)
plt.title("Spectral Centroid Dağılımı (Normalize)")
plt.xlabel("Centroid Mean")
plt.ylabel("Frekans")
plt.savefig("grafikler/spectral_centroid_histogram.png")
plt.close()

plt.figure(figsize=(8,5))
plt.hist(df["pitch_mean"], bins=20, color="orange", alpha=0.7)
plt.title("Pitch Dağılımı (Normalize)")
plt.xlabel("Pitch Mean")
plt.ylabel("Frekans")
plt.savefig("grafikler/pitch_histogram.png")
plt.close()

plt.figure(figsize=(10,5))
plt.boxplot(df["rms"])
plt.title("RMS Boxplot (Normalize)")
plt.savefig("grafikler/rms_boxplot.png")
plt.close()

# -----------------------------------------------------
# YENİ FAZ 3 HISTOGRAMLARI
# -----------------------------------------------------
# 1. Delta MFCC Histogramı: Sesin tınısal hız değişiminin veri setindeki dağılım histogramını üretir
plt.figure(figsize=(8,5))
plt.hist(df["delta_mfcc_mean_1"], bins=20, color="purple", alpha=0.7)
plt.title("Delta MFCC Mean 1 Dağılımı (Ses Değişim Hızı)")
plt.xlabel("Delta MFCC Mean 1")
plt.ylabel("Frekans")
plt.savefig("grafikler/delta_mfcc_histogram.png")
plt.close()

# 2. Spectral Contrast Histogramı: Sesin frekans vadi-tepe fark dağılım grafiğini üretir
plt.figure(figsize=(8,5))
plt.hist(df["spectral_contrast_mean_1"], bins=20, color="brown", alpha=0.7)
plt.title("Spectral Contrast Mean 1 Dağılımı")
plt.xlabel("Spectral Contrast Mean 1")
plt.ylabel("Frekans")
plt.savefig("grafikler/spectral_contrast_histogram.png")
plt.close()

# 3. Tonnetz Histogramı: Seslerin harmonik/tonal akor merkezi dağılım grafiğini üretir
plt.figure(figsize=(8,5))
plt.hist(df["tonnetz_mean_1"], bins=20, color="teal", alpha=0.7)
plt.title("Tonnetz Mean 1 Dağılımı (Tonal Yapı)")
plt.xlabel("Tonnetz Mean 1")
plt.ylabel("Frekans")
plt.savefig("grafikler/tonnetz_histogram.png")
plt.close()

# =====================================================
# DUYGU BAZLI KARŞILAŞTIRMALAR (Faz 2 Yapısı Korundu)
# =====================================================
# Grafik eksenlerinde duyguların düzgün görünmesi için etiket karşılıkları tanımlanır
emotion_names = {0: "Mutlu", 1: "Uzgun", 2: "Ofkeli", 3: "Notr", 4: "Saskin"}
# İstatistiksel gruplama yapabilmek için dataframe tablosuna hedef etiket (label) sütunu eklenir
df["label"] = y

# Duygu Bazlı MFCC Kıyaslaması: Her bir duygu grubuna ait seslerin MFCC Ortalama 1 değerlerinin ortalamasını hesaplayıp bar grafik çizer
mfcc_means = [df[df["label"] == i]["mfcc_mean_1"].mean() for i in range(5)]
plt.figure(figsize=(10,5))
plt.bar(list(emotion_names.values()), mfcc_means, color="darkred")
plt.title("Duygu Bazli MFCC Mean 1 Karsilastirmasi")
plt.xlabel("Duygular")
plt.ylabel("MFCC Mean 1")
plt.savefig("grafikler/duygu_bazli_mfcc_karsilastirma.png")
plt.close()

# Duygu Bazlı Spectral Centroid Kıyaslaması: Duyguların ses parlaklığı (Centroid) bazında ortalama farklarını kıyaslayan bar grafiği üretir
centroid_means = [df[df["label"] == i]["centroid_mean"].mean() for i in range(5)]
plt.figure(figsize=(10,5))
plt.bar(list(emotion_names.values()), centroid_means, color="darkblue")
plt.title("Duygu Bazli Spectral Centroid Karsilastirmasi")
plt.xlabel("Duygular")
plt.ylabel("Centroid Mean")
plt.savefig("grafikler/spectral_centroid_karsilastirma.png")
plt.close()

# İşlemlerin bittiğini bildiren terminal çıktıları
print("\nBütün grafikler ve duygu bazlı karşılaştırmalar güncellendi.")
print("Nihai Faz 3 Dosyaları Başarıyla Hazırlandı!")
print("\nFAZ 3 Sinyal İşleme ve Veri Süreci Başarıyla Tamamlandı.")