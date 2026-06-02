import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Veriyi ve değerlendireceğimiz en güçlü modelimizi (Ensemble) yükle
X_train, X_test, y_train, y_test = joblib.load('split_data.pkl')
model = joblib.load('ensemble_model.pkl')

print("Performans değerlendirmesi yapılıyor...")
# Tahmin yap
y_pred = model.predict(X_test)

# Metrikleri hesapla
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Nihai Başarı Oranı (Accuracy): {acc:.4f}")

# 1. Model Sonuçlarını CSV olarak kaydet
sonuclar = pd.DataFrame([{'Model': 'Ensemble (SVM+RF+XGB)', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}])
sonuclar.to_csv('model_sonuclari.csv', index=False)

# 2. Classification Report'u TXT olarak kaydet
rapor = classification_report(y_test, y_pred)
with open('classification_report.txt', 'w') as f:
    f.write(rapor)

# 3. Confusion Matrix'i PNG olarak kaydet
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.savefig('confusion_matrix.png')
plt.close()

# 4. En iyi modeli best_model.pkl olarak kaydet
joblib.dump(model, 'best_model.pkl')

print("İşlem tamam! Çıktılar klasörüne kaydedildi.")