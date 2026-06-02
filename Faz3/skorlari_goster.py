from sklearn.metrics import precision_score, recall_score, f1_score

# Ensemble modelin tahminlerini al
y_pred_ens = ensemble.predict(X_test)

# Metrikleri hesapla (Ağırlıklı ortalama)
ens_prec = precision_score(y_test, y_pred_ens, average='weighted', zero_division=0)
ens_rec = recall_score(y_test, y_pred_ens, average='weighted', zero_division=0)
ens_f1 = f1_score(y_test, y_pred_ens, average='weighted', zero_division=0)

print("\n--- 2. TABLO İÇİN (Ensemble Sınıflandırma Raporu) ---")
print(f"Precision: {ens_prec:.4f}")
print(f"Recall:    {ens_rec:.4f}")
print(f"F1-Score:  {ens_f1:.4f}")
print("-" * 30)