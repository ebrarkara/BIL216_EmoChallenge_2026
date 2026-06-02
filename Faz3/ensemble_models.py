from sklearn.ensemble import VotingClassifier
import joblib

# Veriyi ve önceden eğittiğimiz modelleri yükle
X_train, X_test, y_train, y_test = joblib.load('split_data.pkl')
svm_model = joblib.load('svm_base.pkl')
xgb_model = joblib.load('xgb_base.pkl')
best_rf = joblib.load('best_rf_tuned.pkl') # Bir önceki adımda optimize ettiğimiz model

print("Ensemble (Topluluk) model kuruluyor...")

# Modelleri birleştir (Oylama sistemi)
ensemble_model = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('rf', best_rf),
        ('xgb', xgb_model)
    ],
    voting='soft' # 'soft' olasılıkları toplar, 'hard' direkt etiketleri sayar
)

ensemble_model.fit(X_train, y_train)

# Ensemble modeli kaydedelim
joblib.dump(ensemble_model, 'ensemble_model.pkl')
print("Ensemble model eğitildi ve kaydedildi!")