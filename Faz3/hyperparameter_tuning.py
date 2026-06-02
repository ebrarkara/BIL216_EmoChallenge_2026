from sklearn.model_selection import GridSearchCV
import joblib

# Veriyi ve modelleri geri yükle
X_train, X_test, y_train, y_test = joblib.load('split_data.pkl')
rf_model = joblib.load('rf_base.pkl')

print("Random Forest için en iyi ayarlar aranıyor (Bu işlem biraz sürebilir)...")

# Random Forest için denenecek ayarlar (Bilgisayarı yormamak için basit tuttum)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

# GridSearchCV ile 5-Fold Cross Validation uyguluyoruz
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"En iyi Random Forest parametreleri: {grid_search.best_params_}")

# Optimize edilmiş modeli kaydedelim
joblib.dump(best_rf, 'best_rf_tuned.pkl')
print("Optimize edilmiş model kaydedildi!")