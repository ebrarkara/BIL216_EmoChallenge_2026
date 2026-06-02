import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

# 1. VERİYİ YÜKLE
print("Veriler yükleniyor...")
X = np.load('X_faz3.npy')
y = np.load('y_faz3.npy')

# 2. VERİYİ PARÇALA (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Diğer dosyalarda kullanmak üzere ayırdığımız verileri kaydedelim
joblib.dump((X_train, X_test, y_train, y_test), 'split_data.pkl')

# 3. MODELLERİ TANIMLA VE EĞİT
print("Modeller eğitiliyor...")
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Eğitilen modelleri kaydedelim
joblib.dump(svm_model, 'svm_base.pkl')
joblib.dump(rf_model, 'rf_base.pkl')
joblib.dump(xgb_model, 'xgb_base.pkl')

print("Temel modeller eğitildi ve kaydedildi!")