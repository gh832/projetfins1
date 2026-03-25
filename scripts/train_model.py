
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '..', 'data_clean', 'cardio_clean.csv')

df = pd.read_csv(data_path)

df['age_years'] = (df['age'] / 365).round(1)
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

feature_cols = ['age_years', 'BMI', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'gender']

X = df[feature_cols]
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train_scaled, y_train)

model_path = os.path.join(BASE_DIR, '..', 'model.pkl')
scaler_path = os.path.join(BASE_DIR, '..', 'scaler.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print("Modèle et scaler cardio entraînés et sauvegardés.")
