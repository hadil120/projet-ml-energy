import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error

# Charger les données
df = pd.read_csv("data/processed/full_processed.csv")

# Target
target = "Heating_Load"

# 🔥 IMPORTANT : mêmes features que train.py
X = df.drop(columns=["Heating_Load", "Cooling_Load"])
y = df[target]

# Charger le modèle
model = joblib.load("models/best_model.pkl")

# Prédiction
y_pred = model.predict(X)

# Métriques
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("===== Evaluation Model =====")
print(f"R2 Score : {r2:.4f}")
print(f"MSE : {mse:.4f}")