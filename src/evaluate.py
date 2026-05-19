import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─── MLflow setup ─────────────────────────────────────────────────────────────
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("energy_efficiency_heating")

# ─── Données ──────────────────────────────────────────────────────────────────
df     = pd.read_csv("data/processed/full_processed.csv")
target = "Heating_Load"
X      = df.drop(columns=["Heating_Load", "Cooling_Load"])
y      = df[target]

# ─── Modèle ───────────────────────────────────────────────────────────────────
model = joblib.load("models/best_model.pkl")
y_pred = model.predict(X)

# ─── Métriques ────────────────────────────────────────────────────────────────
r2   = r2_score(y, y_pred)
mae  = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("===== Évaluation du meilleur modèle =====")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# ─── Artefacts ────────────────────────────────────────────────────────────────
# 1. Graphique Prédictions vs Réel
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y, y_pred, alpha=0.5, color="#4C72B0")
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
axes[0].set_xlabel("Valeurs réelles")
axes[0].set_ylabel("Prédictions")
axes[0].set_title("Prédictions vs Réel")

# 2. Distribution des résidus
residuals = y.values - y_pred
axes[1].hist(residuals, bins=40, color="#4C72B0", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_xlabel("Résidu")
axes[1].set_ylabel("Fréquence")
axes[1].set_title("Distribution des résidus")

plt.suptitle(f"Évaluation — R²={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}")
plt.tight_layout()
plt.savefig("outputs/evaluation_plot.png", dpi=100)
plt.close()

# ─── Log dans MLflow ──────────────────────────────────────────────────────────
with mlflow.start_run(run_name="evaluation_best_model"):
    mlflow.log_metrics({"r2": r2, "mae": mae, "rmse": rmse})
    mlflow.log_artifact("outputs/evaluation_plot.png")
    print("✔ Métriques et artefacts loggés dans MLflow.")

print("✔ Graphique sauvegardé → outputs/evaluation_plot.png")