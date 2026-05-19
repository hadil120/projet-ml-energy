import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ─── MLflow setup ────────────────────────────────────────────────────────────
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
print(f"MLflow tracking URI: {tracking_uri}")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("energy_efficiency_heating")   # ← AJOUT : expérience nommée

# ─── Données ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/full_processed.csv")
target = "Heating_Load"
X = df.drop(columns=["Heating_Load", "Cooling_Load"])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── Suivi du meilleur modèle ─────────────────────────────────────────────────
best_score = -1
best_model = None
best_name  = ""
best_run_id = None

# ─── Helpers artefacts ────────────────────────────────────────────────────────
def log_residuals_plot(y_test, y_pred, run_name):
    """Génère et logge un graphique des résidus."""
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Résidus vs prédictions
    axes[0].scatter(y_pred, residuals, alpha=0.5, color="#4C72B0")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Prédictions")
    axes[0].set_ylabel("Résidus")
    axes[0].set_title(f"Résidus — {run_name}")

    # Distribution des résidus
    axes[1].hist(residuals, bins=30, color="#4C72B0", edgecolor="white")
    axes[1].set_xlabel("Résidu")
    axes[1].set_ylabel("Fréquence")
    axes[1].set_title("Distribution des résidus")

    plt.tight_layout()
    path = "residuals_plot.png"
    plt.savefig(path, dpi=100)
    plt.close()
    mlflow.log_artifact(path)
    os.remove(path)


def log_feature_importance(model, feature_names, run_name):
    """Logge l'importance des features si disponible (RF, XGBoost, AdaBoost)."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], color="#4C72B0")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Feature Importance — {run_name}")
    ax.set_ylabel("Importance")
    plt.tight_layout()

    path = "feature_importance.png"
    plt.savefig(path, dpi=100)
    plt.close()
    mlflow.log_artifact(path)
    os.remove(path)


# ─── Fonction d'évaluation ────────────────────────────────────────────────────
def evaluate_model(model, model_name, params=None):
    global best_score, best_model, best_name, best_run_id

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métriques
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{model_name:40s} -> R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    with mlflow.start_run(run_name=model_name) as run:
        # Paramètres
        mlflow.log_params(params or {})
        mlflow.log_param("model_type", model_name.split("_")[0])
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Métriques
        mlflow.log_metrics({"r2": r2, "mae": mae, "rmse": rmse})

        # Artefacts
        log_residuals_plot(y_test, y_pred, model_name)
        log_feature_importance(model, list(X.columns), model_name)

        # Modèle
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id

    # Meilleur modèle ?
    if r2 > best_score:
        best_score  = r2
        best_model  = model
        best_name   = model_name
        best_run_id = run_id


# ─── Entraînement des modèles ─────────────────────────────────────────────────

# Linear Regression
evaluate_model(LinearRegression(), "LinearRegression")

# SVR
for c in [0.1, 1, 10]:
    evaluate_model(SVR(C=c), f"SVR_C_{str(c).replace('.', '_')}", {"C": c})

# Random Forest
for n in [50, 100, 200]:
    evaluate_model(
        RandomForestRegressor(n_estimators=n, random_state=42),
        f"RandomForest_{n}",
        {"n_estimators": n}
    )

# AdaBoost
for n in [50, 100, 200]:
    evaluate_model(
        AdaBoostRegressor(n_estimators=n, random_state=42),
        f"AdaBoost_{n}",
        {"n_estimators": n}
    )

# XGBoost
for n in [50, 100, 200]:
    for lr in [0.01, 0.1]:
        evaluate_model(
            XGBRegressor(
                n_estimators=n, learning_rate=lr,
                objective="reg:squarederror",
                random_state=42, n_jobs=-1, verbosity=0,
            ),
            f"XGBoost_{n}_lr_{str(lr).replace('.', '_')}",
            {"n_estimators": n, "learning_rate": lr, "objective": "reg:squarederror"}
        )


# ─── Sauvegarde du meilleur modèle ────────────────────────────────────────────
print(f"\nBest model: {best_name} avec R2 = {best_score:.4f}")
print(f"Run ID     : {best_run_id}")

joblib.dump(best_model, "models/best_model.pkl")

# Sauvegarder l'ID du meilleur run pour register_model.py
with open("models/best_run_id.txt", "w") as f:
    f.write(f"{best_run_id}\n{best_name}\n")

print("Modèle sauvegardé dans models/best_model.pkl")
print("Run ID sauvegardé dans models/best_run_id.txt")