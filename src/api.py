"""
API Backend — FastAPI
Connecte le frontend React aux données MLflow et au modèle en production.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import numpy as np
import os

# ─── Setup ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Projet ML API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

MODEL_NAME = "heating_load_model"

# ─── Schémas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    Compacite: float
    Surface_Totale: float
    Surface_Murs: float
    Surface_Toit: float
    Hauteur: float
    Orientation: float
    Surface_Vitree: float
    Distribution_Vitree: float

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Projet ML API opérationnelle"}


@app.get("/api/runs")
def get_runs():
    """Retourne tous les runs MLflow de l'expérience principale."""
    try:
        experiment = client.get_experiment_by_name("energy_efficiency_heating")
        if not experiment:
            return {"runs": []}
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.r2 DESC"],
            max_results=20,
        )
        result = []
        for run in runs:
            result.append({
                "id":     run.info.run_id[:8],
                "run_id": run.info.run_id,
                "model":  run.data.tags.get("mlflow.runName", "unknown"),
                "r2":     round(run.data.metrics.get("r2", 0), 4),
                "mae":    round(run.data.metrics.get("mae", 0), 4),
                "rmse":   round(run.data.metrics.get("rmse", 0), 4),
                "status": run.info.status,
                "date":   str(run.info.start_time),
            })
        return {"runs": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/best-run")
def get_best_run():
    """Retourne le meilleur run (R² max)."""
    try:
        experiment = client.get_experiment_by_name("energy_efficiency_heating")
        if not experiment:
            raise HTTPException(status_code=404, detail="Expérience introuvable")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.r2 DESC"],
            max_results=1,
        )
        if not runs:
            raise HTTPException(status_code=404, detail="Aucun run trouvé")
        run = runs[0]
        return {
            "run_id": run.info.run_id,
            "model":  run.data.tags.get("mlflow.runName", "unknown"),
            "r2":     round(run.data.metrics.get("r2", 0), 4),
            "mae":    round(run.data.metrics.get("mae", 0), 4),
            "rmse":   round(run.data.metrics.get("rmse", 0), 4),
            "params": run.data.params,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/registry")
def get_registry():
    """Retourne les versions du modèle dans le Registry."""
    try:
        versions = client.get_latest_versions(MODEL_NAME)
        return {
            "model_name": MODEL_NAME,
            "versions": [
                {
                    "version": v.version,
                    "stage":   v.current_stage,
                    "run_id":  v.run_id[:8],
                    "status":  v.status,
                }
                for v in versions
            ],
        }
    except Exception as e:
        return {"model_name": MODEL_NAME, "versions": [], "error": str(e)}


@app.get("/api/drift")
def get_drift():
    """Retourne les dernières métriques de drift."""
    try:
        experiment = client.get_experiment_by_name("monitoring_drift")
        if not experiment:
            return {"drift_share": None, "message": "Aucune détection de drift lancée"}
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return {"drift_share": None}
        run = runs[0]
        return {
            "drift_share":       run.data.metrics.get("drift_share"),
            "drifted_columns":   run.data.metrics.get("drifted_columns"),
            "total_columns":     run.data.metrics.get("total_columns"),
            "retrain_triggered": run.data.metrics.get("retrain_triggered"),
            "date":              str(run.info.start_time),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
def predict(data: PredictRequest):
    """Prédit la charge de chauffage à partir des features du bâtiment."""
    try:
        # Charger le modèle et le scaler
        model  = joblib.load("models/best_model.pkl")
        scaler = joblib.load("data/processed/scaler.pkl")

        # Construire le vecteur de features
        features = np.array([[
            data.Compacite,
            data.Surface_Totale,
            data.Surface_Murs,
            data.Surface_Toit,
            data.Hauteur,
            data.Orientation,
            data.Surface_Vitree,
            data.Distribution_Vitree,
        ]])

        # Normaliser comme pendant l'entraînement
        features_scaled = scaler.transform(features)

        # Prédiction
        prediction = model.predict(features_scaled)[0]

        return {
            "prediction": round(float(prediction), 2),
            "unit": "kWh/m²"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Lancement ────────────────────────────────────────────────────────────────
# uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload