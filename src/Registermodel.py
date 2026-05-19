"""
Partie 3 — Model Registry
Enregistre le meilleur modèle dans MLflow Registry et gère son cycle de vie :
  None → Staging → Production
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

# ─── Config ───────────────────────────────────────────────────────────────────
tracking_uri   = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME     = "heating_load_model"
SEUIL_PROD     = 0.85   # R² minimum pour passer en Production

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

# ─── 1. Récupérer le meilleur run ─────────────────────────────────────────────
# Lire le run_id sauvegardé par train.py (si disponible)
best_run_id   = None
best_run_name = None

run_id_file = "models/best_run_id.txt"
if os.path.exists(run_id_file):
    with open(run_id_file) as f:
        lines = f.read().strip().splitlines()
        best_run_id   = lines[0]
        best_run_name = lines[1] if len(lines) > 1 else "unknown"
    print(f"Run ID chargé depuis fichier : {best_run_id} ({best_run_name})")
else:
    # Recherche programmatique du meilleur run
    print("Fichier best_run_id.txt introuvable → recherche programmatique...")
    experiment = client.get_experiment_by_name("energy_efficiency_heating")
    if experiment is None:
        raise RuntimeError("Expérience 'energy_efficiency_heating' introuvable. Lance d'abord train.py.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2 DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("Aucun run trouvé. Lance d'abord train.py.")

    best_run      = runs[0]
    best_run_id   = best_run.info.run_id
    best_run_name = best_run.data.tags.get("mlflow.runName", "unknown")
    print(f"Meilleur run trouvé : {best_run_id} ({best_run_name})")

# Récupérer les métriques du run
run_data = client.get_run(best_run_id)
r2  = run_data.data.metrics.get("r2", 0)
mae = run_data.data.metrics.get("mae", None)
print(f"  R²  : {r2:.4f}")
if mae:
    print(f"  MAE : {mae:.4f}")

# ─── 2. Enregistrement dans le Registry ───────────────────────────────────────
model_uri = f"runs:/{best_run_id}/model"
print(f"\nEnregistrement dans le Registry → '{MODEL_NAME}'...")

registered = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)
version = registered.version
print(f"✔ Version enregistrée : v{version}")

# ─── 3. Description et tags ───────────────────────────────────────────────────
client.update_registered_model(
    name=MODEL_NAME,
    description=(
        "Modèle de régression pour la prédiction de la charge de chauffage "
        "(Energy Efficiency Dataset). Entraîné avec MLflow Tracking."
    ),
)
client.set_model_version_tag(MODEL_NAME, version, "validated_by",   "equipe_data")
client.set_model_version_tag(MODEL_NAME, version, "best_run_name",  best_run_name)
client.set_model_version_tag(MODEL_NAME, version, "r2_score",       f"{r2:.4f}")
print("✔ Description et tags ajoutés.")

# ─── 4. Promotion en Staging ──────────────────────────────────────────────────
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Staging",
    archive_existing_versions=False,
)
print(f"✔ Modèle v{version} promu en Staging.")

# ─── 5. Validation et promotion en Production ────────────────────────────────
print(f"\nValidation : R²={r2:.4f} vs seuil={SEUIL_PROD}")
if r2 >= SEUIL_PROD:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,   # archive les anciennes versions prod
    )
    print(f"✔ Modèle v{version} promu en Production.")
else:
    print(f"✘ Modèle NON promu : R²={r2:.3f} < seuil={SEUIL_PROD}")
    print("  Le modèle reste en Staging pour révision.")

# ─── 6. Résumé ────────────────────────────────────────────────────────────────
print("\n===== Résumé Registry =====")
latest = client.get_latest_versions(MODEL_NAME)
for mv in latest:
    print(f"  Version {mv.version} | Stage: {mv.current_stage} | Run: {mv.run_id[:8]}...")