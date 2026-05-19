"""
Partie 6 — Détection du Data Drift
Utilise Evidently et le test KS pour détecter le drift entre les données
d'entraînement (référence) et une simulation de données de production.
Déclenche automatiquement le ré-entraînement si le drift dépasse le seuil.
"""

import os
import subprocess
import numpy as np
import pandas as pd
import mlflow
from scipy import stats
from sklearn.model_selection import train_test_split

# ─── Config ───────────────────────────────────────────────────────────────────
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("monitoring_drift")

SEUIL_DRIFT = 0.30   # 30 % de features driftées → ré-entraînement
SEUIL_WARN  = 0.15   # 15 % → alerte seulement

# ─── 1. Chargement des données ────────────────────────────────────────────────
print("===== Détection du Data Drift =====\n")
df = pd.read_csv("data/processed/full_processed.csv")
X  = df.drop(columns=["Heating_Load", "Cooling_Load"])
y  = df["Heating_Load"]

X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── 2. Simulation du drift (données de production artificielles) ─────────────
print("Simulation du drift sur les données de production...")
X_prod    = X_test.copy()
num_cols  = X_prod.select_dtypes(include=np.number).columns

for col in num_cols[:2]:   # drift sur les 2 premières features numériques
    X_prod[col] = X_prod[col] * 1.6 + np.random.normal(0, 0.5, len(X_prod))

for col in num_cols[:2]:
    print(f"  {col:<30} | Ref mean: {X_train[col].mean():.3f}  Prod mean: {X_prod[col].mean():.3f}")

# ─── 3. Rapport Evidently ─────────────────────────────────────────────────────
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric

    print("\nGénération du rapport Evidently...")
    with mlflow.start_run(run_name="drift_check_evidently"):

        # Rapport HTML visuel complet
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=X_train, current_data=X_prod)
        report.save_html("outputs/drift_report.html")
        mlflow.log_artifact("outputs/drift_report.html")
        print("✔ Rapport HTML sauvegardé → outputs/drift_report.html")

        # Extraction des scores numériques
        score_report = Report(metrics=[DatasetDriftMetric()])
        score_report.run(reference_data=X_train, current_data=X_prod)
        result      = score_report.as_dict()
        drift_data  = result["metrics"][0]["result"]

        drift_share    = drift_data["drift_share"]
        dataset_drift  = drift_data["dataset_drift"]
        n_drifted      = drift_data["number_of_drifted_columns"]
        n_total        = drift_data["number_of_columns"]

        mlflow.log_metric("drift_share",      drift_share)
        mlflow.log_metric("drifted_columns",  n_drifted)
        mlflow.log_metric("total_columns",    n_total)
        mlflow.log_metric("dataset_drifted",  int(dataset_drift))

        print(f"  Drift share       : {drift_share:.2%}")
        print(f"  Colonnes driftées : {n_drifted}/{n_total}")
        print(f"  Dataset drifté    : {dataset_drift}")

except ImportError:
    print("⚠ Evidently non installé. Lance : pip install evidently")
    print("  Continuation avec KS-test uniquement...\n")
    drift_share = None

# ─── 4. KS-test par feature ───────────────────────────────────────────────────
print("\nKS-test statistique par feature...")

ks_results = []
with mlflow.start_run(run_name="drift_check_ks_test"):
    for col in X_train.select_dtypes(include="number").columns:
        stat, pvalue = stats.ks_2samp(X_train[col], X_prod[col])
        drifted = pvalue < 0.05
        ks_results.append({
            "feature":  col,
            "ks_stat":  round(stat,   4),
            "p_value":  round(pvalue, 4),
            "drifted":  drifted,
        })
        mlflow.log_metric(f"ks_pvalue_{col}", pvalue)
        mlflow.log_metric(f"ks_stat_{col}",   stat)

    df_ks = pd.DataFrame(ks_results)
    df_ks.to_csv("outputs/ks_drift_results.csv", index=False)
    mlflow.log_artifact("outputs/ks_drift_results.csv")

    n_ks_drifted = df_ks["drifted"].sum()
    ks_drift_share = n_ks_drifted / len(df_ks)
    mlflow.log_metric("ks_drift_share",    ks_drift_share)
    mlflow.log_metric("ks_drifted_columns", n_ks_drifted)

print(df_ks.to_string(index=False))
print(f"\n  KS drift share : {ks_drift_share:.2%} ({n_ks_drifted}/{len(df_ks)} features)")
print("✔ Résultats KS sauvegardés → outputs/ks_drift_results.csv")

# ─── 5. Déclenchement automatique du ré-entraînement ─────────────────────────
# Utilise le drift Evidently si disponible, sinon le KS-test
active_drift_share = drift_share if drift_share is not None else ks_drift_share

print(f"\n===== Décision automatique (drift={active_drift_share:.2%}) =====")

with mlflow.start_run(run_name="drift_decision"):
    mlflow.log_metric("active_drift_share", active_drift_share)

    if active_drift_share > SEUIL_DRIFT:
        print(f"🔴 CRITIQUE : drift {active_drift_share:.2%} > seuil {SEUIL_DRIFT:.0%}")
        print("   → Déclenchement du ré-entraînement automatique...")
        mlflow.log_metric("retrain_triggered", 1)
        mlflow.log_param("decision", "retrain")
        result = subprocess.run(["python", "src/train.py"], check=False)
        if result.returncode == 0:
            print("✔ Ré-entraînement terminé avec succès.")
        else:
            print("✘ Erreur lors du ré-entraînement.")

    elif active_drift_share > SEUIL_WARN:
        print(f"🟡 AVERTISSEMENT : drift {active_drift_share:.2%} — surveillance renforcée recommandée")
        mlflow.log_metric("retrain_triggered", 0)
        mlflow.log_param("decision", "warning")

    else:
        print(f"🟢 OK : drift {active_drift_share:.2%} — modèle stable, aucune action requise")
        mlflow.log_metric("retrain_triggered", 0)
        mlflow.log_param("decision", "stable")

print("\n===== Détection du drift terminée =====")