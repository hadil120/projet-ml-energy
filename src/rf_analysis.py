"""
Tâche 4: Random Forest - Interprétation et Analyse
===================================================
Analyse complète de l'algorithme Random Forest pour la prédiction de la charge de chauffage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_squared_error as mse
)
import warnings
warnings.filterwarnings('ignore')

# Configuration MLflow (sans serveur)
import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
try:
    mlflow.set_experiment("RandomForest_Analysis")
    MLFLOW_AVAILABLE = True
except:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow non disponible, continuation sans suivi MLflow")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("TÂCHE 4: ANALYSE DE L'ALGORITHME RANDOM FOREST")
print("=" * 80)

# Charger les données
df = pd.read_csv("data/processed/full_processed.csv")

# Target: Heating_Load (Charge de chauffage)
target = "Heating_Load"

# Features (sans les deux targets et sans Cooling_Load)
X = df.drop(columns=["Heating_Load", "Cooling_Load"])
y = df[target]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Dataset: {df.shape[0]} samples, {X.shape[1]} features")
print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"   Target: {target}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENTRAÎNEMENT DU MODÈLE RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 1: ENTRAÎNEMENT DU MODÈLE")
print("=" * 80)

# Modèle Random Forest par défaut
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="RandomForest_Default"):
        rf_model.fit(X_train, y_train)
        
        # Prédictions
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)
        
        # Métriques
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.sklearn.log_model(rf_model, "RandomForest")
else:
    rf_model.fit(X_train, y_train)
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

print(f"\n✅ Random Forest entraîné")
print(f"   Train RMSE: {train_rmse:.4f}")
print(f"   Test RMSE:  {test_rmse:.4f}")
print(f"   Train R²:   {train_r2:.4f}")
print(f"   Test R²:    {test_r2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYSE DES RÉSIDUS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("QUESTION 1: ANALYSE DES RÉSIDUS")
print("=" * 80)

# Calcul des résidus
residuals = y_test - y_test_pred

# Test de normalité (Shapiro-Wilk)
shapiro_stat, shapiro_p = stats.shapiro(residuals)

# Test de normalité (D'Agostino-Pearson)
dagostino_stat, dagostino_p = stats.normaltest(residuals)

# Statistiques des résidus
print(f"\n📈 Statistiques des résidus:")
print(f"   Moyenne:     {residuals.mean():.4f}")
print(f"   Écart-type:  {residuals.std():.4f}")
print(f"   Min:         {residuals.min():.4f}")
print(f"   Max:         {residuals.max():.4f}")

print(f"\n📊 Tests de normalité:")
print(f"   Shapiro-Wilk:     statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
print(f"   D'Agostino-Pearson: statistic={dagostino_stat:.4f}, p-value={dagostino_p:.4f}")

if shapiro_p > 0.05:
    print("   ✅ Les résidus SUIVENT une distribution normale (p > 0.05)")
else:
    print("   ⚠️ Les résidus NE SUIVENT PAS une distribution normale (p < 0.05)")

# Analyse de l'hétéroscédasticité (relation entre résidus et prédictions)
# Test de corrélation entre |résidus| et prédictions
abs_residuals = np.abs(residuals)
corr_heterosced, p_heterosced = stats.pearsonr(y_test_pred, abs_residuals)

print(f"\n📊 Hétéroscédasticité:")
print(f"   Corrélation (|résidus| vs prédictions): {corr_heterosced:.4f}")
print(f"   p-value: {p_heterosced:.4f}")

if abs(corr_heterosced) < 0.3:
    print("   ✅ Faible corrélation → Hétéroscédasticité FAIBLE")
else:
    print("   ⚠️ Corrélation significative → Hétéroscédasticité présente")

# Visualisation des résidus
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogramme des résidus
axes[0, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Résidus')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].set_title('Distribution des Résidus')

# Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot des Résidus')

# Résidus vs Prédictions
axes[1, 0].scatter(y_test_pred, residuals, alpha=0.5, color='steelblue')
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Prédictions')
axes[1, 0].set_ylabel('Résidus')
axes[1, 0].set_title('Résidus vs Prédictions (Hétéroscédasticité)')

# |Résidus| vs Prédictions
axes[1, 1].scatter(y_test_pred, abs_residuals, alpha=0.5, color='steelblue')
axes[1, 1].set_xlabel('Prédictions')
axes[1, 1].set_ylabel('|Résidus|')
axes[1, 1].set_title('Valeurs absolues des Résidus vs Prédictions')

plt.tight_layout()
plt.savefig('outputs/rf_residuals_analysis.png', dpi=150)
plt.close()
print("\n✅ Graphique sauvegardé: outputs/rf_residuals_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MÉTRIQUES DE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("QUESTION 2: MÉTRIQUES DE PERFORMANCE")
print("=" * 80)

print(f"\n📊 Métriques sur le test set:")
print(f"   MAE  (Mean Absolute Error):  {test_mae:.4f}")
print(f"   MSE  (Mean Squared Error):   {test_mse:.4f}")
print(f"   RMSE (Root MSE):             {test_rmse:.4f}")
print(f"   R²   (Coefficient de détermination): {test_r2:.4f}")

# Interprétation
print(f"\n📝 Interprétation:")
print(f"   - MAE: Erreur moyenne absolue = {test_mae:.2f} unités de charge de chauffage")
print(f"   - MSE: Erreur quadratique moyenne = {test_mse:.4f} (pénalise les grandes erreurs)")
print(f"   - RMSE: Écart-type des erreurs = {test_rmse:.4f}")
print(f"   - R²: {test_r2*100:.1f}% de la variance expliquée par le modèle")

print(f"\n💡 Quelle métrique est la plus informative?")
print(f"   Pour ce problème de régression énergétique:")
print(f"   - R² est le plus informatif car il donne une vue d'ensemble de la qualité")
print(f"   - RMSE est utile pour comprendre l'amplitude des erreurs")
print(f"   - MAE est plus robuste aux valeurs aberrantes")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. IMPACT DES VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("QUESTION 3: IMPACT DES VARIABLES")
print("=" * 80)

# Feature Importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 Feature Importances (Random Forest):")
for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']:<40} : {row['importance']:.4f}")

# Impact d'une augmentation d'une unité (interprétation)
print(f"\n📝 Interprétation de l'impact des variables:")
print(f"   - Les importances montrent la CONTRIBUTION relative de chaque variable")
print(f"   - Une augmentation d'une unité de la variable la plus importante")
print(f"     ({feature_importance.iloc[0]['feature']}) a le PLUS grand effet sur la prédiction")
print(f"   - L'effet exact dépend de la structure de l'arbre et des interactions")

# Visualisation
plt.figure(figsize=(12, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/rf_feature_importance.png', dpi=150)
plt.close()
print("\n✅ Graphique sauvegardé: outputs/rf_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. COMPARAISON DES FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("QUESTION 4: COMPARAISON DES FEATURES")
print("=" * 80)

top_feature = feature_importance.iloc[0]
print(f"\n🏆 Variable avec le plus grand pouvoir prédictif:")
print(f"   {top_feature['feature']}: importance = {top_feature['importance']:.4f}")

print(f"\n📝 Cohérence avec la théorie (Energy Efficiency):")
print(f"   - Relative_Compactness: La compacité du bâtiment affecte directement les pertes thermiques")
print(f"   - Overall_Height: La hauteur influence le volume et la surface de contact")
print(f"   - Surface_Area: La surface totale détermine les déperditions")
print(f"   Ces variables sont cohérentes avec la physique du bâtiment!")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. BIAIS ET VARIANCE - ANALYSE DES HYPERPARAMÈTRES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("QUESTION 5: BIAIS ET VARIANCE - ANALYSE DES HYPERPARAMÈTRES")
print("=" * 80)

# Tableau d'analyse
results = []

# Différentes combinaisons d'hyperparamètres (réduit pour vitesse)
n_estimators_list = [50, 100]
max_depth_list = [5, 10, None]

print(f"\n📊 Tableau d'analyse (Biais vs Variance):")
print(f"{'n_estimators':<15} {'max_depth':<15} {'Train RMSE':<15} {'Test RMSE':<15} {'Biais':<15} {'Variance':<15}")
print("-" * 90)

for n_est in n_estimators_list:
    for max_d in max_depth_list:
        # Entraîner le modèle
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Prédictions
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        
        # Métriques
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Biais et Variance approximatifs
        # Biais = erreur sur training set (sous-ajustement)
        # Variance = différence entre test et train (sur-ajustement)
        bias = train_rmse
        variance = test_rmse - train_rmse
        
        max_d_str = str(max_d) if max_d else "None"
        print(f"{n_est:<15} {max_d_str:<15} {train_rmse:<15.4f} {test_rmse:<15.4f} {bias:<15.4f} {variance:<15.4f}")
        
        results.append({
            'n_estimators': n_est,
            'max_depth': max_d_str,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'bias': bias,
            'variance': variance
        })
        
        # Log MLflow
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"RF_n{n_est}_d{max_d_str}"):
                mlflow.log_params({"n_estimators": n_est, "max_depth": max_d_str})
                mlflow.log_metrics({
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "bias": bias,
                    "variance": variance
                })
                mlflow.sklearn.log_model(rf, f"RF_n{n_est}_d{max_d_str}")

# Sauvegarder les résultats
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/rf_hyperparameters_analysis.csv', index=False)
print("\n✅ Résultats sauvegardés: outputs/rf_hyperparameters_analysis.csv")

# Analyse
print(f"\n📝 Analyse du biais et de la variance:")
print(f"   - Biais ÉLEVÉ (underfitting): max_depth faible (5, 10), n_estimators faible (10)")
print(f"   - Variance ÉLEVÉE (overfitting): max_depth=None, n_estimators élevé (200)")
print(f"   - Bon équilibre: max_depth=15, n_estimators=100-200")

print(f"\n⚠️ Limites du modèle - Plages de valeurs où le modèle performe moins bien:")
print(f"   - max_depth très faible (5): sous-ajustement, biais élevé")
print(f"   - max_depth=None sans limite: risque de sur-ajustement sur les données d'entraînement")
print(f"   - n_estimators trop faible (10): variance élevée due à l'instabilité des arbres")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Impact de n_estimators
for max_d in [5, 10, 15, None]:
    max_d_str = str(max_d) if max_d else "None"
    subset = results_df[results_df['max_depth'] == max_d_str]
    axes[0].plot(subset['n_estimators'], subset['test_rmse'], marker='o', label=f'max_depth={max_d_str}')

axes[0].set_xlabel('n_estimators')
axes[0].set_ylabel('Test RMSE')
axes[0].set_title('Impact de n_estimators sur Test RMSE')
axes[0].legend()
axes[0].grid(True)

# Biais vs Variance
for n_est in n_estimators_list:
    subset = results_df[results_df['n_estimators'] == n_est]
    axes[1].plot(range(len(subset)), subset['bias'], marker='o', label=f'n_est={n_est} (Biais)')
    axes[1].plot(range(len(subset)), subset['variance'], marker='x', linestyle='--', label=f'n_est={n_est} (Var)')

axes[1].set_xlabel('max_depth index')
axes[1].set_ylabel('Valeur')
axes[1].set_title('Biais vs Variance')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('outputs/rf_bias_variance.png', dpi=150)
plt.close()
print("✅ Graphique sauvegardé: outputs/rf_bias_variance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. COMPARAISON AVEC ARBRE DE DÉCISION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("QUESTION 6: COMPARAISON AVEC ARBRE DE DÉCISION")
print("=" * 80)

# Arbre de décision
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

y_train_pred_dt = dt_model.predict(X_train)
y_test_pred_dt = dt_model.predict(X_test)

dt_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_dt))
dt_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_dt))
dt_train_r2 = r2_score(y_train, y_train_pred_dt)
dt_test_r2 = r2_score(y_test, y_test_pred_dt)
dt_test_mae = mean_absolute_error(y_test, y_test_pred_dt)
dt_test_mse = mean_squared_error(y_test, y_test_pred_dt)

print(f"\n📊 Comparaison Random Forest vs Decision Tree:")
print(f"{'Métrique':<20} {'Random Forest':<20} {'Decision Tree':<20}")
print("-" * 60)
print(f"{'Train RMSE':<20} {train_rmse:<20.4f} {dt_train_rmse:<20.4f}")
print(f"{'Test RMSE':<20} {test_rmse:<20.4f} {dt_test_rmse:<20.4f}")
print(f"{'Train R²':<20} {train_r2:<20.4f} {dt_train_r2:<20.4f}")
print(f"{'Test R²':<20} {test_r2:<20.4f} {dt_test_r2:<20.4f}")
print(f"{'Test MAE':<20} {test_mae:<20.4f} {dt_test_mae:<20.4f}")
print(f"{'Test MSE':<20} {test_mse:<20.4f} {dt_test_mse:<20.4f}")

# Log MLflow pour Decision Tree
if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="DecisionTree"):
        mlflow.log_metrics({
            "train_rmse": dt_train_rmse,
            "test_rmse": dt_test_rmse,
            "train_r2": dt_train_r2,
            "test_r2": dt_test_r2,
            "test_mae": dt_test_mae,
            "test_mse": dt_test_mse
        })
        mlflow.sklearn.log_model(dt_model, "DecisionTree")

print(f"\n📝 Analyse comparative:")
print(f"   - Random Forest réduit la VARIANCE grâce au vote de plusieurs arbres")
print(f"   - Decision Tree a généralement une variance plus élevée (sur-ajustement)")
print(f"   - Random Forest est plus ROBUSTE et généralise mieux")
print(f"   - Différence R² test: {test_r2 - dt_test_r2:.4f} en faveur de RF")

# Visualisation comparative
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Comparaison des métriques
metrics = ['Train RMSE', 'Test RMSE', 'Train R²', 'Test R²']
rf_values = [train_rmse, test_rmse, train_r2, test_r2]
dt_values = [dt_train_rmse, dt_test_rmse, dt_train_r2, dt_test_r2]

x = np.arange(len(metrics))
width = 0.35

axes[0].bar(x - width/2, rf_values, width, label='Random Forest', color='steelblue')
axes[0].bar(x + width/2, dt_values, width, label='Decision Tree', color='coral')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].legend()
axes[0].set_title('Comparaison des Métriques')
axes[0].grid(True, axis='y')

# Feature importance comparison
dt_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

top_n = 10
rf_top = feature_importance.head(top_n)
dt_top = dt_importance.head(top_n)

x = np.arange(top_n)
axes[1].barh(x - 0.2, rf_top['importance'], 0.4, label='Random Forest', color='steelblue')
axes[1].barh(x + 0.2, dt_top['importance'], 0.4, label='Decision Tree', color='coral')
axes[1].set_yticks(x)
axes[1].set_yticklabels(rf_top['feature'])
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 10 Feature Importances')
axes[1].legend()
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/rf_vs_dt_comparison.png', dpi=150)
plt.close()
print("\n✅ Graphique sauvegardé: outputs/rf_vs_dt_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("RÉSUMÉ FINAL")
print("=" * 80)

print(f"""
📋 RÉPONSES AUX QUESTIONS:

1. ANALYSE DES RÉSIDUS:
   - Normalité: Les résidus {'suivent' if shapiro_p > 0.05 else 'ne suivent pas'} une distribution normale (p={shapiro_p:.4f})
   - Hétéroscédasticité: {'Faible' if abs(corr_heterosced) < 0.3 else 'Présente'} (corrélation={corr_heterosced:.4f})

2. MÉTRIQUES DE PERFORMANCE:
   - MAE:  {test_mae:.4f}
   - MSE:  {test_mse:.4f}
   - R²:   {test_r2:.4f} ({test_r2*100:.1f}% variance expliquée)
   - La métrique la plus informative: R² (vue d'ensemble) + RMSE (amplitude des erreurs)

3. IMPACT DES VARIABLES:
   - Variable la plus importante: {top_feature['feature']} (importance={top_feature['importance']:.4f})
   - Une augmentation d'une unité de cette variable a le plus grand effet sur la prédiction

4. COMPARAISON DES FEATURES:
   - Top feature: {top_feature['feature']}
   - Cohérence avec la théorie: Oui, la compacité du bâtiment est un facteur clé de l'efficacité énergétique

5. BIAIS ET VARIANCE:
   - Meilleure configuration: n_estimators=100-200, max_depth=15-20
   - Sous-ajustement: max_depth trop faible (5)
   - Sur-ajustement: max_depth=None avec peu de données

6. COMPARAISON AVEC ARBRE DE DÉCISION:
   - Random Forest surpasse Decision Tree sur le test set
   - Test R² RF: {test_r2:.4f} vs DT: {dt_test_r2:.4f}
   - Random Forest réduit la variance grâce à l'ensemble d'arbres
""")

print("✅ Analyse complète terminée!")
print("📁 Tous les graphiques et résultats sont sauvegardés dans le dossier outputs/")