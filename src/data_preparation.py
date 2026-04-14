import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── 1. Chargement ──────────────────────────────────────────────────────────────
df = pd.read_excel("data/raw/energy+efficiency/ENB2012_data.xlsx")

df.columns = [
    "Compacite", "Surface_Totale", "Surface_Murs", "Surface_Toit",
    "Hauteur", "Orientation", "Surface_Vitree", "Distribution_Vitree",
    "Charge_Chauffage", "Charge_Climatisation"
]

# ── 2. Rapport initial ─────────────────────────────────────────────────────────
print("=== STATISTIQUES ===")
print(df.describe().round(2))
print(f"\nValeurs manquantes : {df.isnull().sum().sum()}")
print(f"Doublons           : {df.duplicated().sum()}")

# ── 3. Nettoyage ───────────────────────────────────────────────────────────────
df = df.drop_duplicates().dropna()
print(f"\n✔ Données nettoyées : {df.shape[0]} lignes")

# ── 4. Analyse des outliers (IQR) ──────────────────────────────────────────────
print("\n=== OUTLIERS (IQR) ===")
for col in df.columns:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col:<30} : {n} outlier(s)")
# Les outliers sont conservés car ce sont des simulations légitimes

# ── 5. Visualisations ──────────────────────────────────────────────────────────

# Matrice de corrélation
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr().round(2), annot=True, cmap="RdYlGn", center=0)
plt.title("Matrice de Corrélation")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png")
plt.close()

# Distribution des targets
df[["Charge_Chauffage", "Charge_Climatisation"]].hist(bins=30, figsize=(10, 4), color="steelblue")
plt.suptitle("Distribution des Targets")
plt.tight_layout()
plt.savefig("outputs/distribution_targets.png")
plt.close()

# Boxplots des features
df.drop(columns=["Charge_Chauffage", "Charge_Climatisation"]).boxplot(figsize=(12, 5), rot=30)
plt.title("Boxplots des Features")
plt.tight_layout()
plt.savefig("outputs/boxplots.png")
plt.close()

print("\n✔ Visualisations sauvegardées dans outputs/")

# ── 6. Séparation + Split ──────────────────────────────────────────────────────
X = df.drop(columns=["Charge_Chauffage", "Charge_Climatisation"])
y = df[["Charge_Chauffage", "Charge_Climatisation"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n✔ Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

# ── 7. Normalisation ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

# Sauvegarde du scaler pour usage futur
joblib.dump(scaler, "data/processed/scaler.pkl")
print("✔ Scaler sauvegardé → data/processed/scaler.pkl")

# ── 8. Sauvegarde des données ──────────────────────────────────────────────────
X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
X_test_scaled.to_csv( "data/processed/X_test.csv",  index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv( "data/processed/y_test.csv",  index=False)

print("✔ Données sauvegardées dans data/processed/")
