import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 🔥 MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Charger les données
df = pd.read_csv("data/processed/full_processed.csv")

# Target
target = "Heating_Load"

# 🔥 IMPORTANT : enlever Cooling_Load
X = df.drop(columns=["Heating_Load", "Cooling_Load"])
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Variables pour meilleur modèle
best_score = -1
best_model = None
best_name = ""

# Fonction d’évaluation
def evaluate_model(model, model_name, params=None):
    global best_score, best_model, best_name

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} -> R2: {r2:.4f}")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_metric("r2", r2)
        if params:
            mlflow.log_params(params)
        mlflow.sklearn.log_model(model, model_name)

    # 🔥 garder le meilleur modèle
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = model_name


# 🔹 Modèles

# Linear Regression
evaluate_model(LinearRegression(), "LinearRegression")

# SVR
for c in [0.1, 1, 10]:
    name = f"SVR_C_{str(c).replace('.', '_')}"
    evaluate_model(SVR(C=c), name, {"C": c})

# Random Forest
for n in [50, 100, 200]:
    name = f"RandomForest_{n}"
    evaluate_model(RandomForestRegressor(n_estimators=n), name, {"n_estimators": n})


# 🔥 Sauvegarde du meilleur modèle
print(f"\nBest model: {best_name} avec R2 = {best_score:.4f}")

joblib.dump(best_model, "models/best_model.pkl")

print("Modèle sauvegardé dans models/")