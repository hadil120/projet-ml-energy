# ProjetML

Ce projet Python/React couvre la préparation des données, l'entraînement d'un modèle de régression, l'évaluation et une interface frontend légère.

## Structure du projet

- `data/raw/energy+efficiency/ENB2012_data.xlsx` : données brutes du jeu de données Energy Efficiency.
- `data/processed/` : données prétraitées et fichiers CSV générés.
- `models/` : modèle entraîné sauvegardé (`best_model.pkl`).
- `outputs/` : graphiques et visualisations générés pendant le prétraitement.
- `src/` : scripts Python principaux.
  - `data_preparation.py` : nettoyage, analyse, normalisation et sauvegarde des données traitées.
  - `train.py` : entraînement de plusieurs modèles, suivi MLflow et sauvegarde du meilleur modèle.
  - `evaluate.py` : évaluation du modèle sauvegardé sur le jeu de données complet.
- `frontend/` : application React/Vite pour la visualisation ou le prototype frontend.

## Prérequis

- Python 3.8+ recommandé
- Node.js 18+ pour le frontend

## Installation Python

1. Créez un environnement virtuel:

```bash
python -m venv .venv
```

2. Activez l'environnement:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Installez les dépendances:

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Prétraitement des données

Exécutez `data_preparation.py` pour charger les données brutes, nettoyer, analyser, normaliser et sauvegarder les jeux de données:

```bash
python src/data_preparation.py
```

Les fichiers créés sont :
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `data/processed/scaler.pkl`

### 2. Entraînement du modèle

Exécutez `train.py` pour entraîner plusieurs modèles, suivre les expériences avec MLflow et sauvegarder le meilleur modèle:

```bash
python src/train.py
```

Le script entraîne désormais plusieurs modèles :
- `LinearRegression`
- `SVR`
- `RandomForest`
- `AdaBoost`
- `XGBoost`

Le meilleur modèle est sauvegardé dans :
- `models/best_model.pkl`

> Assurez-vous que le serveur MLflow est disponible si vous souhaitez utiliser le suivi MLflow.

### 2.1. Afficher les expériences MLflow

Avant d’exécuter `src/train.py`, lancez d’abord le serveur MLflow dans un terminal séparé :

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

Si la commande `mlflow ui` plante dans PowerShell, utilisez directement :

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

Ensuite exécutez l’entraînement dans un autre terminal :

```bash
python src/train.py
```

Puis ouvrez le navigateur sur :

```text
http://127.0.0.1:5000
```

Si vous préférez utiliser une URI MLflow différente, définissez-la avant l’exécution :

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python src/train.py
```

### 3. Évaluation du modèle

Pour évaluer le modèle sauvegardé sur l'ensemble de données complet :

```bash
python src/evaluate.py
```

### 4. Frontend

Le dossier `frontend/` contient une application React avec Vite. Pour lancer le frontend :

```bash
cd frontend
npm install
npm run dev
```

## Notes importantes

- Les scripts `train.py` et `evaluate.py` utilisent la même target `Heating_Load`.
- `train.py` exclut à la fois `Heating_Load` et `Cooling_Load` des features.
- `data_preparation.py` génère des visualisations dans `outputs/`.

## Dépendances principales

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- openpyxl
- xgboost
- joblib

