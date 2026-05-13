# Rapport - Tâche 4: Random Forest
## Interprétation et Analyse des Algorithmes d'Ensemble

---

## Contexte du Projet

Ce projet porte sur la prédiction de la **charge de chauffage** (Heating_Load) des bâtiments basée sur leurs caractéristiques physiques. Le jeu de données **Energy Efficiency** contient 768 échantillons avec 19 features.

---

## 1. Analyse des Résidus

### Question: Les résidus sont-ils distribués normalement? Y a-t-il des patterns (hétéroscédasticité)?

### Résultats:

| Test | Statistique | p-value | Interprétation |
|------|-------------|---------|----------------|
| Shapiro-Wilk | 0.9549 | 0.0001 | Non normal |
| D'Agostino-Pearson | 1.6062 | 0.4479 | Normal |

**Statistiques des résidus:**
- Moyenne: -0.0192 (proche de 0)
- Écart-type: 0.5339
- Min: -1.4101 | Max: 1.3610

### Analyse:
- **Normalité**: Le test de Shapiro-Wilk rejette l'hypothèse de normalité (p < 0.05), mais le test D'Agostino-Pearson ne la rejette pas (p = 0.4479). La distribution des résidus est **approximativement normale** mais pas parfaitement.
- **Hétéroscédasticité**: Corrélation entre |résidus| et prédictions = 0.5258 (p < 0.0001). Il y a une **hétéroscédasticité présente** - la variance des erreurs augmente avec les prédictions.

### Conclusion:
Les résidus ne suivent pas parfaitement une distribution normale, ce qui est fréquent avec les modèles d'ensemble. L'hétéroscédasticité suggère que le modèle a plus de difficulté à prédire les valeurs extrêmes.

---

## 2. Métriques de Performance

### Question: Comparez MAE, MSE et R². Quelle métrique est la plus informative pour votre problème?

### Résultats:

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **MAE** | 0.3686 | Erreur moyenne absolue |
| **MSE** | 0.2836 | Erreur quadratique moyenne |
| **RMSE** | 0.5326 | Écart-type des erreurs |
| **R²** | 0.9973 | 99.7% de la variance expliquée |

### Analyse:

Pour ce problème de régression énergétique:

1. **R² est la métrique la plus informative** car:
   - Elle donne une vue d'ensemble de la qualité du modèle
   - 99.7% de la variance est expliquée → modèle excellent
   - Facile à interpréter pour les parties prenantes

2. **RMSE** est utile pour:
   - Comprendre l'amplitude des erreurs en unités réelles
   - Détecter les grandes erreurs (via le carré)

3. **MAE** est plus robuste aux valeurs aberrantes mais moins sensible aux erreurs systématiques.

---

## 3. Impact des Variables

### Question: Comment une augmentation d'une unité de la variable X affecte-t-elle la variable cible, toutes choses égales par ailleurs?

### Feature Importances (Top 10):

| Feature | Importance |
|---------|------------|
| Relative_Compactness | 0.1970 |
| Height_x_Compactness | 0.1704 |
| Surface_Area | 0.1583 |
| Wall_to_Surface_Ratio | 0.1294 |
| Roof_to_Surface_Ratio | 0.0987 |
| Glazing_Area | 0.0760 |
| Roof_Area | 0.0719 |
| Overall_Height | 0.0559 |
| Wall_Area | 0.0262 |
| Glazing_Area_Distribution_0 | 0.0145 |

### Analyse:

- **Relative_Compactness** (compacité relative) a le plus grand impact: une augmentation d'une unité de cette variable a le **plus grand effet** sur la prédiction de la charge de chauffage.
- L'effet exact dépend des interactions dans les arbres, mais cette variable capture la relation surface/volume du bâtiment.
- Les variables d'orientation et de distribution de vitrage ont un impact minimal.

---

## 4. Comparaison de Features

### Question: Quelle variable a le plus grand pouvoir prédictif? Est-ce cohérent avec la théorie?

### Réponse:

**Variable avec le plus grand pouvoir prédictif:** `Relative_Compactness` (importance = 0.1970)

### Cohérence avec la théorie:

**Oui, parfaitement cohérent!** La compacité du bâtiment est un facteur clé en efficacité énergétique car:

1. **Physique du bâtiment**: Un bâtiment compact (ratio surface/volume faible) perd moins de chaleur car il y a moins de surface exposée à l'extérieur.

2. **Théorie thermique**: Q = U × A × ΔT
   - Q = perte de chaleur
   - U = coefficient de transmission thermique
   - A = surface de déperditions
   - ΔT = différence de température

   La compacité réduit A pour un même volume, donc réduit Q.

3. **Variables cohérentes**: 
   - `Surface_Area` (3ème plus importante)
   - `Overall_Height` et `Height_x_Compactness` (2ème)
   
   Toutes ces variables sont liées à la géométrie du bâtiment et à ses déperditions thermiques.

---

## 5. Biais et Variance

### Question: Tester différents hyperparamètres et observer leur impact. Étudier le biais et la variance.

### Tableau d'Analyse:

| n_estimators | max_depth | Train RMSE | Test RMSE | Biais | Variance |
|--------------|-----------|------------|-----------|-------|----------|
| 50 | 5 | 0.8423 | 1.0563 | 0.8423 | 0.2140 |
| 50 | 10 | 0.2092 | 0.5324 | 0.2092 | 0.3232 |
| 50 | None | 0.1868 | 0.5353 | 0.1868 | 0.3484 |
| 100 | 5 | 0.8375 | 1.0498 | 0.8375 | 0.2123 |
| 100 | 10 | 0.2095 | 0.5287 | 0.2095 | 0.3192 |
| 100 | None | 0.1880 | 0.5326 | 0.1880 | 0.3446 |

### Analyse:

**Sous-ajustement (High Bias):**
- `max_depth = 5`: Biais élevé (~0.84), le modèle est trop simple
- Train RMSE ≈ Test RMSE → sous-ajustement

**Sur-ajustement (High Variance):**
- `max_depth = None`: Variance élevée (~0.34-0.35)
- Train RMSE << Test RMSE → sur-ajustement

**Configuration optimale:**
- `n_estimators = 100`, `max_depth = 10`: Bon équilibre
- Test RMSE le plus bas: 0.5287

### Limites du modèle - Plages de valeurs où le modèle performe moins bien:

1. **max_depth très faible (5)**:
   - Sous-ajustement important
   - Le modèle ne capture pas les relations complexes
   - Biais élevé

2. **max_depth = None (sans limite)**:
   - Risque de sur-ajustement
   - Chaque arbre apprend trop细节 des données d'entraînement
   - Variance plus élevée

3. **n_estimators trop faible (10)**:
   - Instabilité des arbres
   - Variance élevée due au manque de diversité

---

## 6. Comparaison avec Arbre de Décision

### Question: Comparez les résultats avec ceux de l'algorithme Arbre de décision.

### Résultats:

| Métrique | Random Forest | Decision Tree |
|----------|---------------|---------------|
| Train RMSE | 0.1880 | 0.0000 |
| Test RMSE | 0.5326 | 0.6498 |
| Train R² | 0.9996 | 1.0000 |
| Test R² | 0.9973 | 0.9959 |
| Test MAE | 0.3686 | 0.4275 |
| Test MSE | 0.2836 | 0.4222 |

### Analyse:

1. **Random Forest surpasse Decision Tree** sur le test set:
   - Test RMSE: 0.5326 vs 0.6498 (18% meilleur)
   - Test R²: 0.9973 vs 0.9959

2. **Decision Tree montre un sur-ajustement évident**:
   - Train RMSE = 0 (apprentissage parfait!)
   - Mais Test RMSE = 0.6498

3. **Random Forest réduit la variance** grâce à:
   - Vote/moyenne de plusieurs arbres
   - Feature subsampling (random features à chaque split)
   - Sample bootstrapping

4. **Conclusion**: Random Forest est plus **robuste** et **généralise mieux** que l'arbre de décision unique.

---

## Résumé des Réponses

| Question | Réponse |
|----------|---------|
| 1. Résidus normaux? | Non parfaitement (Shapiro p=0.0001), hétéroscédasticité présente |
| 2. Métrique la plus informative? | **R²** (99.7% variance expliquée) + RMSE |
| 3. Impact d'une unité X? | Relative_Compactness a le plus grand effet |
| 4. Variable la plus prédictive? | **Relative_Compactness** (0.197), cohérente avec la théorie thermique |
| 5. Biais/Variance? | max_depth=10 optimal, sous-ajustement si trop faible, sur-ajustement si None |
| 6. RF vs DT? | RF meilleur (Test R² 0.9973 vs 0.9959), variance réduite |

---

## Fichiers Générés

- `outputs/rf_residuals_analysis.png` - Analyse des résidus
- `outputs/rf_feature_importance.png` - Importance des features
- `outputs/rf_bias_variance.png` - Analyse biais-variance
- `outputs/rf_vs_dt_comparison.png` - Comparaison RF vs DT
- `outputs/rf_hyperparameters_analysis.csv` - Tableau des hyperparamètres

---

## Conclusion

Le modèle **Random Forest** atteint des performances excellentes avec un R² de **99.73%** sur le test set. L'analyse montre que:

1. La **compacité relative** du bâtiment est le facteur le plus important pour prédire la charge de chauffage
2. Le modèle présente une légère hétéroscédasticité mais reste très performant
3. L'équilibre biais-variance est optimal avec `n_estimators=100` et `max_depth=10`
4. Random Forest surpasse significativement l'arbre de décision en réduisant la variance

Le suivi MLflow a permis de documenter toutes les expérimentations avec les métriques et les modèles sauvegardés.