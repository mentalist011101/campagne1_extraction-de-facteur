"""Modèles classiques de comparaison (sklearn, hyperparamètres par défaut).

Données brutes : standardisation apprise sur l'apprentissage seul (dans le Pipeline).
Données binarisées : contextes de M1, utilisés tels quels.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

NOMS_MODELES = ["regression_logistique", "arbre_decision", "knn", "foret_aleatoire", "mlp"]


def creer_modeles(donnees: str) -> dict[str, Pipeline]:
	"""Les 5 modèles, non entraînés. donnees = "brutes" (standardisées) ou "binarisees"."""
	modeles_seuls = {
		"regression_logistique": LogisticRegression(random_state=42),
		"arbre_decision": DecisionTreeClassifier(random_state=42),
		"knn": KNeighborsClassifier(),
		"foret_aleatoire": RandomForestClassifier(random_state=42),
		"mlp": MLPClassifier(hidden_layer_sizes=(16,), max_iter=3000, random_state=42),
	}
	modeles = {}
	for nom, modele in modeles_seuls.items():
		if donnees == "brutes":
			etapes = [("standardisation", StandardScaler()), ("modele", modele)]
		else:
			etapes = [("modele", modele)]
		modeles[nom] = Pipeline(etapes)
	return modeles


def entrainer_et_predire(modeles: dict[str, Pipeline], X_app, y_app, X_eval) -> dict[str, np.ndarray]:
	"""Entraîne chaque modèle sur l'apprentissage et prédit X_eval."""
	predictions = {}
	for nom, modele in modeles.items():
		modele.fit(X_app, y_app)
		predictions[nom] = modele.predict(X_eval)
	return predictions
