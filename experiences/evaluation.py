"""Mesures d'évaluation communes aux classifieurs factoriels et aux modèles classiques."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from m3_classifieurs.classifieur_factoriel import PredictionFactorielle


def metriques(y_vrai, y_pred) -> dict:
	"""Accuracy, F1 macro, et nombre de x prédits en classe 1."""
	return {
		"accuracy": float(accuracy_score(y_vrai, y_pred)),
		"f1_macro": float(f1_score(y_vrai, y_pred, average="macro")),
		"predits_classe_1": int((np.asarray(y_pred) == 1).sum()),
	}


def taux_accord(predictions_a, predictions_b) -> float:
	"""Proportion de x pour lesquels les deux méthodes prédisent la même classe."""
	return float((np.asarray(predictions_a) == np.asarray(predictions_b)).mean())


def accuracy_decides(y_vrai, predictions: list[PredictionFactorielle]) -> float:
	"""Accuracy sur les seuls x couverts et sans égalité (NaN si aucun)."""
	y_vrai = np.asarray(y_vrai)
	decides = np.array([not (p.non_couvert or p.egalite) for p in predictions])
	if not decides.any():
		return float("nan")
	classes_predites = np.array([p.classe for p in predictions])
	return float((classes_predites[decides] == y_vrai[decides]).mean())
