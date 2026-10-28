"""Stacking sur les facteurs : régression logistique sur la matrice A (objets × facteurs).

Les facteurs deviennent des attributs ; leurs poids sont appris au lieu d'être fixés par
une heuristique (précision, accuracy, lift). A est calculée sans la classe (GreConD sur
les attributs seuls) : seule la régression voit y, d'où l'absence de fuite (DECISIONS.md).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from m2_facteurs.fca import Concept, extension


def matrice_facteurs(I: np.ndarray, facteurs: list[Concept]) -> np.ndarray:
	"""A[i, f] = 1 si l'objet i possède tous les attributs de l'intention du facteur f."""
	return np.column_stack([extension(I, facteur.intention) for facteur in facteurs])


def empiler(
	facteurs: list[Concept],
	I_app: np.ndarray,
	y_app,
	I_eval: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression]:
	"""Entraîne la régression logistique sur (A_app, y_app) et prédit A_eval."""
	modele = LogisticRegression(max_iter=1000, random_state=42)
	modele.fit(matrice_facteurs(I_app, facteurs), y_app)
	return modele.predict(matrice_facteurs(I_eval, facteurs)), modele
