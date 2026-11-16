"""Bagging du classifieur factoriel M3 (best-vote + précision), pour deux classes ou plus.

Chaque sac est un tirage avec remise de la partie d'apprentissage ; dans chaque sac,
GreConD (sans la classe) puis M3, classes et poids calculés sur le sac. Les prédictions
des sacs sont agrégées par vote majoritaire. Choix détaillés : DECISIONS.md (étape 17 ;
égalités à plusieurs classes : campagne 1, étape 3).
Référence : L. Breiman, « Bagging predictors », Machine Learning 24(2), 1996.
"""

import time
from dataclasses import dataclass

import numpy as np

from m2_facteurs.grecond import DelaiDepasse, grecond
from m3_classifieurs.classifieur_factoriel import compter, construire_regles, departager, predire


@dataclass(frozen=True)
class ResultatBagging:
	predictions: np.ndarray
	# Proportion de sacs qui prédisent chaque classe (colonnes dans l'ordre de classes) : signal possible pour M5
	proportions: np.ndarray
	non_couverts_par_sac: list[int]
	temps_par_sac: list[float]


def tirer_sac(n_app: int, numero_sac: int, graine_base: int = 42) -> np.ndarray:
	"""n_app indices tirés avec remise, graine graine_base + numero_sac."""
	return np.random.default_rng(graine_base + numero_sac).integers(0, n_app, n_app)


def agreger(predictions_par_sac: np.ndarray, classes: list, classe_par_defaut) -> tuple[np.ndarray, np.ndarray]:
	"""Vote majoritaire des sacs (lignes) ; égalité → règle de predire (classe_par_defaut, puis ordre de classes)."""
	proportions = np.column_stack([(predictions_par_sac == c).mean(axis=0) for c in classes])
	predictions = []
	for ligne in proportions:
		gagnantes = [c for c, p in zip(classes, ligne) if p == ligne.max()]
		predictions.append(departager(gagnantes, classe_par_defaut))
	return np.array(predictions), proportions


def bagging_m3(
	I_app: np.ndarray,
	y_app,
	I_eval: np.ndarray,
	classes: list,
	classe_par_defaut,
	nb_sacs: int = 25,
	graine_base: int = 42,
	delai_max_s: float | None = None,
) -> ResultatBagging:
	"""B sacs, un M3 (best-vote + précision) par sac, agrégation par vote majoritaire.

	DelaiDepasse (avec le numéro du sac) si le GreConD d'un sac dépasse delai_max_s.
	"""
	y_app = np.asarray(y_app)
	predictions_par_sac = []
	non_couverts_par_sac = []
	temps_par_sac = []
	for numero_sac in range(nb_sacs):
		debut = time.perf_counter()
		indices = tirer_sac(len(y_app), numero_sac, graine_base)
		I_sac, y_sac = I_app[indices], y_app[indices]
		try:
			facteurs = grecond(I_sac, delai_max_s=delai_max_s).facteurs
		except DelaiDepasse as erreur:
			raise DelaiDepasse(erreur.duree_s, erreur.nb_facteurs, numero_sac) from erreur
		regles, ecartes = construire_regles(facteurs, y_sac, I_sac, y_sac, mesure="precision")
		predictions = predire(regles, I_eval, classes, classe_par_defaut, vote="best-vote")
		predictions_par_sac.append([p.classe for p in predictions])
		non_couverts_par_sac.append(compter(predictions)["nb_non_couverts"])
		temps_par_sac.append(time.perf_counter() - debut)

	predictions, proportions = agreger(np.array(predictions_par_sac), classes, classe_par_defaut)
	return ResultatBagging(predictions, proportions, non_couverts_par_sac, temps_par_sac)
