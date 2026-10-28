"""Bagging du classifieur factoriel M3 (best-vote + précision), pour une classe binaire 0 / 1.

Chaque sac est un tirage avec remise de la partie d'apprentissage ; dans chaque sac,
GreConD (sans la classe) puis M3, classes et poids calculés sur le sac. Les prédictions
des sacs sont agrégées par vote majoritaire. Choix détaillés : DECISIONS.md (étape 17).
Référence : L. Breiman, « Bagging predictors », Machine Learning 24(2), 1996.
"""

import time
from dataclasses import dataclass

import numpy as np

from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import compter, construire_regles, predire


@dataclass(frozen=True)
class ResultatBagging:
	predictions: np.ndarray
	# Proportion de sacs qui prédisent la classe 1 : signal possible pour M5
	proportion_classe_1: np.ndarray
	non_couverts_par_sac: list[int]
	temps_par_sac: list[float]


def tirer_sac(n_app: int, numero_sac: int, graine_base: int = 42) -> np.ndarray:
	"""n_app indices tirés avec remise, graine graine_base + numero_sac."""
	return np.random.default_rng(graine_base + numero_sac).integers(0, n_app, n_app)


def agreger(predictions_par_sac: np.ndarray, classe_par_defaut: int) -> tuple[np.ndarray, np.ndarray]:
	"""Vote majoritaire des sacs (lignes) ; égalité → classe_par_defaut."""
	proportion_classe_1 = predictions_par_sac.mean(axis=0)
	predictions = np.where(proportion_classe_1 > 0.5, 1, 0)
	predictions[proportion_classe_1 == 0.5] = classe_par_defaut
	return predictions, proportion_classe_1


def bagging_m3(
	I_app: np.ndarray,
	y_app,
	I_eval: np.ndarray,
	classe_par_defaut: int,
	nb_sacs: int = 25,
	graine_base: int = 42,
) -> ResultatBagging:
	"""B sacs, un M3 (best-vote + précision) par sac, agrégation par vote majoritaire."""
	y_app = np.asarray(y_app)
	predictions_par_sac = []
	non_couverts_par_sac = []
	temps_par_sac = []
	for numero_sac in range(nb_sacs):
		debut = time.perf_counter()
		indices = tirer_sac(len(y_app), numero_sac, graine_base)
		I_sac, y_sac = I_app[indices], y_app[indices]
		facteurs = grecond(I_sac).facteurs
		regles, ecartes = construire_regles(facteurs, y_sac, I_sac, y_sac, mesure="precision")
		predictions = predire(regles, I_eval, [0, 1], classe_par_defaut, vote="best-vote")
		predictions_par_sac.append([p.classe for p in predictions])
		non_couverts_par_sac.append(compter(predictions)["nb_non_couverts"])
		temps_par_sac.append(time.perf_counter() - debut)

	predictions, proportion_classe_1 = agreger(np.array(predictions_par_sac), classe_par_defaut)
	return ResultatBagging(predictions, proportion_classe_1, non_couverts_par_sac, temps_par_sac)
