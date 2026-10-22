"""Contextes binaires de Pima (M1), partagés par les expériences."""

import numpy as np
import pandas as pd

from donnees.pima import COLONNES_ZERO_MANQUANT, charger_decoupages, charger_pima
from m1_binarisation.binarisation import apprendre_seuils, binariser


def contextes_pima() -> tuple[dict[str, np.ndarray], list[str], dict[str, pd.Series], dict[str, tuple[float, float]]]:
	"""Contextes I et classes y de chaque découpage, noms d'attributs, seuils.

	Les seuils sont appris sur l'apprentissage seul, puis appliqués aux trois découpages.
	"""
	X, y = charger_pima()
	decoupages = charger_decoupages()
	seuils = apprendre_seuils(X.iloc[decoupages["apprentissage"]], COLONNES_ZERO_MANQUANT)

	contextes = {}
	classes = {}
	for nom, indices in decoupages.items():
		contextes[nom], noms_attributs = binariser(X.iloc[indices], seuils, COLONNES_ZERO_MANQUANT)
		classes[nom] = y.iloc[indices]
	return contextes, noms_attributs, classes, seuils


def contextes_pli(
	indices_app: np.ndarray,
	indices_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series, pd.Series, dict[str, tuple[float, float]]]:
	"""Contextes d'un pli de validation croisée : seuils réappris sur la partie d'apprentissage du pli.

	Retourne I_app, I_eval, noms d'attributs, y_app, y_eval, seuils (option b pour les zéros manquants).
	"""
	X, y = charger_pima()
	seuils = apprendre_seuils(X.iloc[indices_app], COLONNES_ZERO_MANQUANT)
	I_app, noms_attributs = binariser(X.iloc[indices_app], seuils, COLONNES_ZERO_MANQUANT)
	I_eval, _ = binariser(X.iloc[indices_eval], seuils, COLONNES_ZERO_MANQUANT)
	return I_app, I_eval, noms_attributs, y.iloc[indices_app], y.iloc[indices_eval], seuils
