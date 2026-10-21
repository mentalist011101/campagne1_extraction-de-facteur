"""Contextes binaires de Pima (M1), partagés par les expériences des étapes 5 à 9."""

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
