"""M1 : binarisation des variables numériques par terciles.

Chaque variable donne 3 attributs binaires : bas (x <= s1), moyen (s1 < x <= s2),
haut (x > s2), avec s1 et s2 les quantiles 1/3 et 2/3 de l'apprentissage.
Les seuils sont appris une fois, puis appliqués tels quels à tout nouvel x.

Zéros manquants (voir DECISIONS.md) : pour les colonnes listées, un zéro signifie
« valeur inconnue » ; il est exclu du calcul des seuils et ne reçoit aucun attribut.
"""

import numpy as np
import pandas as pd


def apprendre_seuils(
	X_apprentissage: pd.DataFrame,
	colonnes_zero_manquant: list[str],
) -> dict[str, tuple[float, float]]:
	"""Quantiles 1/3 et 2/3 de chaque variable, calculés sur l'apprentissage seul."""
	seuils = {}
	for colonne in X_apprentissage.columns:
		valeurs = X_apprentissage[colonne]
		if colonne in colonnes_zero_manquant:
			valeurs = valeurs[valeurs != 0]
		seuils[colonne] = (float(valeurs.quantile(1 / 3)), float(valeurs.quantile(2 / 3)))
	return seuils


def binariser(
	X: pd.DataFrame,
	seuils: dict[str, tuple[float, float]],
	colonnes_zero_manquant: list[str],
	echelle: str = "nominale",
) -> tuple[np.ndarray, list[str]]:
	"""Contexte formel et noms lisibles des attributs.

	echelle = "nominale" : 3 attributs par variable, intervalles disjoints (bas, moyen, haut).
	echelle = "interordinale" : 4 attributs par variable, v<=t1, v<=t2, v>t1, v>t2 ;
	l'intervalle moyen s'écrit alors comme la conjonction {v>t1, v<=t2}.
	Les seuils ne sont jamais recalculés ici : X peut être un seul objet.
	"""
	colonnes_binaires = []
	noms_attributs = []
	for colonne, (bas, haut) in seuils.items():
		valeurs = X[colonne].to_numpy()
		if colonne in colonnes_zero_manquant:
			connue = valeurs != 0
		else:
			connue = np.ones(len(valeurs), dtype=bool)

		# Seuils affichés avec 4 chiffres significatifs ; la comparaison utilise la valeur exacte
		if echelle == "nominale":
			colonnes_binaires += [
				connue & (valeurs <= bas),
				connue & (valeurs > bas) & (valeurs <= haut),
				connue & (valeurs > haut),
			]
			noms_attributs += [
				f"{colonne}<={bas:.4g}",
				f"{bas:.4g}<{colonne}<={haut:.4g}",
				f"{colonne}>{haut:.4g}",
			]
		else:
			# Ordre des colonnes fixé (voir DECISIONS.md) : il décide des égalités de GreConD
			colonnes_binaires += [
				connue & (valeurs <= bas),
				connue & (valeurs <= haut),
				connue & (valeurs > bas),
				connue & (valeurs > haut),
			]
			noms_attributs += [
				f"{colonne}<={bas:.4g}",
				f"{colonne}<={haut:.4g}",
				f"{colonne}>{bas:.4g}",
				f"{colonne}>{haut:.4g}",
			]
	return np.column_stack(colonnes_binaires), noms_attributs
