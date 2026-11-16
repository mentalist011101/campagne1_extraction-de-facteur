"""M1 : binarisation des variables numériques par terciles.

Chaque variable donne 3 attributs binaires : bas (x <= s1), moyen (s1 < x <= s2),
haut (x > s2), avec s1 et s2 les quantiles 1/3 et 2/3 de l'apprentissage.
Les seuils sont appris une fois, puis appliqués tels quels à tout nouvel x.

Zéros manquants (voir DECISIONS.md) : pour les colonnes listées, un zéro signifie
« valeur inconnue » ; il est exclu du calcul des seuils et ne reçoit aucun attribut.

Campagne 1 (apprendre_binarisation, binariser_jeu) : variables numériques et catégorielles
mêlées, attributs constants retirés et attributs identiques fusionnés sur l'apprentissage
(voir DECISIONS.md, 2026-11-16).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_VALEURS_CATEGORIELLE = 5


@dataclass(frozen=True)
class Binarisation:
	"""Binarisation apprise sur la partie d'apprentissage d'un pli, appliquée telle quelle ensuite."""

	seuils: dict[str, tuple[float, float]]
	modalites: dict[str, list]
	colonnes: list[str]
	gardes: np.ndarray
	noms: list[str]
	nb_constants: int
	nb_fusions: int


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


def apprendre_modalites(X_apprentissage: pd.DataFrame) -> dict[str, list[str]]:
	"""Valeurs de chaque variable catégorielle, apprises sur l'apprentissage seul.

	« ? » (valeur manquante) n'est pas une modalité. Une valeur présente chez tous les
	objets donnerait un attribut constant : elle est retirée (voir DECISIONS.md, étape 14).
	"""
	modalites = {}
	for colonne in X_apprentissage.columns:
		effectifs = X_apprentissage[colonne][X_apprentissage[colonne] != "?"].value_counts()
		modalites[colonne] = sorted(v for v, n in effectifs.items() if n < len(X_apprentissage))
	return modalites


def binariser_categoriel(X: pd.DataFrame, modalites: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
	"""Un attribut « variable=valeur » par modalité apprise.

	« ? » et toute valeur absente de l'apprentissage ne reçoivent aucun attribut.
	"""
	colonnes_binaires = []
	noms_attributs = []
	for colonne, valeurs in modalites.items():
		for valeur in valeurs:
			colonnes_binaires.append((X[colonne] == valeur).to_numpy())
			noms_attributs.append(f"{colonne}={valeur}")
	return np.column_stack(colonnes_binaires), noms_attributs


def apprendre_binarisation(X_app: pd.DataFrame, types: dict[str, str]) -> Binarisation:
	"""Catégorielle ou numérique à au plus 5 valeurs : un attribut par valeur ; sinon interordinale.

	Les attributs suivent l'ordre des colonnes de X_app (il décide des égalités de GreConD).
	Puis, sur l'apprentissage : attributs constants retirés, attributs identiques fusionnés
	(le premier dans l'ordre des colonnes est gardé). Manquant (NaN) : aucun attribut.
	"""
	seuils, modalites = {}, {}
	for colonne in X_app.columns:
		valeurs = X_app[colonne]
		if types[colonne] == "numerique" and valeurs.nunique() > MAX_VALEURS_CATEGORIELLE:
			seuils[colonne] = apprendre_seuils(X_app[[colonne]], colonnes_zero_manquant=[])[colonne]
		else:
			modalites[colonne] = apprendre_modalites(X_app[[colonne]])[colonne]
	colonnes = list(X_app.columns)
	I_complet, noms_complets = _binariser_complet(X_app, seuils, modalites, colonnes)

	non_constants = I_complet.any(axis=0) & ~I_complet.all(axis=0)
	gardes, deja_vues = [], set()
	for j in np.flatnonzero(non_constants):
		cle = I_complet[:, j].tobytes()
		if cle not in deja_vues:
			deja_vues.add(cle)
			gardes.append(j)
	return Binarisation(
		seuils, modalites, colonnes, np.array(gardes, dtype=int), [noms_complets[j] for j in gardes],
		nb_constants=int((~non_constants).sum()), nb_fusions=int(non_constants.sum()) - len(gardes),
	)


def binariser_jeu(X: pd.DataFrame, binarisation: Binarisation) -> np.ndarray:
	"""Contexte formel de X avec les attributs gardés ; rien n'est réappris ici."""
	I_complet, _ = _binariser_complet(X, binarisation.seuils, binarisation.modalites, binarisation.colonnes)
	return I_complet[:, binarisation.gardes]


def _binariser_complet(X, seuils, modalites, colonnes) -> tuple[np.ndarray, list[str]]:
	blocs, noms = [], []
	for colonne in colonnes:
		if colonne in seuils:
			bloc, noms_bloc = binariser(X[[colonne]], {colonne: seuils[colonne]}, [], echelle="interordinale")
		elif modalites[colonne]:
			bloc, noms_bloc = binariser_categoriel(X[[colonne]], {colonne: modalites[colonne]})
		else:
			continue
		blocs.append(bloc)
		noms += noms_bloc
	return np.column_stack(blocs), noms
