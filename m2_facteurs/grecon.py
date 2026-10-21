"""GreCon : factorisation booléenne gloutonne parmi tous les concepts formels.

Référence : R. Belohlavek, V. Vychodil, « Discovery of optimal factors in binary
data via a novel method of matrix decomposition », JCSS 76(1), 2010.

Fonctionne sur n'importe quel contexte binaire (échantillon, sous-ensemble d'attributs…).
"""

import numpy as np

from m2_facteurs.factorisation import ResultatFactorisation, construire_resultat
from m2_facteurs.fca import Concept, enumerer_concepts


def grecon(
	I: np.ndarray,
	couverture_cible: float = 1.0,
	nb_max_facteurs: int | None = None,
) -> ResultatFactorisation:
	"""Choisit, tant qu'il reste des croix, le concept qui en couvre le plus de non couvertes.

	Arrêt : couverture ≥ couverture_cible, ou nombre de facteurs = nb_max_facteurs.
	"""
	# Pas 1 : tous les concepts du contexte
	concepts = enumerer_concepts(I)

	non_couvertes = I.copy()
	total_croix = int(I.sum())
	facteurs: list[Concept] = []
	croix_nouvelles: list[int] = []

	# Pas 2 : boucle gloutonne
	while non_couvertes.any():
		meilleur = concepts[0]
		meilleur_compte = _compter_non_couvertes(non_couvertes, meilleur)
		for concept in concepts[1:]:
			compte = _compter_non_couvertes(non_couvertes, concept)
			if _est_meilleur(compte, concept, meilleur_compte, meilleur):
				meilleur, meilleur_compte = concept, compte

		facteurs.append(meilleur)
		croix_nouvelles.append(meilleur_compte)
		# Les croix du rectangle extension × intention sont désormais couvertes
		non_couvertes[np.ix_(meilleur.extension, meilleur.intention)] = False

		# Pas 3 : critères d'arrêt
		couverture = sum(croix_nouvelles) / total_croix
		if couverture >= couverture_cible:
			break
		if nb_max_facteurs is not None and len(facteurs) >= nb_max_facteurs:
			break

	# Pas 4 : facteurs, A, B, couverture cumulée
	return construire_resultat(I, facteurs, croix_nouvelles)


def _compter_non_couvertes(non_couvertes: np.ndarray, concept: Concept) -> int:
	"""Nombre de croix encore non couvertes dans le rectangle du concept."""
	return int(non_couvertes[np.ix_(concept.extension, concept.intention)].sum())


def _est_meilleur(compte: int, concept: Concept, meilleur_compte: int, meilleur: Concept) -> bool:
	"""Vrai si le candidat doit remplacer le meilleur courant.

	Règle d'égalité (pas 5, voir DECISIONS.md) :
	  1. plus de croix non couvertes ;
	  2. à égalité, plus grande extension ;
	  3. à égalité encore, intention (liste croissante d'indices) première dans l'ordre lexicographique.
	"""
	if compte != meilleur_compte:
		return compte > meilleur_compte

	taille_ext = int(concept.extension.sum())
	taille_ext_meilleur = int(meilleur.extension.sum())
	if taille_ext != taille_ext_meilleur:
		return taille_ext > taille_ext_meilleur

	# Les listes Python se comparent dans l'ordre lexicographique : [0, 1, 2] < [0, 3, 4]
	indices = np.flatnonzero(concept.intention).tolist()
	indices_meilleur = np.flatnonzero(meilleur.intention).tolist()
	return indices < indices_meilleur
