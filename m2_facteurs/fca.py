"""Outils de base de l'analyse formelle de concepts (FCA).

Contexte formel : matrice booléenne I (objets × attributs).
Ensembles d'objets ou d'attributs : masques booléens.

Énumération des concepts : algorithme Close-by-One (CbO).
Référence : S. O. Kuznetsov, « A fast algorithm for computing all intersections
of objects from an arbitrary semilattice », Nauchno-Tekhnicheskaya Informatsiya,
Seriya 2, 1993. Présentation reprise dans R. Belohlavek, V. Vychodil (2010).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Concept:
	"""Concept formel (extension, intention) : deux masques booléens."""

	extension: np.ndarray
	intention: np.ndarray


def extension(I: np.ndarray, attributs: np.ndarray) -> np.ndarray:
	"""Objets qui possèdent tous les attributs donnés (opérateur ↓)."""
	# Un objet est gardé si, sur les colonnes demandées, il n'a que des 1
	return I[:, attributs].all(axis=1)


def intention(I: np.ndarray, objets: np.ndarray) -> np.ndarray:
	"""Attributs communs à tous les objets donnés (opérateur ↑)."""
	return I[objets, :].all(axis=0)


def est_concept(I: np.ndarray, objets: np.ndarray, attributs: np.ndarray) -> bool:
	"""Vrai si objets↑ = attributs et attributs↓ = objets."""
	return bool(
		np.array_equal(intention(I, objets), attributs)
		and np.array_equal(extension(I, attributs), objets)
	)


def enumerer_concepts(I: np.ndarray) -> list[Concept]:
	"""Tous les concepts formels de I, par Close-by-One.

	On part du concept du haut (∅↓, ∅↓↑). Depuis un concept (A, B), on essaie
	d'ajouter chaque attribut j > y (y = dernier attribut ajouté) absent de B :
	  C = A ∩ {j}↓,  D = C↑.
	Le test canonique garde (C, D) seulement si D n'ajoute aucun attribut
	d'indice < j absent de B ; sinon ce concept est atteint par une autre branche.
	Chaque concept est ainsi produit exactement une fois.
	"""
	nb_attributs = I.shape[1]
	concepts: list[Concept] = []

	aucun_attribut = np.zeros(nb_attributs, dtype=bool)
	ext_haut = extension(I, aucun_attribut)
	int_haut = intention(I, ext_haut)
	_generer_depuis(I, ext_haut, int_haut, 0, concepts)
	return concepts


def _generer_depuis(
	I: np.ndarray,
	ext_courante: np.ndarray,
	int_courante: np.ndarray,
	premier_attribut: int,
	concepts: list[Concept],
) -> None:
	"""Étape récursive de CbO : enregistre (A, B) puis explore ses descendants."""
	concepts.append(Concept(ext_courante, int_courante))
	nb_attributs = I.shape[1]

	for j in range(premier_attribut, nb_attributs):
		if int_courante[j]:
			continue
		nouvelle_ext = ext_courante & I[:, j]
		nouvelle_int = intention(I, nouvelle_ext)
		# Test canonique : sur les attributs d'indice < j, rien de nouveau
		if np.array_equal(nouvelle_int[:j], int_courante[:j]):
			_generer_depuis(I, nouvelle_ext, nouvelle_int, j + 1, concepts)
