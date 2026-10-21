"""Résultat commun aux algorithmes de factorisation booléenne (GreCon, GreConD).

I ≈ A ∘ B, avec A (objets × facteurs) et B (facteurs × attributs).
Référence : R. Belohlavek, V. Vychodil, « Discovery of optimal factors in binary
data via a novel method of matrix decomposition », JCSS 76(1), 2010.
"""

from dataclasses import dataclass

import numpy as np

from m2_facteurs.fca import Concept


@dataclass(frozen=True)
class PasTrace:
	"""Un pas de la construction d'un facteur par GreConD (option trace=True)."""

	numero_facteur: int
	numero_pas: int
	intention_depart: np.ndarray
	# valeurs[j] : croix non couvertes de la fermeture de D ∪ {j} ; None si j est déjà dans D
	valeurs: list[int | None]
	# None quand aucun attribut n'améliore V : le facteur est terminé
	attribut_retenu: int | None


@dataclass(frozen=True)
class ResultatFactorisation:
	"""Facteurs trouvés, matrices A et B, et suivi de la couverture."""

	facteurs: list[Concept]
	A: np.ndarray
	B: np.ndarray
	croix_nouvelles: list[int]
	couverture_cumulee: list[float]
	trace: list[PasTrace] | None = None


def produit_booleen(A: np.ndarray, B: np.ndarray) -> np.ndarray:
	"""(A ∘ B)_ij = OU sur k de (A_ik ET B_kj)."""
	# Le produit entier compte les facteurs qui couvrent (i, j) ; > 0 donne le OU
	return (A.astype(int) @ B.astype(int)) > 0


def construire_resultat(
	I: np.ndarray,
	facteurs: list[Concept],
	croix_nouvelles: list[int],
	trace: list[PasTrace] | None = None,
) -> ResultatFactorisation:
	"""Assemble A, B et la couverture cumulée à partir de la liste des facteurs."""
	nb_objets, nb_attributs = I.shape
	if facteurs:
		A = np.column_stack([f.extension for f in facteurs])
		B = np.vstack([f.intention for f in facteurs])
	else:
		A = np.zeros((nb_objets, 0), dtype=bool)
		B = np.zeros((0, nb_attributs), dtype=bool)

	total_croix = int(I.sum())
	couverture_cumulee = (np.cumsum(croix_nouvelles) / total_croix).tolist()
	return ResultatFactorisation(facteurs, A, B, croix_nouvelles, couverture_cumulee, trace)
