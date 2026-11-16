"""GreConD : factorisation booléenne gloutonne, facteurs construits à la demande.

Contrairement à GreCon, on n'énumère pas tous les concepts : chaque facteur est
construit en ajoutant un attribut à la fois à son intention.

Référence : R. Belohlavek, V. Vychodil, « Discovery of optimal factors in binary
data via a novel method of matrix decomposition », JCSS 76(1), 2010.

Fonctionne sur n'importe quel contexte binaire (échantillon, sous-ensemble d'attributs…).
"""

import time
from dataclasses import dataclass

import numpy as np

from m2_facteurs.factorisation import PasTrace, ResultatFactorisation, construire_resultat
from m2_facteurs.fca import Concept, extension, intention


class DelaiDepasse(Exception):
	"""GreConD a dépassé delai_max_s (vérifié après chaque facteur) ; numero_sac renseigné par le bagging."""

	def __init__(self, duree_s: float, nb_facteurs: int, numero_sac: int | None = None):
		super().__init__(f"GreConD : {duree_s:.0f} s après {nb_facteurs} facteurs")
		self.duree_s = duree_s
		self.nb_facteurs = nb_facteurs
		self.numero_sac = numero_sac


@dataclass(frozen=True)
class ChoixEgalite:
	"""Un pas de GreConD où le meilleur compte était atteint par plusieurs attributs."""

	numero_facteur: int
	numero_pas: int
	attributs_a_egalite: list[int]
	valeur: int
	# Faux si tous les attributs à égalité mènent à la même fermeture : l'égalité ne change rien
	fermetures_differentes: bool


def grecond(
	I: np.ndarray,
	couverture_cible: float = 1.0,
	nb_max_facteurs: int | None = None,
	trace: bool = False,
	delai_max_s: float | None = None,
) -> ResultatFactorisation:
	"""Construit les facteurs un par un jusqu'à couverture ≥ couverture_cible ou nb_max_facteurs atteint.

	Avec trace=True, chaque pas (valeur de chaque attribut candidat) est enregistré.
	Avec delai_max_s, DelaiDepasse est levée si la durée le dépasse après un facteur.
	"""
	debut = time.perf_counter()
	non_couvertes = I.copy()
	total_croix = int(I.sum())
	facteurs: list[Concept] = []
	croix_nouvelles: list[int] = []
	pas_trace: list[PasTrace] | None = [] if trace else None

	while non_couvertes.any():
		facteur, compte = _construire_un_facteur(I, non_couvertes, len(facteurs) + 1, pas_trace)
		facteurs.append(facteur)
		croix_nouvelles.append(compte)
		# Pas 4 : les croix du rectangle du facteur sont désormais couvertes
		non_couvertes[np.ix_(facteur.extension, facteur.intention)] = False
		duree = time.perf_counter() - debut
		if delai_max_s is not None and duree > delai_max_s:
			raise DelaiDepasse(duree, len(facteurs))

		couverture = sum(croix_nouvelles) / total_croix
		if couverture >= couverture_cible:
			break
		if nb_max_facteurs is not None and len(facteurs) >= nb_max_facteurs:
			break

	return construire_resultat(I, facteurs, croix_nouvelles, pas_trace)


def _construire_un_facteur(
	I: np.ndarray,
	non_couvertes: np.ndarray,
	numero_facteur: int,
	pas_trace: list[PasTrace] | None,
) -> tuple[Concept, int]:
	"""Pas 1 à 3 : agrandit l'intention D tant que cela couvre plus de croix non couvertes."""
	nb_attributs = I.shape[1]

	# Pas 1 : intention vide, V = 0
	D = np.zeros(nb_attributs, dtype=bool)
	V = 0
	numero_pas = 1

	while True:
		# Pas 2 : valeur de chaque attribut j absent de D
		valeurs: list[int | None] = []
		meilleur_j = None
		meilleure_valeur = 0
		meilleure_intention = D
		for j in range(nb_attributs):
			if D[j]:
				valeurs.append(None)
				continue
			candidat = D.copy()
			candidat[j] = True
			ext_j = extension(I, candidat)
			int_j = intention(I, ext_j)
			valeur = int(non_couvertes[np.ix_(ext_j, int_j)].sum())
			valeurs.append(valeur)
			# Pas 5 : « > » strict, donc à égalité on garde le premier attribut dans l'ordre des colonnes
			if valeur > meilleure_valeur:
				meilleur_j, meilleure_valeur, meilleure_intention = j, valeur, int_j

		# Pas 3 : on n'avance que si le meilleur compte dépasse V
		ameliore = meilleure_valeur > V
		if pas_trace is not None:
			pas_trace.append(
				PasTrace(numero_facteur, numero_pas, D, valeurs, meilleur_j if ameliore else None)
			)
		if not ameliore:
			break
		D, V = meilleure_intention, meilleure_valeur
		numero_pas += 1

	# Pas 4 : le facteur est (D↓, D)
	return Concept(extension(I, D), D), V


def choix_sur_egalite(I: np.ndarray, trace: list[PasTrace]) -> list[ChoixEgalite]:
	"""Pas qui ont retenu un attribut alors que d'autres avaient la même valeur.

	Les fermetures sont recalculées après coup depuis l'intention de départ du pas.
	"""
	choix = []
	for pas in trace:
		if pas.attribut_retenu is None:
			continue
		meilleure_valeur = pas.valeurs[pas.attribut_retenu]
		a_egalite = [j for j, valeur in enumerate(pas.valeurs) if valeur == meilleure_valeur]
		if len(a_egalite) < 2:
			continue
		fermetures = set()
		for j in a_egalite:
			candidat = pas.intention_depart.copy()
			candidat[j] = True
			fermetures.add(tuple(intention(I, extension(I, candidat))))
		choix.append(ChoixEgalite(
			pas.numero_facteur, pas.numero_pas, a_egalite, meilleure_valeur, len(fermetures) > 1
		))
	return choix
