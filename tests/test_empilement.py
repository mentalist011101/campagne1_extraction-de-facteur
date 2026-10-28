"""Tests de l'étape 16 : matrice des facteurs et stacking."""

import numpy as np

from donnees.exemple import charger_exemple_10_personnes
from m2_facteurs.fca import Concept, extension
from m2_facteurs.grecond import grecond
from m3_classifieurs.empilement import empiler, matrice_facteurs


def test_colonnes_egales_aux_extensions_sur_l_exemple():
	# Un facteur est un concept : les objets qui ont toute l'intention forment exactement l'extension
	I, objets, attributs, classes = charger_exemple_10_personnes()
	facteurs = grecond(I).facteurs
	A = matrice_facteurs(I, facteurs)
	assert A.shape == (10, len(facteurs))
	for f, facteur in enumerate(facteurs):
		assert np.array_equal(A[:, f], facteur.extension)


def test_nouveaux_objets_et_ordre_des_colonnes():
	facteurs = [
		Concept(np.array([True, False]), np.array([True, True, False])),   # intention {a1, a2}
		Concept(np.array([True, True]), np.array([False, False, True])),   # intention {a3}
	]
	I_eval = np.array([
		[1, 1, 1],   # tout : couvert par les deux
		[1, 0, 1],   # manque a2 : pas le facteur 1
		[0, 0, 0],   # rien
	], dtype=bool)
	A = matrice_facteurs(I_eval, facteurs)
	assert A.shape == (3, 2)
	assert A.astype(int).tolist() == [[1, 1], [0, 1], [0, 0]]
	for f, facteur in enumerate(facteurs):
		assert np.array_equal(A[:, f], extension(I_eval, facteur.intention))


def test_empiler_dimensions():
	I, objets, attributs, classes = charger_exemple_10_personnes()
	facteurs = grecond(I).facteurs
	y = (classes == "D").astype(int)
	y_pred, modele = empiler(facteurs, I, y, I[:4])
	assert y_pred.shape == (4,)
	assert modele.coef_.shape == (1, len(facteurs))
