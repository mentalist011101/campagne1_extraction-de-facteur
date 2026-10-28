"""Tests de l'étape 17 : bagging du M3 de référence (best-vote + précision)."""

import numpy as np

from donnees.exemple import charger_exemple_10_personnes
from m3_classifieurs.ensachage import agreger, bagging_m3, tirer_sac


def test_sac_reproductible_avec_remise_et_de_bonne_taille():
	sac = tirer_sac(100, numero_sac=0)
	assert np.array_equal(sac, tirer_sac(100, numero_sac=0))
	assert len(sac) == 100
	assert sac.min() >= 0 and sac.max() < 100
	assert len(np.unique(sac)) < 100          # avec remise : des doublons
	assert not np.array_equal(sac, tirer_sac(100, numero_sac=1))
	assert np.array_equal(sac, np.random.default_rng(42).integers(0, 100, 100))


def test_agreger_vote_majoritaire_et_proportion():
	predictions_par_sac = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]])
	predictions, proportion = agreger(predictions_par_sac, classe_par_defaut=0)
	assert predictions.tolist() == [1, 0, 1]
	assert np.allclose(proportion, [2 / 3, 1 / 3, 2 / 3])


def test_agreger_egalite_classe_par_defaut():
	# B = 2 : premier x à égalité (1 contre 1) → classe par défaut
	predictions_par_sac = np.array([[1, 0], [0, 0]])
	predictions, proportion = agreger(predictions_par_sac, classe_par_defaut=1)
	assert predictions.tolist() == [1, 0]
	assert proportion.tolist() == [0.5, 0.0]


def test_bagging_reproductible_et_dimensions():
	I, objets, attributs, classes = charger_exemple_10_personnes()
	y = (classes == "D").astype(int)
	premier = bagging_m3(I, y, I[:4], classe_par_defaut=0, nb_sacs=3)
	second = bagging_m3(I, y, I[:4], classe_par_defaut=0, nb_sacs=3)
	assert np.array_equal(premier.predictions, second.predictions)
	assert np.array_equal(premier.proportion_classe_1, second.proportion_classe_1)
	assert premier.predictions.shape == (4,)
	assert premier.proportion_classe_1.shape == (4,)
	assert len(premier.non_couverts_par_sac) == 3
	assert len(premier.temps_par_sac) == 3
