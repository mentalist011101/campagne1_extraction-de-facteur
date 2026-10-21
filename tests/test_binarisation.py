"""Tests de l'étape 5 : M1, binarisation par terciles appris sur l'apprentissage."""

import numpy as np
import pandas as pd
import pytest

from donnees.pima import COLONNES_ZERO_MANQUANT, charger_decoupages, charger_pima
from m1_binarisation.binarisation import apprendre_seuils, binariser


def test_seuils_et_noms_sur_donnees_simples():
	X = pd.DataFrame({"v": [0, 1, 2, 3, 4, 5, 6]})
	seuils = apprendre_seuils(X, colonnes_zero_manquant=[])
	assert seuils["v"] == (2.0, 4.0)
	I, noms = binariser(X, seuils, colonnes_zero_manquant=[])
	assert noms == ["v<=2", "2<v<=4", "v>4"]


def test_bornes_des_intervalles():
	X = pd.DataFrame({"v": [0, 1, 2, 3, 4, 5, 6]})
	seuils = apprendre_seuils(X, colonnes_zero_manquant=[])
	nouveaux = pd.DataFrame({"v": [2, 3, 4, 5]})
	I, noms = binariser(nouveaux, seuils, colonnes_zero_manquant=[])
	assert I.tolist() == [
		[True, False, False],   # 2 → bas
		[False, True, False],   # 3 → moyen
		[False, True, False],   # 4 → moyen
		[False, False, True],   # 5 → haut
	]


def test_zero_manquant_exclu_des_seuils_et_sans_attribut():
	# Sans le zéro : 1..7, quantiles 1/3 et 2/3 = 3 et 5
	X = pd.DataFrame({"w": [0, 1, 2, 3, 4, 5, 6, 7]})
	seuils = apprendre_seuils(X, colonnes_zero_manquant=["w"])
	assert seuils["w"] == (3.0, 5.0)
	I, noms = binariser(X, seuils, colonnes_zero_manquant=["w"])
	assert not I[0].any()
	assert (I[1:].sum(axis=1) == 1).all()


@pytest.fixture
def pima_binarise():
	X, y = charger_pima()
	d = charger_decoupages()
	X_app = X.iloc[d["apprentissage"]]
	seuils = apprendre_seuils(X_app, COLONNES_ZERO_MANQUANT)
	contextes = {
		nom: binariser(X.iloc[indices], seuils, COLONNES_ZERO_MANQUANT)
		for nom, indices in d.items()
	}
	return X, d, seuils, contextes


def test_seuils_appris_sur_l_apprentissage_seulement(pima_binarise):
	X, d, seuils, contextes = pima_binarise
	insuline_app = X["insuline"].iloc[d["apprentissage"]]
	non_nulles = insuline_app[insuline_app != 0]
	assert seuils["insuline"] == (non_nulles.quantile(1 / 3), non_nulles.quantile(2 / 3))
	age_app = X["age"].iloc[d["apprentissage"]]
	assert seuils["age"] == (age_app.quantile(1 / 3), age_app.quantile(2 / 3))


def test_formes_des_trois_contextes(pima_binarise):
	X, d, seuils, contextes = pima_binarise
	assert contextes["apprentissage"][0].shape == (460, 24)
	assert contextes["validation"][0].shape == (154, 24)
	assert contextes["test"][0].shape == (154, 24)
	noms_app = contextes["apprentissage"][1]
	assert contextes["validation"][1] == noms_app
	assert contextes["test"][1] == noms_app


def test_un_attribut_par_variable_sauf_valeur_manquante(pima_binarise):
	X, d, seuils, contextes = pima_binarise
	for nom, (I, noms) in contextes.items():
		X_part = X.iloc[d[nom]]
		for numero, colonne in enumerate(X.columns):
			par_variable = I[:, 3 * numero : 3 * numero + 3].sum(axis=1)
			if colonne in COLONNES_ZERO_MANQUANT:
				attendu = (X_part[colonne] != 0).astype(int).to_numpy()
			else:
				attendu = np.ones(len(X_part), dtype=int)
			assert np.array_equal(par_variable, attendu), (nom, colonne)


def test_une_seule_ligne_donne_la_meme_binarisation(pima_binarise):
	"""Aucun recalcul : binariser un x seul ou dans le tableau donne la même ligne."""
	X, d, seuils, contextes = pima_binarise
	I_test, noms = contextes["test"]
	for position in [0, 50, 153]:
		ligne = X.iloc[[d["test"][position]]]
		I_ligne, _ = binariser(ligne, seuils, COLONNES_ZERO_MANQUANT)
		assert np.array_equal(I_ligne[0], I_test[position])
