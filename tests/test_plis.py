"""Tests de l'étape 10 : plis de validation croisée et contextes réappris par pli."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from donnees.pima import (
	COLONNES_ZERO_MANQUANT,
	charger_decoupages,
	charger_pima,
	charger_plis_cv10,
	creer_plis_cv10,
)
from experiences.contextes_pima import contextes_pli


@pytest.fixture(scope="module")
def plis():
	return charger_plis_cv10()


def indices_app_val():
	d = charger_decoupages()
	return np.concatenate([d["apprentissage"], d["validation"]])


def test_dix_plis_parties_evaluees_disjointes_et_completes(plis):
	assert len(plis) == 10
	evalues = np.concatenate([evaluation for apprentissage, evaluation in plis])
	assert len(evalues) == 614
	assert set(evalues) == set(indices_app_val())


def test_apprentissage_et_evaluation_disjoints_et_complets(plis):
	tous = set(indices_app_val())
	for apprentissage, evaluation in plis:
		assert not set(apprentissage) & set(evaluation)
		assert set(apprentissage) | set(evaluation) == tous


def test_le_test_reste_scelle(plis):
	test = set(charger_decoupages()["test"])
	for apprentissage, evaluation in plis:
		assert not set(apprentissage) & test
		assert not set(evaluation) & test


def test_taille_et_stratification(plis):
	X, y = charger_pima()
	proportion = y.iloc[indices_app_val()].mean()
	for apprentissage, evaluation in plis:
		assert len(evaluation) in (61, 62)
		positifs = int(y.iloc[evaluation].sum())
		assert abs(positifs - proportion * len(evaluation)) <= 1


def test_plis_reproductibles(plis):
	X, y = charger_pima()
	tous = indices_app_val()
	decoupeur = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	for (app, ev), (pos_app, pos_ev) in zip(plis, decoupeur.split(tous, y.iloc[tous])):
		assert set(app) == set(tous[pos_app])
		assert set(ev) == set(tous[pos_ev])


def test_plis_jamais_ecrases():
	with pytest.raises(FileExistsError):
		creer_plis_cv10()


def test_contextes_pli_seuils_appris_sur_la_partie_apprentissage(plis):
	X, y = charger_pima()
	apprentissage, evaluation = plis[0]
	I_app, I_eval, noms, y_app, y_eval, seuils = contextes_pli(apprentissage, evaluation)
	insuline = X["insuline"].iloc[apprentissage]
	connues = insuline[insuline != 0]
	assert seuils["insuline"] == (connues.quantile(1 / 3), connues.quantile(2 / 3))
	assert I_app.shape == (len(apprentissage), 24)
	assert I_eval.shape == (len(evaluation), 24)
	assert np.array_equal(y_app.to_numpy(), y.iloc[apprentissage].to_numpy())


def test_contextes_pli_garde_l_option_b(plis):
	X, y = charger_pima()
	apprentissage, evaluation = plis[0]
	I_app, I_eval, noms, y_app, y_eval, seuils = contextes_pli(apprentissage, evaluation)
	X_eval = X.iloc[evaluation]
	for numero, colonne in enumerate(X.columns):
		par_variable = I_eval[:, 3 * numero : 3 * numero + 3].sum(axis=1)
		if colonne in COLONNES_ZERO_MANQUANT:
			assert np.array_equal(par_variable, (X_eval[colonne] != 0).astype(int).to_numpy())
