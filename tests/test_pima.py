"""Tests de l'étape 4 : jeu Pima Indians Diabetes et découpages figés."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from donnees.pima import charger_decoupages, charger_pima, creer_decoupages

COLONNES_ATTENDUES = [
	"grossesses", "glucose", "pression", "pli_cutane", "insuline", "imc", "pedigree", "age",
]


def test_dimensions_et_colonnes():
	X, y = charger_pima()
	assert X.shape == (768, 8)
	assert X.columns.tolist() == COLONNES_ATTENDUES


def test_classe_codee_en_0_1_avec_500_negatifs_et_268_positifs():
	X, y = charger_pima()
	assert set(y.unique()) == {0, 1}
	assert (y == 0).sum() == 500
	assert (y == 1).sum() == 268


def test_zeros_par_variable():
	"""Comptes connus des valeurs manquantes codées 0."""
	X, y = charger_pima()
	zeros = (X == 0).sum()
	assert zeros["glucose"] == 5
	assert zeros["pression"] == 35
	assert zeros["pli_cutane"] == 227
	assert zeros["insuline"] == 374
	assert zeros["imc"] == 11


def test_tailles_des_decoupages():
	decoupages = charger_decoupages()
	assert len(decoupages["apprentissage"]) == 460
	assert len(decoupages["validation"]) == 154
	assert len(decoupages["test"]) == 154


def test_decoupages_disjoints_et_complets():
	d = charger_decoupages()
	app, val, test = set(d["apprentissage"]), set(d["validation"]), set(d["test"])
	assert not app & val
	assert not app & test
	assert not val & test
	assert app | val | test == set(range(768))


def test_stratification():
	X, y = charger_pima()
	proportion_globale = 268 / 768
	for nom, indices in charger_decoupages().items():
		positifs = int(y.iloc[indices].sum())
		# À un objet près de la proportion globale
		assert abs(positifs - proportion_globale * len(indices)) <= 1, nom


def test_decoupages_reproductibles():
	"""Refaire le découpage avec random_state=42 redonne les indices sauvegardés."""
	X, y = charger_pima()
	tous = np.arange(len(y))
	app, reste = train_test_split(tous, test_size=0.4, stratify=y, random_state=42)
	val, test = train_test_split(reste, test_size=0.5, stratify=y.iloc[reste], random_state=42)
	d = charger_decoupages()
	assert np.array_equal(d["apprentissage"], app)
	assert np.array_equal(d["validation"], val)
	assert np.array_equal(d["test"], test)


def test_decoupages_jamais_ecrases():
	with pytest.raises(FileExistsError):
		creer_decoupages()
