"""Tests de l'étape 14 : jeu mushroom, découpages, plis et binarisation catégorielle."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from donnees.mushroom import (
	charger_decoupages_mushroom,
	charger_mushroom,
	charger_plis_cv10_mushroom,
	creer_decoupages_mushroom,
	creer_plis_cv10_mushroom,
)
from m1_binarisation.binarisation import apprendre_modalites, binariser_categoriel


# --- Données (valeurs fixées par l'étudiant) ---

def test_dimensions_et_classes():
	X, y = charger_mushroom()
	assert X.shape == (8124, 22)
	assert (y == 0).sum() == 4208   # comestibles
	assert (y == 1).sum() == 3916   # vénéneux


def test_points_d_interrogation_tous_dans_stalk_root():
	X, y = charger_mushroom()
	par_colonne = (X == "?").sum()
	assert par_colonne["stalk-root"] == 2480
	assert par_colonne.sum() == 2480


# --- Découpages et plis ---

def test_decoupages():
	d = charger_decoupages_mushroom()
	assert [len(d[n]) for n in ["apprentissage", "validation", "test"]] == [4874, 1625, 1625]
	app, val, test = set(d["apprentissage"]), set(d["validation"]), set(d["test"])
	assert not (app & val or app & test or val & test)
	assert app | val | test == set(range(8124))


def test_plis_complets_et_test_scelle():
	d = charger_decoupages_mushroom()
	tous = set(d["apprentissage"]) | set(d["validation"])
	test = set(d["test"])
	plis = charger_plis_cv10_mushroom()
	assert len(plis) == 10
	assert set(np.concatenate([ev for app, ev in plis])) == tous
	for app, ev in plis:
		assert len(ev) in (649, 650)
		assert set(app) | set(ev) == tous
		assert not (set(app) & set(ev) or set(app) & test or set(ev) & test)


def test_plis_reproductibles():
	X, y = charger_mushroom()
	d = charger_decoupages_mushroom()
	tous = np.concatenate([d["apprentissage"], d["validation"]])
	decoupeur = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	for (app, ev), (pos_app, pos_ev) in zip(charger_plis_cv10_mushroom(), decoupeur.split(tous, y.iloc[tous])):
		assert set(ev) == set(tous[pos_ev])


def test_jamais_ecrases():
	with pytest.raises(FileExistsError):
		creer_decoupages_mushroom()
	with pytest.raises(FileExistsError):
		creer_plis_cv10_mushroom()


# --- Binarisation catégorielle ---

def test_binarisation_categorielle_sur_donnees_simples():
	X_app = pd.DataFrame({"a": ["x", "y", "x", "?"], "b": ["k", "k", "k", "k"]})
	modalites = apprendre_modalites(X_app)
	# « ? » n'est pas une modalité ; b = k est présent partout : attribut constant retiré
	assert modalites == {"a": ["x", "y"], "b": []}
	X_eval = pd.DataFrame({"a": ["y", "?", "z"], "b": ["k", "k", "m"]})
	I, noms = binariser_categoriel(X_eval, modalites)
	assert noms == ["a=x", "a=y"]
	assert I.astype(int).tolist() == [
		[0, 1],   # y
		[0, 0],   # ? : aucun attribut
		[0, 0],   # z, absent de l'apprentissage : aucun attribut
	]


def test_au_plus_un_attribut_par_variable_dans_un_pli():
	X, y = charger_mushroom()
	app, ev = charger_plis_cv10_mushroom()[0]
	modalites = apprendre_modalites(X.iloc[app])
	assert modalites["veil-type"] == []
	I_app, noms = binariser_categoriel(X.iloc[app], modalites)
	debut = 0
	for colonne in X.columns:
		nb = len(modalites[colonne])
		par_variable = I_app[:, debut : debut + nb].sum(axis=1)
		assert par_variable.max(initial=0) <= 1
		if nb and colonne != "stalk-root":
			# Toute valeur de l'apprentissage est une modalité (sauf constante) : exactement un attribut
			assert (par_variable == 1).all(), colonne
		debut += nb
	assert not I_app.all(axis=0).any()   # aucun attribut constant
