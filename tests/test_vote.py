"""Tests du jeu congressional voting (phase 3) : données, découpages et plis figés."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from donnees.vote import (
	charger_decoupages_vote,
	charger_plis_cv10_vote,
	charger_vote,
	creer_decoupages_vote,
	creer_plis_cv10_vote,
)


def test_dimensions_et_classes():
	X, y = charger_vote()
	assert X.shape == (435, 16)
	assert (y == 0).sum() == 267   # démocrates
	assert (y == 1).sum() == 168   # républicains
	assert set(np.unique(X.to_numpy())) == {"y", "n", "?"}


def test_decoupages():
	d = charger_decoupages_vote()
	assert [len(d[n]) for n in ["apprentissage", "validation", "test"]] == [261, 87, 87]
	app, val, test = set(d["apprentissage"]), set(d["validation"]), set(d["test"])
	assert not (app & val or app & test or val & test)
	assert app | val | test == set(range(435))


def test_plis_complets_et_test_scelle():
	d = charger_decoupages_vote()
	tous = set(d["apprentissage"]) | set(d["validation"])
	test = set(d["test"])
	plis = charger_plis_cv10_vote()
	assert len(plis) == 10
	assert set(np.concatenate([ev for app, ev in plis])) == tous
	for app, ev in plis:
		assert len(ev) in (34, 35)
		assert set(app) | set(ev) == tous
		assert not (set(app) & set(ev) or set(app) & test or set(ev) & test)


def test_plis_stratifies_et_reproductibles():
	X, y = charger_vote()
	d = charger_decoupages_vote()
	tous = np.concatenate([d["apprentissage"], d["validation"]])
	proportion = y.iloc[tous].mean()
	decoupeur = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	for (app, ev), (pos_app, pos_ev) in zip(charger_plis_cv10_vote(), decoupeur.split(tous, y.iloc[tous])):
		assert set(ev) == set(tous[pos_ev])
		assert abs(y.iloc[ev].sum() - proportion * len(ev)) <= 1


def test_jamais_ecrases():
	with pytest.raises(FileExistsError):
		creer_decoupages_vote()
	with pytest.raises(FileExistsError):
		creer_plis_cv10_vote()
