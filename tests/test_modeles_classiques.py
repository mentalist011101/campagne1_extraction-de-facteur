"""Tests de l'étape 8 : modèles classiques de comparaison.

Le MLP prend plusieurs secondes : chaque série de modèles est entraînée une seule
fois par module (fixture), puis une seconde fois pour le test de reproductibilité.
"""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from donnees.pima import charger_decoupages, charger_pima
from experiences.contextes_pima import contextes_pima
from modeles_ml.modeles_classiques import NOMS_MODELES, creer_modeles, entrainer_et_predire


@pytest.fixture(scope="module")
def jeux():
	X, y = charger_pima()
	d = charger_decoupages()
	contextes, noms, classes, seuils = contextes_pima()
	return {
		"brutes": (X.iloc[d["apprentissage"]], y.iloc[d["apprentissage"]], X.iloc[d["validation"]]),
		"binarisees": (contextes["apprentissage"], classes["apprentissage"], contextes["validation"]),
	}


@pytest.fixture(scope="module")
def entrainements(jeux):
	"""Pour chaque type de données : modèles entraînés, prédictions, avertissements émis."""
	resultats = {}
	for donnees, (X_app, y_app, X_val) in jeux.items():
		modeles = creer_modeles(donnees)
		with warnings.catch_warnings(record=True) as avertissements:
			warnings.simplefilter("always")
			predictions = entrainer_et_predire(modeles, X_app, y_app, X_val)
		resultats[donnees] = (modeles, predictions, list(avertissements))
	return resultats


@pytest.mark.parametrize("donnees", ["brutes", "binarisees"])
def test_cinq_modeles(donnees):
	assert list(creer_modeles(donnees)) == NOMS_MODELES
	assert len(NOMS_MODELES) == 5


@pytest.mark.parametrize("donnees", ["brutes", "binarisees"])
def test_predictions_de_la_bonne_taille(donnees, entrainements):
	modeles, predictions, avertissements = entrainements[donnees]
	for nom in NOMS_MODELES:
		assert predictions[nom].shape == (154,)
		assert set(np.unique(predictions[nom])) <= {0, 1}


@pytest.mark.parametrize("donnees", ["brutes", "binarisees"])
def test_aucun_avertissement_de_convergence(donnees, entrainements):
	modeles, predictions, avertissements = entrainements[donnees]
	assert not [a for a in avertissements if issubclass(a.category, ConvergenceWarning)]


@pytest.mark.parametrize("donnees", ["brutes", "binarisees"])
def test_reproductibilite(donnees, jeux, entrainements):
	X_app, y_app, X_val = jeux[donnees]
	modeles, premier, avertissements = entrainements[donnees]
	second = entrainer_et_predire(creer_modeles(donnees), X_app, y_app, X_val)
	for nom in NOMS_MODELES:
		assert np.array_equal(premier[nom], second[nom]), nom


def test_standardisation_apprise_sur_l_apprentissage_seul(jeux, entrainements):
	X_app, y_app, X_val = jeux["brutes"]
	modeles, predictions, avertissements = entrainements["brutes"]
	X_tout, y_tout = charger_pima()
	for nom, pipeline in modeles.items():
		moyennes = pipeline.named_steps["standardisation"].mean_
		assert np.allclose(moyennes, X_app.mean().to_numpy()), nom
		assert not np.allclose(moyennes, X_tout.mean().to_numpy()), nom


def test_pas_de_standardisation_sur_donnees_binarisees():
	for pipeline in creer_modeles("binarisees").values():
		assert "standardisation" not in pipeline.named_steps
