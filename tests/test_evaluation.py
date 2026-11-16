"""Tests de l'étape 9 : mesures d'évaluation, sur de petits cas calculés à la main."""

import math

import numpy as np
import pytest

from experiences.evaluation import accuracy_decides, metriques, taux_accord
from m3_classifieurs.classifieur_factoriel import PredictionFactorielle


def test_metriques():
	# Classe 0 : précision 1, rappel 1/2 → F1 2/3 ; classe 1 : précision 2/3, rappel 1 → F1 4/5
	resultat = metriques(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
	assert resultat["accuracy"] == pytest.approx(0.75)
	assert resultat["f1_macro"] == pytest.approx((2 / 3 + 4 / 5) / 2)
	assert resultat["predits_classe_1"] == 3


def test_taux_accord():
	assert taux_accord(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])) == pytest.approx(0.75)


def prediction(classe, non_couvert=False, egalite=False):
	return PredictionFactorielle(classe, {}, [], non_couvert, egalite)


def test_accuracy_decides_ignore_non_couverts_et_egalites():
	predictions = [
		prediction(1),                       # décidé, juste
		prediction(0),                       # décidé, faux
		prediction(0, non_couvert=True),     # ignoré
		prediction(0, egalite=True),         # ignoré
	]
	y = np.array([1, 1, 0, 0])
	assert accuracy_decides(y, predictions) == pytest.approx(0.5)


def test_accuracy_decides_sans_aucun_x_decide():
	predictions = [prediction(0, non_couvert=True)]
	assert math.isnan(accuracy_decides(np.array([0]), predictions))


def test_f1_macro_sur_la_reunion_des_classes_reelles_et_predites():
	# Classe 2 présente mais jamais prédite : F1 = 0 ; classe 0 : F1 = 1 ; classe 1 : précision 2/3, rappel 1 → 4/5
	resultat = metriques(np.array([0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 1]))
	assert resultat["f1_macro"] == pytest.approx((1 + 4 / 5 + 0) / 3)
	# Une classe absente du pli évalué et des prédictions n'entre pas dans la moyenne
	assert metriques(np.array([0, 0, 1]), np.array([0, 0, 1]))["f1_macro"] == pytest.approx(1.0)
