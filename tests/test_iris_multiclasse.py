"""Campagne 1, étape 3 : M3, bagging et stacking à plusieurs classes sur iris (pli 0)."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from donnees.catalogue import charger_jeu, lire_variables
from m1_binarisation.binarisation import apprendre_binarisation, binariser_jeu
from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import classes_par_frequence, construire_regles, predire
from m3_classifieurs.empilement import empiler
from m3_classifieurs.ensachage import bagging_m3


@pytest.fixture(scope="module")
def iris_pli_0():
	X, y = charger_jeu("iris")
	types = lire_variables("iris").set_index("variable")["type"].to_dict()
	app, ev = next(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(X, y))
	binarisation = apprendre_binarisation(X.iloc[app], types)
	I_app = binariser_jeu(X.iloc[app], binarisation)
	I_eval = binariser_jeu(X.iloc[ev], binarisation)
	y_app = y.iloc[app].to_numpy()
	return I_app, y_app, I_eval, classes_par_frequence(y_app), grecond(I_app).facteurs


def test_iris_les_3_classes_dans_les_predictions_de_m3_all_votes(iris_pli_0):
	I_app, y_app, I_eval, classes, facteurs = iris_pli_0
	regles, _ = construire_regles(facteurs, y_app, I_app, y_app, mesure="precision")
	predictions = predire(regles, I_eval, classes, classes[0], vote="all-votes")
	assert {p.classe for p in predictions} == {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}


def test_iris_best_vote_bagging_stacking_predisent_des_classes_de_l_apprentissage(iris_pli_0):
	I_app, y_app, I_eval, classes, facteurs = iris_pli_0
	regles, _ = construire_regles(facteurs, y_app, I_app, y_app, mesure="precision")
	best_vote = [p.classe for p in predire(regles, I_eval, classes, classes[0], vote="best-vote")]
	bagging = bagging_m3(I_app, y_app, I_eval, classes, classes[0], nb_sacs=5)
	stacking, _ = empiler(facteurs, I_app, y_app, I_eval)
	for predictions in [best_vote, bagging.predictions, stacking]:
		assert len(predictions) == 15
		assert set(predictions) <= set(classes)
	assert bagging.proportions.shape == (15, 3)
