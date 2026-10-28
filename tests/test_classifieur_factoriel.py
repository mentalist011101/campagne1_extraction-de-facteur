"""Tests de l'étape 7 : M3, facteurs → classifieurs (Makhalova & Trnecka, 2019)."""

import numpy as np
import pytest

from donnees.exemple import charger_exemple_10_personnes
from m2_facteurs.fca import Concept
from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import (
	RAISON_EGALITE,
	RAISON_NON_COUVERT,
	Regle,
	compter,
	construire_regles,
	decrire_facteurs,
	predire,
	regles_qui_votent,
)


@pytest.fixture
def exemple():
	I, objets, attributs, classes = charger_exemple_10_personnes()
	facteurs = grecond(I).facteurs
	return I, objets, attributs, classes, facteurs


def regle_par_numero(regles, numero):
	return next(r for r in regles if r.numero_facteur == numero)


# --- Valeurs attendues des instructions et des décisions (bibliographie) ---

def test_f3_normal_classe_N_poids_precision_075(exemple):
	I, objets, attributs, classes, facteurs = exemple
	regles, ecartes = construire_regles(facteurs, classes, I, classes, mesure="precision")
	f3 = regle_par_numero(regles, 3)
	assert f3.classe == "N"
	assert f3.poids == pytest.approx(0.75)


def test_f3_normal_poids_accuracy_070(exemple):
	I, objets, attributs, classes, facteurs = exemple
	regles, ecartes = construire_regles(facteurs, classes, I, classes, mesure="accuracy")
	assert regle_par_numero(regles, 3).poids == pytest.approx(0.70)


def test_f1_adult_male_ecarte_pour_egalite(exemple):
	I, objets, attributs, classes, facteurs = exemple
	regles, ecartes = construire_regles(facteurs, classes, I, classes, mesure="precision")
	assert 1 not in [r.numero_facteur for r in regles]
	assert ecartes[1] == RAISON_EGALITE


# --- Valeurs calculées à la main (hors bibliographie) ---

def test_f2_elderly_female_classe_D_poids_deux_tiers(exemple):
	# Extension P3 (D), P9 (D), P10 (N)
	I, objets, attributs, classes, facteurs = exemple
	regles, ecartes = construire_regles(facteurs, classes, I, classes, mesure="precision")
	f2 = regle_par_numero(regles, 2)
	assert f2.classe == "D"
	assert f2.poids == pytest.approx(2 / 3)


def test_prediction_de_P1_all_votes_et_best_vote(exemple):
	# P1 = {Adult, Male, Normal} : couvert par F1 (écarté), F3 {Normal} → N 0,75 et F8 {Male} → N 3/5
	I, objets, attributs, classes, facteurs = exemple
	regles, ecartes = construire_regles(facteurs, classes, I, classes, mesure="precision")
	p1 = I[[0]]

	tous = predire(regles, p1, classes=["D", "N"], classe_par_defaut="N", vote="all-votes")[0]
	assert tous.classe == "N"
	assert tous.scores["N"] == pytest.approx(1.35)
	assert tous.scores["D"] == 0
	assert [r.numero_facteur for r in tous.regles_couvrantes] == [3, 8]
	assert not tous.non_couvert and not tous.egalite

	meilleur = predire(regles, p1, classes=["D", "N"], classe_par_defaut="N", vote="best-vote")[0]
	assert meilleur.classe == "N"
	assert meilleur.scores["N"] == pytest.approx(0.75)


def test_poids_sur_validation_et_regle_sans_objet_couvert(exemple):
	# « Validation » artificielle : P1 à P5. F3 {Normal} y couvre P1 (N), P2 (N), P3 (D) → 2/3.
	# F4 {Elderly, Underweight} = {P6, P10} n'y couvre personne → écarté.
	I, objets, attributs, classes, facteurs = exemple
	I_val, y_val = I[:5], classes[:5]
	regles, ecartes = construire_regles(facteurs, classes, I_val, y_val, mesure="precision")
	f3 = regle_par_numero(regles, 3)
	assert f3.classe == "N"
	assert f3.poids == pytest.approx(2 / 3)
	assert ecartes[4] == RAISON_NON_COUVERT
	assert ecartes[1] == RAISON_EGALITE

	regles_acc, ecartes_acc = construire_regles(facteurs, classes, I_val, y_val, mesure="accuracy")
	assert ecartes_acc[4] == RAISON_NON_COUVERT


def test_parametre_k(exemple):
	I, objets, attributs, classes, facteurs = exemple
	regles, ecartes = construire_regles(facteurs, classes, I, classes, mesure="precision", k=3)
	assert sorted([r.numero_facteur for r in regles] + list(ecartes)) == [1, 2, 3]


# --- Petits cas construits ---

def regle(numero, attributs_masque, classe, poids):
	return Regle(numero, np.array(attributs_masque, dtype=bool), classe, poids)


def test_all_votes_et_best_vote_different():
	regles = [
		regle(1, [1, 0, 0], "A", 0.6),
		regle(2, [0, 1, 0], "A", 0.5),
		regle(3, [0, 0, 1], "B", 0.9),
	]
	x = np.array([[1, 1, 1]], dtype=bool)
	assert predire(regles, x, ["A", "B"], "A", vote="all-votes")[0].classe == "A"   # 1,1 contre 0,9
	assert predire(regles, x, ["A", "B"], "A", vote="best-vote")[0].classe == "B"   # 0,6 contre 0,9


def test_non_couvert_recoit_la_classe_par_defaut():
	regles = [regle(1, [1, 0], "B", 0.8)]
	x = np.array([[0, 1]], dtype=bool)
	prediction = predire(regles, x, ["A", "B"], "A")[0]
	assert prediction.non_couvert
	assert not prediction.egalite
	assert prediction.classe == "A"
	assert prediction.regles_couvrantes == []


def test_egalite_de_scores_recoit_la_classe_par_defaut():
	regles = [regle(1, [1, 0], "A", 0.5), regle(2, [0, 1], "B", 0.5)]
	x = np.array([[1, 1]], dtype=bool)
	prediction = predire(regles, x, ["A", "B"], "B")[0]
	assert prediction.egalite
	assert not prediction.non_couvert
	assert prediction.classe == "B"


def test_couvert_seulement_par_un_facteur_ecarte_compte_comme_non_couvert():
	I = np.array([[1, 0], [1, 0], [0, 1]], dtype=bool)
	y = np.array(["A", "B", "A"])
	facteurs = [
		Concept(np.array([True, True, False]), np.array([True, False])),   # 1 A, 1 B → écarté
		Concept(np.array([False, False, True]), np.array([False, True])),  # A
	]
	regles, ecartes = construire_regles(facteurs, y, I, y)
	assert ecartes == {1: RAISON_EGALITE}
	prediction = predire(regles, I[[0]], ["A", "B"], "A")[0]
	assert prediction.non_couvert


def test_compter():
	regles = [regle(1, [1, 0], "A", 0.5), regle(2, [0, 1], "B", 0.5)]
	X = np.array([[1, 0], [1, 1], [0, 0], [0, 0]], dtype=bool)
	comptes = compter(predire(regles, X, ["A", "B"], "A"))
	assert comptes["nb_objets"] == 4
	assert comptes["nb_non_couverts"] == 2
	assert comptes["taux_non_couverts"] == pytest.approx(0.5)
	assert comptes["nb_egalites"] == 1
	assert comptes["taux_egalites"] == pytest.approx(0.25)


def test_decrire_facteurs(exemple):
	I, objets, attributs, classes, facteurs = exemple
	table = decrire_facteurs(facteurs, attributs, classes, I[:5], classes[:5])
	assert len(table) == 8
	f1, f3, f4 = table.loc["F1"], table.loc["F3"], table.loc["F4"]
	assert f1["ecarte"] and f1["raison"] == RAISON_EGALITE
	assert f3["intention"] == "Normal"
	assert f3["taille_extension"] == 4
	assert f3["classe"] == "N"
	assert f3["couverts_validation"] == 3
	assert f3["precision_apprentissage"] == pytest.approx(0.75)
	assert f3["accuracy_apprentissage"] == pytest.approx(0.70)
	assert f3["precision_validation"] == pytest.approx(2 / 3)
	assert not f3["ecarte"]
	assert f4["ecarte"] and f4["raison"] == RAISON_NON_COUVERT


# --- Étape 13 : lift et règles qui votent effectivement ---

def test_f3_normal_lift_15(exemple):
	# Précision 0,75 ; P(N) = 5/10 → lift 1,5
	I, objets, attributs, classes, facteurs = exemple
	regles, ecartes = construire_regles(facteurs, classes, I, classes, mesure="lift")
	assert regle_par_numero(regles, 3).poids == pytest.approx(1.5)


def test_lift_egal_precision_sur_proportion_de_classe(exemple):
	I, objets, attributs, classes, facteurs = exemple
	precisions, _ = construire_regles(facteurs, classes, I, classes, mesure="precision")
	lifts, _ = construire_regles(facteurs, classes, I, classes, mesure="lift")
	for p, l in zip(precisions, lifts):
		assert l.classe == p.classe
		assert l.poids == pytest.approx(p.poids / np.mean(np.asarray(classes) == p.classe))


def test_regles_qui_votent_all_votes_toutes_les_couvrantes():
	regles = [regle(1, [1, 0, 0], "A", 0.6), regle(2, [0, 1, 0], "B", 0.9), regle(3, [0, 0, 1], "A", 0.5)]
	x = np.array([[1, 1, 1]], dtype=bool)
	prediction = predire(regles, x, ["A", "B"], "A", vote="all-votes")[0]
	assert [r.numero_facteur for r in regles_qui_votent(prediction, "all-votes")] == [1, 2, 3]


def test_regles_qui_votent_best_vote_la_regle_decisive():
	regles = [regle(1, [1, 0, 0], "A", 0.6), regle(2, [0, 1, 0], "B", 0.9), regle(3, [0, 0, 1], "B", 0.9)]
	x = np.array([[1, 1, 1]], dtype=bool)
	prediction = predire(regles, x, ["A", "B"], "A", vote="best-vote")[0]
	assert prediction.classe == "B"
	# Règles 2 et 3 au même poids maximal : la première dans l'ordre des facteurs
	assert [r.numero_facteur for r in regles_qui_votent(prediction, "best-vote")] == [2]


def test_regles_qui_votent_vide_si_non_couvert_ou_egalite():
	regles = [regle(1, [1, 0], "A", 0.5), regle(2, [0, 1], "B", 0.5)]
	non_couvert = predire(regles, np.array([[0, 0]], dtype=bool), ["A", "B"], "A", vote="best-vote")[0]
	egalite = predire(regles, np.array([[1, 1]], dtype=bool), ["A", "B"], "A", vote="best-vote")[0]
	assert regles_qui_votent(non_couvert, "best-vote") == []
	assert regles_qui_votent(egalite, "best-vote") == []
	assert regles_qui_votent(non_couvert, "all-votes") == []
