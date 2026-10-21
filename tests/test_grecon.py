"""Tests de l'étape 2 : GreCon, sur l'exemple des 10 personnes."""

import numpy as np
import pytest

from donnees.exemple import charger_exemple_10_personnes
from m2_facteurs.factorisation import nb_facteurs_pour_couverture, produit_booleen
from m2_facteurs.fca import est_concept
from m2_facteurs.grecon import grecon


@pytest.fixture
def exemple():
	I, objets, attributs, classes = charger_exemple_10_personnes()
	return I, objets, attributs


def noms(masque_bool, tous_les_noms):
	"""Retourne l'ensemble des noms sélectionnés par un masque."""
	return {nom for nom, garde in zip(tous_les_noms, masque_bool) if garde}


def test_quatre_premiers_facteurs(exemple):
	I, objets, attributs = exemple
	resultat = grecon(I)
	attendus = [
		({"P1", "P4", "P5", "P8"}, {"Adult", "Male"}),
		({"P3", "P9", "P10"}, {"Elderly", "Female"}),
		({"P1", "P2", "P3", "P8"}, {"Normal"}),
		({"P2", "P7"}, {"Adult", "Female"}),
	]
	for facteur, (ext_attendue, int_attendue) in zip(resultat.facteurs, attendus):
		assert noms(facteur.extension, objets) == ext_attendue
		assert noms(facteur.intention, attributs) == int_attendue
	assert resultat.croix_nouvelles[:4] == [8, 6, 4, 4]
	assert resultat.couverture_cumulee[3] == pytest.approx(22 / 30)


def test_arret_par_nombre_maximal_de_facteurs(exemple):
	I, objets, attributs = exemple
	resultat = grecon(I, nb_max_facteurs=4)
	assert len(resultat.facteurs) == 4
	assert resultat.couverture_cumulee[-1] == pytest.approx(22 / 30)


def test_arret_par_couverture_cible(exemple):
	I, objets, attributs = exemple
	# Couvertures cumulées : 8, 14, 18, 22 croix sur 30 ; 22/30 ≈ 0,733 est la première ≥ 0,7
	resultat = grecon(I, couverture_cible=0.7)
	assert len(resultat.facteurs) == 4


def test_couverture_complete_en_8_facteurs(exemple):
	I, objets, attributs = exemple
	resultat = grecon(I)
	assert len(resultat.facteurs) == 8
	assert sum(resultat.croix_nouvelles) == 30
	assert resultat.couverture_cumulee[-1] == pytest.approx(1.0)
	assert np.array_equal(produit_booleen(resultat.A, resultat.B), I)


def test_formes_de_A_et_B(exemple):
	I, objets, attributs = exemple
	resultat = grecon(I)
	nb_facteurs = len(resultat.facteurs)
	assert resultat.A.shape == (10, nb_facteurs)
	assert resultat.B.shape == (nb_facteurs, 8)


def test_chaque_facteur_est_un_concept(exemple):
	I, objets, attributs = exemple
	for facteur in grecon(I).facteurs:
		assert est_concept(I, facteur.extension, facteur.intention)


def test_jamais_de_croix_absente(exemple):
	"""A ∘ B ≤ I pour toute factorisation partielle (k = 1 à 8 facteurs)."""
	I, objets, attributs = exemple
	for k in range(1, 9):
		resultat = grecon(I, nb_max_facteurs=k)
		reconstruction = produit_booleen(resultat.A, resultat.B)
		assert not (reconstruction & ~I).any()


def test_egalite_plus_grande_extension():
	# {o1,o2}×{a2} et {o1}×{a1,a2} couvrent 2 croix chacun : on garde la plus grande extension
	I = np.array([[1, 1], [0, 1]], dtype=bool)
	premier = grecon(I, nb_max_facteurs=1).facteurs[0]
	assert premier.extension.tolist() == [True, True]
	assert premier.intention.tolist() == [False, True]


def test_egalite_intention_premiere_dans_l_ordre_des_colonnes():
	# {o1}×{a1} et {o2}×{a2} : même nombre de croix, même taille d'extension → intention [0] avant [1]
	I = np.array([[1, 0], [0, 1]], dtype=bool)
	premier = grecon(I, nb_max_facteurs=1).facteurs[0]
	assert premier.intention.tolist() == [True, False]


def test_egalite_ordre_lexicographique_des_intentions():
	# o1 = {a1,a4,a5} → [0,3,4] et o2 = {a1,a2,a3} → [0,1,2] : 3 croix chacun, extension de taille 1
	# ({o1,o2}×{a1} ne couvre que 2 croix). Même premier indice : l'ordre lexicographique garde [0,1,2]
	I = np.array([[1, 0, 0, 1, 1], [1, 1, 1, 0, 0]], dtype=bool)
	premier = grecon(I, nb_max_facteurs=1).facteurs[0]
	assert premier.intention.tolist() == [True, True, True, False, False]


def test_nb_facteurs_pour_couverture(exemple):
	# Couvertures cumulées : 8, 14, 18, 22, 25, 27, 29, 30 croix sur 30
	I, objets, attributs = exemple
	resultat = grecon(I)
	assert nb_facteurs_pour_couverture(resultat, 0.8) == 5
	assert nb_facteurs_pour_couverture(resultat, 0.9) == 6   # 27/30 : cible atteinte exactement
	assert nb_facteurs_pour_couverture(resultat, 1.0) == 8
