"""Tests de l'étape 3 : GreConD, sur l'exemple des 10 personnes."""

import numpy as np
import pytest

from donnees.exemple import charger_exemple_10_personnes
from m2_facteurs.factorisation import produit_booleen
from m2_facteurs.fca import est_concept
from m2_facteurs.grecon import grecon
from m2_facteurs.grecond import grecond


@pytest.fixture
def exemple():
	I, objets, attributs, classes = charger_exemple_10_personnes()
	return I, objets, attributs


def noms(masque_bool, tous_les_noms):
	"""Retourne l'ensemble des noms sélectionnés par un masque."""
	return {nom for nom, garde in zip(tous_les_noms, masque_bool) if garde}


def test_trois_premiers_facteurs_identiques_a_grecon(exemple):
	I, objets, attributs = exemple
	facteurs_d = grecond(I).facteurs[:3]
	facteurs_c = grecon(I).facteurs[:3]
	for fd, fc in zip(facteurs_d, facteurs_c):
		assert np.array_equal(fd.extension, fc.extension)
		assert np.array_equal(fd.intention, fc.intention)


def test_quatrieme_facteur(exemple):
	I, objets, attributs = exemple
	resultat = grecond(I)
	f4 = resultat.facteurs[3]
	assert noms(f4.extension, objets) == {"P6", "P10"}
	assert noms(f4.intention, attributs) == {"Elderly", "Underweight"}
	assert resultat.croix_nouvelles[:4] == [8, 6, 4, 3]
	assert resultat.couverture_cumulee[3] == pytest.approx(21 / 30)


def test_couverture_complete_en_8_facteurs(exemple):
	I, objets, attributs = exemple
	resultat = grecond(I)
	assert len(resultat.facteurs) == 8
	assert sum(resultat.croix_nouvelles) == 30
	assert resultat.couverture_cumulee[-1] == pytest.approx(1.0)
	assert np.array_equal(produit_booleen(resultat.A, resultat.B), I)


def test_arret_par_nombre_maximal_de_facteurs(exemple):
	I, objets, attributs = exemple
	resultat = grecond(I, nb_max_facteurs=4)
	assert len(resultat.facteurs) == 4
	assert resultat.couverture_cumulee[-1] == pytest.approx(21 / 30)


def test_chaque_facteur_est_un_concept(exemple):
	I, objets, attributs = exemple
	for facteur in grecond(I).facteurs:
		assert est_concept(I, facteur.extension, facteur.intention)


def test_jamais_de_croix_absente(exemple):
	"""A ∘ B ≤ I pour toute factorisation partielle (k = 1 à 8 facteurs)."""
	I, objets, attributs = exemple
	for k in range(1, 9):
		resultat = grecond(I, nb_max_facteurs=k)
		reconstruction = produit_booleen(resultat.A, resultat.B)
		assert not (reconstruction & ~I).any()


def test_egalite_premier_attribut_dans_l_ordre_des_colonnes():
	# j = a1 donne {o1}×{a1,a2}, j = a2 donne {o1,o2}×{a2} : 2 croix chacun.
	# GreConD garde a1 (premier attribut) ; GreCon garderait {o1,o2}×{a2} (plus grande extension).
	I = np.array([[1, 1], [0, 1]], dtype=bool)
	premier = grecond(I, nb_max_facteurs=1).facteurs[0]
	assert premier.extension.tolist() == [True, False]
	assert premier.intention.tolist() == [True, True]


def test_sans_trace_par_defaut(exemple):
	I, objets, attributs = exemple
	assert grecond(I).trace is None


def test_trace_des_deux_premiers_pas(exemple):
	I, objets, attributs = exemple
	trace = grecond(I, trace=True).trace

	# Facteur 1, pas 1 : D = ∅ ; Adult et Overweight à égalité (6), Adult est retenu
	pas_1 = trace[0]
	assert pas_1.numero_facteur == 1
	assert pas_1.numero_pas == 1
	assert not pas_1.intention_depart.any()
	assert pas_1.valeurs == [6, 4, 5, 5, 4, 4, 6, 4]
	assert attributs[pas_1.attribut_retenu] == "Adult"

	# Facteur 1, pas 2 : D = {Adult} ; Male donne {P1,P4,P5,P8}×{Adult,Male} = 8 croix
	pas_2 = trace[1]
	assert pas_2.numero_facteur == 1
	assert pas_2.numero_pas == 2
	assert pas_2.valeurs[0] is None
	assert attributs[pas_2.attribut_retenu] == "Male"
	assert pas_2.valeurs[2] == 8


def test_trace_dernier_pas_d_un_facteur_sans_attribut_retenu(exemple):
	I, objets, attributs = exemple
	trace = grecond(I, trace=True).trace
	pas_facteur_1 = [p for p in trace if p.numero_facteur == 1]
	assert pas_facteur_1[-1].attribut_retenu is None
