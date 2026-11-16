"""Tests de l'étape 3 : GreConD, sur l'exemple des 10 personnes."""

import numpy as np
import pytest

from donnees.exemple import charger_exemple_10_personnes
from m2_facteurs.factorisation import nb_facteurs_pour_couverture, produit_booleen
from m2_facteurs.fca import est_concept
from m2_facteurs.grecon import grecon
from m2_facteurs.grecond import choix_sur_egalite, grecond


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


def test_nb_facteurs_pour_couverture(exemple):
	# Couvertures cumulées : 8, 14, 18, 21, 24, 27, 29, 30 croix sur 30
	I, objets, attributs = exemple
	resultat = grecond(I)
	assert nb_facteurs_pour_couverture(resultat, 0.8) == 5   # 24/30 : cible atteinte exactement
	assert nb_facteurs_pour_couverture(resultat, 0.9) == 6
	assert nb_facteurs_pour_couverture(resultat, 1.0) == 8


def test_choix_sur_egalite_dans_l_exemple(exemple):
	# F1 pas 1 : Adult et Overweight à 6, fermetures {Adult} et {Adult, Male, Overweight}
	# F4 pas 1 : Underweight et Obese à 3, fermetures {Elderly, Underweight} et {Female, Obese}
	I, objets, attributs = exemple
	resultat = grecond(I, trace=True)
	choix = {(c.numero_facteur, c.numero_pas): c for c in choix_sur_egalite(I, resultat.trace)}
	f1 = choix[(1, 1)]
	assert [attributs[j] for j in f1.attributs_a_egalite] == ["Adult", "Overweight"]
	assert f1.valeur == 6
	assert f1.fermetures_differentes
	f4 = choix[(4, 1)]
	assert [attributs[j] for j in f4.attributs_a_egalite] == ["Underweight", "Obese"]
	assert f4.valeur == 3
	assert f4.fermetures_differentes


def test_choix_sur_egalite_sans_consequence():
	# a1 et a2 vont toujours ensemble : même fermeture {a1, a2}, l'égalité ne change rien
	I = np.array([[1, 1], [1, 1], [0, 0]], dtype=bool)
	resultat = grecond(I, trace=True)
	choix = choix_sur_egalite(I, resultat.trace)
	assert len(choix) == 1
	assert choix[0].attributs_a_egalite == [0, 1]
	assert not choix[0].fermetures_differentes


# --- Campagne 1, étape 5 : limite de temps ---

from m2_facteurs.grecond import DelaiDepasse


def test_delai_none_ne_change_rien():
	I = np.random.default_rng(0).random((30, 8)) < 0.4
	sans = grecond(I).facteurs
	avec = grecond(I, delai_max_s=None).facteurs
	assert len(sans) == len(avec)
	for a, b in zip(sans, avec):
		assert np.array_equal(a.extension, b.extension) and np.array_equal(a.intention, b.intention)


def test_delai_zero_leve_delai_depasse_apres_le_premier_facteur():
	I = np.random.default_rng(0).random((30, 8)) < 0.4
	with pytest.raises(DelaiDepasse) as erreur:
		grecond(I, delai_max_s=0)
	assert erreur.value.nb_facteurs == 1
	assert erreur.value.duree_s >= 0
