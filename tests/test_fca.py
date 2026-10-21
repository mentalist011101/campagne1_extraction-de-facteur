"""Tests de l'étape 1 : outils FCA de base, sur l'exemple des 10 personnes."""

import numpy as np
import pytest
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

from donnees.exemple import charger_exemple_10_personnes
from m2_facteurs.fca import enumerer_concepts, est_concept, extension, intention


@pytest.fixture
def exemple():
	I, objets, attributs, classes = charger_exemple_10_personnes()
	return I, objets, attributs


def masque(noms_choisis, tous_les_noms):
	"""Construit un masque booléen à partir d'une liste de noms."""
	return np.array([nom in noms_choisis for nom in tous_les_noms])


def noms(masque_bool, tous_les_noms):
	"""Retourne l'ensemble des noms sélectionnés par un masque."""
	return {nom for nom, garde in zip(tous_les_noms, masque_bool) if garde}


def test_chargement_exemple(exemple):
	I, objets, attributs = exemple
	assert I.shape == (10, 8)
	assert I.dtype == bool
	assert I.sum() == 30


def test_extension_ensemble_vide_donne_tous_les_objets(exemple):
	I, objets, attributs = exemple
	assert extension(I, masque([], attributs)).all()


def test_intention_de_P1(exemple):
	I, objets, attributs = exemple
	resultat = intention(I, masque(["P1"], objets))
	assert noms(resultat, attributs) == {"Adult", "Male", "Normal"}


def test_extension_elderly_female(exemple):
	I, objets, attributs = exemple
	resultat = extension(I, masque(["Elderly", "Female"], attributs))
	assert noms(resultat, objets) == {"P3", "P9", "P10"}


def test_est_concept_vrai_et_faux(exemple):
	I, objets, attributs = exemple
	elderly_female = masque(["Elderly", "Female"], attributs)
	assert est_concept(I, masque(["P3", "P9", "P10"], objets), elderly_female)
	# {P3, P9} n'est pas fermé : P10 possède aussi Elderly et Female
	assert not est_concept(I, masque(["P3", "P9"], objets), elderly_female)


def test_22_concepts_distincts(exemple):
	I, objets, attributs = exemple
	concepts = enumerer_concepts(I)
	assert len(concepts) == 22
	intentions = {tuple(c.intention) for c in concepts}
	assert len(intentions) == 22


def test_chaque_concept_est_ferme(exemple):
	I, objets, attributs = exemple
	for concept in enumerer_concepts(I):
		assert est_concept(I, concept.extension, concept.intention)


def test_concepts_haut_et_bas_et_20_non_triviaux(exemple):
	I, objets, attributs = exemple
	concepts = enumerer_concepts(I)
	est_haut = [c.extension.all() and not c.intention.any() for c in concepts]
	est_bas = [not c.extension.any() and c.intention.all() for c in concepts]
	assert sum(est_haut) == 1
	assert sum(est_bas) == 1
	assert len(concepts) - sum(est_haut) - sum(est_bas) == 20


def test_meme_concepts_que_fcapy(exemple):
	"""Validation croisée : fcapy doit trouver exactement les mêmes concepts."""
	I, objets, attributs = exemple

	contexte = FormalContext(data=I.tolist(), object_names=objets, attribute_names=attributs)
	treillis = ConceptLattice.from_context(contexte)
	attendus = {
		(frozenset(c.extent), frozenset(c.intent)) for c in treillis
	}

	obtenus = {
		(frozenset(noms(c.extension, objets)), frozenset(noms(c.intention, attributs)))
		for c in enumerer_concepts(I)
	}
	assert obtenus == attendus
