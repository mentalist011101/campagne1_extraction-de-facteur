import hashlib

import numpy as np
import pandas as pd
import pytest

from donnees.catalogue import CHEMIN_CATALOGUE, JEUX, charger_jeu, chemin_copie, lire_variables

pytestmark = pytest.mark.lent

NOMS = [jeu.nom for jeu in JEUX]


def test_dix_neuf_jeux_uniques():
	assert len(JEUX) == 19
	assert len(set(NOMS)) == 19


def test_catalogue_une_ligne_par_jeu():
	catalogue = pd.read_csv(CHEMIN_CATALOGUE)
	assert list(catalogue["jeu"]) == NOMS


@pytest.mark.parametrize("nom", NOMS)
def test_copie_figee_conforme_au_catalogue(nom):
	ligne = pd.read_csv(CHEMIN_CATALOGUE).set_index("jeu").loc[nom]
	assert hashlib.sha256(chemin_copie(nom).read_bytes()).hexdigest() == ligne["sha256"]
	X, y = charger_jeu(nom)
	assert len(X) == len(y) == ligne["taille"]
	assert y.nunique() == ligne["nb_classes"] >= 2
	assert not y.isna().any()


@pytest.mark.parametrize("nom", NOMS)
def test_chaque_variable_a_un_type(nom):
	X, _ = charger_jeu(nom)
	types = lire_variables(nom).set_index("variable")["type"]
	for colonne in X.columns:
		if types[colonne] == "numerique":
			assert pd.api.types.is_float_dtype(X[colonne])
		else:
			assert types[colonne] == "categorielle"


def test_iris():
	X, y = charger_jeu("iris")
	assert X.shape == (150, 4)
	assert sorted(y.value_counts().tolist()) == [50, 50, 50]


def test_colonnes_retirees_au_chargement():
	X_zoo, _ = charger_jeu("zoo")
	assert "animal" not in X_zoo.columns
	assert X_zoo.shape == (101, 16)
	X_vowel, _ = charger_jeu("vowel")
	assert "Speaker_Number" not in X_vowel.columns
	assert "Train_or_Test" not in X_vowel.columns
	assert "Sex" in X_vowel.columns
	assert "animal" in pd.read_csv(chemin_copie("zoo")).columns
	X_colic, y_colic = charger_jeu("colic")
	assert "outcome" not in X_colic.columns
	assert X_colic.shape == (368, 21)
	assert y_colic.value_counts().to_dict() == {"yes": 232, "no": 136}
	assert lire_variables("colic").set_index("role").loc["classe", "variable"] == "surgical_lesion"


def test_valeurs_manquantes_des_copies_existantes():
	assert charger_jeu("vote")[0].isna().sum().sum() == 392
	X_mushroom, _ = charger_jeu("mushroom")
	assert X_mushroom.isna().sum().sum() == X_mushroom["stalk-root"].isna().sum() == 2480
	X_pima, _ = charger_jeu("pima")
	assert X_pima.isna().sum().sum() == 652
	assert not (X_pima[["glucose", "pression", "pli_cutane", "insuline", "imc"]] == 0).any().any()
	assert (X_pima["grossesses"] == 0).sum() > 0


def test_aucune_valeur_manquante_codee_en_texte():
	for nom in NOMS:
		X, _ = charger_jeu(nom)
		categorielles = X.select_dtypes(exclude=np.number)
		assert not categorielles.isin(["?", ""]).any().any(), nom
