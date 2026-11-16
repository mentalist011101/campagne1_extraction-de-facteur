"""Tests de l'étape 5 : M1, binarisation par terciles appris sur l'apprentissage."""

import numpy as np
import pandas as pd
import pytest

from donnees.pima import COLONNES_ZERO_MANQUANT, charger_decoupages, charger_pima, charger_plis_cv10
from m1_binarisation.binarisation import apprendre_seuils, binariser


def test_seuils_et_noms_sur_donnees_simples():
	X = pd.DataFrame({"v": [0, 1, 2, 3, 4, 5, 6]})
	seuils = apprendre_seuils(X, colonnes_zero_manquant=[])
	assert seuils["v"] == (2.0, 4.0)
	I, noms = binariser(X, seuils, colonnes_zero_manquant=[])
	assert noms == ["v<=2", "2<v<=4", "v>4"]


def test_bornes_des_intervalles():
	X = pd.DataFrame({"v": [0, 1, 2, 3, 4, 5, 6]})
	seuils = apprendre_seuils(X, colonnes_zero_manquant=[])
	nouveaux = pd.DataFrame({"v": [2, 3, 4, 5]})
	I, noms = binariser(nouveaux, seuils, colonnes_zero_manquant=[])
	assert I.tolist() == [
		[True, False, False],   # 2 → bas
		[False, True, False],   # 3 → moyen
		[False, True, False],   # 4 → moyen
		[False, False, True],   # 5 → haut
	]


def test_zero_manquant_exclu_des_seuils_et_sans_attribut():
	# Sans le zéro : 1..7, quantiles 1/3 et 2/3 = 3 et 5
	X = pd.DataFrame({"w": [0, 1, 2, 3, 4, 5, 6, 7]})
	seuils = apprendre_seuils(X, colonnes_zero_manquant=["w"])
	assert seuils["w"] == (3.0, 5.0)
	I, noms = binariser(X, seuils, colonnes_zero_manquant=["w"])
	assert not I[0].any()
	assert (I[1:].sum(axis=1) == 1).all()


@pytest.fixture
def pima_binarise():
	X, y = charger_pima()
	d = charger_decoupages()
	X_app = X.iloc[d["apprentissage"]]
	seuils = apprendre_seuils(X_app, COLONNES_ZERO_MANQUANT)
	contextes = {
		nom: binariser(X.iloc[indices], seuils, COLONNES_ZERO_MANQUANT)
		for nom, indices in d.items()
	}
	return X, d, seuils, contextes


def test_seuils_appris_sur_l_apprentissage_seulement(pima_binarise):
	X, d, seuils, contextes = pima_binarise
	insuline_app = X["insuline"].iloc[d["apprentissage"]]
	non_nulles = insuline_app[insuline_app != 0]
	assert seuils["insuline"] == (non_nulles.quantile(1 / 3), non_nulles.quantile(2 / 3))
	age_app = X["age"].iloc[d["apprentissage"]]
	assert seuils["age"] == (age_app.quantile(1 / 3), age_app.quantile(2 / 3))


def test_formes_des_trois_contextes(pima_binarise):
	X, d, seuils, contextes = pima_binarise
	assert contextes["apprentissage"][0].shape == (460, 24)
	assert contextes["validation"][0].shape == (154, 24)
	assert contextes["test"][0].shape == (154, 24)
	noms_app = contextes["apprentissage"][1]
	assert contextes["validation"][1] == noms_app
	assert contextes["test"][1] == noms_app


def test_un_attribut_par_variable_sauf_valeur_manquante(pima_binarise):
	X, d, seuils, contextes = pima_binarise
	for nom, (I, noms) in contextes.items():
		X_part = X.iloc[d[nom]]
		for numero, colonne in enumerate(X.columns):
			par_variable = I[:, 3 * numero : 3 * numero + 3].sum(axis=1)
			if colonne in COLONNES_ZERO_MANQUANT:
				attendu = (X_part[colonne] != 0).astype(int).to_numpy()
			else:
				attendu = np.ones(len(X_part), dtype=int)
			assert np.array_equal(par_variable, attendu), (nom, colonne)


def test_une_seule_ligne_donne_la_meme_binarisation(pima_binarise):
	"""Aucun recalcul : binariser un x seul ou dans le tableau donne la même ligne."""
	X, d, seuils, contextes = pima_binarise
	I_test, noms = contextes["test"]
	for position in [0, 50, 153]:
		ligne = X.iloc[[d["test"][position]]]
		I_ligne, _ = binariser(ligne, seuils, COLONNES_ZERO_MANQUANT)
		assert np.array_equal(I_ligne[0], I_test[position])


# --- Étape 11 : échelle interordinale ---

def test_interordinale_tableau_des_instructions():
	# Valeurs 1 à 6 et une valeur manquante (codée 0), t1 = 2, t2 = 4
	X = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 0]})
	I, noms = binariser(X, {"v": (2.0, 4.0)}, colonnes_zero_manquant=["v"], echelle="interordinale")
	assert noms == ["v<=2", "v<=4", "v>2", "v>4"]
	assert I.astype(int).tolist() == [
		[1, 1, 0, 0],   # 1
		[1, 1, 0, 0],   # 2
		[0, 1, 1, 0],   # 3
		[0, 1, 1, 0],   # 4
		[0, 0, 1, 1],   # 5
		[0, 0, 1, 1],   # 6
		[0, 0, 0, 0],   # manquant
	]


def test_nominale_reste_la_valeur_par_defaut():
	X = pd.DataFrame({"v": [0, 1, 2, 3, 4, 5, 6]})
	seuils = apprendre_seuils(X, colonnes_zero_manquant=[])
	I_defaut, noms_defaut = binariser(X, seuils, colonnes_zero_manquant=[])
	I_nominale, noms_nominale = binariser(X, seuils, colonnes_zero_manquant=[], echelle="nominale")
	assert np.array_equal(I_defaut, I_nominale)
	assert noms_defaut == noms_nominale


@pytest.fixture
def pima_deux_echelles():
	X, y = charger_pima()
	d = charger_decoupages()
	X_app = X.iloc[d["apprentissage"]]
	seuils = apprendre_seuils(X_app, COLONNES_ZERO_MANQUANT)
	I_nom, _ = binariser(X_app, seuils, COLONNES_ZERO_MANQUANT, echelle="nominale")
	I_inter, noms_inter = binariser(X_app, seuils, COLONNES_ZERO_MANQUANT, echelle="interordinale")
	return X_app, I_nom, I_inter, noms_inter


def test_interordinale_et_nominale_coincident(pima_deux_echelles):
	"""bas = v<=t1 ; haut = v>t2 ; milieu = v>t1 ET v<=t2."""
	X_app, I_nom, I_inter, noms = pima_deux_echelles
	for numero in range(8):
		bas, milieu, haut = I_nom[:, 3 * numero], I_nom[:, 3 * numero + 1], I_nom[:, 3 * numero + 2]
		inf_t1, inf_t2, sup_t1, sup_t2 = (I_inter[:, 4 * numero + k] for k in range(4))
		assert np.array_equal(bas, inf_t1)
		assert np.array_equal(haut, sup_t2)
		assert np.array_equal(milieu, sup_t1 & inf_t2)


def test_interordinale_deux_attributs_par_variable_sauf_manquant(pima_deux_echelles):
	X_app, I_nom, I_inter, noms = pima_deux_echelles
	for numero, colonne in enumerate(X_app.columns):
		par_variable = I_inter[:, 4 * numero : 4 * numero + 4].sum(axis=1)
		if colonne in COLONNES_ZERO_MANQUANT:
			attendu = 2 * (X_app[colonne] != 0).astype(int).to_numpy()
		else:
			attendu = np.full(len(X_app), 2)
		assert np.array_equal(par_variable, attendu), colonne


def test_interordinale_formes_et_densite(pima_deux_echelles):
	X_app, I_nom, I_inter, noms = pima_deux_echelles
	assert I_inter.shape == (460, 32)
	assert len(noms) == 32
	assert I_inter.sum() == 2 * I_nom.sum() == 6584
	assert I_inter.mean() == pytest.approx(6584 / 14720)


def test_seuils_strictement_croissants_apprentissage_et_plis():
	"""t1 < t2 pour chaque variable ; toute égalité est signalée (variable, pli), rien n'est corrigé."""
	X, y = charger_pima()
	d = charger_decoupages()
	parties = {"apprentissage complet": d["apprentissage"]}
	for numero_pli, (indices_app, indices_eval) in enumerate(charger_plis_cv10()):
		parties[f"pli {numero_pli}"] = indices_app
	egalites = []
	for nom_partie, indices in parties.items():
		for variable, (t1, t2) in apprendre_seuils(X.iloc[indices], COLONNES_ZERO_MANQUANT).items():
			if not t1 < t2:
				egalites.append((variable, nom_partie, t1, t2))
	assert not egalites, f"Seuils non strictement croissants : {egalites}"


# --- Campagne 1, étape 3 : binarisation mixte (DECISIONS.md, 2026-11-16) ---

from m1_binarisation.binarisation import apprendre_binarisation, binariser_jeu


def test_campagne_numerique_a_au_plus_5_valeurs_devient_categorielle():
	X = pd.DataFrame({"v": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})
	binarisation = apprendre_binarisation(X, {"v": "numerique"})
	assert binarisation.noms == ["v=1.0", "v=2.0", "v=3.0"]
	assert binariser_jeu(X, binarisation)[:3].tolist() == [
		[True, False, False], [False, True, False], [False, False, True],
	]


def test_campagne_numerique_a_7_valeurs_donne_4_attributs_interordinaux():
	X = pd.DataFrame({"v": [0.0, 1, 2, 3, 4, 5, 6]})
	binarisation = apprendre_binarisation(X, {"v": "numerique"})
	assert binarisation.noms == ["v<=2", "v<=4", "v>2", "v>4"]
	assert (binarisation.nb_constants, binarisation.nb_fusions) == (0, 0)


def test_campagne_t1_egal_t2_donne_2_attributs_et_non_4():
	# 20 valeurs dont 14 zéros : quantiles 1/3 et 2/3 nuls, t1 = t2 = 0
	X = pd.DataFrame({"v": [0.0] * 14 + [1, 2, 3, 4, 5, 6]})
	binarisation = apprendre_binarisation(X, {"v": "numerique"})
	assert binarisation.seuils["v"] == (0.0, 0.0)
	assert binarisation.noms == ["v<=0", "v>0"]
	assert binarisation.nb_fusions == 2
	assert binariser_jeu(X, binarisation).shape == (20, 2)


def test_campagne_doublon_entre_deux_variables_fusionne_premier_garde():
	X = pd.DataFrame({"a": ["x", "y", "x"], "b": ["u", "w", "u"]})
	binarisation = apprendre_binarisation(X, {"a": "categorielle", "b": "categorielle"})
	# a=x ≡ b=u et a=y ≡ b=w sur l'apprentissage
	assert binarisation.noms == ["a=x", "a=y"]
	assert binarisation.nb_fusions == 2


def test_campagne_attributs_constants_retires():
	# Quantiles : 1/3 → 4,667 ; 2/3 → 6 = maximum, donc v<=6 toujours vrai et v>6 toujours faux
	X = pd.DataFrame({"v": [1.0, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6]})
	binarisation = apprendre_binarisation(X, {"v": "numerique"})
	assert binarisation.noms == ["v<=4.667", "v>4.667"]
	assert binarisation.nb_constants == 2


def test_campagne_valeur_manquante_sans_attribut():
	X = pd.DataFrame({
		"v": [0.0, 1, 2, 3, 4, 5, 6, np.nan],
		"c": ["a", "b", "a", np.nan, "b", "a", "b", "a"],
	})
	binarisation = apprendre_binarisation(X, {"v": "numerique", "c": "categorielle"})
	assert binarisation.seuils["v"] == (2.0, 4.0)
	assert binarisation.noms == ["v<=2", "v<=4", "v>2", "v>4", "c=a", "c=b"]
	I = binariser_jeu(X, binarisation)
	assert not I[7, :4].any()
	assert not I[3, 4:].any()


def test_campagne_seuils_et_modalites_de_l_apprentissage_appliques_tels_quels():
	X_app = pd.DataFrame({"v": [0.0, 1, 2, 3, 4, 5, 6], "c": ["a", "b", "a", "b", "a", "b", "a"]})
	binarisation = apprendre_binarisation(X_app, {"v": "numerique", "c": "categorielle"})
	X_nouveau = pd.DataFrame({"v": [10.0, -1.0], "c": ["z", "b"]})
	I = binariser_jeu(X_nouveau, binarisation)
	assert I.tolist() == [
		[False, False, True, True, False, False],   # v = 10 ; « z » absente de l'apprentissage
		[True, True, False, False, False, True],    # v = -1 ; c = b
	]
