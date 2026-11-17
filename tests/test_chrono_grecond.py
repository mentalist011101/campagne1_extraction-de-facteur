"""Campagne 1, étape 6 : chronométrage du GreConD principal et vérification du facteur 26."""

import pandas as pd
import pytest

from experiences.campagne1 import RACINE, Config, METHODES, lancer, lire_config
from experiences.chrono_grecond import FACTEUR_PLI, NB_PLIS, SEUIL_RISQUE_S, chronometrer, verifier_facteur

pytestmark = pytest.mark.lent


def test_constantes():
	assert (FACTEUR_PLI, NB_PLIS, SEUIL_RISQUE_S) == (26, 10, 600.0)


def test_essai_local_memes_reglages_que_la_campagne():
	essai = lire_config(RACINE / "experiences" / "essai_local.yaml")
	campagne = lire_config(RACINE / "experiences" / "campagne1.yaml")
	assert essai.jeux == ["iris", "wine", "zoo"]
	assert essai.dossier_resultats == RACINE / "resultats" / "essai_local"
	assert (essai.methodes, essai.graine, essai.nb_sacs, essai.delai_grecond_s) == (
		campagne.methodes, campagne.graine, campagne.nb_sacs, campagne.delai_grecond_s)


def test_chronometrer_iris():
	table = chronometrer(["iris"], delai_max_s=1200.0)
	ligne = table.iloc[0]
	assert ligne["jeu"] == "iris" and ligne["n_app"] == 135
	assert ligne["nb_attributs"] == 14 and ligne["nb_facteurs"] == 20
	assert not ligne["depasse_limite"] and not ligne["a_risque"]
	assert ligne["estimation_jeu_s"] == pytest.approx(ligne["temps_grecond_s"] * 26 * 10)
	for colonne in ["machine", "commit", "modifications_non_commitees", "densite"]:
		assert colonne in table.columns, colonne


def test_limite_depassee_signalee():
	table = chronometrer(["iris"], delai_max_s=0.0)
	ligne = table.iloc[0]
	assert ligne["depasse_limite"] and ligne["a_risque"]
	assert ligne["nb_facteurs"] >= 1


def test_verifier_facteur(tmp_path):
	config = Config(jeux=["iris"], methodes=list(METHODES), graine=42, nb_sacs=3, delai_grecond_s=1200.0,
		dossier_resultats=tmp_path)
	lancer(config)
	table = verifier_facteur(pd.read_csv(tmp_path / "campagne1_par_pli.csv"))
	ligne = table.set_index("jeu").loc["iris"]
	assert ligne["nb_plis"] == 10
	assert ligne["temps_pli_s"] > ligne["temps_grecond_s"] > 0
	assert ligne["facteur_observe"] == pytest.approx(ligne["temps_pli_s"] / ligne["temps_grecond_s"])
