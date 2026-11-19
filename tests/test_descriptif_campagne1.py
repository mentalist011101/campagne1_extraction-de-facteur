"""Campagne 1, étape 8 : vérifications descriptives (classes prédites par M3, règles, essai local contre Kaggle)."""

import numpy as np
import pandas as pd
import pytest

from experiences.campagne1 import RACINE
from experiences.descriptif_campagne1 import (
	classes_predites, comparer_plis, controler, lire_csv, lire_plis, recalculer_pli, resumer_regles,
)
from m3_classifieurs.classifieur_factoriel import Regle


def predictions_synthetiques():
	return pd.DataFrame({
		"jeu": ["j"] * 6, "pli": [0, 0, 0, 1, 1, 1], "methode": ["M3-BV"] * 6,
		"reelle": ["a", "b", "b", "a", "b", "a"], "predite": ["a", "a", "a", "a", "b", "a"],
	})


def test_classes_predites_sommees_sur_les_plis():
	table = classes_predites(predictions_synthetiques()).set_index("classe")
	assert table.loc["a", "n_predit"] == 5 and table.loc["a", "n_reel"] == 3
	assert table.loc["b", "n_predit"] == 1 and table.loc["b", "n_reel"] == 3
	assert table["part_predite"].sum() == pytest.approx(1.0)


def test_resumer_regles():
	regles = [Regle(1, np.array([True]), "a", 0.9), Regle(2, np.array([True]), "b", 0.8),
		Regle(4, np.array([True]), "a", 0.7)]
	resume = resumer_regles(regles, {3: "égalité"}, "a", nb_facteurs=4)
	assert resume == {"nb_facteurs": 4, "nb_regles": 3, "nb_ecartes": 1, "classe_dominante": "a",
		"nb_regles_dominante": 2, "part_regles_dominante": pytest.approx(2 / 3)}


def test_controle_egal_et_ecart():
	stockees = pd.DataFrame({"jeu": ["j", "j"], "pli": [0, 1], "methode": ["M3-BV", "M3-BV"],
		"accuracy": [1 / 3, 0.5], "f1_macro": [0.25, 0.5]})
	table = controler(predictions_synthetiques(), stockees).set_index("pli")
	assert table.loc[0, "accuracy_recalculee"] == pytest.approx(1 / 3) and table.loc[0, "accuracy_egale"]
	assert table.loc[0, "f1_recalculee"] == pytest.approx(0.25) and table.loc[0, "f1_egale"]
	assert table.loc[1, "accuracy_recalculee"] == 1.0 and not table.loc[1, "accuracy_egale"]
	assert table.loc[1, "ecart_accuracy"] == pytest.approx(0.5)


def test_comparer_plis_exact():
	essai = pd.DataFrame({"jeu": ["iris"] * 2, "pli": [0, 1], "methode": ["RF", "RF"],
		"f1_macro": [0.9, 0.8], "accuracy": [0.91, 0.81]})
	campagne = essai.assign(f1_macro=[0.9, 0.81])
	table = comparer_plis(essai, campagne, ["iris"]).set_index("pli")
	assert table.loc[0, "f1_egale"] and table.loc[0, "accuracy_egale"] and table.loc[0, "ecart_f1"] == 0
	assert not table.loc[1, "f1_egale"] and table.loc[1, "ecart_f1"] == pytest.approx(0.01)
	assert table.loc[1, "accuracy_egale"]


def test_lecture_exacte_des_flottants(tmp_path):
	pd.DataFrame({"accuracy": [14 / 15]}).to_csv(tmp_path / "a.csv", index=False)
	assert lire_csv(tmp_path / "a.csv")["accuracy"].iloc[0] == 14 / 15


def test_lire_plis_valeurs_exactes(tmp_path):
	for jeu, k, valeur in [("a", 0, 14 / 15), ("a", 1, 1 / 3), ("b", 0, 2 / 7)]:
		(tmp_path / jeu).mkdir(exist_ok=True)
		pd.DataFrame({"jeu": [jeu], "pli": [k], "accuracy": [valeur]}).to_csv(tmp_path / jeu / f"pli_{k}.csv", index=False)
	plis = lire_plis(tmp_path, ["a", "b"])
	assert plis["accuracy"].tolist() == [14 / 15, 1 / 3, 2 / 7]


@pytest.mark.lent
def test_recalcul_retrouve_l_essai_local_iris_pli_0():
	predictions, resume = recalculer_pli("iris", 0)
	stockees = lire_csv(RACINE / "resultats" / "essai_local" / "iris" / "pli_0.csv")
	table = controler(predictions, stockees)
	assert set(table["methode"]) == {"M3-AV", "M3-BV"}
	assert table["accuracy_egale"].all() and table["f1_egale"].all()
	assert resume["nb_facteurs"] == 20 and resume["nb_regles"] + resume["nb_ecartes"] == 20
