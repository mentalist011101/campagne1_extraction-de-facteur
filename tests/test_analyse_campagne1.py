"""Campagne 1, étape 6 bis : script d'analyse (H8 à H10, descriptif, traçabilité)."""

import numpy as np
import pandas as pd
import pytest

from experiences.analyse_campagne1 import (
	ResultatsIncoherents, analyser, bootstrap_moyenne, friedman_nemenyi, groupes_non_distincts, holm,
	appliquer_wilcoxon, verdict_superiorite,
)
from experiences.campagne1 import COLONNES_TRACE, METHODES, RACINE, Config, assembler

TRACE = {"commit": "abc", "modifications_non_commitees": False, "version_python": "3.12",
	"version_numpy": "2.5.0", "version_pandas": "3.0.3", "version_sklearn": "1.9.0",
	"processeur": "cpu", "plateforme": "os", "date_calcul": "2026-11-18T00:00:00"}


def ecrire_campagne(dossier, d_stack, nb_plis=None, sans_bag=(), modifie=None):
	"""Campagne synthétique : F1 constante sur les plis ; STACK = M3-BV + d_stack[j] ; RF = STACK - 0,01."""
	noms = [f"jeu{j:02d}" for j in range(len(d_stack))]
	for j, nom in enumerate(noms):
		m3bv = 0.6 + 0.01 * j
		f1 = {"MAJ": 0.3, "M3-AV": m3bv - 0.02, "M3-BV": m3bv, "STACK": m3bv + d_stack[j], "LR": m3bv + 0.03,
			"RF": m3bv + d_stack[j] - 0.01, "NB": m3bv - 0.05, "BAG": m3bv + 0.005}
		for k in range((nb_plis or {}).get(nom, 10)):
			lignes = []
			for methode in METHODES:
				abandonne = methode == "BAG" and nom in sans_bag
				lignes.append({"jeu": nom, "pli": k, "methode": methode, "statut": "abandonne" if abandonne else "ok",
					"accuracy": np.nan if abandonne else f1[methode] + 0.05,
					"f1_macro": np.nan if abandonne else f1[methode],
					"taux_non_couverts": 0.01 if methode.startswith("M3") else np.nan,
					"taux_egalites": 0.02 if methode.startswith("M3") else np.nan, "temps_s": 0.1,
					"nb_attributs": 10, "nb_fusions": 0, "nb_facteurs": 5, "part_facteurs_un_attribut": 0.2,
					"temps_grecond_s": 0.1, **TRACE,
					"modifications_non_commitees": modifie == (nom, k)})
			(dossier / nom).mkdir(parents=True, exist_ok=True)
			pd.DataFrame(lignes).to_csv(dossier / nom / f"pli_{k}.csv", index=False)
		if nom in sans_bag:
			pd.DataFrame([{"jeu": nom, "pli": 0, "sac": 3, "duree_s": 1250.0}]).to_csv(
				dossier / nom / "abandon_bag.csv", index=False)
	assembler(Config(jeux=noms, methodes=list(METHODES), graine=42, nb_sacs=25, delai_grecond_s=1200.0,
		dossier_resultats=dossier))
	return noms


D_DISTINCTES = 0.01 + 0.001 * np.arange(19)
D_SYMETRIQUES = np.arange(-9, 10) / 100


@pytest.mark.lent
def test_essai_local_non_concluant(tmp_path):
	resultat = analyser(RACINE / "resultats" / "essai_local", tmp_path)
	assert not resultat["concluante"] and resultat["retenus"] == ["iris", "wine", "zoo"]
	rapport = (tmp_path / "rapport.md").read_text(encoding="utf-8")
	assert "Campagne non concluante (3 jeux terminés < 15)" in rapport
	assert not (tmp_path / "hypotheses.csv").exists()
	for fichier in ["f1_par_jeu.csv", "rangs.csv", "jeux_exclus.csv", "tracabilite.csv", "diagramme_cd.png"]:
		assert (tmp_path / fichier).exists(), fichier
	assert "F1 macro par jeu" in rapport and "Friedman" in rapport


@pytest.mark.lent
def test_h8_confirmee_quand_stack_domine(tmp_path):
	ecrire_campagne(tmp_path, D_DISTINCTES)
	resultat = analyser(tmp_path, tmp_path / "analyse")
	h8 = resultat["hypotheses"].set_index("hypothese").loc["H8"]
	assert resultat["concluante"] and h8["n_jeux"] == 19 and h8["n_effectifs"] == 19
	assert h8["p_brut"] == pytest.approx(2 / 2**19, rel=1e-6)
	assert (h8["victoires"], h8["defaites"], h8["egalites"]) == (19, 0, 0)
	assert h8["verdict"] == "confirmée"


@pytest.mark.lent
def test_h8_non_confirmee_si_differences_symetriques(tmp_path):
	ecrire_campagne(tmp_path, D_SYMETRIQUES)
	h8 = analyser(tmp_path, tmp_path / "analyse")["hypotheses"].set_index("hypothese").loc["H8"]
	assert h8["mediane_d"] == pytest.approx(0, abs=1e-12)
	assert h8["verdict"] == "rejetée"


def test_holm_calcule_a_la_main():
	assert holm({"H8": 0.01, "H9": 0.04}) == pytest.approx({"H8": 0.02, "H9": 0.04})
	assert holm({"H8": 0.03, "H9": 0.02}) == pytest.approx({"H8": 0.04, "H9": 0.04})
	assert holm({"H8": 0.6, "H9": 0.7}) == pytest.approx({"H8": 1.0, "H9": 1.0})


def test_verdicts_et_differences_toutes_nulles():
	d = pd.Series(D_DISTINCTES)
	assert verdict_superiorite(0.01, d) == "confirmée"
	assert verdict_superiorite(0.2, d) == "rejetée"
	assert verdict_superiorite(0.01, -d) == "rejetée (différence significative dans le sens contraire)"
	nulles = appliquer_wilcoxon(pd.Series(np.zeros(19)))
	assert nulles["n_effectifs"] == 0 and np.isnan(nulles["p_brut"])
	assert verdict_superiorite(np.nan, pd.Series(np.zeros(19))) == "p non défini, hypothèse rejetée (aucune différence)"


def test_bootstrap_reproductible():
	d = pd.Series(np.linspace(-0.03, 0.05, 19))
	premier, second = bootstrap_moyenne(d), bootstrap_moyenne(d)
	assert premier == second
	moyenne, basse, haute = premier
	assert moyenne == pytest.approx(d.mean()) and basse < moyenne < haute


@pytest.mark.lent
def test_pli_modifie_signale(tmp_path):
	ecrire_campagne(tmp_path, D_DISTINCTES, modifie=("jeu04", 7))
	resultat = analyser(tmp_path, tmp_path / "analyse")
	anomalies = pd.read_csv(tmp_path / "analyse" / "tracabilite.csv")
	assert ((anomalies["type"] == "modifications_non_commitees") & (anomalies["jeu"] == "jeu04")
		& (anomalies["pli"] == 7)).any()
	rapport = (tmp_path / "analyse" / "rapport.md").read_text(encoding="utf-8")
	assert "## Anomalies" in rapport and "jeu04" in rapport
	assert resultat["concluante"]


@pytest.mark.lent
def test_seuil_de_quinze_jeux(tmp_path):
	incomplets = {f"jeu{j:02d}": 5 for j in range(14, 19)}
	ecrire_campagne(tmp_path / "a", D_DISTINCTES, nb_plis=incomplets)
	assert not analyser(tmp_path / "a", tmp_path / "a" / "analyse")["concluante"]
	incomplets = {f"jeu{j:02d}": 5 for j in range(15, 19)}
	ecrire_campagne(tmp_path / "b", D_DISTINCTES, nb_plis=incomplets)
	resultat = analyser(tmp_path / "b", tmp_path / "b" / "analyse")
	assert resultat["concluante"] and len(resultat["retenus"]) == 15


@pytest.mark.lent
def test_jeu_sans_bag_exclu_et_liste(tmp_path):
	ecrire_campagne(tmp_path, D_DISTINCTES, sans_bag=("jeu03",))
	resultat = analyser(tmp_path, tmp_path / "analyse")
	assert "jeu03" not in resultat["retenus"] and len(resultat["retenus"]) == 18
	exclus = pd.read_csv(tmp_path / "analyse" / "jeux_exclus.csv").set_index("jeu")
	assert exclus.loc["jeu03", "statut"] == "termine_sans_bag"
	assert (resultat["hypotheses"]["n_jeux"] == 18).all()


def test_nemenyi_et_groupes():
	f1 = pd.DataFrame(np.tile(np.arange(8, 0, -1) / 10, (19, 1)), columns=METHODES)
	_, _, rangs, cd = friedman_nemenyi(f1)
	assert cd == pytest.approx(3.031 * np.sqrt(72 / 114))
	assert cd == pytest.approx(2.409, abs=1e-3)
	assert rangs.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
	rangs = pd.Series({"A": 1.0, "B": 2.0, "C": 3.5, "D": 6.0})
	assert groupes_non_distincts(rangs, 2.0) == [["A", "B"], ["B", "C"]]


@pytest.mark.lent
def test_fichiers_incoherents_refuses(tmp_path):
	ecrire_campagne(tmp_path, D_DISTINCTES)
	par_jeu = pd.read_csv(tmp_path / "campagne1_par_jeu.csv")
	par_jeu.loc[0, "f1_macro"] += 0.01
	par_jeu.to_csv(tmp_path / "campagne1_par_jeu.csv", index=False)
	with pytest.raises(ResultatsIncoherents):
		analyser(tmp_path, tmp_path / "analyse")


@pytest.mark.lent
def test_table3_seulement_poids_precision(tmp_path):
	ecrire_campagne(tmp_path, D_DISTINCTES)
	chemin = tmp_path / "table3.csv"
	chemin.write_text("# référence\njeu,vote,ponderation,mesure,valeur\n"
		"jeu00,all-votes,precision,accuracy,0.70\njeu00,best-vote,precision,accuracy,0.72\n"
		"jeu00,best-vote,lift,accuracy,0.99\n", encoding="utf-8")
	analyser(tmp_path, tmp_path / "analyse", chemin_table3=chemin)
	table3 = pd.read_csv(tmp_path / "analyse" / "table3.csv")
	assert len(table3) == 2 and set(table3["methode"]) == {"M3-AV", "M3-BV"}
	ligne = table3.set_index("methode").loc["M3-BV"]
	assert ligne["valeur_article"] == 0.72 and ligne["accuracy_campagne"] == pytest.approx(0.65)
	rapport = (tmp_path / "analyse" / "rapport.md").read_text(encoding="utf-8")
	assert "indicative" in rapport
	analyser(tmp_path, tmp_path / "sans", chemin_table3=tmp_path / "absent.csv")
	assert "Table 3 non fournie" in (tmp_path / "sans" / "rapport.md").read_text(encoding="utf-8")
