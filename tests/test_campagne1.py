"""Campagne 1, étape 5 : plis figés, script de campagne, reprise et abandons (iris, B = 3)."""

import os

import numpy as np
import pandas as pd
import pytest

import experiences.campagne1 as campagne1
import m3_classifieurs.ensachage as ensachage
from donnees.catalogue import JEUX, PlisInvalides, charger_jeu, charger_plis, figer_plis_campagne1
from experiences.campagne1 import COLONNES_TRACE, METHODES, Config, assembler, lancer
from m2_facteurs.grecond import DelaiDepasse

pytestmark = pytest.mark.lent


def config_iris(dossier, jeux=("iris",), nb_sacs=3, delai=1200.0):
	return Config(jeux=list(jeux), methodes=list(METHODES), graine=42, nb_sacs=nb_sacs,
		delai_grecond_s=delai, dossier_resultats=dossier)


def test_plis_figes_stratifies_disjoints_et_non_ecrases(tmp_path):
	figer_plis_campagne1(["iris"], dossier=tmp_path)
	with pytest.raises(FileExistsError):
		figer_plis_campagne1(["iris"], dossier=tmp_path)
	_, y = charger_jeu("iris")
	plis = charger_plis("iris", y, dossier=tmp_path)
	assert len(plis) == 10
	evalues = np.concatenate([ev for _, ev in plis])
	assert sorted(evalues.tolist()) == list(range(150))
	for app, ev in plis:
		assert len(set(app) & set(ev)) == 0 and len(app) + len(ev) == 150
		assert sorted(y.iloc[ev].value_counts().tolist()) == [5, 5, 5]


def test_plis_alteres_refuses(tmp_path):
	figer_plis_campagne1(["iris"], dossier=tmp_path)
	_, y = charger_jeu("iris")
	table = pd.read_csv(tmp_path / "iris.csv", dtype={"classe": str})
	table.loc[0, "classe"] = "Iris-virginica" if table.loc[0, "classe"] != "Iris-virginica" else "Iris-setosa"
	table.to_csv(tmp_path / "iris.csv", index=False)
	with pytest.raises(PlisInvalides):
		charger_plis("iris", y, dossier=tmp_path)
	table.iloc[1:].to_csv(tmp_path / "iris.csv", index=False)
	with pytest.raises(PlisInvalides):
		charger_plis("iris", y, dossier=tmp_path)


def test_plis_figes_des_19_jeux_valides():
	for jeu in JEUX:
		_, y = charger_jeu(jeu.nom)
		assert len(charger_plis(jeu.nom, y)) == 10


def test_reprise_apres_interruption_rien_n_est_recalcule(tmp_path):
	config = config_iris(tmp_path)
	assert lancer(config, arreter_apres=2) == {"calcules": 2, "sautes": 0}
	premiers = {k: (tmp_path / "iris" / f"pli_{k}.csv") for k in (0, 1)}
	avant = {k: (os.stat(c).st_mtime_ns, c.read_bytes()) for k, c in premiers.items()}
	assert lancer(config) == {"calcules": 8, "sautes": 2}
	assert len(list((tmp_path / "iris").glob("pli_*.csv"))) == 10
	for k, chemin in premiers.items():
		assert (os.stat(chemin).st_mtime_ns, chemin.read_bytes()) == avant[k]
	assert not list(tmp_path.rglob("*.tmp"))


def test_contenu_d_un_pli(tmp_path):
	lancer(config_iris(tmp_path), arreter_apres=1)
	pli = pd.read_csv(tmp_path / "iris" / "pli_0.csv")
	assert pli["methode"].tolist() == ["MAJ", "M3-AV", "M3-BV", "STACK", "LR", "RF", "NB", "BAG"]
	assert (pli["statut"] == "ok").all()
	assert pli["f1_macro"].between(0, 1).all() and pli["accuracy"].between(0, 1).all()
	for colonne in ["temps_s", "n_app", "n_eval", "nb_attributs", "nb_constants", "nb_fusions", "nb_facteurs",
			"part_facteurs_un_attribut", "temps_binarisation_s", "temps_grecond_s",
			"taux_non_couverts", "taux_egalites", "avertissements_convergence", *COLONNES_TRACE]:
		assert colonne in pli.columns, colonne
	assert (pli["n_app"] == 135).all() and (pli["n_eval"] == 15).all()
	assert pli["commit"].iloc[0] != "" and pli["version_sklearn"].iloc[0] != ""


def test_grecond_principal_trop_long_abandon_du_jeu_et_suite(tmp_path, monkeypatch):
	vrai_grecond = campagne1.grecond
	appels = []

	def grecond_lent_pour_iris(I, **options):
		appels.append(I.shape)
		if len(appels) == 1:
			raise DelaiDepasse(1300.0, 4)
		return vrai_grecond(I, **options)

	monkeypatch.setattr(campagne1, "grecond", grecond_lent_pour_iris)
	assert lancer(config_iris(tmp_path, jeux=("iris", "wine")), arreter_apres=1)["calcules"] == 1
	abandon = pd.read_csv(tmp_path / "iris" / "abandon.csv")
	assert abandon.loc[0, "pli"] == 0 and abandon.loc[0, "duree_s"] == 1300.0
	assert not list((tmp_path / "iris").glob("pli_*.csv"))
	assert (tmp_path / "wine" / "pli_0.csv").exists()
	assert lancer(config_iris(tmp_path, jeux=("iris",)))["calcules"] == 0


def test_grecond_d_un_sac_trop_long_bag_abandonne_les_autres_ecrites(tmp_path, monkeypatch):
	appels_sacs = []

	def grecond_de_sac_trop_long(I, **options):
		appels_sacs.append(I.shape)
		raise DelaiDepasse(1250.0, 7)

	monkeypatch.setattr(ensachage, "grecond", grecond_de_sac_trop_long)
	lancer(config_iris(tmp_path), arreter_apres=2)
	pli_0 = pd.read_csv(tmp_path / "iris" / "pli_0.csv").set_index("methode")
	assert (pli_0.drop(index="BAG")["statut"] == "ok").all()
	assert pli_0.drop(index="BAG")["f1_macro"].notna().all()
	assert pli_0.loc["BAG", "statut"] == "abandonne" and np.isnan(pli_0.loc["BAG", "f1_macro"])
	abandon_bag = pd.read_csv(tmp_path / "iris" / "abandon_bag.csv")
	assert abandon_bag.iloc[0][["pli", "sac", "duree_s"]].tolist() == [0, 0, 1250.0]
	assert len(appels_sacs) == 1
	pli_1 = pd.read_csv(tmp_path / "iris" / "pli_1.csv").set_index("methode")
	assert pli_1.loc["BAG", "statut"] == "abandonne"
	lancer(config_iris(tmp_path))
	assert len(appels_sacs) == 1
	par_jeu = pd.read_csv(tmp_path / "campagne1_par_jeu.csv").set_index("methode")
	assert set(par_jeu["statut"]) == {"termine_sans_bag"}
	assert np.isnan(par_jeu.loc["BAG", "f1_macro"]) and par_jeu.loc["BAG", "nb_plis"] == 0
	assert par_jeu.drop(index="BAG")["f1_macro"].notna().all()


def test_assemblage_et_statuts(tmp_path):
	config = config_iris(tmp_path)
	lancer(config, arreter_apres=2)
	par_jeu = assembler(config)
	assert set(par_jeu["statut"]) == {"incomplet"}
	assert par_jeu["f1_macro"].isna().all() and (par_jeu["nb_plis"] == 2).all()
	lancer(config)
	par_jeu = pd.read_csv(tmp_path / "campagne1_par_jeu.csv")
	assert len(par_jeu) == 8 and set(par_jeu["statut"]) == {"termine"}
	assert (par_jeu["nb_plis"] == 10).all() and par_jeu["f1_macro"].notna().all()
	par_pli = pd.read_csv(tmp_path / "campagne1_par_pli.csv")
	assert len(par_pli) == 80
	assert np.isclose(par_jeu.set_index("methode").loc["M3-AV", "f1_macro"],
		par_pli[par_pli["methode"] == "M3-AV"]["f1_macro"].mean())


def test_configuration_de_la_campagne():
	config = campagne1.lire_config(campagne1.RACINE / "experiences" / "campagne1.yaml")
	assert sorted(config.jeux) == sorted(jeu.nom for jeu in JEUX)
	assert config.methodes == METHODES
	assert (config.graine, config.nb_sacs, config.delai_grecond_s) == (42, 25, 1200.0)
	assert config.dossier_resultats == campagne1.RACINE / "resultats" / "campagne1"
