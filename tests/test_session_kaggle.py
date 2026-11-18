"""Campagne 1, étape 7 : configurations, description de session, versions, reprise, dossier sessions/ ignoré."""

import filecmp
import json
import platform
import re
import shutil
import sys
from datetime import datetime

import pandas as pd
import pytest

from experiences.analyse_campagne1 import analyser
from experiences.campagne1 import RACINE, Config, METHODES, assembler, lire_config
from experiences.session_kaggle import (
	CampagneEchouee, RepriseIncompatible, commande_campagne, comparer_versions, decrire_session, executer_campagne,
	reprendre, valeur_proc,
)
from tests.test_analyse_campagne1 import ecrire_campagne

JEUX_CAMPAGNE = {
	"anneal", "breast-w", "hepatitis", "colic", "iris", "led7", "mushroom", "nursery", "page-blocks", "pima",
	"wine", "balance-scale", "breast-cancer", "credit-approval", "glass", "heart-statlog", "vowel", "zoo", "vote",
}

CPUINFO = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\nprocessor\t: 1\n"
MEMINFO = "MemTotal:       32873000 kB\nMemFree:        30000000 kB\nMemAvailable:   31000000 kB\n"


def test_campagne_memes_jeux_mushroom_en_dernier():
	jeux = lire_config(RACINE / "experiences" / "campagne1.yaml").jeux
	assert len(jeux) == 19 and set(jeux) == JEUX_CAMPAGNE
	assert jeux[-1] == "mushroom"


def test_essai_kaggle_jeux_de_l_essai_local_memes_reglages():
	essai = lire_config(RACINE / "experiences" / "essai_kaggle.yaml")
	campagne = lire_config(RACINE / "experiences" / "campagne1.yaml")
	assert essai.jeux == ["iris", "wine", "zoo"] == lire_config(RACINE / "experiences" / "essai_local.yaml").jeux
	assert essai.dossier_resultats == RACINE / "resultats" / "essai_kaggle"
	assert (essai.methodes, essai.graine, essai.nb_sacs, essai.delai_grecond_s) == (
		campagne.methodes, campagne.graine, campagne.nb_sacs, campagne.delai_grecond_s)


def test_valeur_proc():
	assert valeur_proc(CPUINFO, "model name") == "Intel(R) Xeon(R) CPU @ 2.20GHz"
	assert valeur_proc(MEMINFO, "MemTotal") == "32873000 kB"
	assert valeur_proc(MEMINFO, "SwapTotal") is None
	assert valeur_proc(None, "MemTotal") is None


def test_versions_conformes_et_ecarts(tmp_path):
	comparaison = comparer_versions(RACINE / "requirements.txt")
	assert set(comparaison) == {"numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "PyYAML"}
	assert all(attendue == installee for attendue, installee in comparaison.values())
	dev = (RACINE / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
	assert dev[0] == "-r requirements.txt"
	assert set(comparer_versions(RACINE / "requirements-dev.txt")) == {"pytest", "fcapy"}
	faux = tmp_path / "requirements.txt"
	faux.write_text("numpy==0.0.1\npaquet-inexistant-xyz==1.0\n", encoding="utf-8")
	comparaison = comparer_versions(faux)
	assert comparaison["numpy"][0] == "0.0.1" and comparaison["numpy"][1] != "0.0.1"
	assert comparaison["paquet-inexistant-xyz"] == ("1.0", None)


def test_description_de_session(tmp_path):
	config = Config(jeux=["iris"], methodes=list(METHODES), graine=42, nb_sacs=25, delai_grecond_s=1200.0,
		dossier_resultats=tmp_path / "essai")
	chemin = decrire_session(config, "experiences/essai_kaggle.yaml", datetime(2026, 11, 18, 14, 5, 9))
	assert chemin == tmp_path / "essai" / "sessions" / "20261118T140509.txt"
	texte = chemin.read_text(encoding="utf-8")
	for cle in ["date: 2026-11-18T14:05:09", "configuration: experiences/essai_kaggle.yaml", "commit: ",
			"modifications_non_commitees: ", "nb_coeurs: ", "processeur: ", "memoire_totale: ",
			"memoire_disponible: ", "plateforme: ", "python: ", "version_numpy: ", "ecarts_versions: aucun"]:
		assert cle in texte, cle
	assert platform.node() not in texte


def ecrire_pli(dossier, commit, contenu="a"):
	dossier.mkdir(parents=True, exist_ok=True)
	pd.DataFrame({"commit": [commit, commit], "valeur": [contenu, contenu]}).to_csv(dossier / "pli_0.csv", index=False)


def test_reprise_copie_sans_ecraser(tmp_path):
	source, destination = tmp_path / "source", tmp_path / "destination"
	ecrire_pli(source / "iris", "abc")
	ecrire_pli(source / "wine", "abc")
	(source / "wine" / "pli_1.csv.tmp").write_text("coupé", encoding="utf-8")
	(source / "sessions").mkdir()
	(source / "sessions" / "20261118T140509.txt").write_text("date: x\n", encoding="utf-8")
	ecrire_pli(destination / "iris", "abc", contenu="deja_la")
	assert reprendre(source, destination, "abc") == (2, 1)
	assert pd.read_csv(destination / "iris" / "pli_0.csv")["valeur"].iloc[0] == "deja_la"
	assert filecmp.cmp(source / "wine" / "pli_0.csv", destination / "wine" / "pli_0.csv", shallow=False)
	assert (destination / "sessions" / "20261118T140509.txt").exists()
	assert not (destination / "wine" / "pli_1.csv.tmp").exists()


def test_reprise_refuse_un_autre_commit(tmp_path):
	ecrire_pli(tmp_path / "source" / "iris", "ancien")
	with pytest.raises(RepriseIncompatible, match="ancien"):
		reprendre(tmp_path / "source", tmp_path / "destination", "nouveau")
	assert not (tmp_path / "destination" / "iris" / "pli_0.csv").exists()


def test_assembler_ignore_le_dossier_sessions(tmp_path):
	noms = ecrire_campagne(tmp_path, [0.02])
	config = Config(jeux=noms, methodes=list(METHODES), graine=42, nb_sacs=25, delai_grecond_s=1200.0,
		dossier_resultats=tmp_path)
	reference = {nom: (tmp_path / nom).read_bytes() for nom in ["campagne1_par_pli.csv", "campagne1_par_jeu.csv"]}
	(tmp_path / "sessions").mkdir()
	(tmp_path / "sessions" / "20261118T140509.txt").write_text("date: x\nnb_coeurs: 4\n", encoding="utf-8")
	(tmp_path / "sessions" / "pli_0.csv").write_text("piège\n1\n", encoding="utf-8")
	assembler(config)
	assert {nom: (tmp_path / nom).read_bytes() for nom in reference} == reference


def test_commande_de_campagne_sous_timeout():
	assert commande_campagne("experiences/campagne1.yaml", "11h") == [
		"timeout", "11h", sys.executable, "-m", "experiences.campagne1", "--config", "experiences/campagne1.yaml"]


def test_code_124_arret_normal_autre_code_erreur():
	simuler = lambda code: [sys.executable, "-c", f"import sys; sys.exit({code})"]  # noqa: E731
	assert executer_campagne(simuler(124)) is True
	assert executer_campagne(simuler(0)) is False
	with pytest.raises(CampagneEchouee, match="code 1"):
		executer_campagne(simuler(1))


def test_notebook_kaggle():
	notebook = json.loads((RACINE / "kaggle" / "campagne1.ipynb").read_text(encoding="utf-8"))
	code = ["".join(cellule["source"]) for cellule in notebook["cells"] if cellule["cell_type"] == "code"]
	for source in code:
		compile(source, "cellule", "exec")
	assert all(not cellule.get("outputs") for cellule in notebook["cells"])
	tete = code[0]
	for attendu in ['CONFIG = "experiences/essai_kaggle.yaml"', 'COMMIT_ATTENDU = ""', "REPRISE = None",
			'DUREE_MAX = "11h"', "https://github.com/mentalist011101/campagne1_extraction-de-facteur.git"]:
		assert attendu in tete, attendu
	tout = "\n".join(code)
	for attendu in ["experiences.session_kaggle", '"decrire"', '"reprendre"', '"lancer"', '"--duree-max", DUREE_MAX',
			'"--assembler"', "make_archive"]:
		assert attendu in tout, attendu
	assert not re.search(r"(?i)token|password|mot de passe|ghp_|secret", tout)


@pytest.mark.lent
def test_analyse_ignore_le_dossier_sessions(tmp_path):
	source = RACINE / "resultats" / "essai_local"
	copie = tmp_path / "essai"
	shutil.copytree(source, copie, ignore=shutil.ignore_patterns("analyse"))
	(copie / "sessions").mkdir()
	(copie / "sessions" / "20261118T140509.txt").write_text("date: x\n", encoding="utf-8")
	resultat = analyser(copie, tmp_path / "analyse")
	assert resultat["retenus"] == ["iris", "wine", "zoo"] and resultat["anomalies"].empty
	for fichier in ["f1_par_jeu.csv", "rangs.csv", "jeux_exclus.csv", "tracabilite.csv"]:
		assert filecmp.cmp(source / "analyse" / fichier, tmp_path / "analyse" / fichier, shallow=False), fichier
