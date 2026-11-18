"""Campagne 1 : le drapeau modifications_non_commitees ignore les fichiers non suivis (résultats)."""

import subprocess

from experiences.campagne1 import modifications_non_commitees


def git(depot, *arguments):
	subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@test", *arguments],
		cwd=depot, check=True, capture_output=True)


def test_fichier_non_suivi_ignore_fichier_suivi_modifie_detecte(tmp_path):
	git(tmp_path, "init", "-q")
	(tmp_path / "code.py").write_text("x = 1\n")
	git(tmp_path, "add", "code.py")
	git(tmp_path, "commit", "-q", "-m", "initial")
	assert not modifications_non_commitees(tmp_path)
	(tmp_path / "resultats").mkdir()
	(tmp_path / "resultats" / "pli_0.csv").write_text("a\n1\n")
	assert not modifications_non_commitees(tmp_path)
	(tmp_path / "code.py").write_text("x = 2\n")
	assert modifications_non_commitees(tmp_path)
