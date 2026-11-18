"""Campagne 1, étape 7 : outils d'une session Kaggle, appelés par kaggle/campagne1.ipynb.

- decrire : vérifie les versions installées contre requirements.txt et écrit la description de la machine
  dans <dossier_resultats>/sessions/<horodatage>.txt (le nom de la machine n'y figure pas) ;
- reprendre : copie les résultats d'une session précédente sans rien écraser, après avoir vérifié qu'ils
  viennent du même commit ;
- lancer : la campagne sous `timeout` ; le code 124 (coupure à la durée maximale) est un arrêt normal, pour que
  l'assemblage et l'archive s'exécutent ensuite sous Save & Run All ; tout autre code non nul est une erreur.

Lancement : python -m experiences.session_kaggle decrire --config experiences/essai_kaggle.yaml
            python -m experiences.session_kaggle reprendre --config ... --source /kaggle/input/...
            python -m experiences.session_kaggle lancer --config ... --duree-max 11h
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from experiences.campagne1 import RACINE, Config, _decrire_machine, _git, lire_config


CODE_TIMEOUT = 124


class RepriseIncompatible(Exception):
	pass


class CampagneEchouee(Exception):
	pass


def commande_campagne(nom_config: str, duree_max: str) -> list[str]:
	return ["timeout", duree_max, sys.executable, "-m", "experiences.campagne1", "--config", nom_config]


def executer_campagne(commande: list[str]) -> bool:
	"""True si la commande a été coupée par timeout (code 124, arrêt normal), False si elle s'est terminée ;
	tout autre code non nul est une erreur."""
	code = subprocess.run(commande, cwd=RACINE).returncode
	if code == CODE_TIMEOUT:
		return True
	if code != 0:
		raise CampagneEchouee(f"campagne terminée avec le code {code}")
	return False


def comparer_versions(chemin_requirements: Path) -> dict[str, tuple[str, str | None]]:
	"""Paquet → (version attendue, version installée, None si absent)."""
	comparaison = {}
	for ligne in Path(chemin_requirements).read_text(encoding="utf-8").splitlines():
		if "==" not in ligne:
			continue
		paquet, attendue = (morceau.strip() for morceau in ligne.split("=="))
		try:
			installee = version(paquet)
		except PackageNotFoundError:
			installee = None
		comparaison[paquet] = (attendue, installee)
	return comparaison


def decrire_session(config: Config, nom_config: str, maintenant: datetime | None = None) -> Path:
	maintenant = maintenant or datetime.now()
	machine = _decrire_machine()
	cpuinfo, meminfo = _lire("/proc/cpuinfo"), _lire("/proc/meminfo")
	comparaison = comparer_versions(RACINE / "requirements.txt")
	ecarts = [f"{paquet} attendu {attendue}, installé {installee}"
		for paquet, (attendue, installee) in comparaison.items() if attendue != installee]
	champs = {
		"date": maintenant.isoformat(timespec="seconds"),
		"configuration": nom_config,
		"commit": machine["commit"],
		"modifications_non_commitees": machine["modifications_non_commitees"],
		"nb_coeurs": os.cpu_count(),
		"processeur": valeur_proc(cpuinfo, "model name") or machine["processeur"],
		"memoire_totale": valeur_proc(meminfo, "MemTotal") or "inconnue",
		"memoire_disponible": valeur_proc(meminfo, "MemAvailable") or "inconnue",
		"plateforme": machine["plateforme"],
		"python": platform.python_version(),
		**{f"version_{paquet}": installee for paquet, (_, installee) in comparaison.items()},
		"ecarts_versions": " ; ".join(ecarts) or "aucun",
	}
	chemin = config.dossier_resultats / "sessions" / f"{maintenant.strftime('%Y%m%dT%H%M%S')}.txt"
	chemin.parent.mkdir(parents=True, exist_ok=True)
	chemin.write_text("".join(f"{cle}: {valeur}\n" for cle, valeur in champs.items()), encoding="utf-8")
	return chemin


def valeur_proc(texte: str | None, cle: str) -> str | None:
	"""Première valeur de « cle : valeur » dans un fichier de /proc (cpuinfo, meminfo)."""
	for ligne in (texte or "").splitlines():
		nom, _, valeur = ligne.partition(":")
		if nom.strip() == cle:
			return valeur.strip()
	return None


def reprendre(source: Path, destination: Path, commit: str) -> tuple[int, int]:
	"""Copie les fichiers de source absents de destination ; renvoie (copiés, déjà présents).
	Les fichiers temporaires d'une écriture interrompue (*.tmp) ne sont pas copiés."""
	source, destination = Path(source), Path(destination)
	commits = {valeur for chemin in source.rglob("pli_*.csv") for valeur in pd.read_csv(chemin)["commit"].astype(str)}
	if commits - {commit}:
		raise RepriseIncompatible(f"plis de la session précédente aux commits {sorted(commits)}, commit courant {commit}")
	copies, presents = 0, 0
	for chemin in sorted(source.rglob("*")):
		if not chemin.is_file() or chemin.suffix == ".tmp":
			continue
		cible = destination / chemin.relative_to(source)
		if cible.exists():
			presents += 1
			continue
		cible.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(chemin, cible)
		copies += 1
	return copies, presents


def main() -> None:
	parseur = argparse.ArgumentParser(description="Campagne 1 : outils d'une session Kaggle")
	parseur.add_argument("action", choices=["decrire", "reprendre", "lancer"])
	parseur.add_argument("--config", required=True, type=Path)
	parseur.add_argument("--source", type=Path, help="reprendre : dossier de résultats de la session précédente")
	parseur.add_argument("--duree-max", default="11h", help="lancer : durée passée à timeout")
	arguments = parseur.parse_args()
	config = lire_config(arguments.config)
	if arguments.action == "lancer":
		try:
			coupee = executer_campagne(commande_campagne(arguments.config.as_posix(), arguments.duree_max))
		except CampagneEchouee as erreur:
			sys.exit(f"ERREUR : {erreur}")
		print(f"Arrêt à {arguments.duree_max} : reprendre dans une nouvelle session (REPRISE)." if coupee
			else "Campagne terminée.", flush=True)
		return
	if arguments.action == "decrire":
		chemin = decrire_session(config, arguments.config.as_posix())
		texte = chemin.read_text(encoding="utf-8")
		print(f"{chemin.relative_to(RACINE).as_posix()}\n{texte}", flush=True)
		if "ecarts_versions: aucun" not in texte:
			sys.exit("Versions installées différentes de requirements.txt : arrêt.")
		return
	if arguments.source is None or not arguments.source.is_dir():
		sys.exit(f"--source introuvable : {arguments.source}")
	copies, presents = reprendre(arguments.source, config.dossier_resultats, _git(["rev-parse", "HEAD"]) or "inconnu")
	print(f"Reprise : {copies} fichiers copiés, {presents} déjà présents (non écrasés)", flush=True)


def _lire(chemin: str) -> str | None:
	try:
		return Path(chemin).read_text(encoding="utf-8")
	except OSError:
		return None


if __name__ == "__main__":
	main()
