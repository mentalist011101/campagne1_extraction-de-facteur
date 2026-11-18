"""Campagne 1 : méthodes d'ensemble sur les facteurs, 19 jeux, 10 plis figés par jeu.

Protocole, méthodes et hypothèses : docs/instructions_campagne1.md et DECISIONS.md (campagne 1).
Chaque pli est écrit dès qu'il est fini (resultats/campagne1/<jeu>/pli_<k>.csv) ; au redémarrage,
un pli déjà écrit est sauté. GreConD au-delà de delai_grecond_s : jeu abandonné (GreConD principal)
ou BAG abandonné pour le reste du jeu (GreConD d'un sac).

Lancement : python -m experiences.campagne1 --config experiences/campagne1.yaml [--arreter-apres N] [--assembler]
"""

import argparse
import os
import platform
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import BernoulliNB

from donnees.catalogue import charger_jeu, charger_plis, lire_variables
from m1_binarisation.binarisation import apprendre_binarisation, binariser_jeu
from m2_facteurs.grecond import DelaiDepasse, grecond
from m3_classifieurs.classifieur_factoriel import classes_par_frequence, compter, construire_regles, predire
from m3_classifieurs.empilement import empiler
from m3_classifieurs.ensachage import bagging_m3

RACINE = Path(__file__).parent.parent
METHODES = ["MAJ", "M3-AV", "M3-BV", "STACK", "LR", "RF", "NB", "BAG"]
NB_PLIS = 10
COLONNES_TRACE = [
	"commit", "modifications_non_commitees", "version_python", "version_numpy", "version_pandas",
	"version_sklearn", "processeur", "plateforme", "date_calcul",
]


@dataclass(frozen=True)
class Config:
	jeux: list[str]
	methodes: list[str]
	graine: int
	nb_sacs: int
	delai_grecond_s: float
	dossier_resultats: Path


def lire_config(chemin: Path) -> Config:
	contenu = yaml.safe_load(Path(chemin).read_text(encoding="utf-8"))
	if contenu["methodes"] != METHODES:
		raise ValueError(f"Liste fermée des méthodes : {METHODES}, reçu {contenu['methodes']}")
	dossier = Path(contenu["dossier_resultats"])
	return Config(
		jeux=contenu["jeux"], methodes=contenu["methodes"], graine=contenu["graine"], nb_sacs=contenu["nb_sacs"],
		delai_grecond_s=float(contenu["delai_grecond_s"]),
		dossier_resultats=dossier if dossier.is_absolute() else RACINE / dossier,
	)


def lancer(config: Config, arreter_apres: int | None = None) -> dict[str, int]:
	"""Calcule les plis manquants ; s'arrête après arreter_apres plis calculés (test de reprise)."""
	trace = _decrire_machine()
	comptes = {"calcules": 0, "sautes": 0}
	for nom in config.jeux:
		dossier = config.dossier_resultats / nom
		if (dossier / "abandon.csv").exists():
			print(f"{nom} : abandonné lors d'une session précédente, sauté", flush=True)
			continue
		X, y = charger_jeu(nom)
		types = lire_variables(nom).set_index("variable")["type"].to_dict()
		for k, (indices_app, indices_eval) in enumerate(charger_plis(nom, y)):
			chemin = dossier / f"pli_{k}.csv"
			if chemin.exists():
				comptes["sautes"] += 1
				print(f"{nom} pli {k} : sauté (déjà calculé)", flush=True)
				continue
			if arreter_apres is not None and comptes["calcules"] >= arreter_apres:
				return comptes
			debut = time.perf_counter()
			sans_bag = (dossier / "abandon_bag.csv").exists()
			try:
				lignes, abandon_bag = calculer_pli(nom, k, X, y, types, indices_app, indices_eval, config, sans_bag)
			except DelaiDepasse as erreur:
				_ecrire_atomique(pd.DataFrame([{
					"jeu": nom, "pli": k, "appel": "GreConD principal", "duree_s": round(erreur.duree_s, 1),
					"nb_facteurs": erreur.nb_facteurs, "raison": f"GreConD au-delà de {config.delai_grecond_s:.0f} s",
				}]), dossier / "abandon.csv")
				print(f"{nom} pli {k} : ABANDON du jeu ({erreur})", flush=True)
				break
			if abandon_bag is not None:
				_ecrire_atomique(pd.DataFrame([abandon_bag]), dossier / "abandon_bag.csv")
				print(f"{nom} pli {k} : BAG abandonné pour la suite du jeu (sac {abandon_bag['sac']})", flush=True)
			_ecrire_atomique(lignes.assign(**trace, date_calcul=datetime.now().isoformat(timespec="seconds")), chemin)
			comptes["calcules"] += 1
			print(f"{nom} pli {k} : calculé en {time.perf_counter() - debut:.1f} s "
				f"({int(lignes['nb_facteurs'].iloc[0])} facteurs)", flush=True)
	assembler(config)
	return comptes


def calculer_pli(nom, k, X, y, types, indices_app, indices_eval, config: Config, sans_bag: bool):
	"""Les 8 méthodes sur un pli ; une ligne par méthode, et l'abandon de BAG s'il a lieu.

	temps_s : pour M3-AV et M3-BV, règles et prédiction, sans le GreConD partagé (compté dans
	temps_grecond_s) ; pour STACK, régression seule ; pour BAG, ses nb_sacs GreConD compris.
	Lève DelaiDepasse si le GreConD principal dépasse config.delai_grecond_s.
	"""
	X_app, X_eval = X.iloc[indices_app], X.iloc[indices_eval]
	y_app, y_eval = y.iloc[indices_app].to_numpy(), y.iloc[indices_eval].to_numpy()
	debut = time.perf_counter()
	binarisation = apprendre_binarisation(X_app, types)
	I_app, I_eval = binariser_jeu(X_app, binarisation), binariser_jeu(X_eval, binarisation)
	temps_binarisation = time.perf_counter() - debut
	debut = time.perf_counter()
	facteurs = grecond(I_app, delai_max_s=config.delai_grecond_s).facteurs
	temps_grecond = time.perf_counter() - debut
	classes = classes_par_frequence(y_app)
	commun = {
		"jeu": nom, "pli": k, "n_app": len(y_app), "n_eval": len(y_eval), "nb_attributs": len(binarisation.noms),
		"nb_constants": binarisation.nb_constants, "nb_fusions": binarisation.nb_fusions,
		"nb_facteurs": len(facteurs),
		"part_facteurs_un_attribut": float(np.mean([f.intention.sum() == 1 for f in facteurs])),
		"temps_binarisation_s": round(temps_binarisation, 3), "temps_grecond_s": round(temps_grecond, 3),
	}
	lignes = []

	def ajouter(methode, debut_methode, y_pred, **autres):
		lignes.append({**commun, "methode": methode, "statut": "ok",
			"accuracy": float(accuracy_score(y_eval, y_pred)),
			"f1_macro": float(f1_score(y_eval, y_pred, average="macro", zero_division=0)),
			"temps_s": round(time.perf_counter() - debut_methode, 3), **autres})

	debut = time.perf_counter()
	ajouter("MAJ", debut, np.full(len(y_eval), classes[0], dtype=object))
	for methode, vote in [("M3-AV", "all-votes"), ("M3-BV", "best-vote")]:
		debut = time.perf_counter()
		regles, _ = construire_regles(facteurs, y_app, I_app, y_app, mesure="precision")
		predictions = predire(regles, I_eval, classes, classes[0], vote=vote)
		comptes = compter(predictions)
		ajouter(methode, debut, [p.classe for p in predictions],
			taux_non_couverts=comptes["taux_non_couverts"], taux_egalites=comptes["taux_egalites"])
	debut = time.perf_counter()
	with warnings.catch_warnings(record=True) as avertissements:
		warnings.simplefilter("always", ConvergenceWarning)
		y_pred, _ = empiler(facteurs, I_app, y_app, I_eval)
	ajouter("STACK", debut, y_pred, avertissements_convergence=_nb_convergence(avertissements))
	references = {
		"LR": LogisticRegression(max_iter=1000, random_state=config.graine),
		"RF": RandomForestClassifier(n_estimators=100, random_state=config.graine),
		"NB": BernoulliNB(),
	}
	for methode, modele in references.items():
		debut = time.perf_counter()
		with warnings.catch_warnings(record=True) as avertissements:
			warnings.simplefilter("always", ConvergenceWarning)
			y_pred = modele.fit(I_app, y_app).predict(I_eval)
		ajouter(methode, debut, y_pred, avertissements_convergence=_nb_convergence(avertissements))

	abandon_bag = None
	if sans_bag:
		lignes.append({**commun, "methode": "BAG", "statut": "abandonne"})
		return pd.DataFrame(lignes), abandon_bag
	debut = time.perf_counter()
	try:
		bagging = bagging_m3(I_app, y_app, I_eval, classes, classes[0], nb_sacs=config.nb_sacs,
			graine_base=config.graine, delai_max_s=config.delai_grecond_s)
	except DelaiDepasse as erreur:
		abandon_bag = {"jeu": nom, "pli": k, "sac": erreur.numero_sac, "duree_s": round(erreur.duree_s, 1)}
		lignes.append({**commun, "methode": "BAG", "statut": "abandonne", "temps_s": round(time.perf_counter() - debut, 3)})
		return pd.DataFrame(lignes), abandon_bag
	proportions = bagging.proportions
	egalites = (proportions == proportions.max(axis=1, keepdims=True)).sum(axis=1) > 1
	ajouter("BAG", debut, bagging.predictions,
		taux_non_couverts=float(np.mean(bagging.non_couverts_par_sac)) / len(y_eval),
		taux_egalites=float(egalites.mean()))
	return pd.DataFrame(lignes), abandon_bag


def assembler(config: Config) -> pd.DataFrame:
	"""campagne1_par_pli.csv (tous les plis) et campagne1_par_jeu.csv (moyennes et statut par jeu)."""
	plis = [pd.read_csv(chemin) for nom in config.jeux
		for chemin in sorted((config.dossier_resultats / nom).glob("pli_*.csv"))]
	par_pli = pd.concat(plis, ignore_index=True) if plis else pd.DataFrame(columns=["jeu", "pli", "methode", "statut"])
	config.dossier_resultats.mkdir(parents=True, exist_ok=True)
	par_pli.to_csv(config.dossier_resultats / "campagne1_par_pli.csv", index=False)
	lignes = []
	colonnes = ["accuracy", "f1_macro", "taux_non_couverts", "taux_egalites", "temps_s", "nb_attributs",
		"nb_fusions", "nb_facteurs", "part_facteurs_un_attribut", "temps_grecond_s"]
	for nom in config.jeux:
		du_jeu = par_pli[par_pli["jeu"] == nom]
		statut = _statut(config.dossier_resultats / nom, du_jeu)
		for methode in METHODES:
			ok = du_jeu[(du_jeu["methode"] == methode) & (du_jeu["statut"] == "ok")]
			ligne = {"jeu": nom, "methode": methode, "statut": statut, "nb_plis": len(ok)}
			for colonne in colonnes:
				complet = colonne in ok and len(ok) == NB_PLIS
				ligne[colonne] = ok[colonne].mean() if complet else np.nan
			lignes.append(ligne)
	par_jeu = pd.DataFrame(lignes)
	par_jeu.to_csv(config.dossier_resultats / "campagne1_par_jeu.csv", index=False)
	return par_jeu


def main() -> None:
	parseur = argparse.ArgumentParser(description="Campagne 1 : méthodes d'ensemble sur les facteurs")
	parseur.add_argument("--config", required=True, type=Path)
	parseur.add_argument("--arreter-apres", type=int, default=None, help="arrêt après N plis calculés")
	parseur.add_argument("--assembler", action="store_true", help="assembler seulement, sans calcul")
	arguments = parseur.parse_args()
	config = lire_config(arguments.config)
	if arguments.assembler:
		assembler(config)
		return
	comptes = lancer(config, arguments.arreter_apres)
	print(f"\nPlis calculés : {comptes['calcules']}, sautés : {comptes['sautes']}")


def _statut(dossier: Path, du_jeu: pd.DataFrame) -> str:
	if (dossier / "abandon.csv").exists():
		return "abandonne"
	if du_jeu["pli"].nunique() < NB_PLIS:
		return "incomplet"
	if (du_jeu["statut"] == "abandonne").any():
		return "termine_sans_bag"
	return "termine"


def modifications_non_commitees(racine: Path = RACINE) -> bool:
	"""Fichiers suivis modifiés par rapport au commit. Les fichiers non suivis (résultats écrits par la
	campagne, y compris lors d'une reprise sur Kaggle) sont ignorés ; limite : un nouveau fichier de code
	non suivi n'est pas détecté non plus."""
	return _git(["status", "--porcelain", "--untracked-files=no"], racine) not in ("", None)


def _nb_convergence(avertissements) -> int:
	return sum(issubclass(a.category, ConvergenceWarning) for a in avertissements)


def _ecrire_atomique(table: pd.DataFrame, chemin: Path) -> None:
	chemin.parent.mkdir(parents=True, exist_ok=True)
	temporaire = chemin.with_suffix(".csv.tmp")
	table.to_csv(temporaire, index=False)
	os.replace(temporaire, chemin)


def _decrire_machine() -> dict:
	return {
		"commit": _git(["rev-parse", "HEAD"]) or "inconnu",
		"modifications_non_commitees": modifications_non_commitees(),
		"version_python": platform.python_version(), "version_numpy": np.__version__,
		"version_pandas": pd.__version__, "version_sklearn": sklearn.__version__,
		"processeur": platform.processor() or platform.machine(), "plateforme": platform.platform(),
	}


def _git(arguments: list[str], racine: Path = RACINE) -> str | None:
	try:
		sortie = subprocess.run(["git", *arguments], cwd=racine, capture_output=True, text=True, timeout=10)
	except (OSError, subprocess.TimeoutExpired):
		return None
	return sortie.stdout.strip() if sortie.returncode == 0 else None


if __name__ == "__main__":
	main()
