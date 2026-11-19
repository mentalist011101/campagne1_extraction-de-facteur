"""Campagne 1, étape 8 : vérifications descriptives. Aucun verdict ne change ; experiences/analyse_campagne1.py
n'est pas touché.

a. M3-AV et M3-BV recalculés à l'identique de experiences/campagne1.py (calculer_pli : binarisation du pli,
   GreConD principal, règles pondérées par la précision sur l'apprentissage) sur les jeux où M3 égale MAJ :
   classes prédites contre classes réelles, part des règles dont la classe est la classe dominante de
   l'apprentissage, et contrôle (accuracy et F1 recalculées égales à celles des fichiers pli_k.csv de Kaggle).
b. (sans calcul) LR, RF et NB sont entraînés sur I_app, la matrice binarisée du pli, entrée du GreConD
   principal ; STACK seul est entraîné sur la matrice des facteurs (calculer_pli).
c. Essai local (Windows) contre campagne (Kaggle) : F1 et accuracy, pli par pli, d'iris, wine et zoo.

Lancement : python -m experiences.descriptif_campagne1
Sorties : resultats/campagne1/descriptif/
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from donnees.catalogue import charger_jeu, charger_plis, lire_variables
from experiences.campagne1 import RACINE
from m1_binarisation.binarisation import apprendre_binarisation, binariser_jeu
from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import classes_par_frequence, construire_regles, predire

JEUX_EGAUX_MAJ = ["breast-cancer", "hepatitis", "anneal", "page-blocks", "nursery"]
JEUX_ESSAI = ["iris", "wine", "zoo"]
VOTES = {"M3-AV": "all-votes", "M3-BV": "best-vote"}
DOSSIER_CAMPAGNE = RACINE / "resultats" / "campagne1"
DOSSIER_ESSAI = RACINE / "resultats" / "essai_local"
DOSSIER_SORTIE = DOSSIER_CAMPAGNE / "descriptif"


def recalculer_pli(nom: str, k: int) -> tuple[pd.DataFrame, dict]:
	"""Prédictions de M3-AV et M3-BV sur le pli k (une ligne par objet et méthode), et résumé des règles."""
	X, y = charger_jeu(nom)
	types = lire_variables(nom).set_index("variable")["type"].to_dict()
	indices_app, indices_eval = charger_plis(nom, y)[k]
	X_app, X_eval = X.iloc[indices_app], X.iloc[indices_eval]
	y_app, y_eval = y.iloc[indices_app].to_numpy(), y.iloc[indices_eval].to_numpy()
	binarisation = apprendre_binarisation(X_app, types)
	I_app, I_eval = binariser_jeu(X_app, binarisation), binariser_jeu(X_eval, binarisation)
	facteurs = grecond(I_app).facteurs
	classes = classes_par_frequence(y_app)
	regles, ecartes = construire_regles(facteurs, y_app, I_app, y_app, mesure="precision")
	morceaux = []
	for methode, vote in VOTES.items():
		predites = [p.classe for p in predire(regles, I_eval, classes, classes[0], vote=vote)]
		morceaux.append(pd.DataFrame({"jeu": nom, "pli": k, "methode": methode, "reelle": y_eval, "predite": predites}))
	resume = resumer_regles(regles, ecartes, classes[0], nb_facteurs=len(facteurs))
	return pd.concat(morceaux, ignore_index=True), resume


def resumer_regles(regles, ecartes: dict, classe_dominante, nb_facteurs: int) -> dict:
	nb_dominante = sum(regle.classe == classe_dominante for regle in regles)
	return {"nb_facteurs": nb_facteurs, "nb_regles": len(regles), "nb_ecartes": len(ecartes),
		"classe_dominante": classe_dominante, "nb_regles_dominante": nb_dominante,
		"part_regles_dominante": nb_dominante / len(regles) if regles else float("nan")}


def classes_predites(predictions: pd.DataFrame) -> pd.DataFrame:
	"""Par jeu, méthode et classe : nombres prédit et réel, sommés sur les plis, et leurs parts."""
	cles = ["jeu", "methode"]
	predit = predictions.groupby([*cles, "predite"]).size().rename_axis(index={"predite": "classe"})
	reel = predictions.groupby([*cles, "reelle"]).size().rename_axis(index={"reelle": "classe"})
	table = (pd.concat({"n_predit": predit, "n_reel": reel}, axis=1).fillna(0).astype(int).reset_index()
		.sort_values([*cles, "classe"], key=lambda colonne: colonne.astype(str), ignore_index=True))
	totaux = table.groupby(cles)[["n_predit", "n_reel"]].transform("sum")
	return table.assign(part_predite=table["n_predit"] / totaux["n_predit"], part_reelle=table["n_reel"] / totaux["n_reel"])


def controler(predictions: pd.DataFrame, stockees: pd.DataFrame) -> pd.DataFrame:
	"""Accuracy et F1 macro recalculées (comme calculer_pli) contre les valeurs stockées, à l'égalité exacte."""
	lignes = []
	for (jeu, pli, methode), groupe in predictions.groupby(["jeu", "pli", "methode"], sort=False):
		reelles, predites = groupe["reelle"].to_numpy(), groupe["predite"].to_numpy()
		lignes.append({"jeu": jeu, "pli": pli, "methode": methode,
			"accuracy_recalculee": float(accuracy_score(reelles, predites)),
			"f1_recalculee": float(f1_score(reelles, predites, average="macro", zero_division=0))})
	table = pd.DataFrame(lignes).merge(stockees[["jeu", "pli", "methode", "accuracy", "f1_macro"]],
		on=["jeu", "pli", "methode"], how="left")
	return table.assign(
		accuracy_egale=table["accuracy_recalculee"] == table["accuracy"],
		f1_egale=table["f1_recalculee"] == table["f1_macro"],
		ecart_accuracy=table["accuracy_recalculee"] - table["accuracy"],
		ecart_f1=table["f1_recalculee"] - table["f1_macro"])


def comparer_plis(essai: pd.DataFrame, campagne: pd.DataFrame, jeux: list[str]) -> pd.DataFrame:
	"""F1 et accuracy pli par pli : essai local contre campagne, à l'égalité exacte."""
	colonnes = ["jeu", "pli", "methode", "f1_macro", "accuracy"]
	table = essai.loc[essai["jeu"].isin(jeux), colonnes].merge(
		campagne.loc[campagne["jeu"].isin(jeux), colonnes], on=["jeu", "pli", "methode"], how="outer",
		suffixes=("_essai", "_campagne"), validate="one_to_one")
	return table.assign(
		f1_egale=table["f1_macro_essai"] == table["f1_macro_campagne"],
		accuracy_egale=table["accuracy_essai"] == table["accuracy_campagne"],
		ecart_f1=table["f1_macro_campagne"] - table["f1_macro_essai"],
		ecart_accuracy=table["accuracy_campagne"] - table["accuracy_essai"])


def lire_csv(chemin) -> pd.DataFrame:
	"""Le lecteur par défaut de pandas peut s'écarter d'un ulp du flottant écrit (14/15 relu 0.9333333333333332) :
	les comparaisons exactes exigent la relecture exacte."""
	return pd.read_csv(chemin, float_precision="round_trip")


def lire_plis(dossier, jeux: list[str]) -> pd.DataFrame:
	"""Les fichiers pli_k.csv, valeurs exactes. campagne1_par_pli.csv ne convient pas aux comparaisons exactes :
	assembler relit les plis avec le lecteur par défaut, qui peut décaler une valeur d'un ulp."""
	return pd.concat([lire_csv(chemin) for nom in jeux for chemin in sorted((dossier / nom).glob("pli_*.csv"))],
		ignore_index=True)


def main() -> None:
	par_pli = lire_plis(DOSSIER_CAMPAGNE, [*JEUX_EGAUX_MAJ, *JEUX_ESSAI])
	predictions, resumes = [], []
	for nom in JEUX_EGAUX_MAJ:
		for k in range(10):
			du_pli, resume = recalculer_pli(nom, k)
			predictions.append(du_pli)
			resumes.append({"jeu": nom, "pli": k, **resume})
		print(f"{nom} : 10 plis recalculés", flush=True)
	predictions = pd.concat(predictions, ignore_index=True)
	tables = {
		"classes_predites": classes_predites(predictions),
		"regles_principales": pd.DataFrame(resumes),
		"controle_m3": controler(predictions, par_pli),
		"essai_local_contre_kaggle": comparer_plis(lire_plis(DOSSIER_ESSAI, JEUX_ESSAI), par_pli, JEUX_ESSAI),
	}
	DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
	for nom, table in tables.items():
		table.to_csv(DOSSIER_SORTIE / f"{nom}.csv", index=False, lineterminator="\n")
	with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_rows", 500):
		for nom, table in tables.items():
			print(f"\n{nom}\n{table.round(4).to_string(index=False)}")


if __name__ == "__main__":
	main()
