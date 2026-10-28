"""Étape 13 : grille M3 en validation croisée à 10 plis, références, critères de H2 et H3.

Dans chaque pli, tout est réappris sur sa partie d'apprentissage : seuils, GreConD
(100 % de couverture), classes et poids des facteurs, modèles. Le test reste scellé.
Critères figés : voir DECISIONS.md (2026-10-22, hypothèses de la phase 2).

Lancement : python -m experiences.cv_grille_pima
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

from donnees.pima import charger_pima, charger_plis_cv10
from experiences.contextes_pima import contextes_pli
from experiences.evaluation import metriques
from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import compter, construire_regles, predire, regles_qui_votent
from modeles_ml.modeles_classiques import creer_modeles, entrainer_et_predire

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"
CLASSES = [0, 1]
ECHELLES = {"nominale": 3, "interordinale": 4}   # attributs par variable
VOTES = ["all-votes", "best-vote"]
MESURES = ["precision", "accuracy", "lift"]


def main() -> None:
	X, y = charger_pima()
	lignes = []
	for numero_pli, (indices_app, indices_eval) in enumerate(charger_plis_cv10()):
		y_app, y_eval = y.iloc[indices_app], y.iloc[indices_eval]
		classe_majoritaire = int(y_app.mode()[0])
		commun = {"pli": numero_pli, "reels_classe_1": int(y_eval.sum())}
		lignes.append({**commun, "methode": "majoritaire",
			**metriques(y_eval, np.full(len(y_eval), classe_majoritaire))})

		for echelle, attributs_par_variable in ECHELLES.items():
			I_app, I_eval, noms, y_app, y_eval, seuils = contextes_pli(indices_app, indices_eval, echelle)
			facteurs = grecond(I_app).facteurs
			for mesure in MESURES:
				regles, ecartes = construire_regles(facteurs, y_app, I_app, y_app, mesure=mesure)
				for vote in VOTES:
					predictions = predire(regles, I_eval, CLASSES, classe_majoritaire, vote=vote)
					comptes = compter(predictions)
					lignes.append({
						**commun,
						"methode": f"{echelle} {vote} {mesure}",
						"echelle": echelle, "vote": vote, "mesure": mesure,
						**metriques(y_eval, [p.classe for p in predictions]),
						"taux_non_couverts": comptes["taux_non_couverts"],
						"taux_egalites": comptes["taux_egalites"],
						"nb_facteurs": len(facteurs),
						**_structure_des_votants(predictions, vote, attributs_par_variable),
					})

			references = {
				"naive_bayes_bernoulli": Pipeline([("modele", BernoulliNB())]),
				**{nom: m for nom, m in creer_modeles("binarisees").items()
					if nom in ("regression_logistique", "foret_aleatoire")},
			}
			for nom, y_pred in entrainer_et_predire(references, I_app, y_app, I_eval).items():
				lignes.append({**commun, "methode": f"{nom} ({echelle})", **metriques(y_eval, y_pred)})

		bruts = {nom: m for nom, m in creer_modeles("brutes").items()
			if nom in ("regression_logistique", "foret_aleatoire")}
		for nom, y_pred in entrainer_et_predire(bruts, X.iloc[indices_app], y_app, X.iloc[indices_eval]).items():
			lignes.append({**commun, "methode": f"{nom} (brutes)", **metriques(y_eval, y_pred)})

	par_pli = pd.DataFrame(lignes)
	jour = date.today().isoformat()
	par_pli.to_csv(DOSSIER_RESULTATS / f"pima_cv10_grille_par_pli_{jour}.csv", index=False)

	colonnes = ["accuracy", "f1_macro", "predits_classe_1", "reels_classe_1", "taux_non_couverts",
		"taux_egalites", "nb_facteurs", "longueur_votants", "variables_votants"]
	groupes = par_pli.groupby("methode", sort=False)[colonnes]
	resume = groupes.mean().round(3).astype(str) + " ± " + groupes.std().round(3).astype(str)
	resume.to_csv(DOSSIER_RESULTATS / f"pima_cv10_grille_resume_{jour}.csv")
	with pd.option_context("display.width", 300, "display.max_columns", None):
		print("Moyenne ± écart-type sur les 10 plis\n")
		print(resume.to_string(), "\n")

	h2 = _tableau_h2(par_pli)
	h2.to_csv(DOSSIER_RESULTATS / f"pima_cv10_h2_{jour}.csv")
	print("H2 : F1 macro interordinale − nominale, pli par pli (victoire = strictement > 0)")
	print(h2.round(3).to_string(), "\n")

	h3 = _tableau_h3(par_pli)
	h3.to_csv(DOSSIER_RESULTATS / f"pima_cv10_h3_{jour}.csv")
	print("H3 : lift contre précision, pour chaque échelle × vote")
	print(h3.round(3).to_string())


def _structure_des_votants(predictions, vote: str, attributs_par_variable: int) -> dict:
	"""Longueur et nombre de variables distinctes des règles qui votent, moyennés sur les couples (x, règle)."""
	longueurs, variables = [], []
	for prediction in predictions:
		for regle in regles_qui_votent(prediction, vote):
			indices = np.flatnonzero(regle.intention)
			longueurs.append(len(indices))
			variables.append(len({j // attributs_par_variable for j in indices}))
	return {
		"longueur_votants": float(np.mean(longueurs)) if longueurs else np.nan,
		"variables_votants": float(np.mean(variables)) if variables else np.nan,
	}


def _tableau_h2(par_pli: pd.DataFrame) -> pd.DataFrame:
	"""Écart de F1 macro (interordinale − nominale) par pli, pour chaque vote × mesure ; victoires strictes."""
	lignes = {}
	for vote in VOTES:
		for mesure in MESURES:
			f1 = {}
			for echelle in ECHELLES:
				choix = par_pli[(par_pli["echelle"] == echelle) & (par_pli["vote"] == vote) & (par_pli["mesure"] == mesure)]
				f1[echelle] = choix.set_index("pli")["f1_macro"]
			ecart = f1["interordinale"] - f1["nominale"]
			principale = " (principale)" if (vote, mesure) == ("best-vote", "precision") else ""
			lignes[f"{vote} {mesure}{principale}"] = {
				**{f"pli_{p}": v for p, v in ecart.items()},
				"victoires": int((ecart > 0).sum()),
			}
	return pd.DataFrame(lignes).T


def _tableau_h3(par_pli: pd.DataFrame) -> pd.DataFrame:
	"""Critères (i) écart moyen |prédits 1 − réels| lift < précision et (ii) accuracy lift ≥ majoritaire."""
	accuracy_majoritaire = par_pli.loc[par_pli["methode"] == "majoritaire", "accuracy"].mean()
	lignes = {}
	for echelle in ECHELLES:
		for vote in VOTES:
			mesures = {}
			for mesure in ["precision", "lift"]:
				choix = par_pli[(par_pli["echelle"] == echelle) & (par_pli["vote"] == vote) & (par_pli["mesure"] == mesure)]
				mesures[mesure] = {
					"ecart": (choix["predits_classe_1"] - choix["reels_classe_1"]).abs().mean(),
					"accuracy": choix["accuracy"].mean(),
				}
			critere_i = mesures["lift"]["ecart"] < mesures["precision"]["ecart"]
			critere_ii = mesures["lift"]["accuracy"] >= accuracy_majoritaire
			lignes[f"{echelle} {vote}"] = {
				"ecart_precision": mesures["precision"]["ecart"],
				"ecart_lift": mesures["lift"]["ecart"],
				"critere_i": critere_i,
				"accuracy_lift": mesures["lift"]["accuracy"],
				"accuracy_majoritaire": accuracy_majoritaire,
				"critere_ii": critere_ii,
				"les_deux": critere_i and critere_ii,
			}
	return pd.DataFrame(lignes).T


if __name__ == "__main__":
	main()
