"""Étape 14 : M3 sur mushroom en validation croisée à 10 plis (le test reste scellé).

Dans chaque pli : modalités apprises sur la partie d'apprentissage (attributs constants
retirés), GreConD à 100 % de couverture (temps mesuré), puis all-votes / best-vote avec
les poids précision, accuracy et lift, et la règle majoritaire.
Références classiques sur les mêmes attributs binaires : régression logistique,
forêt aléatoire, Naive Bayes de Bernoulli (hyperparamètres par défaut, random_state=42).
Critère H4 (figé) : accuracy moyenne ≥ 0,90 avec best-vote + précision.

Lancement : python -m experiences.cv_mushroom
"""

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

from donnees.mushroom import charger_mushroom, charger_plis_cv10_mushroom
from experiences.evaluation import metriques
from m1_binarisation.binarisation import apprendre_modalites, binariser_categoriel
from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import compter, construire_regles, predire
from modeles_ml.modeles_classiques import creer_modeles, entrainer_et_predire

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"
CLASSES = [0, 1]
DUREE_MAX_PAR_PLI_S = 30 * 60


def main() -> None:
	X, y = charger_mushroom()
	lignes = []
	plis = []
	for numero_pli, (indices_app, indices_eval) in enumerate(charger_plis_cv10_mushroom()):
		X_app, X_eval = X.iloc[indices_app], X.iloc[indices_eval]
		y_app, y_eval = y.iloc[indices_app], y.iloc[indices_eval]
		modalites = apprendre_modalites(X_app)
		I_app, noms = binariser_categoriel(X_app, modalites)
		I_eval, _ = binariser_categoriel(X_eval, modalites)
		classe_majoritaire = int(y_app.mode()[0])

		debut = time.perf_counter()
		facteurs = grecond(I_app).facteurs
		temps_grecond = time.perf_counter() - debut
		longueurs = np.array([int(f.intention.sum()) for f in facteurs])
		plis.append({
			"pli": numero_pli,
			"attributs": I_app.shape[1],
			"densite": round(float(I_app.mean()), 4),
			"facteurs": len(facteurs),
			"part_un_attribut": round(float((longueurs == 1).mean()), 3),
			"longueur_moyenne": round(float(longueurs.mean()), 2),
			"temps_grecond_s": round(temps_grecond, 1),
		})
		print(f"Pli {numero_pli} : {I_app.shape[1]} attributs, {len(facteurs)} facteurs, "
			f"GreConD en {temps_grecond:.1f} s", flush=True)

		commun = {"pli": numero_pli, "reels_classe_1": int(y_eval.sum())}
		lignes.append({**commun, "methode": "majoritaire",
			**metriques(y_eval, np.full(len(y_eval), classe_majoritaire))})
		for mesure in ["precision", "accuracy", "lift"]:
			regles, ecartes = construire_regles(facteurs, y_app, I_app, y_app, mesure=mesure)
			for vote in ["all-votes", "best-vote"]:
				predictions = predire(regles, I_eval, CLASSES, classe_majoritaire, vote=vote)
				comptes = compter(predictions)
				lignes.append({
					**commun,
					"methode": f"{vote} {mesure}",
					**metriques(y_eval, [p.classe for p in predictions]),
					"taux_non_couverts": comptes["taux_non_couverts"],
					"taux_egalites": comptes["taux_egalites"],
					"nb_regles": len(regles),
					"nb_ecartes": len(ecartes),
				})

		references = {
			"naive_bayes_bernoulli": Pipeline([("modele", BernoulliNB())]),
			**{nom: m for nom, m in creer_modeles("binarisees").items()
				if nom in ("regression_logistique", "foret_aleatoire")},
		}
		for nom, y_pred in entrainer_et_predire(references, I_app, y_app, I_eval).items():
			lignes.append({**commun, "methode": nom, **metriques(y_eval, y_pred)})

		if temps_grecond > DUREE_MAX_PAR_PLI_S:
			print(f"ARRÊT : GreConD a dépassé 30 minutes au pli {numero_pli}.")
			break

	jour = date.today().isoformat()
	par_pli = pd.DataFrame(lignes)
	par_pli.to_csv(DOSSIER_RESULTATS / f"mushroom_cv10_par_pli_{jour}.csv", index=False)
	table_plis = pd.DataFrame(plis).set_index("pli")
	table_plis.to_csv(DOSSIER_RESULTATS / f"mushroom_cv10_facteurs_{jour}.csv")

	colonnes = ["accuracy", "f1_macro", "predits_classe_1", "reels_classe_1",
		"taux_non_couverts", "taux_egalites", "nb_regles", "nb_ecartes"]
	groupes = par_pli.groupby("methode", sort=False)[colonnes]
	resume = groupes.mean().round(3).astype(str) + " ± " + groupes.std().round(3).astype(str)
	resume.to_csv(DOSSIER_RESULTATS / f"mushroom_cv10_resume_{jour}.csv")

	print("\nFacteurs par pli")
	print(table_plis.to_string())
	print("\nMoyenne ± écart-type sur les plis")
	with pd.option_context("display.width", 250):
		print(resume.to_string())
	accuracy_h4 = par_pli.loc[par_pli["methode"] == "best-vote precision", "accuracy"].mean()
	print(f"\nH4 : accuracy moyenne best-vote + précision = {accuracy_h4:.4f} "
		f"(critère ≥ 0,90 : {'atteint' if accuracy_h4 >= 0.90 else 'non atteint'})")


if __name__ == "__main__":
	main()
