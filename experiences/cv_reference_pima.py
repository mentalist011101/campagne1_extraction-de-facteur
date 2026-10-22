"""Étape 10 : configuration de la phase 1 en validation croisée à 10 plis (le test reste scellé).

Échelle nominale, GreConD à 100 % de couverture, all-votes / best-vote, poids précision / accuracy,
plus la règle « toujours la classe majoritaire de la partie d'apprentissage du pli ».
Tout est réappris dans chaque pli sur sa partie d'apprentissage.

Lancement : python -m experiences.cv_reference_pima
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from donnees.pima import charger_plis_cv10
from experiences.contextes_pima import contextes_pli
from experiences.evaluation import metriques
from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import compter, construire_regles, predire

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"
CLASSES = [0, 1]


def main() -> None:
	lignes = []
	for numero_pli, (indices_app, indices_eval) in enumerate(charger_plis_cv10()):
		I_app, I_eval, noms, y_app, y_eval, seuils = contextes_pli(indices_app, indices_eval)
		classe_majoritaire = int(y_app.mode()[0])
		commun = {"pli": numero_pli, "reels_classe_1": int(y_eval.sum())}

		lignes.append({
			**commun,
			"methode": "majoritaire",
			**metriques(y_eval, np.full(len(y_eval), classe_majoritaire)),
		})

		facteurs = grecond(I_app).facteurs
		for mesure in ["precision", "accuracy"]:
			regles, ecartes = construire_regles(facteurs, y_app, I_app, y_app, mesure=mesure)
			for vote in ["all-votes", "best-vote"]:
				predictions = predire(regles, I_eval, CLASSES, classe_majoritaire, vote=vote)
				comptes = compter(predictions)
				lignes.append({
					**commun,
					"methode": f"nominale {vote} {mesure}",
					**metriques(y_eval, [p.classe for p in predictions]),
					"taux_non_couverts": comptes["taux_non_couverts"],
					"taux_egalites": comptes["taux_egalites"],
					"nb_facteurs": len(facteurs),
					"nb_regles": len(regles),
				})

	par_pli = pd.DataFrame(lignes)
	jour = date.today().isoformat()
	par_pli.to_csv(DOSSIER_RESULTATS / f"pima_cv10_reference_par_pli_{jour}.csv", index=False)

	colonnes = ["accuracy", "f1_macro", "predits_classe_1", "reels_classe_1",
		"taux_non_couverts", "taux_egalites", "nb_facteurs"]
	groupes = par_pli.groupby("methode", sort=False)[colonnes]
	moyennes, ecarts = groupes.mean(), groupes.std()
	resume = moyennes.round(3).astype(str) + " ± " + ecarts.round(3).astype(str)
	resume.to_csv(DOSSIER_RESULTATS / f"pima_cv10_reference_resume_{jour}.csv")

	print("Moyenne ± écart-type sur les 10 plis\n")
	with pd.option_context("display.width", 250):
		print(resume.to_string(), "\n")
	print("Accuracy pli par pli")
	print(par_pli.pivot(index="methode", columns="pli", values="accuracy").round(3).to_string())


if __name__ == "__main__":
	main()
