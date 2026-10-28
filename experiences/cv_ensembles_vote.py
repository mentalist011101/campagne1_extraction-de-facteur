"""Phase 3 sur congressional voting, en validation croisée à 10 plis (le test reste scellé).

Étape 16 : stacking (régression logistique sur la matrice A des facteurs), comparé au M3 de
référence (best-vote + précision), aux références classiques sur les attributs et à la règle
majoritaire. Dans chaque pli, tout est réappris sur sa partie d'apprentissage.

Lancement : python -m experiences.cv_ensembles_vote
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

from donnees.vote import charger_plis_cv10_vote, charger_vote
from experiences.evaluation import metriques
from m1_binarisation.binarisation import apprendre_modalites, binariser_categoriel
from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import compter, construire_regles, predire
from m3_classifieurs.empilement import empiler
from modeles_ml.modeles_classiques import creer_modeles, entrainer_et_predire

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"
CLASSES = [0, 1]


def main() -> None:
	X, y = charger_vote()
	lignes = []
	for numero_pli, (indices_app, indices_eval) in enumerate(charger_plis_cv10_vote()):
		X_app, X_eval = X.iloc[indices_app], X.iloc[indices_eval]
		y_app, y_eval = y.iloc[indices_app], y.iloc[indices_eval]
		modalites = apprendre_modalites(X_app)
		I_app, noms = binariser_categoriel(X_app, modalites)
		I_eval, _ = binariser_categoriel(X_eval, modalites)
		classe_majoritaire = int(y_app.mode()[0])
		facteurs = grecond(I_app).facteurs
		commun = {"pli": numero_pli, "reels_classe_1": int(y_eval.sum()), "nb_facteurs": len(facteurs)}

		lignes.append({**commun, "methode": "majoritaire",
			**metriques(y_eval, np.full(len(y_eval), classe_majoritaire))})

		regles, ecartes = construire_regles(facteurs, y_app, I_app, y_app, mesure="precision")
		predictions = predire(regles, I_eval, CLASSES, classe_majoritaire, vote="best-vote")
		comptes = compter(predictions)
		lignes.append({**commun, "methode": "M3 best-vote precision",
			**metriques(y_eval, [p.classe for p in predictions]),
			"taux_non_couverts": comptes["taux_non_couverts"], "taux_egalites": comptes["taux_egalites"]})

		y_pred_stacking, modele_stacking = empiler(facteurs, I_app, y_app, I_eval)
		lignes.append({**commun, "methode": "stacking", **metriques(y_eval, y_pred_stacking)})
		if numero_pli == 0:
			_afficher_facteurs_du_stacking(modele_stacking, facteurs, noms, y_app.to_numpy())

		references = {
			"naive_bayes_bernoulli": Pipeline([("modele", BernoulliNB())]),
			**{nom: m for nom, m in creer_modeles("binarisees").items()
				if nom in ("regression_logistique", "foret_aleatoire")},
		}
		for nom, y_pred in entrainer_et_predire(references, I_app, y_app, I_eval).items():
			lignes.append({**commun, "methode": nom, **metriques(y_eval, y_pred)})

	par_pli = pd.DataFrame(lignes)
	jour = date.today().isoformat()
	par_pli.to_csv(DOSSIER_RESULTATS / f"vote_cv10_ensembles_par_pli_{jour}.csv", index=False)

	colonnes = ["accuracy", "f1_macro", "predits_classe_1", "reels_classe_1",
		"taux_non_couverts", "taux_egalites", "nb_facteurs"]
	groupes = par_pli.groupby("methode", sort=False)[colonnes]
	resume = groupes.mean().round(3).astype(str) + " ± " + groupes.std().round(3).astype(str)
	resume.to_csv(DOSSIER_RESULTATS / f"vote_cv10_ensembles_resume_{jour}.csv")
	print("\nMoyenne ± écart-type sur les 10 plis")
	with pd.option_context("display.width", 250):
		print(resume.to_string())
	print("\nF1 macro pli par pli")
	print(par_pli.pivot(index="methode", columns="pli", values="f1_macro").round(3).to_string())


def _afficher_facteurs_du_stacking(modele, facteurs, noms_attributs, y_app: np.ndarray) -> None:
	"""Pli 0 : les 5 coefficients les plus positifs (républicain) et les 5 plus négatifs (démocrate)."""
	coefficients = modele.coef_[0]
	ordre = np.argsort(coefficients)
	lignes = []
	for sens, indices in [("républicain (+)", ordre[::-1][:5]), ("démocrate (−)", ordre[:5])]:
		for f in indices:
			facteur = facteurs[f]
			lignes.append({
				"sens": sens,
				"facteur": f"F{f + 1}",
				"coefficient": round(float(coefficients[f]), 3),
				"taille_extension": int(facteur.extension.sum()),
				"proportion_republicains": round(float(y_app[facteur.extension].mean()), 3),
				"intention": " ET ".join(n for n, g in zip(noms_attributs, facteur.intention) if g),
			})
	print("Pli 0 : 10 facteurs du stacking (extension et proportion sur l'apprentissage du pli)")
	print("Coefficients individuels peu fiables si les facteurs se recouvrent : "
		"illustration de la lisibilité, pas classement d'importance.")
	with pd.option_context("display.width", 300, "display.max_colwidth", 200):
		print(pd.DataFrame(lignes).set_index("facteur").to_string())


if __name__ == "__main__":
	main()
