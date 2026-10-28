"""Phase 3 sur congressional voting, en validation croisée à 10 plis (le test reste scellé).

Étape 16 : stacking (régression logistique sur la matrice A des facteurs), comparé au M3 de
référence (best-vote + précision), aux références classiques sur les attributs et à la règle
majoritaire. Étape 17 : bagging du M3 de référence (B = 25 sacs, B = 10 si le total prévu
dépasse 30 minutes). Dans chaque pli, tout est réappris sur sa partie d'apprentissage.

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
from m3_classifieurs.ensachage import bagging_m3
from modeles_ml.modeles_classiques import creer_modeles, entrainer_et_predire

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"
CLASSES = [0, 1]
NB_SACS_PREVU = 25
NB_SACS_REDUIT = 10
DUREE_MAX_TOTALE_S = 30 * 60


def main() -> None:
	X, y = charger_vote()
	plis = charger_plis_cv10_vote()
	nb_sacs = _choisir_nb_sacs(X, y, plis)
	lignes = []
	proportions = []
	non_couverts = []
	for numero_pli, (indices_app, indices_eval) in enumerate(plis):
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

		bagging = bagging_m3(I_app, y_app, I_eval, classe_majoritaire, nb_sacs=nb_sacs)
		lignes.append({**commun, "methode": f"M3 best-vote precision + bagging (B={nb_sacs})",
			**metriques(y_eval, bagging.predictions)})
		proportions.append(pd.DataFrame({
			"pli": numero_pli, "indice": indices_eval, "classe_reelle": y_eval.to_numpy(),
			"prediction_bagging": bagging.predictions, "proportion_sacs_classe_1": bagging.proportion_classe_1,
		}))
		non_couverts += [{"pli": numero_pli, "sac": b, "non_couverts": n, "temps_s": round(s, 2)}
			for b, (n, s) in enumerate(zip(bagging.non_couverts_par_sac, bagging.temps_par_sac))]
		print(f"Pli {numero_pli} : bagging en {sum(bagging.temps_par_sac):.1f} s", flush=True)

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
	pd.concat(proportions).to_csv(DOSSIER_RESULTATS / f"vote_cv10_bagging_proportions_{jour}.csv", index=False)
	table_non_couverts = pd.DataFrame(non_couverts)
	table_non_couverts.to_csv(DOSSIER_RESULTATS / f"vote_cv10_bagging_non_couverts_{jour}.csv", index=False)
	print("\nNon-couverts par sac (sur 34 ou 35 représentants évalués) : "
		f"moyenne {table_non_couverts['non_couverts'].mean():.2f}, maximum {table_non_couverts['non_couverts'].max()}")

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


def _choisir_nb_sacs(X, y, plis) -> int:
	"""Mesure un sac du pli 0 ; B = 10 si 25 sacs × 10 plis dépasseraient 30 minutes."""
	indices_app, indices_eval = plis[0]
	modalites = apprendre_modalites(X.iloc[indices_app])
	I_app, _ = binariser_categoriel(X.iloc[indices_app], modalites)
	I_eval, _ = binariser_categoriel(X.iloc[indices_eval], modalites)
	essai = bagging_m3(I_app, y.iloc[indices_app], I_eval, int(y.iloc[indices_app].mode()[0]), nb_sacs=1)
	prevu = essai.temps_par_sac[0] * NB_SACS_PREVU * len(plis)
	nb_sacs = NB_SACS_PREVU if prevu <= DUREE_MAX_TOTALE_S else NB_SACS_REDUIT
	print(f"Un sac : {essai.temps_par_sac[0]:.2f} s ; total prévu pour B = {NB_SACS_PREVU} : "
		f"{prevu / 60:.1f} min → B = {nb_sacs}", flush=True)
	return nb_sacs


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
