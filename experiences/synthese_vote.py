"""Étape 18 : synthèse de la phase 3 sur congressional voting (test scellé).

Relit les résultats pli par pli des étapes 16 et 17, puis donne le tableau final, les
verdicts de H5, H6 et H7 selon les critères figés (DECISIONS.md), D1, et, à titre
descriptif, le test du signe bilatéral (égalités exclues). Reprend les 10 facteurs du
stacking du pli 0. Tout est exporté dans un seul CSV pour les diapositives.

Lancement : python -m experiences.synthese_vote
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from donnees.vote import charger_plis_cv10_vote, charger_vote
from m1_binarisation.binarisation import apprendre_modalites, binariser_categoriel
from m2_facteurs.grecond import grecond
from m3_classifieurs.empilement import empiler

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"
JOUR = "2026-10-28"
M3 = "M3 best-vote precision"
BAGGING = "M3 best-vote precision + bagging (B=25)"
STACKING = "stacking"
REFERENCES = ["regression_logistique", "foret_aleatoire", "naive_bayes_bernoulli"]
ORDRE_TABLEAU = ["majoritaire", M3, BAGGING, STACKING, *REFERENCES]
MARGE_H7 = 0.02


def main() -> None:
	par_pli = pd.read_csv(DOSSIER_RESULTATS / f"vote_cv10_ensembles_par_pli_{JOUR}.csv")
	f1 = par_pli.pivot(index="pli", columns="methode", values="f1_macro")

	tableau = _tableau_final(par_pli)
	comparaisons = [
		_comparer_pli_par_pli("H5", "bagging contre M3 seul", f1[BAGGING], f1[M3], critere=True),
		_comparer_pli_par_pli("H6", "stacking contre M3 seul", f1[STACKING], f1[M3], critere=True),
		_comparer_pli_par_pli("D1", "stacking contre régression logistique (descriptif)",
			f1[STACKING], f1["regression_logistique"], critere=False),
	]
	non_inferiorite = [_non_inferiorite("H7", nom, f1) for nom in [STACKING, BAGGING]]

	synthese = pd.concat([
		tableau.reset_index().assign(section="tableau final"),
		pd.DataFrame(comparaisons).assign(section="comparaison pli par pli"),
		pd.DataFrame(non_inferiorite).assign(section="non-infériorité"),
	], ignore_index=True)
	colonnes = ["section"] + [c for c in synthese.columns if c != "section"]
	synthese[colonnes].to_csv(DOSSIER_RESULTATS / f"vote_synthese_{JOUR}.csv", index=False)

	with pd.option_context("display.width", 250, "display.max_columns", None):
		print("=== Tableau final (moyenne et écart-type sur les 10 plis) ===")
		print(tableau.round(3).to_string(), "\n")
		print("=== H5, H6 (critère : au moins 8 victoires strictes) et D1, sur les F1 macro exacts ===")
		print(pd.DataFrame(comparaisons).set_index("hypothese").to_string(), "\n")
		print("=== H7 : F1 moyen ≥ meilleure référence − 0,02 (valeurs à 4 décimales) ===")
		print(pd.DataFrame(non_inferiorite).set_index("methode").to_string(), "\n")
		print("=== Écarts de F1 macro pli par pli ===")
		ecarts = pd.DataFrame({
			"H5 bagging − M3": f1[BAGGING] - f1[M3],
			"H6 stacking − M3": f1[STACKING] - f1[M3],
			"D1 stacking − RL": f1[STACKING] - f1["regression_logistique"],
		}).T
		print(ecarts.round(4).to_string(), "\n")
	_facteurs_du_stacking_pli_0()


def _tableau_final(par_pli: pd.DataFrame) -> pd.DataFrame:
	groupes = par_pli.groupby("methode")
	tableau = pd.DataFrame({
		"accuracy_moyenne": groupes["accuracy"].mean(),
		"accuracy_ecart_type": groupes["accuracy"].std(),
		"f1_macro_moyen": groupes["f1_macro"].mean(),
		"f1_macro_ecart_type": groupes["f1_macro"].std(),
		"predits_classe_1_moyen": groupes["predits_classe_1"].mean(),
		"reels_classe_1_moyen": groupes["reels_classe_1"].mean(),
	})
	tableau.index.name = "methode"
	return tableau.loc[ORDRE_TABLEAU]


def _comparer_pli_par_pli(hypothese: str, description: str, a: pd.Series, b: pd.Series, critere: bool) -> dict:
	"""Victoires strictes de a sur b, défaites, égalités exactes, test du signe bilatéral."""
	ecart = a - b
	victoires, defaites = int((ecart > 0).sum()), int((ecart < 0).sum())
	egalites = int((ecart == 0).sum())
	p_valeur = binomtest(victoires, victoires + defaites, 0.5).pvalue if victoires + defaites else np.nan
	ligne = {
		"hypothese": hypothese,
		"comparaison": description,
		"victoires": victoires,
		"defaites": defaites,
		"egalites": egalites,
		"p_valeur_signe": round(float(p_valeur), 4),
		"ecart_moyen_f1": round(float(ecart.mean()), 4),
	}
	if critere:
		ligne["verdict"] = "confirmée" if victoires >= 8 else "rejetée"
	return ligne


def _non_inferiorite(hypothese: str, methode: str, f1: pd.DataFrame) -> dict:
	moyennes_references = f1[REFERENCES].mean()
	meilleure = moyennes_references.idxmax()
	seuil = moyennes_references.max() - MARGE_H7
	f1_moyen = f1[methode].mean()
	return {
		"hypothese": hypothese,
		"methode": methode,
		"f1_macro_moyen": round(float(f1_moyen), 4),
		"meilleure_reference": meilleure,
		"f1_meilleure_reference": round(float(moyennes_references.max()), 4),
		"seuil": round(float(seuil), 4),
		"verdict": "confirmée" if f1_moyen >= seuil else "rejetée",
	}


def _facteurs_du_stacking_pli_0() -> None:
	"""Les 10 facteurs du stacking du pli 0 (étape 16), recalculés à l'identique."""
	X, y = charger_vote()
	indices_app, indices_eval = charger_plis_cv10_vote()[0]
	X_app, y_app = X.iloc[indices_app], y.iloc[indices_app].to_numpy()
	modalites = apprendre_modalites(X_app)
	I_app, noms = binariser_categoriel(X_app, modalites)
	I_eval, _ = binariser_categoriel(X.iloc[indices_eval], modalites)
	facteurs = grecond(I_app).facteurs
	y_pred, modele = empiler(facteurs, I_app, y_app, I_eval)
	coefficients = modele.coef_[0]
	ordre = np.argsort(coefficients)
	lignes = []
	for sens, indices in [("républicain (+)", ordre[::-1][:5]), ("démocrate (−)", ordre[:5])]:
		for f in indices:
			lignes.append({
				"sens": sens,
				"facteur": f"F{f + 1}",
				"coefficient": round(float(coefficients[f]), 3),
				"taille_extension": int(facteurs[f].extension.sum()),
				"proportion_republicains": round(float(y_app[facteurs[f].extension].mean()), 3),
				"intention": " ET ".join(n for n, g in zip(noms, facteurs[f].intention) if g),
			})
	table = pd.DataFrame(lignes).set_index("facteur")
	table.to_csv(DOSSIER_RESULTATS / f"vote_stacking_facteurs_pli0_{JOUR}.csv")
	print("=== Pli 0 : 10 facteurs du stacking ===")
	print("Coefficients individuels peu fiables si les facteurs se recouvrent : "
		"illustration de la lisibilité, pas classement d'importance.")
	with pd.option_context("display.width", 300, "display.max_colwidth", 200):
		print(table.to_string())


if __name__ == "__main__":
	main()
