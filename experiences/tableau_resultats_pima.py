"""Tableau final de la phase 1 sur le TEST de Pima : classifieurs factoriels et modèles classiques.

Tout est appris sur l'apprentissage (et la validation pour les poids) ; le test sert
seulement à prédire et à mesurer.

Lancement : python -m experiences.tableau_resultats_pima
"""

from datetime import date
from pathlib import Path

import pandas as pd

from donnees.pima import charger_decoupages, charger_pima
from experiences.contextes_pima import contextes_pima
from experiences.evaluation import accuracy_decides, metriques, taux_accord
from m2_facteurs.grecon import grecon
from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import compter, construire_regles, predire
from modeles_ml.modeles_classiques import creer_modeles, entrainer_et_predire

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"
CLASSES = [0, 1]


def main() -> None:
	X, y = charger_pima()
	d = charger_decoupages()
	contextes, noms_attributs, classes, seuils = contextes_pima()
	I_app, I_val, I_test = contextes["apprentissage"], contextes["validation"], contextes["test"]
	y_app, y_val, y_test = classes["apprentissage"], classes["validation"], classes["test"]
	classe_par_defaut = int(y_app.mode()[0])

	# Modèles classiques : entraînés sur l'apprentissage seul, prédictions sur le test
	predictions_classiques = {
		"brutes": entrainer_et_predire(
			creer_modeles("brutes"), X.iloc[d["apprentissage"]], y_app, X.iloc[d["test"]]
		),
		"binarisees": entrainer_et_predire(creer_modeles("binarisees"), I_app, y_app, I_test),
	}
	references = {
		f"accord_{modele}_{donnees}": predictions_classiques[donnees][nom]
		for donnees in ["brutes", "binarisees"]
		for modele, nom in [("rf", "foret_aleatoire"), ("mlp", "mlp")]
	}

	lignes = []
	for algorithme, fonction in [("GreCon", grecon), ("GreConD", grecond)]:
		facteurs = fonction(I_app).facteurs
		for mesure in ["precision", "accuracy"]:
			for source, (I_poids, y_poids) in [("apprentissage", (I_app, y_app)), ("validation", (I_val, y_val))]:
				regles, ecartes = construire_regles(facteurs, y_app, I_poids, y_poids, mesure=mesure)
				for vote in ["all-votes", "best-vote"]:
					predictions = predire(regles, I_test, CLASSES, classe_par_defaut, vote=vote)
					y_pred = [p.classe for p in predictions]
					comptes = compter(predictions)
					ligne = {
						"methode": f"{algorithme} {vote} {mesure} {source}",
						**metriques(y_test, y_pred),
						"accuracy_decides": accuracy_decides(y_test, predictions),
						"taux_non_couverts": comptes["taux_non_couverts"],
						"taux_egalites": comptes["taux_egalites"],
						"nb_facteurs": len(facteurs),
						"nb_regles": len(regles),
					}
					for colonne, reference in references.items():
						ligne[colonne] = taux_accord(y_pred, reference)
					lignes.append(ligne)

	for donnees, predictions in predictions_classiques.items():
		for nom, y_pred in predictions.items():
			lignes.append({"methode": f"{nom} ({donnees})", **metriques(y_test, y_pred)})

	tableau = pd.DataFrame(lignes).set_index("methode").astype(
		{"predits_classe_1": "Int64", "nb_facteurs": "Int64", "nb_regles": "Int64"}
	)
	jour = date.today().isoformat()
	tableau.to_csv(DOSSIER_RESULTATS / f"pima_resultats_test_{jour}.csv")

	print(f"Test : {len(y_test)} patientes, dont {int(y_test.sum())} de classe 1 ; "
		f"classe par défaut (majoritaire en apprentissage) : {classe_par_defaut}\n")
	with pd.option_context("display.width", 250, "display.max_columns", None):
		print(tableau.round(3).to_string())


if __name__ == "__main__":
	main()
