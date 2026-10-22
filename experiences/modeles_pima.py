"""Modèles classiques sur Pima : scores sur la VALIDATION seulement (le test est réservé à l'étape 9).

Lancement : python -m experiences.modeles_pima
"""

import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from donnees.pima import charger_decoupages, charger_pima
from experiences.contextes_pima import contextes_pima
from modeles_ml.modeles_classiques import creer_modeles, entrainer_et_predire


def main() -> None:
	X, y = charger_pima()
	d = charger_decoupages()
	contextes, noms_attributs, classes, seuils = contextes_pima()
	y_val = classes["validation"]

	jeux = {
		"brutes": (X.iloc[d["apprentissage"]], y.iloc[d["apprentissage"]], X.iloc[d["validation"]]),
		"binarisees": (contextes["apprentissage"], classes["apprentissage"], contextes["validation"]),
	}

	lignes = []
	for donnees, (X_app, y_app, X_val) in jeux.items():
		debut = time.perf_counter()
		predictions = entrainer_et_predire(creer_modeles(donnees), X_app, y_app, X_val)
		duree = time.perf_counter() - debut
		for nom, y_pred in predictions.items():
			lignes.append({
				"modele": nom,
				"donnees": donnees,
				"accuracy_val": round(accuracy_score(y_val, y_pred), 3),
				"f1_macro_val": round(f1_score(y_val, y_pred, average="macro"), 3),
				"predits_classe_1": int(y_pred.sum()),
			})
		print(f"Données {donnees} : 5 modèles entraînés en {duree:.1f} s")

	print(f"\nValidation : {len(y_val)} patientes, dont {int(y_val.sum())} de classe 1\n")
	print(pd.DataFrame(lignes).set_index(["donnees", "modele"]).to_string())


if __name__ == "__main__":
	main()
