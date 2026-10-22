"""M3 sur Pima : tableau des facteurs vus comme règles, pour GreCon et GreConD.

Facteurs appris sur l'apprentissage (sans la classe), classes sur l'apprentissage,
précision et accuracy sur l'apprentissage et la validation. Le test n'est pas utilisé.

Lancement : python -m experiences.regles_pima
"""

from datetime import date
from pathlib import Path

import pandas as pd

from experiences.contextes_pima import contextes_pima
from m2_facteurs.grecon import grecon
from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import RAISON_EGALITE, RAISON_NON_COUVERT, decrire_facteurs

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"


def main() -> None:
	contextes, noms_attributs, classes, seuils = contextes_pima()
	I_app, y_app = contextes["apprentissage"], classes["apprentissage"]
	I_val, y_val = contextes["validation"], classes["validation"]
	jour = date.today().isoformat()

	bilan = []
	for nom, algorithme in [("GreCon", grecon), ("GreConD", grecond)]:
		facteurs = algorithme(I_app).facteurs
		table = decrire_facteurs(facteurs, noms_attributs, y_app, I_val, y_val)
		table.to_csv(DOSSIER_RESULTATS / f"pima_m3_regles_{nom.lower()}_{jour}.csv")

		nb_egalite = int((table["raison"] == RAISON_EGALITE).sum())
		nb_non_couvert = int((table["raison"] == RAISON_NON_COUVERT).sum())
		bilan.append({
			"algorithme": nom,
			"facteurs": len(facteurs),
			"écartés_égalité": nb_egalite,
			"écartés_non_couverts_validation": nb_non_couvert,
			"règles_poids_apprentissage": len(facteurs) - nb_egalite,
			"règles_poids_validation": len(facteurs) - nb_egalite - nb_non_couvert,
		})

		print(f"=== {nom} : {len(facteurs)} facteurs ===")
		with pd.option_context("display.width", 250, "display.max_colwidth", 45):
			print(table.round(3).to_string(), "\n")

	table_bilan = pd.DataFrame(bilan).set_index("algorithme")
	print("Bilan")
	print(table_bilan.to_string())
	table_bilan.to_csv(DOSSIER_RESULTATS / f"pima_m3_bilan_regles_{jour}.csv")


if __name__ == "__main__":
	main()
