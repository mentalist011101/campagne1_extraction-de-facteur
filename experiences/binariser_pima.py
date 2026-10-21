"""M1 sur Pima : seuils appris sur l'apprentissage, 3 contextes binaires, tableau récapitulatif.

Lancement : python -m experiences.binariser_pima
"""

from datetime import date
from pathlib import Path

import pandas as pd

from donnees.pima import charger_decoupages
from experiences.contextes_pima import contextes_pima

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"


def main() -> None:
	contextes, noms_attributs, classes, seuils = contextes_pima()
	decoupages = charger_decoupages()
	jour = date.today().isoformat()

	table_seuils = pd.DataFrame(seuils, index=["seuil_1/3", "seuil_2/3"]).T
	print("Seuils (appris sur l'apprentissage, zéros manquants exclus)")
	print(table_seuils.to_string(), "\n")
	table_seuils.to_csv(DOSSIER_RESULTATS / f"pima_m1_seuils_{jour}.csv")

	recapitulatif = {}
	for nom, I in contextes.items():
		indices = pd.Index(decoupages[nom], name="indice")
		contexte = pd.DataFrame(I.astype(int), columns=noms_attributs, index=indices)
		contexte.to_csv(DOSSIER_RESULTATS / f"pima_m1_contexte_{nom}_{jour}.csv")
		recapitulatif[nom] = {
			"objets": I.shape[0],
			"attributs": I.shape[1],
			"croix": int(I.sum()),
			"densité": round(float(I.mean()), 4),
		}

	table = pd.DataFrame(recapitulatif).T.astype({"objets": int, "attributs": int, "croix": int})
	print("Contextes binaires")
	print(table.to_string(), "\n")
	table.to_csv(DOSSIER_RESULTATS / f"pima_m1_recapitulatif_{jour}.csv")

	print("Attributs :", ", ".join(noms_attributs))


if __name__ == "__main__":
	main()
