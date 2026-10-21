"""M1 sur Pima : seuils appris sur l'apprentissage, 3 contextes binaires, tableau récapitulatif.

Lancement : python -m experiences.binariser_pima
"""

from datetime import date
from pathlib import Path

import pandas as pd

from donnees.pima import COLONNES_ZERO_MANQUANT, charger_decoupages, charger_pima
from m1_binarisation.binarisation import apprendre_seuils, binariser

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"


def main() -> None:
	X, y = charger_pima()
	decoupages = charger_decoupages()
	jour = date.today().isoformat()

	seuils = apprendre_seuils(X.iloc[decoupages["apprentissage"]], COLONNES_ZERO_MANQUANT)
	table_seuils = pd.DataFrame(seuils, index=["seuil_1/3", "seuil_2/3"]).T
	print("Seuils (appris sur l'apprentissage, zéros manquants exclus)")
	print(table_seuils.to_string(), "\n")
	table_seuils.to_csv(DOSSIER_RESULTATS / f"pima_m1_seuils_{jour}.csv")

	recapitulatif = {}
	for nom, indices in decoupages.items():
		I, noms_attributs = binariser(X.iloc[indices], seuils, COLONNES_ZERO_MANQUANT)
		contexte = pd.DataFrame(I.astype(int), columns=noms_attributs, index=pd.Index(indices, name="indice"))
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
