"""Description courte de Pima : effectifs par classe et valeurs manquantes codées 0.

Lancement : python -m experiences.decrire_pima
"""

import pandas as pd

from donnees.pima import charger_decoupages, charger_pima


def main() -> None:
	X, y = charger_pima()
	print(f"Objets : {len(X)}, variables : {X.shape[1]}\n")

	print("Effectifs par classe (0 = négatif, 1 = diabète)")
	lignes = {"total": y}
	for nom, indices in charger_decoupages().items():
		lignes[nom] = y.iloc[indices]
	effectifs = pd.DataFrame(
		{
			nom: {
				"objets": len(classes),
				"classe 0": int((classes == 0).sum()),
				"classe 1": int((classes == 1).sum()),
				"% classe 1": round(100 * classes.mean(), 1),
			}
			for nom, classes in lignes.items()
		}
	).T
	print(effectifs.to_string(), "\n")

	print("Zéros par variable (non corrigés)")
	zeros = pd.DataFrame({"zéros": (X == 0).sum(), "% zéros": (100 * (X == 0).mean()).round(1)})
	print(zeros.to_string(), "\n")

	print("Statistiques descriptives")
	print(X.describe().T.round(2).to_string())


if __name__ == "__main__":
	main()
