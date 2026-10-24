"""Étape 12 : structure des facteurs de GreConD selon l'échelle (nominale ou interordinale).

Sur l'apprentissage de la phase 1 (460 patientes), sans le test ni la validation.
La factorisation ne voit pas la classe ; la proportion de classe 1 par facteur est
calculée après coup, à titre descriptif.

Lancement : python -m experiences.structure_facteurs_pima
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from experiences.contextes_pima import contextes_pima
from m2_facteurs.factorisation import nb_facteurs_pour_couverture
from m2_facteurs.grecond import choix_sur_egalite, grecond

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"
ECHELLES = ["nominale", "interordinale"]


def main() -> None:
	jour = date.today().isoformat()
	comparaison = {}
	longueurs = {}
	for echelle in ECHELLES:
		contextes, noms_attributs, classes, seuils = contextes_pima(echelle)
		I = contextes["apprentissage"]
		y = classes["apprentissage"].to_numpy()

		resultat = grecond(I, trace=True)
		facteurs = _table_facteurs(resultat, noms_attributs, y)
		facteurs.to_csv(DOSSIER_RESULTATS / f"pima_m2_structure_facteurs_{echelle}_{jour}.csv")
		egalites = choix_sur_egalite(I, resultat.trace)

		longueur = facteurs["longueur"]
		proportion = facteurs["proportion_classe_1"]
		comparaison[echelle] = {
			"attributs": I.shape[1],
			"densite": round(float(I.mean()), 4),
			"facteurs_80%": nb_facteurs_pour_couverture(resultat, 0.8),
			"facteurs_90%": nb_facteurs_pour_couverture(resultat, 0.9),
			"facteurs_100%": len(facteurs),
			"part_un_attribut": round(float((longueur == 1).mean()), 3),
			"longueur_moyenne": round(float(longueur.mean()), 2),
			"choix_sur_egalite": len(egalites),
			"dont_fermetures_differentes": sum(c.fermetures_differentes for c in egalites),
			"mediane_proportion_classe_1": round(float(proportion.median()), 3),
			"part_proportion_sous_0.2": round(float((proportion < 0.2).mean()), 3),
			"part_proportion_au_dessus_0.5": round(float((proportion > 0.5).mean()), 3),
		}
		longueurs[echelle] = longueur.value_counts().sort_index()

		print(f"=== {echelle} : 10 premiers facteurs ===")
		with pd.option_context("display.width", 250, "display.max_colwidth", 70):
			print(facteurs.head(10).to_string(), "\n")

	table = pd.DataFrame(comparaison)
	table.to_csv(DOSSIER_RESULTATS / f"pima_m2_structure_comparaison_{jour}.csv")
	table_longueurs = pd.DataFrame(longueurs).fillna(0).astype(int)
	table_longueurs.index.name = "longueur_intention"
	table_longueurs.to_csv(DOSSIER_RESULTATS / f"pima_m2_structure_longueurs_{jour}.csv")

	print("=== Comparaison des échelles (GreConD, apprentissage) ===")
	print(table.to_string(), "\n")
	print("=== Distribution des longueurs d'intention (nombre de facteurs) ===")
	print(table_longueurs.to_string())


def _table_facteurs(resultat, noms_attributs: list[str], y: np.ndarray) -> pd.DataFrame:
	"""Une ligne par facteur : intention, longueur, extension, nouvelles croix, proportion de classe 1."""
	lignes = []
	for numero, (facteur, nouvelles) in enumerate(zip(resultat.facteurs, resultat.croix_nouvelles), start=1):
		lignes.append({
			"facteur": f"F{numero}",
			"intention": " ET ".join(n for n, g in zip(noms_attributs, facteur.intention) if g),
			"longueur": int(facteur.intention.sum()),
			"taille_extension": int(facteur.extension.sum()),
			"croix_nouvelles": nouvelles,
			"proportion_classe_1": round(float(y[facteur.extension].mean()), 3),
		})
	return pd.DataFrame(lignes).set_index("facteur")


if __name__ == "__main__":
	main()
