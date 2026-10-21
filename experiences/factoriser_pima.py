"""M2 sur Pima : GreCon et GreConD sur le contexte d'apprentissage, sans la classe.

Lancement : python -m experiences.factoriser_pima
"""

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from experiences.contextes_pima import contextes_pima
from m2_facteurs.factorisation import ResultatFactorisation, nb_facteurs_pour_couverture, produit_booleen
from m2_facteurs.fca import enumerer_concepts
from m2_facteurs.grecon import grecon
from m2_facteurs.grecond import grecond

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"
CIBLES = [0.8, 0.9, 1.0]


def main() -> None:
	contextes, noms_attributs, classes, seuils = contextes_pima()
	# Seul le contexte binaire est transmis aux algorithmes : la classe n'est jamais lue
	I = contextes["apprentissage"]
	jour = date.today().isoformat()
	print(f"Contexte d'apprentissage : {I.shape[0]} objets × {I.shape[1]} attributs, {int(I.sum())} croix\n")

	debut = time.perf_counter()
	nb_concepts = len(enumerer_concepts(I))
	temps_enumeration = time.perf_counter() - debut

	resultats = {}
	temps = {}
	for nom, algorithme in [("GreCon", grecon), ("GreConD", grecond)]:
		debut = time.perf_counter()
		resultats[nom] = algorithme(I)
		temps[nom] = time.perf_counter() - debut
		_verifier_reconstruction(I, resultats[nom], nom)

	lignes = []
	for nom, resultat in resultats.items():
		ligne = {"algorithme": nom}
		for cible in CIBLES:
			ligne[f"facteurs_{int(cible * 100)}%"] = nb_facteurs_pour_couverture(resultat, cible)
		ligne["temps_s"] = round(temps[nom], 2)
		ligne["concepts"] = nb_concepts if nom == "GreCon" else None
		lignes.append(ligne)
	comparaison = pd.DataFrame(lignes).set_index("algorithme").astype({"concepts": "Int64"})
	print("Comparaison (temps de GreCon énumération comprise ; "
		f"énumération seule : {temps_enumeration:.2f} s)")
	print(comparaison.to_string(), "\n")
	comparaison.to_csv(DOSSIER_RESULTATS / f"pima_m2_comparaison_{jour}.csv")

	for nom, resultat in resultats.items():
		facteurs = _table_facteurs(resultat, noms_attributs)
		facteurs.to_csv(DOSSIER_RESULTATS / f"pima_m2_facteurs_{nom.lower()}_{jour}.csv")
		print(f"10 premiers facteurs de {nom}")
		print(facteurs.head(10).to_string(), "\n")


def _verifier_reconstruction(I: np.ndarray, resultat: ResultatFactorisation, nom: str) -> None:
	"""A ∘ B doit reconstruire I exactement, sans créer de croix absente."""
	reconstruction = produit_booleen(resultat.A, resultat.B)
	croix_en_trop = int((reconstruction & ~I).sum())
	exacte = bool(np.array_equal(reconstruction, I))
	print(f"{nom} : reconstruction exacte = {exacte}, croix créées hors de I = {croix_en_trop}")
	assert exacte and croix_en_trop == 0


def _table_facteurs(resultat: ResultatFactorisation, noms_attributs: list[str]) -> pd.DataFrame:
	"""Une ligne par facteur : intention lisible, taille de l'extension, croix nouvelles, couverture."""
	lignes = []
	for numero, (facteur, nouvelles, couverture) in enumerate(
		zip(resultat.facteurs, resultat.croix_nouvelles, resultat.couverture_cumulee), start=1
	):
		intention = [nom for nom, garde in zip(noms_attributs, facteur.intention) if garde]
		lignes.append({
			"facteur": f"F{numero}",
			"intention": " ET ".join(intention),
			"taille_extension": int(facteur.extension.sum()),
			"croix_nouvelles": nouvelles,
			"couverture_cumulee": round(couverture, 3),
		})
	return pd.DataFrame(lignes).set_index("facteur")


if __name__ == "__main__":
	main()
