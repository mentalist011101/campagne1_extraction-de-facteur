"""Étape 12 : GreCon sur l'échelle interordinale (apprentissage de la phase 1), avec garde-fous.

1. Compte des concepts, abandonné au-delà de 200 000 (TropDeConcepts).
2. Si le compte passe, GreCon, puis comparaison de ses facteurs avec ceux de GreConD.
La limite de 10 minutes est appliquée de l'extérieur, en lançant ce script avec un délai :
  timeout 600 python -m experiences.grecon_interordinale_pima
"""

import time

from experiences.contextes_pima import contextes_pima
from m2_facteurs.fca import TropDeConcepts, enumerer_concepts
from m2_facteurs.grecon import grecon
from m2_facteurs.grecond import grecond

NB_MAX_CONCEPTS = 200_000


def main() -> None:
	contextes, noms_attributs, classes, seuils = contextes_pima("interordinale")
	I = contextes["apprentissage"]

	debut = time.perf_counter()
	try:
		nb_concepts = len(enumerer_concepts(I, nb_max=NB_MAX_CONCEPTS))
	except TropDeConcepts:
		print(f"Plus de {NB_MAX_CONCEPTS} concepts ({time.perf_counter() - debut:.1f} s) : GreCon abandonné.", flush=True)
		return
	print(f"Concepts : {nb_concepts} (énumération en {time.perf_counter() - debut:.1f} s)", flush=True)

	debut = time.perf_counter()
	resultat_grecon = grecon(I)
	print(f"GreCon : {len(resultat_grecon.facteurs)} facteurs en {time.perf_counter() - debut:.1f} s", flush=True)

	resultat_grecond = grecond(I)
	intentions_grecon = {tuple(f.intention) for f in resultat_grecon.facteurs}
	intentions_grecond = {tuple(f.intention) for f in resultat_grecond.facteurs}
	communs = intentions_grecon & intentions_grecond
	print(f"GreConD : {len(resultat_grecond.facteurs)} facteurs")
	print(f"Facteurs communs : {len(communs)} ; propres à GreCon : {len(intentions_grecon - communs)} ; "
		f"propres à GreConD : {len(intentions_grecond - communs)}")
	print(f"Mêmes ensembles de facteurs : {intentions_grecon == intentions_grecond}")


if __name__ == "__main__":
	main()
