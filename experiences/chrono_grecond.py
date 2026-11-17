"""Campagne 1, étape 6 : temps d'un GreConD principal (pli 0) sur les plus grands jeux.

Estimation de la durée d'un jeu : temps × 26 (1 GreConD principal + 25 sacs) × 10 plis.
Le facteur 26 est vérifié sur l'essai local (iris, wine, zoo) : temps moyen d'un pli complet
divisé par le temps moyen de son GreConD principal. Jeu à risque : GreConD principal au-delà
de 600 s (la moitié de la limite), ou limite dépassée, sur cette machine.

Lancement : python -m experiences.chrono_grecond
"""

import time

import numpy as np
import pandas as pd

from donnees.catalogue import charger_jeu, charger_plis, lire_variables
from experiences.campagne1 import RACINE, _decrire_machine, _ecrire_atomique, lire_config
from m1_binarisation.binarisation import apprendre_binarisation, binariser_jeu
from m2_facteurs.grecond import DelaiDepasse, grecond

JEUX_CHRONO = ["mushroom", "nursery", "page-blocks", "led7", "anneal"]
FACTEUR_PLI = 26
NB_PLIS = 10
SEUIL_RISQUE_S = 600.0
DOSSIER = RACINE / "resultats" / "essai_local"


def chronometrer(jeux: list[str], delai_max_s: float) -> pd.DataFrame:
	trace = _decrire_machine()
	lignes = []
	for nom in jeux:
		X, y = charger_jeu(nom)
		types = lire_variables(nom).set_index("variable")["type"].to_dict()
		indices_app, _ = charger_plis(nom, y)[0]
		X_app = X.iloc[indices_app]
		I_app = binariser_jeu(X_app, apprendre_binarisation(X_app, types))
		debut = time.perf_counter()
		try:
			nb_facteurs = len(grecond(I_app, delai_max_s=delai_max_s).facteurs)
			depasse = False
		except DelaiDepasse as erreur:
			nb_facteurs, depasse = erreur.nb_facteurs, True
		duree = round(time.perf_counter() - debut, 3)
		lignes.append({
			"jeu": nom, "pli": 0, "n_app": I_app.shape[0], "nb_attributs": I_app.shape[1],
			"densite": round(float(I_app.mean()), 4), "nb_facteurs": nb_facteurs,
			"temps_grecond_s": duree, "depasse_limite": depasse,
			"estimation_jeu_s": duree * FACTEUR_PLI * NB_PLIS,
			"a_risque": depasse or duree > SEUIL_RISQUE_S,
			"machine": trace["processeur"], "commit": trace["commit"],
			"modifications_non_commitees": trace["modifications_non_commitees"],
		})
		print(f"{nom} : {duree:.1f} s, {nb_facteurs} facteurs{' (limite dépassée)' if depasse else ''}", flush=True)
	return pd.DataFrame(lignes)


def verifier_facteur(par_pli: pd.DataFrame) -> pd.DataFrame:
	"""Par jeu : temps moyen d'un pli complet (binarisation, GreConD, 8 méthodes) et rapport au GreConD."""
	par_pli_jeu = par_pli.groupby(["jeu", "pli"]).agg(
		temps_methodes_s=("temps_s", "sum"), temps_grecond_s=("temps_grecond_s", "first"),
		temps_binarisation_s=("temps_binarisation_s", "first"))
	par_pli_jeu["temps_pli_s"] = par_pli_jeu[["temps_methodes_s", "temps_grecond_s", "temps_binarisation_s"]].sum(axis=1)
	table = par_pli_jeu.groupby("jeu").agg(
		nb_plis=("temps_pli_s", "size"), temps_grecond_s=("temps_grecond_s", "mean"), temps_pli_s=("temps_pli_s", "mean"))
	table["facteur_observe"] = table["temps_pli_s"] / table["temps_grecond_s"]
	return table.reset_index()


def main() -> None:
	delai = lire_config(RACINE / "experiences" / "campagne1.yaml").delai_grecond_s
	chrono = chronometrer(JEUX_CHRONO, delai)
	_ecrire_atomique(chrono, DOSSIER / "chrono_grecond.csv")
	facteur = verifier_facteur(pd.read_csv(DOSSIER / "campagne1_par_pli.csv"))
	_ecrire_atomique(facteur.round(3), DOSSIER / "facteur_pli.csv")
	with pd.option_context("display.width", 200, "display.max_columns", None):
		print(chrono.drop(columns=["commit"]).to_string(index=False))
		print(facteur.round(3).to_string(index=False))
	print(f"Facteur observé moyen : {np.mean(facteur['facteur_observe']):.1f} (prévu : {FACTEUR_PLI})")


if __name__ == "__main__":
	main()
