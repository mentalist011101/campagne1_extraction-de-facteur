"""Compare notre GreConD à celui de PyBMF sur l'exemple des 10 personnes.

PyBMF (https://github.com/PreferredAI/PyBMF, commit 03c9b43) sert seulement de
référence de vérification, il n'est pas une dépendance du projet.

Lancement : python -m experiences.comparer_pybmf
"""

import warnings

import numpy as np
from scipy.sparse import csr_matrix

from donnees.exemple import charger_exemple_10_personnes
from m2_facteurs.factorisation import produit_booleen
from m2_facteurs.fca import est_concept
from m2_facteurs.grecond import grecond


def main() -> None:
	I, objets, attributs, classes = charger_exemple_10_personnes()

	print("=== Notre GreConD ===")
	resultat = grecond(I)
	_afficher_facteurs(I, resultat.A, resultat.B, objets, attributs)

	print("\n=== GreConD de PyBMF ===")
	A_pybmf, B_pybmf = _lancer_pybmf(I)
	_afficher_facteurs(I, A_pybmf, B_pybmf, objets, attributs)


def _lancer_pybmf(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Lance PyBMF et convertit U (objets × k) et V (attributs × k) en A et B booléens."""
	from PyBMF.models import GreConD

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		modele = GreConD(k=None, tol=0)
		# task est obligatoire dans PyBMF, sinon l'évaluation interne plante
		modele.fit(
			csr_matrix(I.astype(float)),
			task="reconstruction",
			show_logs=False,
			save_model=False,
			show_result=False,
		)
	A = modele.U.toarray() > 0
	B = modele.V.toarray().T > 0
	return A, B


def _afficher_facteurs(I, A, B, objets, attributs) -> None:
	"""Affiche chaque facteur avec ses croix nouvelles, puis le bilan de reconstruction."""
	non_couvertes = I.copy()
	for k in range(A.shape[1]):
		ext, inten = A[:, k], B[k, :]
		nouvelles = int(non_couvertes[np.ix_(ext, inten)].sum())
		non_couvertes[np.ix_(ext, inten)] = False
		noms_ext = ", ".join(n for n, g in zip(objets, ext) if g)
		noms_int = ", ".join(n for n, g in zip(attributs, inten) if g)
		concept = "concept" if est_concept(I, ext, inten) else "PAS un concept"
		print(f"F{k + 1}  {{{noms_ext}}} x {{{noms_int}}}   +{nouvelles} croix   ({concept})")

	reconstruction = produit_booleen(A, B)
	print(f"Facteurs : {A.shape[1]}")
	print(f"Croix couvertes : {int((reconstruction & I).sum())}/{int(I.sum())}")
	print(f"Croix créées hors de I : {int((reconstruction & ~I).sum())}")


if __name__ == "__main__":
	main()
