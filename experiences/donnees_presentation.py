"""Données pour la présentation : un seul fichier Markdown, resultats/donnees_presentation.md.

1. Résumé de l'étape 13 (relu depuis resultats/, rien n'est recalculé).
2. Structure des facteurs par jeu : moyenne sur les 10 plis, calculée sur les parties
   d'apprentissage des plis seulement (mushroom : relu depuis resultats/). GreCon : chiffres
   des étapes 6 et 12 (apprentissage de la phase 1).
3. Exemples de facteurs du pli 0 : les 3 meilleurs par classe selon la précision sur la partie
   d'apprentissage du pli (égalité de précision : la plus grande extension d'abord).
Le test n'est jamais lu, ni la partie évaluée des plis.

Lancement : python -m experiences.donnees_presentation
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from donnees.mushroom import charger_mushroom, charger_plis_cv10_mushroom
from donnees.pima import charger_plis_cv10
from donnees.vote import charger_plis_cv10_vote, charger_vote
from experiences.contextes_pima import contextes_pli
from m1_binarisation.binarisation import apprendre_modalites, binariser_categoriel
from m2_facteurs.grecond import grecond
from m3_classifieurs.classifieur_factoriel import construire_regles

DOSSIER_RESULTATS = Path(__file__).parent.parent / "resultats"
SORTIE = DOSSIER_RESULTATS / "donnees_presentation.md"


def main() -> None:
	parties = [
		"# Données pour la présentation\n",
		"Toutes les valeurs viennent de la validation croisée à 10 plis (partie apprentissage + validation "
		"de chaque jeu). **Le test n'a jamais été lu.**\n",
		_section_etape_13(),
		_section_structure(),
		_section_exemples(),
	]
	SORTIE.write_text("\n".join(parties), encoding="utf-8")
	print(f"Écrit : {SORTIE}")


def _section_etape_13() -> str:
	resume = pd.read_csv(DOSSIER_RESULTATS / "pima_cv10_grille_resume_2026-10-24.csv", index_col="methode")
	table = resume[["accuracy", "f1_macro", "predits_classe_1", "reels_classe_1"]]
	return (
		"## 1. Pima, étape 13 : toutes les méthodes (moyenne ± écart-type sur 10 plis)\n\n"
		"Source : `resultats/pima_cv10_grille_resume_2026-10-24.csv`. Chaque pli évalue 61 ou 62 patientes, "
		"dont 21,5 de classe 1 en moyenne.\n\n"
		+ _en_markdown(table) + "\n"
	)


def _section_structure() -> str:
	lignes = [
		_structure_pima("nominale"),
		_structure_pima("interordinale"),
		_structure_mushroom(),
		_structure_categorielle("vote", charger_vote, charger_plis_cv10_vote),
	]
	table = pd.DataFrame(lignes).set_index("jeu")
	return (
		"## 2. Structure des facteurs (GreConD à 100 %, moyenne sur les 10 plis)\n\n"
		"Calculée sur la partie d'apprentissage de chaque pli. Mushroom : relu depuis "
		"`resultats/mushroom_cv10_facteurs_2026-10-28.csv`.\n\n"
		+ _en_markdown(table) + "\n\n"
		"**GreCon (apprentissage de la phase 1, 460 patientes, pas de validation croisée) :**\n\n"
		"| Jeu | Concepts | Temps |\n|---|---|---|\n"
		"| Pima nominal | 7 212 | 17,7 s (étape 6, énumération comprise) |\n"
		"| Pima interordinal | 150 555 | énumération 35 s + GreCon 309 s (étape 12) |\n"
		"| mushroom, vote | non lancé | — |\n"
	)


def _structure_pima(echelle: str) -> dict:
	mesures = []
	for indices_app, indices_eval in charger_plis_cv10():
		I_app, _, noms, y_app, _, _ = contextes_pli(indices_app, indices_eval, echelle)
		mesures.append(_mesurer(I_app))
	return {"jeu": f"Pima {echelle}", **_moyennes(mesures)}


def _structure_categorielle(nom: str, charger, charger_plis) -> dict:
	X, y = charger()
	mesures = []
	for indices_app, indices_eval in charger_plis():
		X_app = X.iloc[indices_app]
		I_app, _ = binariser_categoriel(X_app, apprendre_modalites(X_app))
		mesures.append(_mesurer(I_app))
	return {"jeu": nom, **_moyennes(mesures)}


def _structure_mushroom() -> dict:
	table = pd.read_csv(DOSSIER_RESULTATS / "mushroom_cv10_facteurs_2026-10-28.csv")
	return {
		"jeu": "mushroom",
		"attributs": f"{table['attributs'].mean():.1f}",
		"densite": f"{table['densite'].mean():.3f}",
		"facteurs": f"{table['facteurs'].mean():.1f}",
		"part_un_attribut": f"{table['part_un_attribut'].mean():.3f}",
		"longueur_moyenne": f"{table['longueur_moyenne'].mean():.2f}",
		"temps_grecond_s": f"{table['temps_grecond_s'].mean():.2f} (min {table['temps_grecond_s'].min():.1f}, max {table['temps_grecond_s'].max():.1f})",
	}


def _mesurer(I_app: np.ndarray) -> dict:
	debut = time.perf_counter()
	facteurs = grecond(I_app).facteurs
	temps = time.perf_counter() - debut
	longueurs = np.array([int(f.intention.sum()) for f in facteurs])
	return {
		"attributs": I_app.shape[1],
		"densite": float(I_app.mean()),
		"facteurs": len(facteurs),
		"part_un_attribut": float((longueurs == 1).mean()),
		"longueur_moyenne": float(longueurs.mean()),
		"temps_grecond_s": temps,
	}


def _moyennes(mesures: list[dict]) -> dict:
	table = pd.DataFrame(mesures)
	return {
		"attributs": f"{table['attributs'].mean():.1f}",
		"densite": f"{table['densite'].mean():.3f}",
		"facteurs": f"{table['facteurs'].mean():.1f}",
		"part_un_attribut": f"{table['part_un_attribut'].mean():.3f}",
		"longueur_moyenne": f"{table['longueur_moyenne'].mean():.2f}",
		"temps_grecond_s": f"{table['temps_grecond_s'].mean():.2f} (min {table['temps_grecond_s'].min():.2f}, max {table['temps_grecond_s'].max():.2f})",
	}


def _section_exemples() -> str:
	blocs = ["## 3. Exemples de facteurs, pli 0 (3 meilleurs par classe selon la précision)\n",
		"Précision mesurée sur la partie d'apprentissage du pli 0. À précision égale, la plus grande "
		"extension d'abord.\n"]

	indices_app, indices_eval = charger_plis_cv10()[0]
	I_app, _, noms, y_app, _, _ = contextes_pli(indices_app, indices_eval, "nominale")
	blocs.append(_exemples("Pima (nominale) — classe 1 = diabète", I_app, noms, y_app))

	for titre, charger, charger_plis in [
		("mushroom — classe 1 = vénéneux", charger_mushroom, charger_plis_cv10_mushroom),
		("vote — classe 1 = républicain", charger_vote, charger_plis_cv10_vote),
	]:
		X, y = charger()
		indices_app, _ = charger_plis()[0]
		X_app = X.iloc[indices_app]
		I_app, noms = binariser_categoriel(X_app, apprendre_modalites(X_app))
		blocs.append(_exemples(titre, I_app, noms, y.iloc[indices_app]))
	return "\n".join(blocs)


def _exemples(titre: str, I_app: np.ndarray, noms: list[str], y_app) -> str:
	facteurs = grecond(I_app).facteurs
	regles, ecartes = construire_regles(facteurs, y_app, I_app, y_app, mesure="precision")
	lignes = []
	for regle in regles:
		facteur = facteurs[regle.numero_facteur - 1]
		lignes.append({
			"facteur": f"F{regle.numero_facteur}",
			"classe": regle.classe,
			"precision": round(regle.poids, 3),
			"taille_extension": int(facteur.extension.sum()),
			"intention": " ET ".join(n for n, g in zip(noms, facteur.intention) if g),
		})
	table = pd.DataFrame(lignes).sort_values(["classe", "precision", "taille_extension"], ascending=[True, False, False])
	meilleurs = table.groupby("classe").head(3).set_index("facteur")
	return f"### {titre}\n\n{len(facteurs)} facteurs, {len(ecartes)} écarté(s).\n\n{_en_markdown(meilleurs)}\n"


def _en_markdown(table: pd.DataFrame) -> str:
	"""Tableau Markdown simple, index compris (évite la dépendance tabulate)."""
	entetes = [table.index.name or ""] + [str(c) for c in table.columns]
	lignes = ["| " + " | ".join(entetes) + " |", "|" + "---|" * len(entetes)]
	for index, valeurs in table.iterrows():
		lignes.append("| " + " | ".join([str(index)] + [str(v) for v in valeurs]) + " |")
	return "\n".join(lignes)


if __name__ == "__main__":
	main()
