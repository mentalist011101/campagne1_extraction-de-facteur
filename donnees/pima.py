"""Jeu Pima Indians Diabetes et découpages figés.

Source : CSV de l'UCI, miroir GitHub de J. Brownlee (OpenML inaccessible depuis
notre réseau, voir DECISIONS.md). Téléchargé une seule fois le 2026-10-21, puis lu
depuis la copie figée donnees/pima.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

DOSSIER_DONNEES = Path(__file__).parent
CHEMIN_PIMA = DOSSIER_DONNEES / "pima.csv"
DOSSIER_DECOUPAGES = DOSSIER_DONNEES / "decoupages"
NOMS_DECOUPAGES = ["apprentissage", "validation", "test"]

URL_SOURCE = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# Fichier sans en-tête ; ordre UCI des colonnes : preg, plas, pres, skin, insu, mass, pedi, age, classe
# (classe déjà codée 1 = diabète, 0 = négatif)
COLONNES = [
	"grossesses", "glucose", "pression", "pli_cutane", "insuline", "imc", "pedigree", "age", "classe",
]
# Colonnes où un zéro est médicalement impossible, donc une valeur manquante
COLONNES_ZERO_MANQUANT = ["glucose", "pression", "pli_cutane", "insuline", "imc"]


def telecharger_pima() -> None:
	"""Télécharge le CSV source et écrit la copie figée donnees/pima.csv (avec en-tête)."""
	if CHEMIN_PIMA.exists():
		raise FileExistsError(f"{CHEMIN_PIMA} existe déjà : la copie figée n'est jamais remplacée.")
	table = pd.read_csv(URL_SOURCE, header=None, names=COLONNES)
	table.to_csv(CHEMIN_PIMA, index=False)


def charger_pima() -> tuple[pd.DataFrame, pd.Series]:
	"""Retourne X (768 × 8 variables numériques) et y (0 = négatif, 1 = diabète)."""
	table = pd.read_csv(CHEMIN_PIMA)
	return table.drop(columns="classe"), table["classe"]


def creer_decoupages() -> None:
	"""Découpage stratifié 60/20/20 (random_state=42), sauvegardé une fois pour toutes."""
	chemins = [_chemin_decoupage(nom) for nom in NOMS_DECOUPAGES]
	if any(chemin.exists() for chemin in chemins):
		raise FileExistsError("Les découpages existent déjà : ils ne sont jamais régénérés.")

	X, y = charger_pima()
	tous = np.arange(len(y))
	# 60 % apprentissage, 40 % reste ; puis le reste coupé en deux moitiés égales
	apprentissage, reste = train_test_split(tous, test_size=0.4, stratify=y, random_state=42)
	validation, test = train_test_split(
		reste, test_size=0.5, stratify=y.iloc[reste], random_state=42
	)
	for chemin, indices in zip(chemins, [apprentissage, validation, test]):
		pd.DataFrame({"indice": indices}).to_csv(chemin, index=False)


def charger_decoupages() -> dict[str, np.ndarray]:
	"""Indices (positions de lignes dans pima.csv) de chaque découpage."""
	return {
		nom: pd.read_csv(_chemin_decoupage(nom))["indice"].to_numpy()
		for nom in NOMS_DECOUPAGES
	}


def creer_plis_cv10() -> None:
	"""10 plis stratifiés (random_state=42) sur apprentissage + validation, sauvegardés une fois pour toutes.

	Le test de la phase 1 reste scellé : il n'entre dans aucun pli.
	"""
	chemin = DOSSIER_DECOUPAGES / "cv10_pima.csv"
	if chemin.exists():
		raise FileExistsError(f"{chemin} existe déjà : les plis ne sont jamais régénérés.")
	X, y = charger_pima()
	d = charger_decoupages()
	tous = np.concatenate([d["apprentissage"], d["validation"]])
	decoupeur = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	lignes = []
	for numero_pli, (positions_app, positions_eval) in enumerate(decoupeur.split(tous, y.iloc[tous])):
		lignes += [{"indice": int(i), "pli": numero_pli} for i in tous[positions_eval]]
	pd.DataFrame(lignes).to_csv(chemin, index=False)


def charger_plis_cv10() -> list[tuple[np.ndarray, np.ndarray]]:
	"""Pour chaque pli : (indices de la partie d'apprentissage, indices de la partie évaluée)."""
	table = pd.read_csv(DOSSIER_DECOUPAGES / "cv10_pima.csv")
	plis = []
	for numero_pli in sorted(table["pli"].unique()):
		dans_le_pli = table["pli"] == numero_pli
		plis.append((table.loc[~dans_le_pli, "indice"].to_numpy(), table.loc[dans_le_pli, "indice"].to_numpy()))
	return plis


def _chemin_decoupage(nom: str) -> Path:
	return DOSSIER_DECOUPAGES / f"pima_{nom}.csv"
