"""Jeu mushroom (UCI, aussi sur OpenML) et découpages figés.

Source : OpenML, fetch_openml(name="mushroom", version=1) ; à défaut, le fichier UCI
agaricus-lepiota.data. Téléchargé une seule fois, puis lu depuis la copie figée
donnees/mushroom.csv, où « ? » (valeur manquante de stalk-root) est conservé tel quel.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

DOSSIER_DONNEES = Path(__file__).parent
CHEMIN_MUSHROOM = DOSSIER_DONNEES / "mushroom.csv"
DOSSIER_DECOUPAGES = DOSSIER_DONNEES / "decoupages"
NOMS_DECOUPAGES = ["apprentissage", "validation", "test"]
URL_UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

# Ordre UCI des 22 variables
COLONNES = [
	"cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing",
	"gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
	"stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
	"veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat",
]
CODAGE_CLASSE = {"e": 0, "p": 1}   # 1 = vénéneux


def telecharger_mushroom() -> str:
	"""Écrit la copie figée donnees/mushroom.csv ; retourne la source utilisée."""
	if CHEMIN_MUSHROOM.exists():
		raise FileExistsError(f"{CHEMIN_MUSHROOM} existe déjà : la copie figée n'est jamais remplacée.")
	try:
		X, classe, source = _depuis_openml()
	except Exception as erreur:
		print(f"OpenML indisponible ({type(erreur).__name__}) : fichier UCI utilisé.")
		X, classe, source = _depuis_uci()
	table = X.copy()
	table["classe"] = classe.map(CODAGE_CLASSE)
	table.to_csv(CHEMIN_MUSHROOM, index=False)
	return source


def charger_mushroom() -> tuple[pd.DataFrame, pd.Series]:
	"""X (8124 × 22 variables catégorielles, « ? » conservé) et y (1 = vénéneux)."""
	table = pd.read_csv(CHEMIN_MUSHROOM, dtype=str, keep_default_na=False)
	return table.drop(columns="classe"), table["classe"].astype(int)


def creer_decoupages_mushroom() -> None:
	"""Découpage stratifié 60/20/20 (random_state=42), sauvegardé une fois pour toutes."""
	chemins = [_chemin_decoupage(nom) for nom in NOMS_DECOUPAGES]
	if any(chemin.exists() for chemin in chemins):
		raise FileExistsError("Les découpages de mushroom existent déjà : ils ne sont jamais régénérés.")
	X, y = charger_mushroom()
	tous = np.arange(len(y))
	apprentissage, reste = train_test_split(tous, test_size=0.4, stratify=y, random_state=42)
	validation, test = train_test_split(reste, test_size=0.5, stratify=y.iloc[reste], random_state=42)
	for chemin, indices in zip(chemins, [apprentissage, validation, test]):
		pd.DataFrame({"indice": indices}).to_csv(chemin, index=False)


def charger_decoupages_mushroom() -> dict[str, np.ndarray]:
	return {nom: pd.read_csv(_chemin_decoupage(nom))["indice"].to_numpy() for nom in NOMS_DECOUPAGES}


def creer_plis_cv10_mushroom() -> None:
	"""10 plis stratifiés (random_state=42) sur apprentissage + validation ; le test reste scellé."""
	chemin = DOSSIER_DECOUPAGES / "cv10_mushroom.csv"
	if chemin.exists():
		raise FileExistsError(f"{chemin} existe déjà : les plis ne sont jamais régénérés.")
	X, y = charger_mushroom()
	d = charger_decoupages_mushroom()
	tous = np.concatenate([d["apprentissage"], d["validation"]])
	decoupeur = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	lignes = []
	for numero_pli, (positions_app, positions_eval) in enumerate(decoupeur.split(tous, y.iloc[tous])):
		lignes += [{"indice": int(i), "pli": numero_pli} for i in tous[positions_eval]]
	pd.DataFrame(lignes).to_csv(chemin, index=False)


def charger_plis_cv10_mushroom() -> list[tuple[np.ndarray, np.ndarray]]:
	"""Pour chaque pli : (indices de la partie d'apprentissage, indices de la partie évaluée)."""
	table = pd.read_csv(DOSSIER_DECOUPAGES / "cv10_mushroom.csv")
	plis = []
	for numero_pli in sorted(table["pli"].unique()):
		dans_le_pli = table["pli"] == numero_pli
		plis.append((table.loc[~dans_le_pli, "indice"].to_numpy(), table.loc[dans_le_pli, "indice"].to_numpy()))
	return plis


def _depuis_openml() -> tuple[pd.DataFrame, pd.Series, str]:
	from sklearn.datasets import fetch_openml

	jeu = fetch_openml(name="mushroom", version=1, as_frame=True)
	# OpenML encode « bruises? » en « bruises%3F » ; on vérifie l'ordre UCI avant de renommer
	noms_openml = [nom.replace("%3F", "").replace("?", "") for nom in jeu.data.columns]
	if noms_openml != COLONNES:
		raise ValueError(f"Colonnes OpenML inattendues : {list(jeu.data.columns)}")
	X = jeu.data.astype(object).where(jeu.data.notna(), "?").astype(str)
	X.columns = COLONNES
	return X, jeu.target.astype(str), "OpenML mushroom version 1"


def _depuis_uci() -> tuple[pd.DataFrame, pd.Series, str]:
	table = pd.read_csv(URL_UCI, header=None, names=["classe", *COLONNES], dtype=str)
	return table[COLONNES], table["classe"], URL_UCI


def _chemin_decoupage(nom: str) -> Path:
	return DOSSIER_DECOUPAGES / f"mushroom_{nom}.csv"
