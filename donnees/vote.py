"""Jeu congressional voting records (UCI, OpenML « vote ») et découpages figés.

Source : OpenML, fetch_openml(name="vote", version=1) ; à défaut, le fichier UCI
house-votes-84.data. Téléchargé une seule fois, puis lu depuis la copie figée
donnees/vote.csv : votes « y » / « n », « ? » conservé tel quel (abstention réelle),
classe avec ses libellés d'origine.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

DOSSIER_DONNEES = Path(__file__).parent
CHEMIN_VOTE = DOSSIER_DONNEES / "vote.csv"
DOSSIER_DECOUPAGES = DOSSIER_DONNEES / "decoupages"
NOMS_DECOUPAGES = ["apprentissage", "validation", "test"]
URL_UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"

# Ordre UCI des 16 votes
COLONNES = [
	"handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution",
	"physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools",
	"anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration",
	"synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime",
	"duty-free-exports", "export-administration-act-south-africa",
]
CODAGE_CLASSE = {"democrat": 0, "republican": 1}


def telecharger_vote() -> str:
	"""Écrit la copie figée donnees/vote.csv ; retourne la source utilisée."""
	if CHEMIN_VOTE.exists():
		raise FileExistsError(f"{CHEMIN_VOTE} existe déjà : la copie figée n'est jamais remplacée.")
	try:
		X, classe, source = _depuis_openml()
	except Exception as erreur:
		print(f"OpenML indisponible ({type(erreur).__name__}) : fichier UCI utilisé.")
		X, classe, source = _depuis_uci()
	table = X.copy()
	table["classe"] = classe.to_numpy()
	table.to_csv(CHEMIN_VOTE, index=False)
	return source


def charger_vote() -> tuple[pd.DataFrame, pd.Series]:
	"""X (435 × 16 votes y / n / ?) et y codée par CODAGE_CLASSE."""
	table = pd.read_csv(CHEMIN_VOTE, dtype=str, keep_default_na=False)
	return table.drop(columns="classe"), table["classe"].map(CODAGE_CLASSE).astype(int)


def creer_decoupages_vote() -> None:
	"""Découpage stratifié 60/20/20 (random_state=42), sauvegardé une fois pour toutes."""
	chemins = [_chemin_decoupage(nom) for nom in NOMS_DECOUPAGES]
	if any(chemin.exists() for chemin in chemins):
		raise FileExistsError("Les découpages de vote existent déjà : ils ne sont jamais régénérés.")
	X, y = charger_vote()
	tous = np.arange(len(y))
	apprentissage, reste = train_test_split(tous, test_size=0.4, stratify=y, random_state=42)
	validation, test = train_test_split(reste, test_size=0.5, stratify=y.iloc[reste], random_state=42)
	for chemin, indices in zip(chemins, [apprentissage, validation, test]):
		pd.DataFrame({"indice": indices}).to_csv(chemin, index=False)


def charger_decoupages_vote() -> dict[str, np.ndarray]:
	return {nom: pd.read_csv(_chemin_decoupage(nom))["indice"].to_numpy() for nom in NOMS_DECOUPAGES}


def creer_plis_cv10_vote() -> None:
	"""10 plis stratifiés (random_state=42) sur apprentissage + validation ; le test reste scellé."""
	chemin = DOSSIER_DECOUPAGES / "cv10_vote.csv"
	if chemin.exists():
		raise FileExistsError(f"{chemin} existe déjà : les plis ne sont jamais régénérés.")
	X, y = charger_vote()
	d = charger_decoupages_vote()
	tous = np.concatenate([d["apprentissage"], d["validation"]])
	decoupeur = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	lignes = []
	for numero_pli, (positions_app, positions_eval) in enumerate(decoupeur.split(tous, y.iloc[tous])):
		lignes += [{"indice": int(i), "pli": numero_pli} for i in tous[positions_eval]]
	pd.DataFrame(lignes).to_csv(chemin, index=False)


def charger_plis_cv10_vote() -> list[tuple[np.ndarray, np.ndarray]]:
	"""Pour chaque pli : (indices de la partie d'apprentissage, indices de la partie évaluée)."""
	table = pd.read_csv(DOSSIER_DECOUPAGES / "cv10_vote.csv")
	plis = []
	for numero_pli in sorted(table["pli"].unique()):
		dans_le_pli = table["pli"] == numero_pli
		plis.append((table.loc[~dans_le_pli, "indice"].to_numpy(), table.loc[dans_le_pli, "indice"].to_numpy()))
	return plis


def _depuis_openml() -> tuple[pd.DataFrame, pd.Series, str]:
	from sklearn.datasets import fetch_openml

	jeu = fetch_openml(name="vote", version=1, as_frame=True)
	noms_openml = [nom.replace("%3F", "").replace("?", "") for nom in jeu.data.columns]
	if noms_openml != COLONNES:
		raise ValueError(f"Colonnes OpenML inattendues : {list(jeu.data.columns)}")
	X = jeu.data.astype(object).where(jeu.data.notna(), "?").astype(str)
	X.columns = COLONNES
	return X, jeu.target.astype(str), "OpenML vote version 1"


def _depuis_uci() -> tuple[pd.DataFrame, pd.Series, str]:
	table = pd.read_csv(URL_UCI, header=None, names=["classe", *COLONNES], dtype=str)
	return table[COLONNES], table["classe"], URL_UCI


def _chemin_decoupage(nom: str) -> Path:
	return DOSSIER_DECOUPAGES / f"vote_{nom}.csv"
