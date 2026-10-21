"""Chargeur de l'exemple des 10 personnes (10 objets, 8 attributs binaires, classe D/N)."""

from pathlib import Path

import numpy as np
import pandas as pd

CHEMIN_EXEMPLE = Path(__file__).parent / "exemple_10_personnes.csv"


def charger_exemple_10_personnes() -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
	"""Retourne I (10×8, booléen), noms des objets, noms des attributs, classes (D/N)."""
	table = pd.read_csv(CHEMIN_EXEMPLE)
	objets = table["objet"].tolist()
	classes = table["classe"].to_numpy()
	colonnes_attributs = table.drop(columns=["objet", "classe"])
	attributs = colonnes_attributs.columns.tolist()
	I = colonnes_attributs.to_numpy().astype(bool)
	return I, objets, attributs, classes
