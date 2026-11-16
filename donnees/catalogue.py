"""Catalogue des 19 jeux de la campagne 1 et copies figées.

Source : OpenML (https://www.openml.org). Pour chaque jeu, le CSV complet
(`data/get_csv/<file_id>`, colonnes ignorées par OpenML comprises) et les types des
variables (`api/v1/json/data/features/<id>`), téléchargés une seule fois dans
donnees/brutes/. Pima, mushroom et vote sont lus depuis leurs copies figées des
phases précédentes (voir DECISIONS.md, campagne 1, étape 2).
Comparaison à la Table 1 de T. Makhalova, M. Trnecka, « A Study of Boolean Matrix
Factorization Under Supervised Settings », ICFCA 2019.
"""

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from donnees.pima import COLONNES_ZERO_MANQUANT

DOSSIER_DONNEES = Path(__file__).parent
DOSSIER_BRUTES = DOSSIER_DONNEES / "brutes"
CHEMIN_VARIABLES = DOSSIER_BRUTES / "variables.csv"
CHEMIN_SOURCES = DOSSIER_BRUTES / "sources.csv"
CHEMIN_CATALOGUE = DOSSIER_DONNEES / "catalogue.csv"
CHEMIN_TABLE1 = DOSSIER_DONNEES / "table1_makhalova_trnecka.csv"
DOSSIER_PLIS = DOSSIER_DONNEES / "decoupages" / "campagne1"
API_OPENML = "https://www.openml.org/api/v1/json"
SEUIL_CLASSE_RARE = 10
MAX_VALEURS_CATEGORIELLE = 5
# La Table 1 arrondit à 2 décimales : un écart jusqu'à 0,005 n'est que de l'arrondi
TOLERANCE_ARRONDI = 0.005


@dataclass(frozen=True)
class Jeu:
	nom: str
	id_openml: int
	version: int
	taille_attendue: int
	copie_existante: str | None = None


JEUX = [
	Jeu("anneal", 2, 1, 898),
	Jeu("breast-w", 15, 1, 699),
	Jeu("hepatitis", 55, 1, 155),
	Jeu("colic", 27, 2, 368),
	Jeu("iris", 61, 1, 150),
	Jeu("led7", 40678, 1, 3200),
	Jeu("mushroom", 24, 1, 8124, copie_existante="mushroom.csv"),
	Jeu("nursery", 26, 1, 12960),
	Jeu("page-blocks", 30, 1, 5473),
	Jeu("pima", 37, 1, 768, copie_existante="pima.csv"),
	Jeu("wine", 187, 1, 178),
	Jeu("balance-scale", 11, 1, 625),
	Jeu("breast-cancer", 13, 1, 286),
	Jeu("credit-approval", 29, 1, 690),
	Jeu("glass", 41, 1, 214),
	Jeu("heart-statlog", 53, 1, 270),
	Jeu("vowel", 307, 2, 990),
	Jeu("zoo", 62, 1, 101),
	Jeu("vote", 56, 1, 435, copie_existante="vote.csv"),
]

# Règle : identifiant, marqueur de découpage, ou variable observée après la cible
COLONNES_RETIREES = {
	"zoo": {"animal": "identifiant"},
	"vowel": {"Train_or_Test": "marqueur de découpage", "Speaker_Number": "identifiant"},
	"colic": {"outcome": "observée après la cible"},
}

SOURCES_EXISTANTES = {
	"pima": ("miroir GitHub UCI (J. Brownlee)", "2026-10-21"),
	"mushroom": ("OpenML mushroom version 1", "2026-10-28"),
	"vote": ("OpenML vote version 1", "2026-10-28"),
}


def figer_jeux() -> None:
	"""Télécharge les 16 jeux OpenML et écrit brutes/<jeu>.csv, variables.csv, sources.csv."""
	if DOSSIER_BRUTES.exists() and any(DOSSIER_BRUTES.iterdir()):
		raise FileExistsError(f"{DOSSIER_BRUTES} n'est pas vide : les copies figées ne sont jamais remplacées.")
	DOSSIER_BRUTES.mkdir(exist_ok=True)
	variables, sources = [], []
	for jeu in JEUX:
		if jeu.copie_existante is None:
			sources.append(_figer_depuis_openml(jeu, variables))
		else:
			sources.append(_decrire_copie_existante(jeu, variables))
	pd.DataFrame(variables).to_csv(CHEMIN_VARIABLES, index=False)
	pd.DataFrame(sources).to_csv(CHEMIN_SOURCES, index=False)


def chemin_copie(nom: str) -> Path:
	jeu = _jeu(nom)
	if jeu.copie_existante is None:
		return DOSSIER_BRUTES / f"{nom}.csv"
	return DOSSIER_DONNEES / jeu.copie_existante


def lire_variables(nom: str) -> pd.DataFrame:
	table = pd.read_csv(CHEMIN_VARIABLES, keep_default_na=False)
	return table[table["jeu"] == nom].reset_index(drop=True)


def charger_jeu(nom: str) -> tuple[pd.DataFrame, pd.Series]:
	"""X (sans les colonnes retirées ; manquant = NaN ; numériques en float) et y (libellés en texte)."""
	variables = lire_variables(nom)
	table = _lire_copie(nom)
	retirees = COLONNES_RETIREES.get(nom, {})
	gardees = variables[(variables["role"] == "variable") & ~variables["variable"].isin(retirees)]
	X = table[gardees["variable"].tolist()].copy()
	for colonne in gardees.loc[gardees["type"] == "numerique", "variable"]:
		X[colonne] = pd.to_numeric(X[colonne]).astype(float)
	if nom == "pima":
		X[COLONNES_ZERO_MANQUANT] = X[COLONNES_ZERO_MANQUANT].mask(X[COLONNES_ZERO_MANQUANT] == 0)
	classe = variables.loc[variables["role"] == "classe", "variable"].item()
	return X, table[classe].astype(str)


def ecrire_catalogue() -> None:
	"""Écrit donnees/catalogue.csv à partir des copies figées seules (sans internet)."""
	sources = pd.read_csv(CHEMIN_SOURCES, keep_default_na=False).set_index("jeu")
	table1 = pd.read_csv(CHEMIN_TABLE1, comment="#", keep_default_na=False).set_index("jeu_catalogue")
	lignes = []
	for jeu in JEUX:
		X, y = charger_jeu(jeu.nom)
		variables = lire_variables(jeu.nom)
		numeriques = [c for c in X.columns if pd.api.types.is_float_dtype(X[c])]
		peu_de_valeurs = [c for c in numeriques if X[c].nunique() <= MAX_VALEURS_CATEGORIELLE]
		retirees = [c for c in COLONNES_RETIREES.get(jeu.nom, {}) if c in set(variables["variable"])]
		effectifs = y.value_counts()
		ligne = {
			"jeu": jeu.nom, **sources.loc[jeu.nom].to_dict(),
			"fichier": chemin_copie(jeu.nom).relative_to(DOSSIER_DONNEES.parent).as_posix(),
			"sha256": _sha256(chemin_copie(jeu.nom)),
			"taille": len(X), "taille_attendue": jeu.taille_attendue, "ecart_taille": len(X) - jeu.taille_attendue,
			"nb_variables": X.shape[1], "nb_numeriques": len(numeriques),
			"nb_categorielles": X.shape[1] - len(numeriques),
			"numeriques_au_plus_5_valeurs": ";".join(peu_de_valeurs),
			"colonnes_retirees": ";".join(f"{c} ({COLONNES_RETIREES[jeu.nom][c]})" for c in retirees),
			"ignorees_par_openml": ";".join(variables.loc[variables["ignoree_openml"] == "oui", "variable"]),
			"valeurs_manquantes": int(X.isna().sum().sum()),
			"objets_avec_manquant": int(X.isna().any(axis=1).sum()),
			"nb_classes": len(effectifs),
			"repartition": ";".join(f"{classe}:{n}" for classe, n in effectifs.items()),
			"proportions_triees": "/".join(f"{n / len(y):.2f}" for n in effectifs),
			"classes_rares": ";".join(f"{classe}:{n}" for classe, n in effectifs.items() if n < SEUIL_CLASSE_RARE),
		}
		if jeu.nom in table1.index:
			ligne |= _comparer_table1(table1.loc[jeu.nom], effectifs)
		lignes.append(ligne)
	pd.DataFrame(lignes).convert_dtypes().to_csv(CHEMIN_CATALOGUE, index=False)


class PlisInvalides(Exception):
	"""Le fichier de plis figés ne correspond plus au jeu (nombre de lignes, indices ou classes)."""


def figer_plis_campagne1(noms: list[str] | None = None, dossier: Path = DOSSIER_PLIS) -> None:
	"""10 plis stratifiés (random_state=42) sur chaque jeu entier, écrits une fois pour toutes."""
	dossier.mkdir(parents=True, exist_ok=True)
	for nom in noms or [jeu.nom for jeu in JEUX]:
		chemin = dossier / f"{nom}.csv"
		if chemin.exists():
			raise FileExistsError(f"{chemin} existe déjà : les plis ne sont jamais régénérés.")
		X, y = charger_jeu(nom)
		decoupeur = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
		with warnings.catch_warnings(record=True) as avertissements:
			warnings.simplefilter("always")
			decoupage = list(decoupeur.split(X, y))
		for avertissement in avertissements:
			print(f"SIGNALÉ {nom} : {avertissement.message}")
		lignes = [
			{"indice": int(i), "pli": numero_pli, "classe": y.iloc[i]}
			for numero_pli, (_, evalues) in enumerate(decoupage) for i in evalues
		]
		pd.DataFrame(lignes).to_csv(chemin, index=False)


def charger_plis(nom: str, y: pd.Series, dossier: Path = DOSSIER_PLIS) -> list[tuple[np.ndarray, np.ndarray]]:
	"""Pour chaque pli : (indices de la partie d'apprentissage, indices de la partie évaluée)."""
	chemin = dossier / f"{nom}.csv"
	table = pd.read_csv(chemin, dtype={"classe": str}, keep_default_na=False)
	if sorted(table["indice"]) != list(range(len(y))):
		raise PlisInvalides(f"{chemin} : {len(table)} lignes, indices différents des {len(y)} objets du jeu")
	if not (y.iloc[table["indice"]].to_numpy() == table["classe"].to_numpy()).all():
		raise PlisInvalides(f"{chemin} : classes stockées différentes de celles du jeu")
	plis = []
	for numero_pli in sorted(table["pli"].unique()):
		dans_le_pli = table["pli"] == numero_pli
		plis.append((np.sort(table.loc[~dans_le_pli, "indice"].to_numpy()), table.loc[dans_le_pli, "indice"].to_numpy()))
	return plis


def _comparer_table1(article: pd.Series, effectifs: pd.Series) -> dict:
	"""Objets, nombre de classes et répartition triée, comparés à la Table 1 (écarts signalés, rien corrigé)."""
	n = int(effectifs.sum())
	comparaison = {
		"objets_article": int(article["objets"]), "ecart_objets_article": n - int(article["objets"]),
		"repartition_article": article["repartition_classes"],
	}
	try:
		proportions_article = sorted((float(p) for p in article["repartition_classes"].split("/")), reverse=True)
	except ValueError:
		return comparaison | {"comparaison_repartition": "non comparée : répartition non numérique dans la Table 1"}
	proportions = [k / n for k in effectifs]
	comparaison["nb_classes_article"] = len(proportions_article)
	comparaison["ecart_nb_classes_article"] = len(proportions) - len(proportions_article)
	if len(proportions) != len(proportions_article):
		return comparaison | {"comparaison_repartition": "non comparée : nombres de classes différents"}
	ecart = max(abs(p - q) for p, q in zip(proportions, proportions_article))
	comparaison["ecart_max_repartition"] = round(ecart, 3)
	comparaison["comparaison_repartition"] = "conforme à l'arrondi près" if ecart <= TOLERANCE_ARRONDI else "écart"
	return comparaison


def _lire_copie(nom: str) -> pd.DataFrame:
	if _jeu(nom).copie_existante is not None:
		return pd.read_csv(chemin_copie(nom), dtype=str, keep_default_na=False, na_values=["?"])
	# CSV d'OpenML : en-tête entre guillemets doubles, valeurs entre apostrophes (style ARFF)
	table = pd.read_csv(chemin_copie(nom), dtype=str, keep_default_na=False, na_values=["?"], quotechar="'")
	table.columns = [colonne.strip('"') for colonne in table.columns]
	return table


def _figer_depuis_openml(jeu: Jeu, variables: list[dict]) -> dict:
	description = _lire_json(f"{API_OPENML}/data/{jeu.id_openml}")["data_set_description"]
	if int(description["version"]) != jeu.version:
		raise ValueError(f"{jeu.nom} : version OpenML {description['version']}, attendue {jeu.version}")
	url = f"https://www.openml.org/data/get_csv/{description['file_id']}/{jeu.nom}.csv"
	chemin_copie(jeu.nom).write_bytes(_lire_octets(url))
	caracteristiques = _lire_json(f"{API_OPENML}/data/features/{jeu.id_openml}")["data_features"]["feature"]
	for colonne in COLONNES_RETIREES.get(jeu.nom, {}):
		if colonne not in {c["name"] for c in caracteristiques}:
			print(f"SIGNALÉ {jeu.nom} : colonne à retirer « {colonne} » absente de la version OpenML")
	for c in caracteristiques:
		if c["data_type"] not in ("numeric", "nominal"):
			print(f"SIGNALÉ {jeu.nom} : « {c['name']} » de type OpenML {c['data_type']}")
		ignoree = c["is_ignore"] == "true" or c["is_row_identifier"] == "true"
		variables.append({
			"jeu": jeu.nom, "variable": c["name"],
			"type": "numerique" if c["data_type"] == "numeric" else "categorielle",
			"role": "classe" if c["is_target"] == "true" else "variable",
			"ignoree_openml": "oui" if ignoree else "non",
		})
	return {
		"jeu": jeu.nom, "source": f"OpenML {description['name']} version {description['version']}",
		"id_openml": jeu.id_openml, "nom_openml": description["name"], "version": int(description["version"]),
		"licence": description.get("licence", ""), "url": url, "date_telechargement": date.today().isoformat(),
	}


def _decrire_copie_existante(jeu: Jeu, variables: list[dict]) -> dict:
	type_ = "numerique" if jeu.nom == "pima" else "categorielle"
	for colonne in pd.read_csv(chemin_copie(jeu.nom), nrows=0).columns:
		variables.append({
			"jeu": jeu.nom, "variable": colonne,
			"type": "categorielle" if colonne == "classe" else type_,
			"role": "classe" if colonne == "classe" else "variable", "ignoree_openml": "non",
		})
	source, date_telechargement = SOURCES_EXISTANTES[jeu.nom]
	return {
		"jeu": jeu.nom, "source": source, "id_openml": jeu.id_openml, "nom_openml": "",
		"version": jeu.version, "licence": "", "url": "", "date_telechargement": date_telechargement,
	}


def _jeu(nom: str) -> Jeu:
	return next(jeu for jeu in JEUX if jeu.nom == nom)


def _lire_json(url: str) -> dict:
	return json.loads(_lire_octets(url))


def _lire_octets(url: str) -> bytes:
	with urlopen(url, timeout=120) as reponse:
		return reponse.read()


def _sha256(chemin: Path) -> str:
	return hashlib.sha256(chemin.read_bytes()).hexdigest()
