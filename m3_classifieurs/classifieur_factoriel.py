"""M3 : transformation des facteurs en classifieurs, vote all-votes ou best-vote.

Référence : T. Makhalova, M. Trnecka, « A Study of Boolean Matrix Factorization
Under Supervised Settings », ICFCA 2019.

Chaque facteur devient une règle « intention → classe » pondérée. Une règle couvre x
si x possède tous les attributs de son intention ; sinon elle s'abstient.
Définitions et cas particuliers : voir DECISIONS.md (2026-10-22).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from m2_facteurs.fca import Concept, extension

RAISON_EGALITE = "égalité de classe"
RAISON_NON_COUVERT = "aucun objet couvert en validation"


@dataclass(frozen=True)
class Regle:
	"""Un facteur vu comme règle « intention → classe », avec son poids."""

	numero_facteur: int
	intention: np.ndarray
	classe: object
	poids: float


@dataclass(frozen=True)
class PredictionFactorielle:
	"""Sortie de M3 pour un x (les règles couvrantes serviront à M5 et M6)."""

	classe: object
	scores: dict
	regles_couvrantes: list[Regle]
	non_couvert: bool
	egalite: bool


def construire_regles(
	facteurs: list[Concept],
	y_apprentissage,
	I_poids: np.ndarray,
	y_poids,
	mesure: str = "precision",
	k: int | None = None,
) -> tuple[list[Regle], dict[int, str]]:
	"""Règles des k premiers facteurs, et facteurs écartés {numéro : raison}.

	Classe : majorité de l'extension sur l'apprentissage. Poids : précision, accuracy
	ou lift mesurés sur (I_poids, y_poids), l'apprentissage ou la validation.
	lift = précision / P(c), avec P(c) la proportion de c dans y_apprentissage.
	"""
	y_apprentissage = np.asarray(y_apprentissage)
	y_poids = np.asarray(y_poids)
	regles = []
	ecartes = {}
	for numero, facteur in enumerate(facteurs[:k], start=1):
		classe = _classe_majoritaire(y_apprentissage[facteur.extension])
		if classe is None:
			ecartes[numero] = RAISON_EGALITE
			continue
		couverts = extension(I_poids, facteur.intention)
		if not couverts.any():
			ecartes[numero] = RAISON_NON_COUVERT
			continue
		if mesure == "precision":
			poids = _precision(couverts, y_poids, classe)
		elif mesure == "lift":
			poids = _precision(couverts, y_poids, classe) / float((y_apprentissage == classe).mean())
		else:
			poids = _accuracy(couverts, y_poids, classe)
		regles.append(Regle(numero, facteur.intention, classe, poids))
	return regles, ecartes


def predire(
	regles: list[Regle],
	I: np.ndarray,
	classes: list,
	classe_par_defaut,
	vote: str = "all-votes",
) -> list[PredictionFactorielle]:
	"""Prédiction de chaque ligne de I par all-votes (somme des poids) ou best-vote (maximum).

	Égalité de scores : classe_par_defaut si elle est parmi les gagnantes, sinon la première
	des gagnantes dans l'ordre de classes (DECISIONS.md, campagne 1, étape 3).
	"""
	predictions = []
	for x in I:
		couvrantes = [r for r in regles if x[r.intention].all()]
		if not couvrantes:
			scores = {c: 0.0 for c in classes}
			predictions.append(PredictionFactorielle(classe_par_defaut, scores, [], True, False))
			continue

		scores = {}
		for c in classes:
			poids_c = [r.poids for r in couvrantes if r.classe == c]
			if vote == "all-votes":
				scores[c] = float(sum(poids_c))
			else:
				scores[c] = float(max(poids_c, default=0.0))

		meilleur_score = max(scores.values())
		gagnantes = [c for c in classes if scores[c] == meilleur_score]
		egalite = len(gagnantes) > 1
		classe = departager(gagnantes, classe_par_defaut)
		predictions.append(PredictionFactorielle(classe, scores, couvrantes, False, egalite))
	return predictions


def classes_par_frequence(y) -> list:
	"""Classes triées par effectif décroissant, puis par libellé : ordre de départage des égalités."""
	valeurs, effectifs = np.unique(np.asarray(y), return_counts=True)
	return [valeurs.tolist()[i] for i in sorted(range(len(valeurs)), key=lambda i: -effectifs[i])]


def departager(gagnantes: list, classe_par_defaut):
	"""Classe retenue parmi les gagnantes (dans l'ordre de classes) : règle d'égalité commune à M3 et au bagging."""
	if len(gagnantes) == 1:
		return gagnantes[0]
	return classe_par_defaut if classe_par_defaut in gagnantes else gagnantes[0]


def regles_qui_votent(prediction: PredictionFactorielle, vote: str) -> list[Regle]:
	"""Règles qui votent effectivement pour x (voir DECISIONS.md, étape 13).

	all-votes : toutes les règles couvrantes. best-vote : la seule règle décisive, de poids
	maximal parmi celles de la classe prédite (la première dans l'ordre des facteurs à égalité).
	Aucune règle pour un x non couvert, ni pour une égalité de scores en best-vote.
	"""
	if prediction.non_couvert:
		return []
	if vote == "all-votes":
		return list(prediction.regles_couvrantes)
	if prediction.egalite:
		return []
	de_la_classe = [r for r in prediction.regles_couvrantes if r.classe == prediction.classe]
	# max garde le premier élément en cas d'égalité de poids
	return [max(de_la_classe, key=lambda r: r.poids)]


def compter(predictions: list[PredictionFactorielle]) -> dict:
	"""Nombres et taux de x non couverts et d'égalités, rapportés à part de l'accuracy."""
	nb_objets = len(predictions)
	nb_non_couverts = sum(p.non_couvert for p in predictions)
	nb_egalites = sum(p.egalite for p in predictions)
	return {
		"nb_objets": nb_objets,
		"nb_non_couverts": nb_non_couverts,
		"taux_non_couverts": nb_non_couverts / nb_objets,
		"nb_egalites": nb_egalites,
		"taux_egalites": nb_egalites / nb_objets,
	}


def decrire_facteurs(
	facteurs: list[Concept],
	noms_attributs: list[str],
	y_apprentissage,
	I_validation: np.ndarray,
	y_validation,
) -> pd.DataFrame:
	"""Une ligne par facteur : classe, couverture en validation, précision et accuracy, écart."""
	y_apprentissage = np.asarray(y_apprentissage)
	y_validation = np.asarray(y_validation)
	lignes = []
	for numero, facteur in enumerate(facteurs, start=1):
		classe = _classe_majoritaire(y_apprentissage[facteur.extension])
		couverts_val = extension(I_validation, facteur.intention)
		ligne = {
			"facteur": f"F{numero}",
			"intention": " ET ".join(n for n, g in zip(noms_attributs, facteur.intention) if g),
			"taille_extension": int(facteur.extension.sum()),
			"classe": classe,
			"couverts_validation": int(couverts_val.sum()),
			"precision_apprentissage": np.nan,
			"accuracy_apprentissage": np.nan,
			"precision_validation": np.nan,
			"accuracy_validation": np.nan,
			"ecarte": False,
			"raison": "",
		}
		if classe is None:
			ligne["ecarte"], ligne["raison"] = True, RAISON_EGALITE
		else:
			# Sur l'apprentissage, les objets couverts sont exactement l'extension du facteur
			ligne["precision_apprentissage"] = _precision(facteur.extension, y_apprentissage, classe)
			ligne["accuracy_apprentissage"] = _accuracy(facteur.extension, y_apprentissage, classe)
			if couverts_val.any():
				ligne["precision_validation"] = _precision(couverts_val, y_validation, classe)
				ligne["accuracy_validation"] = _accuracy(couverts_val, y_validation, classe)
			else:
				# Écarté seulement pour les variantes à poids sur la validation
				ligne["ecarte"], ligne["raison"] = True, RAISON_NON_COUVERT
		lignes.append(ligne)
	return pd.DataFrame(lignes).set_index("facteur")


def _classe_majoritaire(classes_extension: np.ndarray):
	"""Classe la plus fréquente ; None en cas d'égalité entre les premières."""
	valeurs, effectifs = np.unique(classes_extension, return_counts=True)
	if (effectifs == effectifs.max()).sum() > 1:
		return None
	return valeurs.tolist()[effectifs.argmax()]


def _precision(couverts: np.ndarray, y: np.ndarray, classe) -> float:
	"""Parmi les objets couverts, proportion de la classe de la règle."""
	return float((y[couverts] == classe).mean())


def _accuracy(couverts: np.ndarray, y: np.ndarray, classe) -> float:
	"""(tp + tn) / |S| : la règle prédit « classe » si couvert, « pas classe » sinon."""
	tp = (couverts & (y == classe)).sum()
	tn = (~couverts & (y != classe)).sum()
	return float((tp + tn) / len(y))
