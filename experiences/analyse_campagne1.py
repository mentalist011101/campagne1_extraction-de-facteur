"""Campagne 1, étape 6 bis : analyse des résultats, selon DECISIONS.md (H8 à H10 et règles d'analyse,
fixées avant toute exécution). Exécuté sans modification après la campagne.

Références : F. Wilcoxon, « Individual comparisons by ranking methods », Biometrics Bulletin 1(6), 1945 ;
S. Holm, « A simple sequentially rejective multiple test procedure », Scand. J. Statistics 6(2), 1979 ;
B. Efron, R. Tibshirani, « An Introduction to the Bootstrap », 1993 (intervalle percentile) ;
J. Demšar, « Statistical Comparisons of Classifiers over Multiple Data Sets », JMLR 7, 2006
(Friedman, Nemenyi, diagramme de la différence critique).

Lancement : python -m experiences.analyse_campagne1 --dossier resultats/campagne1
Sorties : <dossier>/analyse/ (rapport.md, tableaux CSV, diagramme_cd.png).
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from experiences.campagne1 import METHODES, NB_PLIS, RACINE, _git, modifications_non_commitees

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (après le choix du moteur sans écran)

CRITERE_MIN_JEUX = 15
ALPHA = 0.05
SEUIL_H10 = -0.02
NB_TIRAGES = 10_000
GRAINE = 42
Q_NEMENYI = 3.031
TOLERANCE = 1e-12
CHEMIN_TABLE3 = RACINE / "donnees" / "table3_makhalova_trnecka.csv"
COLONNES_UNIQUES = ["commit", "version_python", "version_numpy", "version_pandas", "version_sklearn"]
COLONNES_MACHINE = ["processeur", "plateforme"]
COMPARAISONS = {"H8": ("STACK", "M3-BV"), "H9": ("BAG", "M3-BV"), "H10": ("STACK", "RF")}
VOTES = {"all-votes": "M3-AV", "best-vote": "M3-BV"}


class ResultatsIncoherents(Exception):
	pass


def analyser(dossier_entree: Path, dossier_sortie: Path, chemin_table3: Path = CHEMIN_TABLE3) -> dict:
	dossier_entree, dossier_sortie = Path(dossier_entree), Path(dossier_sortie)
	par_pli = pd.read_csv(dossier_entree / "campagne1_par_pli.csv")
	par_jeu = pd.read_csv(dossier_entree / "campagne1_par_jeu.csv")
	anomalies, machines = verifier_tracabilite(par_pli)
	f1 = moyennes_par_jeu(par_pli, par_jeu, "f1_macro")
	accuracy = moyennes_par_jeu(par_pli, par_jeu, "accuracy")
	retenus, exclus = selectionner_jeux(par_jeu, dossier_entree)
	concluante = len(retenus) >= CRITERE_MIN_JEUX
	hypotheses, differences = tester_hypotheses(f1.loc[retenus]) if concluante else (None, None)
	friedman = friedman_nemenyi(f1.loc[retenus, METHODES])
	groupes = groupes_non_distincts(friedman[2], friedman[3])
	table3 = comparer_table3(accuracy, chemin_table3)

	dossier_sortie.mkdir(parents=True, exist_ok=True)
	f1.to_csv(dossier_sortie / "f1_par_jeu.csv")
	friedman[2].rename("rang_moyen").to_csv(dossier_sortie / "rangs.csv", index_label="methode")
	exclus.to_csv(dossier_sortie / "jeux_exclus.csv", index=False)
	anomalies.to_csv(dossier_sortie / "tracabilite.csv", index=False)
	if concluante:
		hypotheses.to_csv(dossier_sortie / "hypotheses.csv", index=False)
		differences.to_csv(dossier_sortie / "differences.csv", index_label="jeu")
	if table3 is not None:
		table3.to_csv(dossier_sortie / "table3.csv", index=False)
	tracer_cd(friedman[2], friedman[3], groupes, dossier_sortie / "diagramme_cd.png")
	rapport = ecrire_rapport(dossier_entree, retenus, exclus, anomalies, machines, hypotheses, f1, friedman,
		groupes, par_jeu, table3)
	(dossier_sortie / "rapport.md").write_text(rapport, encoding="utf-8")
	return {"concluante": concluante, "retenus": retenus, "hypotheses": hypotheses, "anomalies": anomalies}


def verifier_tracabilite(par_pli: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Anomalies (plis modifiés, écarts de commit ou de versions) et machines utilisées ; rien n'est exclu."""
	plis = par_pli.drop_duplicates(["jeu", "pli"])
	lignes = [{"type": "modifications_non_commitees", "jeu": pli.jeu, "pli": pli.pli, "detail": f"commit {pli.commit}"}
		for pli in plis[plis["modifications_non_commitees"]].itertuples()]
	for colonne in COLONNES_UNIQUES:
		valeurs = plis[colonne].astype(str).value_counts()
		if len(valeurs) > 1:
			lignes.append({"type": f"écart de {colonne}", "jeu": "", "pli": np.nan,
				"detail": " ; ".join(f"{valeur} ({nombre} plis)" for valeur, nombre in valeurs.items())})
	anomalies = pd.DataFrame(lignes, columns=["type", "jeu", "pli", "detail"])
	machines = plis.groupby(COLONNES_MACHINE).size().rename("nb_plis").reset_index()
	return anomalies, machines


def moyennes_par_jeu(par_pli: pd.DataFrame, par_jeu: pd.DataFrame, colonne: str) -> pd.DataFrame:
	"""Moyenne non arrondie sur les 10 plis `ok` (vide sinon) ; doit retrouver campagne1_par_jeu.csv."""
	ok = par_pli[par_pli["statut"] == "ok"].groupby(["jeu", "methode"])[colonne]
	moyennes = ok.mean().where(ok.size() == NB_PLIS).unstack("methode")
	jeux = pd.unique(par_jeu["jeu"])
	table = moyennes.reindex(index=jeux, columns=METHODES)
	stockees = par_jeu.pivot(index="jeu", columns="methode", values=colonne).reindex(index=jeux, columns=METHODES)
	if ((table - stockees).abs() > TOLERANCE).any().any() or (table.isna() != stockees.isna()).any().any():
		raise ResultatsIncoherents(f"{colonne} : campagne1_par_pli.csv et campagne1_par_jeu.csv ne concordent pas")
	return table.rename_axis(index="jeu", columns=None)


def selectionner_jeux(par_jeu: pd.DataFrame, dossier: Path) -> tuple[list[str], pd.DataFrame]:
	"""Jeux `termine` (même ensemble pour tous les tests) ; les autres, avec leur raison."""
	statuts = par_jeu.groupby("jeu", sort=False).agg(statut=("statut", "first"), nb_plis=("nb_plis", "max"))
	retenus = statuts.index[statuts["statut"] == "termine"].tolist()
	exclus = [{"jeu": nom, "statut": ligne.statut, "raison": _raison(dossier / nom, ligne.statut, ligne.nb_plis)}
		for nom, ligne in statuts[statuts["statut"] != "termine"].iterrows()]
	return retenus, pd.DataFrame(exclus, columns=["jeu", "statut", "raison"])


def tester_hypotheses(f1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	differences = pd.DataFrame({nom: f1[a] - f1[b] for nom, (a, b) in COMPARAISONS.items()})
	lignes = {nom: {"hypothese": nom, "comparaison": " - ".join(COMPARAISONS[nom]), **appliquer_wilcoxon(differences[nom])}
		for nom in ("H8", "H9")}
	p_holm = holm({nom: ligne["p_brut"] for nom, ligne in lignes.items()})
	for nom, ligne in lignes.items():
		ligne["p_holm"] = p_holm[nom]
		ligne["verdict"] = verdict_superiorite(p_holm[nom], differences[nom])
	moyenne, basse, haute = bootstrap_moyenne(differences["H10"])
	h10 = {"hypothese": "H10", "comparaison": "STACK - RF", **_resume(differences["H10"]), "moyenne_d": moyenne,
		"borne_basse": basse, "borne_haute": haute, "verdict": "confirmée" if basse > SEUIL_H10 else "rejetée"}
	return pd.DataFrame([lignes["H8"], lignes["H9"], h10]), differences


def appliquer_wilcoxon(d: pd.Series) -> dict:
	"""scipy.stats.wilcoxon(d), réglages par défaut : zero_method="wilcox" (d nulles retirées), method="auto".

	Dans scipy 1.18.1, le mode auto choisit : approximation normale si len(d) > 50 ; sinon calcul exact s'il
	n'y a ni ex-aequo ni d nulle ; sinon permutations exhaustives si len(d) <= 13, approximation normale
	au-delà. Toutes les d nulles : scipy renvoie p = NaN.
	"""
	resultat = wilcoxon(d.to_numpy())
	return {**_resume(d), "W": float(resultat.statistic), "p_brut": float(resultat.pvalue)}


def holm(p_bruts: dict[str, float]) -> dict[str, float]:
	"""Holm sur deux p (DECISIONS.md) ; un p non défini (toutes les d nulles) reste non défini."""
	(nom_1, p_1), (nom_2, p_2) = sorted(p_bruts.items(), key=lambda element: (np.isnan(element[1]), element[1]))
	premier = np.nan if np.isnan(p_1) else min(1.0, 2 * p_1)
	return {nom_1: premier, nom_2: np.nan if np.isnan(p_2) else min(1.0, max(premier, p_2))}


def verdict_superiorite(p_holm: float, d: pd.Series) -> str:
	if np.isnan(p_holm):
		return "p non défini, hypothèse rejetée (aucune différence)"
	if p_holm < ALPHA and d.median() > 0:
		return "confirmée"
	if p_holm < ALPHA and d.median() < 0:
		return "rejetée (différence significative dans le sens contraire)"
	return "rejetée"


def bootstrap_moyenne(d: pd.Series, nb_tirages: int = NB_TIRAGES, graine: int = GRAINE) -> tuple[float, float, float]:
	"""Intervalle percentile à 95 % de la moyenne des d ; tirage des jeux avec remise, en une matrice."""
	valeurs = d.to_numpy()
	indices = np.random.default_rng(graine).integers(0, len(valeurs), size=(nb_tirages, len(valeurs)))
	basse, haute = np.percentile(valeurs[indices].mean(axis=1), [2.5, 97.5])
	return float(valeurs.mean()), float(basse), float(haute)


def friedman_nemenyi(f1: pd.DataFrame) -> tuple[float, float, pd.Series, float]:
	"""Friedman, rangs moyens (1 = meilleure F1, ex-aequo au rang moyen) et CD de Nemenyi (q pour k = 8)."""
	statistique, p = friedmanchisquare(*[f1[methode].to_numpy() for methode in f1.columns])
	rangs = pd.DataFrame(rankdata(-f1.to_numpy(), axis=1), index=f1.index, columns=f1.columns).mean()
	k, n = f1.shape[1], f1.shape[0]
	return float(statistique), float(p), rangs, float(Q_NEMENYI * np.sqrt(k * (k + 1) / (6 * n)))


def groupes_non_distincts(rangs: pd.Series, cd: float) -> list[list[str]]:
	"""Groupes maximaux de méthodes dont les rangs moyens diffèrent de moins de CD."""
	ordre = rangs.sort_values()
	noms, valeurs = ordre.index.tolist(), ordre.to_numpy()
	groupes = []
	for i in range(len(noms)):
		j = i
		while j + 1 < len(noms) and valeurs[j + 1] - valeurs[i] < cd:
			j += 1
		if j > i and not (groupes and groupes[-1][-1] == noms[j]):
			groupes.append(noms[i:j + 1])
	return groupes


def tracer_cd(rangs: pd.Series, cd: float, groupes: list[list[str]], chemin: Path) -> None:
	"""Diagramme de Demšar : axe des rangs, barre CD, groupes non distincts reliés par un trait épais."""
	ordre = rangs.sort_values()
	k, moitie = len(ordre), (len(ordre) + 1) // 2
	figure, axe = plt.subplots(figsize=(8, 2.2 + 0.25 * moitie + 0.1 * len(groupes)))
	axe.axis("off")
	axe.hlines(0, 1, k, color="black")
	for rang in range(1, k + 1):
		axe.vlines(rang, 0, 0.08, color="black")
		axe.text(rang, 0.12, str(rang), ha="center", va="bottom", fontsize=9)
	axe.hlines(0.55, 1, 1 + cd, color="black", linewidth=2)
	axe.text(1 + cd / 2, 0.6, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=9)
	for g, groupe in enumerate(groupes):
		axe.hlines(-0.12 - 0.1 * g, ordre[groupe[0]] - 0.03, ordre[groupe[-1]] + 0.03, color="black", linewidth=3)
	depart = -0.3 - 0.1 * len(groupes)
	for i, (nom, rang) in enumerate(ordre.items()):
		gauche = i < moitie
		y = depart - 0.25 * (i if gauche else k - 1 - i)
		x_texte = 0.6 if gauche else k + 0.4
		axe.vlines(rang, y, 0, color="black", linewidth=0.8)
		axe.hlines(y, min(x_texte, rang), max(x_texte, rang), color="black", linewidth=0.8)
		axe.text(x_texte - 0.05 if gauche else x_texte + 0.05, y, f"{nom} ({rang:.2f})",
			ha="right" if gauche else "left", va="center", fontsize=9)
	axe.set_xlim(-1.5, k + 2.5)
	axe.set_ylim(depart - 0.25 * moitie, 0.9)
	figure.savefig(chemin, dpi=150, bbox_inches="tight")
	plt.close(figure)


def comparer_table3(accuracy: pd.DataFrame, chemin: Path) -> pd.DataFrame | None:
	"""Cases de la Table 3 de même configuration que la campagne (poids par précision) ; None sans fichier."""
	if not Path(chemin).exists():
		return None
	article = pd.read_csv(chemin, comment="#")
	article = article[(article["ponderation"] == "precision") & (article["mesure"] == "accuracy")
		& article["vote"].isin(VOTES)]
	lignes = [{"jeu": ligne.jeu, "vote": ligne.vote, "methode": VOTES[ligne.vote], "valeur_article": ligne.valeur,
		"accuracy_campagne": accuracy.at[ligne.jeu, VOTES[ligne.vote]] if ligne.jeu in accuracy.index else np.nan}
		for ligne in article.itertuples()]
	table = pd.DataFrame(lignes, columns=["jeu", "vote", "methode", "valeur_article", "accuracy_campagne"])
	return table.assign(ecart=table["accuracy_campagne"] - table["valeur_article"])


def ecrire_rapport(dossier, retenus, exclus, anomalies, machines, hypotheses, f1, friedman, groupes, par_jeu,
		table3) -> str:
	statistique, p, rangs, cd = friedman
	n = len(retenus)
	lignes = [
		"# Campagne 1 : analyse", "",
		f"- Résultats analysés : `{_chemin_relatif(dossier)}`",
		f"- Script d'analyse : commit `{_git(['rev-parse', 'HEAD']) or 'inconnu'}`, modifications non commitées : "
		f"{modifications_non_commitees()}",
		f"- Versions : scipy {scipy.__version__}, numpy {np.__version__}, pandas {pd.__version__}",
		f"- Date de l'analyse : {datetime.now().isoformat(timespec='seconds')}", "",
	]
	if n >= CRITERE_MIN_JEUX:
		lignes += [f"**Campagne concluante ({n} jeux terminés >= {CRITERE_MIN_JEUX}).**", ""]
	else:
		lignes += [f"**Campagne non concluante ({n} jeux terminés < {CRITERE_MIN_JEUX}).** H8 à H10 ne reçoivent "
			"pas de verdict ; seuls les résultats descriptifs sont rapportés (DECISIONS.md).", ""]
	lignes += ["## Anomalies de traçabilité", "",
		"Signalées, non bloquantes : aucun pli n'est exclu pour cette raison.", ""]
	lignes += [_tableau_md(anomalies) if len(anomalies) else "Aucune.", "", "Machines utilisées :", "",
		_tableau_md(machines), ""]
	lignes += [f"## Jeux retenus ({n})", "", ", ".join(retenus) or "Aucun.", "", "## Jeux non retenus", "",
		_tableau_md(exclus) if len(exclus) else "Aucun.", ""]
	if hypotheses is not None:
		lignes += ["## Hypothèses H8 à H10", "",
			"H8, H9 : Wilcoxon signé bilatéral (scipy, réglages par défaut), Holm sur les deux ; confirmée si "
			f"p_holm < {ALPHA} et médiane(d) > 0. H10 : intervalle bootstrap percentile à 95 % de la moyenne de "
			f"STACK - RF ({NB_TIRAGES} tirages, graine {GRAINE}) ; confirmée si borne basse > {SEUIL_H10}.", "",
			_tableau_md(hypotheses, decimales=4), ""]
	lignes += ["## F1 macro par jeu (moyenne sur 10 plis)", "",
		"Arrondie à 3 décimales ici ; valeurs exactes dans `f1_par_jeu.csv`.", "", _tableau_md(f1.reset_index()), ""]
	lignes += ["## Friedman et Nemenyi (descriptif, sans verdict)", "",
		f"Friedman : chi2 = {statistique:.3f}, p = {p:.4g}, N = {n} jeux, k = {len(rangs)} méthodes. "
		f"Nemenyi : CD = {Q_NEMENYI}·√(k(k+1)/(6N)) = {cd:.3f}.", "",
		_tableau_md(rangs.sort_values().rename("rang_moyen").reset_index().rename(columns={"index": "methode"})), "",
		"Groupes non distincts (rangs à moins de CD) : "
		+ ("; ".join("{" + ", ".join(groupe) + "}" for groupe in groupes) or "aucun"), "",
		"![Diagramme de la différence critique](diagramme_cd.png)", ""]
	m3 = par_jeu[par_jeu["methode"].isin(VOTES.values())]
	lignes += ["## Non-couverts et égalités de M3-AV et M3-BV", "",
		_tableau_md(m3.pivot(index="jeu", columns="methode", values=["taux_non_couverts", "taux_egalites"])
			.set_axis([f"{a}_{m}" for a, m in m3.pivot(index="jeu", columns="methode",
				values=["taux_non_couverts", "taux_egalites"]).columns], axis=1)
			.reindex(pd.unique(par_jeu["jeu"])).reset_index()), ""]
	lignes += ["## Table 3 de Makhalova & Trnecka (indicative)", ""]
	if table3 is None:
		lignes += ["Table 3 non fournie (`donnees/table3_makhalova_trnecka.csv` absent) : comparaison non faite.", ""]
	else:
		lignes += ["Comparaison indicative : protocole et binarisation différents. Seules les cases de même "
			"configuration (poids par précision) sont comparées.", "", _tableau_md(table3), ""]
	return "\n".join(lignes)


def main() -> None:
	parseur = argparse.ArgumentParser(description="Campagne 1 : analyse (H8 à H10, descriptif)")
	parseur.add_argument("--dossier", required=True, type=Path)
	parseur.add_argument("--sortie", type=Path, default=None, help="par défaut : <dossier>/analyse")
	arguments = parseur.parse_args()
	dossier = arguments.dossier if arguments.dossier.is_absolute() else RACINE / arguments.dossier
	resultat = analyser(dossier, arguments.sortie or dossier / "analyse")
	print(f"Concluante : {resultat['concluante']} ({len(resultat['retenus'])} jeux retenus)")
	if resultat["hypotheses"] is not None:
		print(resultat["hypotheses"][["hypothese", "p_brut", "p_holm", "mediane_d", "verdict"]].to_string(index=False))


def _resume(d: pd.Series) -> dict:
	return {"n_jeux": len(d), "n_effectifs": int((d != 0).sum()), "mediane_d": float(d.median()),
		"moyenne_d": float(d.mean()), "victoires": int((d > 0).sum()), "defaites": int((d < 0).sum()),
		"egalites": int((d == 0).sum())}


def _chemin_relatif(dossier: Path) -> str:
	"""Chemin relatif à la racine du dépôt : le rapport commité ne contient aucun chemin local."""
	dossier = Path(dossier).resolve()
	if dossier.is_relative_to(RACINE.resolve()):
		return dossier.relative_to(RACINE.resolve()).as_posix()
	return dossier.name


def _raison(dossier: Path, statut: str, nb_plis: int) -> str:
	if statut == "abandonne":
		abandon = pd.read_csv(dossier / "abandon.csv").iloc[0]
		return f"{abandon['raison']} (pli {abandon['pli']}, {abandon['duree_s']} s, {abandon['nb_facteurs']} facteurs)"
	if statut == "termine_sans_bag":
		abandon = pd.read_csv(dossier / "abandon_bag.csv").iloc[0]
		return f"GreConD du sac {abandon['sac']} au-delà de la limite (pli {abandon['pli']}, {abandon['duree_s']} s)"
	return f"{nb_plis} plis calculés sur {NB_PLIS}"


def _tableau_md(table: pd.DataFrame, decimales: int = 3) -> str:
	def cellule(valeur):
		if isinstance(valeur, float):
			return "" if np.isnan(valeur) else f"{valeur:.{decimales}g}" if abs(valeur) < 1e-3 and valeur != 0 \
				else f"{valeur:.{decimales}f}"
		return str(valeur)
	lignes = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "---|" * len(table.columns)]
	lignes += ["| " + " | ".join(cellule(valeur) for valeur in ligne) + " |" for ligne in table.itertuples(index=False)]
	return "\n".join(lignes)


if __name__ == "__main__":
	main()
