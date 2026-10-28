# CLAUDE.md — Mémoire M2 : valider les prédictions d'un modèle profond par des facteurs booléens

Ce fichier est lu à chaque session. Il contient ce qui reste vrai pour tout le projet. La tâche en cours est décrite à part, dans `docs/` (voir « Phase en cours »).

## Le projet

- Mémoire de M2 Data Science (ENSPY), sous la direction de Dr TIOGNING et Dr KAMENI.
- Sujet : « Mesure de l'incertitude structurelle par analyse factorielle des données pour la validation des prédictions d'un modèle de Deep Learning ».
- Idée : on extrait des données des régularités symboliques, les facteurs d'une factorisation booléenne (BMF), dans le cadre de l'analyse formelle de concepts (FCA). Puis on confronte la prédiction d'un modèle profond à ces régularités, pour repérer les prédictions douteuses et les expliquer.
- Ce code sert une recherche. Chaque résultat doit pouvoir être défendu devant un jury : clarté et reproductibilité passent avant la performance.

## La chaîne complète

```
données ─► M1 binarisation ─► M2 facteurs ─► M3 classifieurs factoriels ─┐
   └─────► M4 modèle profond : ŷ(x) et confiance ─────────► M5 confrontation ─► M6 explication
                                                                               ─► M7 évaluation
```

| Module | Dossier | Rôle | État |
|---|---|---|---|
| M1 | `m1_binarisation/` | Données → contexte formel binaire | terminé : échelle nominale ou interordinale (numérique), binarisation catégorielle (« ? » sans attribut, attributs constants retirés) |
| M2 | `m2_facteurs/` | I ≈ A ∘ B : GreCon, GreConD (puis éventuellement MDLGreConD) | terminé : GreCon, GreConD, `nb_max` pour l'énumération, comptage des choix sur égalité |
| M3 | `m3_classifieurs/` | Facteur → règle « intention → classe » avec un poids ; vote all-votes / best-vote | terminé : poids précision, accuracy ou lift, all-votes ou best-vote, `regles_qui_votent` |
| — | `m3_classifieurs/` | Méthodes d'ensemble sur les classifieurs factoriels (axe proposé par l'encadreuse) | premiers essais : stacking (`empilement.py`), bagging (`ensachage.py`) sur vote |
| M4 | (à venir) | Modèle profond, d'abord un MLP pour le tabulaire. Jamais modifié par la chaîne. | à venir |
| M5 | (à venir) | Verdict sur ŷ(x) : confirmée, contestée ou inconnue, puis un score | à venir |
| M6 | (à venir) | Explication : facteurs qui soutiennent ŷ, facteurs qui s'y opposent | à venir |
| M7 | `experiences/` | Évaluation : détection d'erreurs, courbes accuracy–rejet, comparaison au softmax | à venir |

## Phase en cours

**Phase 1 terminée ; phase 2 terminée pour les étapes 10 à 14 (H1 confirmée, H2 et H3 rejetées, H4 confirmée) ; phase 3 légère terminée (H5 et H6 rejetées, H7 confirmée). Reste : étape 15 (synthèse de la phase 2), après la présentation.**
Détails : `JOURNAL.md`. Instructions : `docs/instructions_phase2.md` ; phase 3 : `docs/instructions_phase3_ensembles.md` (jeu de données : voir `DECISIONS.md`). Le test reste scellé sur tous les jeux.
Ne code rien qui appartienne à une phase suivante.

## Comment on travaille

L'étudiant décide ; toi, tu proposes, tu expliques et tu codes. Il doit pouvoir suivre et comprendre chaque ligne.

**Cycle obligatoire pour chaque étape :**
1. Présenter le plan : fichiers, signatures, choix. Attendre son accord.
2. Écrire les tests avec les valeurs attendues, avant le code.
3. Coder par morceaux d'environ 150 lignes au maximum, en s'arrêtant pour montrer.
4. Exécuter et montrer la sortie réelle, pas un résumé.
5. Expliquer le lien entre le code et l'algorithme.
6. Attendre « validé ».
7. Faire un commit git, puis ajouter une entrée dans `JOURNAL.md`.

**Règles :**
- Jamais d'étape suivante sans validation.
- Si un résultat diffère d'une valeur attendue : s'arrêter et expliquer. Ne jamais modifier un test pour qu'il passe.
- Si un choix n'est pas fixé : demander, puis l'écrire dans `DECISIONS.md` avec sa raison.
- Répondre aux questions de l'étudiant avant de continuer.

## Règles scientifiques à ne jamais enfreindre

- **Pas de fuite d'information.** Les seuils de binarisation, les facteurs et les poids sont appris sur l'apprentissage, ou sur la validation pour les variantes de poids. Ils ne sont **jamais** appris sur le test.
- **Le test ne sert qu'aux tableaux finaux.** Aucun réglage de paramètre ne s'y fait, et aucune vérification non plus : les contrôles se font sur l'apprentissage ou la validation.
- **Découpages figés.** Les indices apprentissage / validation / test sont sauvegardés dans `donnees/decoupages/` et réutilisés par toute la chaîne. Ne jamais les régénérer.
- **M2 ne voit pas la classe.** La factorisation se fait sur les attributs seuls ; la classe n'intervient qu'en M3.
- **Un facteur est un concept formel**, et A ∘ B ne crée jamais une croix absente de I. Teste-le.
- **Rien ne se cache.** Les x couverts par aucun facteur, et les égalités de vote, sont comptés et rapportés à part. Ils ne sont jamais noyés dans l'accuracy.
- **Les règles d'égalité sont écrites et testées.** GreConD : premier attribut dans l'ordre des colonnes. GreCon : plus grande extension, puis intention première dans l'ordre lexicographique. Classe d'un facteur à égalité : facteur écarté.
- **Données figées.** Pima se lit uniquement depuis la copie figée `donnees/pima.csv`. Les zéros des colonnes de `COLONNES_ZERO_MANQUANT` sont des valeurs inconnues : sans attribut en M1 (option b). La densité du contexte est donc inférieure à 1/3 (0,298 sur l'apprentissage) ; 8/24 n'est atteint que sans traitement des zéros.
- **Comparaison à information égale.** M3 se compare d'abord aux modèles entraînés sur les données binarisées. Les modèles bruts voient les zéros manquants comme des valeurs (voir `DECISIONS.md`) : la comparaison avec eux n'est pas à traitement égal.
- **Hypothèses avant expériences.** Toute hypothèse et son critère sont commités dans `DECISIONS.md` avant la première expérience qui la teste, puis ne sont plus modifiés.
- **Validation croisée figée par jeu.** Chaque jeu a son découpage 60/20/20 et ses 10 plis figés dans `donnees/decoupages/`. Dans chaque pli, tout est réappris sur sa partie d'apprentissage : binarisation, facteurs, poids, modèles.
- **Données figées, au-delà de Pima :** `donnees/mushroom.csv` et `donnees/vote.csv`, lus uniquement depuis ces copies. Pour vote, « ? » = abstention réelle, sans attribut (limite connue).
- **Graine fixe** : `random_state=42` partout.

## Leçons de méthode

- Un critère « au moins 8 plis sur 10 » doit tenir compte de la taille des plis (sur vote, un représentant ≈ 0,03 de F1) ; on donne aussi le test du signe, à titre descriptif. Les critères déjà figés ne sont pas modifiés.
- Les critères de structure des facteurs se formulent en variables distinctes, pas en attributs (la fermeture interordinale ajoute les attributs impliqués, étape 12).

## Conventions de code

- Python 3.10 ou plus récent ; numpy, pandas, scikit-learn, pytest. Les versions sont notées dans `requirements.txt`.
- Noms et commentaires en français. Fonctions courtes et pures quand c'est possible. Pas d'optimisation prématurée.
- Contexte formel : matrice numpy booléenne (objets × attributs). Ensembles d'objets et d'attributs : masques booléens.
- Chaque algorithme cite sa référence dans la docstring du module.
- M2 accepte n'importe quel contexte binaire : un échantillon, ou un sous-ensemble d'attributs, pour les méthodes d'ensemble à venir.
- M3 renvoie, pour chaque x : la classe prédite, le score par classe, la liste des facteurs qui couvrent x (avec leur classe et leur poids), et un drapeau « non couvert » ou « égalité ».
- Les résultats sont sauvegardés en CSV dans `resultats/`, avec un nom qui indique l'expérience et la date.
- Les expériences obtiennent les contextes de Pima par `experiences/contextes_pima.py`, qui n'apprend jamais les seuils hors de l'apprentissage.

## Commandes

```bash
pip install -r requirements.txt
python -m pytest -q                  # tous les tests (115, environ 1 minute)
python -m pytest -q -p no:warnings   # idem, sans les avertissements de skmine (dépendance de fcapy)
python -m experiences.<script>       # une expérience
python -m experiences.cv_ensembles_vote   # phase 3 : stacking et bagging (environ 2 min)
python -m experiences.synthese_vote       # phase 3 : tableau final et verdicts
```

## Fichiers de suivi

- `JOURNAL.md` : une entrée par étape validée (date, ce qui est fait, résultats, questions ouvertes).
- `DECISIONS.md` : chaque choix fait en cours de route, et pourquoi.
- `docs/` : les instructions de chaque phase.

## Références

- R. Belohlavek, V. Vychodil, « Discovery of optimal factors in binary data via a novel method of matrix decomposition », *Journal of Computer and System Sciences* 76(1), 2010 : GreCon et GreConD.
- T. Makhalova, M. Trnecka, « A Study of Boolean Matrix Factorization Under Supervised Settings », ICFCA 2019 : facteurs → classifieurs, all-votes et best-vote.
- PyBMF (https://github.com/PreferredAI/PyBMF), commit 03c9b43 : son GreConD s'écarte de l'article (facteur final perdu, mise à jour au milieu du parcours des attributs, voir JOURNAL étape 3). Il ne sert pas de référence numérique.
- fcapy 0.1.4.5 : référence pour l'énumération des concepts, dans les tests uniquement.
- L. Breiman, « Bagging predictors », *Machine Learning* 24(2), 1996.
- D. H. Wolpert, « Stacked generalization », *Neural Networks* 5(2), 1992, p. 241-259.
- Jeux mushroom et congressional voting (UCI), obtenus par OpenML (version 1).

## Ce fichier

Ne modifie pas `CLAUDE.md` sans accord. À la fin de chaque phase, propose les mises à jour nécessaires : la phase en cours, les modules terminés, les nouvelles règles.
