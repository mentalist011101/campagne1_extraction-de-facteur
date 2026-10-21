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

| Module | Dossier | Rôle |
|---|---|---|
| M1 | `m1_binarisation/` | Données → contexte formel binaire |
| M2 | `m2_facteurs/` | I ≈ A ∘ B : GreCon, GreConD (puis éventuellement MDLGreConD) |
| M3 | `m3_classifieurs/` | Facteur → règle « intention → classe » avec un poids ; vote all-votes / best-vote |
| — | (à venir) | Méthodes d'ensemble sur les classifieurs factoriels (axe proposé par l'encadreuse) |
| M4 | (à venir) | Modèle profond, d'abord un MLP pour le tabulaire. Jamais modifié par la chaîne. |
| M5 | (à venir) | Verdict sur ŷ(x) : confirmée, contestée ou inconnue, puis un score |
| M6 | (à venir) | Explication : facteurs qui soutiennent ŷ, facteurs qui s'y opposent |
| M7 | `experiences/` | Évaluation : détection d'erreurs, courbes accuracy–rejet, comparaison au softmax |

## Phase en cours

**Phase 1 : M1, M2 et M3 sur Pima, comparés à des modèles classiques.**
Instructions détaillées : `docs/instructions_phase1.md`.
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
- **Le test ne sert qu'aux tableaux finaux.** Aucun réglage de paramètre ne s'y fait.
- **Découpages figés.** Les indices apprentissage / validation / test sont sauvegardés dans `donnees/decoupages/` et réutilisés par toute la chaîne. Ne jamais les régénérer.
- **M2 ne voit pas la classe.** La factorisation se fait sur les attributs seuls ; la classe n'intervient qu'en M3.
- **Un facteur est un concept formel**, et A ∘ B ne crée jamais une croix absente de I. Teste-le.
- **Rien ne se cache.** Les x couverts par aucun facteur, et les égalités de vote, sont comptés et rapportés à part. Ils ne sont jamais noyés dans l'accuracy.
- **Les règles d'égalité sont écrites et testées.** GreConD : premier attribut dans l'ordre des colonnes. GreCon : plus grande extension. Classe d'un facteur à égalité : facteur écarté.
- **Graine fixe** : `random_state=42` partout.

## Conventions de code

- Python 3.10 ou plus récent ; numpy, pandas, scikit-learn, pytest. Les versions sont notées dans `requirements.txt`.
- Noms et commentaires en français. Fonctions courtes et pures quand c'est possible. Pas d'optimisation prématurée.
- Contexte formel : matrice numpy booléenne (objets × attributs). Ensembles d'objets et d'attributs : masques booléens.
- Chaque algorithme cite sa référence dans la docstring du module.
- M2 accepte n'importe quel contexte binaire : un échantillon, ou un sous-ensemble d'attributs, pour les méthodes d'ensemble à venir.
- M3 renvoie, pour chaque x : la classe prédite, le score par classe, la liste des facteurs qui couvrent x (avec leur classe et leur poids), et un drapeau « non couvert » ou « égalité ».
- Les résultats sont sauvegardés en CSV dans `resultats/`, avec un nom qui indique l'expérience et la date.

## Commandes

```bash
pip install -r requirements.txt
python -m pytest -q                  # tous les tests
python -m experiences.<script>       # une expérience
```

## Fichiers de suivi

- `JOURNAL.md` : une entrée par étape validée (date, ce qui est fait, résultats, questions ouvertes).
- `DECISIONS.md` : chaque choix fait en cours de route, et pourquoi.
- `docs/` : les instructions de chaque phase.

## Références

- R. Belohlavek, V. Vychodil, « Discovery of optimal factors in binary data via a novel method of matrix decomposition », *Journal of Computer and System Sciences* 76(1), 2010 : GreCon et GreConD.
- T. Makhalova, M. Trnecka, « A Study of Boolean Matrix Factorization Under Supervised Settings », ICFCA 2019 : facteurs → classifieurs, all-votes et best-vote.
- PyBMF (https://github.com/PreferredAI/PyBMF) : sert seulement à vérifier notre GreConD.

## Ce fichier

Ne modifie pas `CLAUDE.md` sans accord. À la fin de chaque phase, propose les mises à jour nécessaires : la phase en cours, les modules terminés, les nouvelles règles.
