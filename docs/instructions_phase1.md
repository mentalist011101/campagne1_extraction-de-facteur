# Instructions pour Claude Code — modules M1 à M3 sur un premier jeu réel

## Contexte

Je suis étudiant en M2 Data Science. Mon mémoire porte sur la fiabilité des prédictions d'un modèle de Deep Learning, jugée à l'aide de facteurs extraits des données par factorisation booléenne (BMF), dans le cadre de l'analyse formelle de concepts (FCA).

Ce premier travail met en place trois modules :
- **M1** : binarisation des données ;
- **M2** : extraction des facteurs ;
- **M3** : transformation des facteurs en classifieurs.

On les compare ensuite à quelques modèles de Machine Learning classiques, sur les mêmes données.

## Le but général : la chaîne complète

Ce fichier ne couvre que le début d'une chaîne plus longue. Tu dois la connaître pour que le code d'aujourd'hui serve la suite. **Ne code que la phase 1.**

```
données ──► M1 binarisation ──► M2 facteurs (BMF) ──► M3 classifieurs factoriels ──┐
   │                                                                               ▼
   └──────► M4 modèle profond : prédiction ŷ(x) et confiance ──────► M5 confrontation facteurs / ŷ
                                                                        │   verdict : confirmée,
                                                                        │   contestée ou inconnue
                                                                        ▼
                                                     M6 explication : facteurs pour et contre ŷ
                                                                        ▼
                                  M7 évaluation : le verdict détecte-t-il les erreurs du modèle ?
```

| Module | Rôle |
|---|---|
| M1 | Transformer les données en table binaire (contexte formel). Seuils appris une fois, appliqués tels quels à chaque nouvel x. |
| M2 | Extraire des facteurs : I ≈ A ∘ B, chaque facteur est un concept formel. |
| M3 | Transformer chaque facteur en règle « intention → classe », avec un poids, puis voter (all-votes, best-vote). |
| M4 | Un modèle profond (d'abord un MLP pour le tabulaire) qui produit ŷ(x) et sa confiance. Il n'est jamais modifié. |
| M5 | Confronter ŷ(x) aux facteurs qui couvrent x. Si les régularités des données contredisent un modèle sûr de lui, la prédiction est contestée. |
| M6 | Expliquer le verdict par les facteurs qui soutiennent ŷ et ceux qui s'y opposent. |
| M7 | Mesurer si les prédictions contestées sont plus souvent fausses : détection d'erreurs, courbes accuracy–rejet, comparaison avec la confiance du softmax. |

Un axe s'ajoute entre M3 et M5 : **combiner les classifieurs factoriels avec des méthodes d'ensemble** (bagging, random subspace, boosting, sélection dynamique…), pour les rendre plus précis. Il viendra après un diagnostic des erreurs de M3.

**Phases :**
1. **Maintenant (ce fichier) :** M1, M2 et M3 sur Pima, comparés à des modèles classiques.
2. Ensuite : diagnostiquer les erreurs de M3 (x non couvert, facteurs en conflit, facteur mal étiqueté), puis tester une stratégie d'ensemble.
3. Puis : M4, M5, M6 et M7.

**Conséquences pour le code d'aujourd'hui :**
- M2 doit accepter **n'importe quel** contexte binaire : un échantillon bootstrap ou un sous-ensemble d'attributs, par exemple. Les méthodes d'ensemble l'appelleront ainsi.
- M3 ne doit pas renvoyer seulement la classe prédite. Pour chaque x, il renvoie aussi le score de chaque classe et **la liste des facteurs qui couvrent x**, avec leur classe et leur poids. M5 et M6 en auront besoin.
- Les découpages apprentissage / validation / test sont sauvegardés une fois pour toutes. Toute la chaîne réutilisera les mêmes.

## Comment on travaille ensemble (important)

Je veux suivre et comprendre tout le travail, de bout en bout. C'est moi qui décide ; toi, tu proposes, tu expliques et tu codes. Les algorithmes et les valeurs attendues viennent de mon travail bibliographique : ils sont dans ce fichier.

**Pour chaque étape, suis ce cycle sans en sauter aucun point :**
1. **Plan.** En quelques lignes : les fichiers, les fonctions avec leur signature, et les choix que tu comptes faire. Attends mon accord.
2. **Tests d'abord.** Écris les tests avec les valeurs attendues données ici, avant le code.
3. **Code.** Par petits morceaux : environ 150 lignes au maximum avant de t'arrêter pour me montrer.
4. **Exécution.** Lance les tests et le script. Montre-moi la sortie réelle, pas un résumé.
5. **Explication.** Explique les passages importants du code, et fais le lien avec l'algorithme (quelle ligne correspond à quel pas).
6. **Validation.** Attends que je dise « validé » ou que je demande une modification.
7. **Trace.** Fais un commit git par étape validée, avec un message clair. Ajoute aussi quelques lignes dans `JOURNAL.md` : ce qui a été fait, les résultats et les questions restées ouvertes.

**Règles :**
- Ne passe **jamais** à l'étape suivante sans ma validation.
- Si un résultat ne correspond pas à une valeur attendue, **arrête-toi** et explique l'écart. Ne modifie jamais un test pour qu'il passe.
- Si un choix n'est pas fixé dans ce fichier (égalités, valeurs manquantes, paramètres…), **demande-moi** au lieu de trancher en silence. Note chaque décision prise dans `DECISIONS.md`.
- Si je pose une question, réponds-y avant de continuer.
- Code lisible par un étudiant : noms explicites, commentaires en français, pas d'astuce cachée, pas d'optimisation prématurée.
- Reproductibilité : graine fixée (`random_state=42`), versions notées dans `requirements.txt`.
- N'ajoute rien qui ne soit pas demandé : pas de méthode d'ensemble ni de réseau profond dans cette phase.

## Environnement

- Python 3.10 ou plus récent, avec `numpy`, `pandas`, `scikit-learn` et `pytest`.
- Pour énumérer les concepts (nécessaire à GreCon), tu peux utiliser `fcapy` ou implémenter Close-by-One. Explique ton choix.
- `PyBMF` (https://github.com/PreferredAI/PyBMF) contient un GreConD. Il sert **seulement à vérifier** le nôtre, pas à le remplacer.

## Structure du projet

```
memoire_fca/
  donnees/          # jeux de données et chargeurs
  m1_binarisation/
  m2_facteurs/      # grecon.py, grecond.py, outils FCA
  m3_classifieurs/  # facteurs → classifieurs, all-votes, best-vote
  modeles_ml/       # modèles classiques de comparaison
  experiences/      # scripts qui produisent les tableaux de résultats
  tests/
  resultats/
  JOURNAL.md        # une entrée par étape validée
  DECISIONS.md      # chaque choix fait en cours de route, avec sa raison
```

## Étape 0 — Mise en place

Initialise un dépôt git. Crée la structure, un `README.md`, un `requirements.txt`, ainsi que `JOURNAL.md` et `DECISIONS.md`, encore vides.

Deux fichiers sont fournis :
- `CLAUDE.md`, à placer à la racine. Il contient les règles permanentes du projet. Ne le modifie pas sans mon accord.
- ce fichier, à placer dans `docs/instructions_phase1.md`.

Sauvegarde les découpages dans `donnees/decoupages/`, comme le demande `CLAUDE.md`. Place le fichier `exemple_10_personnes.csv` (fourni) dans `donnees/` : 10 objets, 8 attributs binaires, une classe D/N.

## Étape 1 — Outils FCA de base

Dans `m2_facteurs/fca.py`, écris :
- `extension(I, attributs)` : les objets qui ont tous ces attributs ;
- `intention(I, objets)` : les attributs communs à tous ces objets ;
- `est_concept(I, objets, attributs)` ;
- l'énumération de tous les concepts d'un contexte.

**Test :** sur l'exemple des 10 personnes, vérifier que chaque concept énuméré satisfait la fermeture. Résultat attendu : 22 concepts, dont le concept du haut (tous les objets) et celui du bas (tous les attributs), soit 20 concepts non triviaux.

## Étape 2 — GreCon

Dans `m2_facteurs/grecon.py` (Belohlavek & Vychodil, 2010) :
1. Calculer tous les concepts.
2. Tant qu'il reste des croix non couvertes : choisir le concept qui couvre le plus de croix **non encore couvertes**, l'ajouter aux facteurs, marquer ses croix comme couvertes.
3. Arrêts possibles : couverture cible (par défaut 100 %) ou nombre maximal de facteurs.
4. Retourner les facteurs, les matrices A (objets × facteurs) et B (facteurs × attributs), et la couverture cumulée.
5. En cas d'égalité entre deux concepts, garder celui qui a la plus grande extension. Écris ce choix dans le code.

**Tests attendus sur l'exemple des 10 personnes :**
- F1 = {P1, P4, P5, P8} × {Adult, Male} : 8 croix ;
- F2 = {P3, P9, P10} × {Elderly, Female} : 6 croix ;
- F3 = {P1, P2, P3, P8} × {Normal} : 4 croix ;
- F4 = {P2, P7} × {Adult, Female} : 4 croix, soit 22 croix sur 30 avec 4 facteurs ;
- 8 facteurs pour couvrir les 30 croix ;
- A ∘ B reconstruit exactement la table, sans jamais créer de croix absente.

## Étape 3 — GreConD

Dans `m2_facteurs/grecond.py` (même article ; D = *on Demand*). Pour chaque facteur :
1. Partir d'une intention vide D, avec V = 0.
2. Pour chaque attribut j absent de D : fermer D ∪ {j} (extension, puis intention) et compter les croix non couvertes du rectangle obtenu.
3. Si le meilleur compte dépasse V : remplacer D par la fermeture choisie, V par ce compte, et recommencer l'étape 2. Sinon, arrêter.
4. Le facteur est (extension de D, D). Marquer ses croix comme couvertes.
5. En cas d'égalité entre attributs, garder le premier dans l'ordre des colonnes. Écris ce choix dans le code.
6. Option `trace=True` : enregistrer, à chaque pas, la valeur de chaque attribut candidat.

**Tests attendus :**
- F1 à F3 identiques à GreCon ;
- F4 = {P6, P10} × {Elderly, Underweight} : 3 croix, soit 21 croix sur 30 avec 4 facteurs ;
- 8 facteurs pour couvrir les 30 croix ;
- chaque facteur est un concept, et la reconstruction est exacte.

Compare aussi tes facteurs avec ceux du GreConD de PyBMF sur cet exemple, et explique les éventuelles différences.

## Étape 4 — Premier jeu réel : Pima Indians Diabetes

- Source : OpenML, jeu « diabetes » (id 37), par exemple via `sklearn.datasets.fetch_openml`. Sinon, le CSV UCI ou Kaggle.
- 768 patientes, 8 variables numériques, classe binaire.
- Découpage stratifié et figé (`random_state=42`) : 60 % apprentissage, 20 % validation, 20 % test. Sauvegarde les indices.
- Fais une courte description : effectifs par classe, valeurs manquantes codées 0 (glucose, pression, IMC…). Ne les corrige pas sans m'en parler.

## Étape 5 — M1 binarisation

Dans `m1_binarisation/` :
- Pour chaque variable numérique, 3 intervalles par quantiles (bas, moyen, haut), soit 3 attributs binaires par variable.
- **Les seuils sont appris sur l'apprentissage uniquement**, puis appliqués tels quels à la validation et au test. Aucun recalcul sur les nouvelles données.
- Noms d'attributs lisibles, par exemple `glucose<=99`, `99<glucose<=126`, `glucose>126`.
- Sortie : les 3 contextes binaires, et un tableau donnant le nombre d'objets, le nombre d'attributs et la densité.

## Étape 6 — GreCon et GreConD sur Pima

Sur le contexte d'apprentissage, **sans utiliser la classe** :
- nombre de facteurs pour atteindre 80, 90 et 100 % de couverture, pour chaque algorithme ;
- temps de calcul, et nombre de concepts pour GreCon ;
- un tableau qui compare les deux, et la liste lisible des 10 premiers facteurs.

## Étape 7 — M3 : facteurs → classifieurs (Makhalova & Trnecka, 2019)

Aucun code public n'existe : implémente d'après ces définitions.
- **Couverture :** un facteur couvre x si x possède **tous** les attributs de son intention. Sinon, il s'abstient.
- **Classe du facteur :** la classe majoritaire de son extension. En cas d'égalité, le facteur est **écarté**.
- **Poids du facteur :** sa précision, et en variante son accuracy, calculées sur l'**apprentissage** (comme l'article) et, en variante, sur la **validation**. Ces options doivent se choisir par paramètre.
- **all-votes :** pour chaque classe c, la somme des poids des facteurs qui couvrent x et désignent c. On prédit la classe de plus grand score.
- **best-vote :** pour chaque classe c, le poids maximal parmi ces facteurs. On prédit la classe de plus grand score.
- **x couvert par aucun facteur :** on le marque « non couvert » et on le compte à part. Pour le calcul de l'accuracy, on lui attribue la classe majoritaire de l'apprentissage, et on le signale.
- **Égalité de score entre deux classes :** on la compte à part, et on choisit la classe majoritaire de l'apprentissage.

**Test sur l'exemple des 10 personnes, avec GreConD et des poids = précision :**
- F3 {Normal} → classe N, poids 0,75 ;
- F1 {Adult, Male} est écarté (2 N, 2 D).

## Étape 8 — Modèles classiques de comparaison

Sur les **mêmes** découpages :
- régression logistique, arbre de décision, kNN, forêt aléatoire, et un petit MLP (sklearn) ;
- entraînés sur les données brutes standardisées, et aussi sur les données binarisées.

## Étape 9 — Tableau de résultats

Un tableau sur le test, avec une ligne par méthode : accuracy, F1 macro, et, pour les classifieurs factoriels, taux de non-couverts, taux d'égalités et nombre de facteurs utilisés.
- Méthodes : GreCon et GreConD, croisés avec all-votes et best-vote, croisés avec poids sur l'apprentissage et sur la validation ; puis les 5 modèles classiques.
- Ajoute pour chaque classifieur factoriel son **taux d'accord** avec la forêt aléatoire et avec le MLP : la proportion de cas où ils prédisent la même classe.
- Sauvegarde le tout en CSV dans `resultats/`, puis commente les résultats en quelques phrases.

## Hors périmètre pour l'instant

Méthodes d'ensemble sur les facteurs, MDLGreConD, GreCon2 / GreCon2Compact, réseau profond, autres jeux de données. Ils viendront après validation de cette première chaîne.
