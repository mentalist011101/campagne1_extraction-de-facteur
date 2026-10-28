# Décisions

## 2026-10-21 — Racine du projet
- **Choix :** `PROTOCOLE_1/` est la racine du dépôt, sans sous-dossier `memoire_fca/`.
- **Raison :** `CLAUDE.md` y était déjà placé ; on garde la structure prévue, directement à la racine.

## 2026-10-21 — Énumération des concepts
- **Choix :** Close-by-One codé dans le projet (`m2_facteurs/fca.py`) comme méthode principale ; `fcapy` (0.1.4.5) sert uniquement de référence dans les tests.
- **Raison :** code court, lisible et défendable ligne par ligne ; travaille directement sur des masques numpy ; la comparaison avec une bibliothèque reconnue garantit la justesse. `fcapy` tire `skmine`, qui utilise des API numpy dépréciées : risque de casse futur, limité aux tests.

## 2026-10-21 — GreCon : départage des égalités
- **Choix :** à nombre égal de croix nouvelles, on garde le concept de plus grande extension (règle fixée par `CLAUDE.md`). Si l'égalité persiste, on garde le concept dont l'intention, lue comme liste croissante d'indices d'attributs, vient en premier dans l'ordre lexicographique.
- **Raison :** la seconde règle rend GreCon indépendant de l'algorithme d'énumération des concepts (CbO, fcapy…), et reste proche de la règle de GreConD (premier attribut dans l'ordre des colonnes).

## 2026-10-21 — GreConD : trace
- **Choix :** la trace est un champ optionnel `trace` (liste de `PasTrace`) de `ResultatFactorisation`, `None` par défaut (donc pour GreCon).
- **Raison :** un seul champ en plus ne justifie pas un type de résultat propre à GreConD ; GreCon et GreConD gardent le même type de sortie.

## 2026-10-21 — PyBMF comme vérification seulement
- **Choix :** PyBMF installé depuis GitHub, commit `03c9b4364d480fa4445c10b6d65e93dcaedc74d6`. Absent de `requirements.txt` et des tests ; utilisé seulement par `experiences/comparer_pybmf.py`.
- **Raison :** sa version de GreConD s'écarte de l'article (voir JOURNAL, étape 3) ; il ne peut pas servir de référence numérique exacte.

## 2026-10-21 — Pima : noms des colonnes et codage de la classe
- **Choix :** colonnes nommées dans la copie figée `donnees/pima.csv` : preg → grossesses, plas → glucose, pres → pression, skin → pli_cutane, insu → insuline, mass → imc, pedi → pedigree, age → age. Classe codée 1 = diabète (`tested_positive` dans OpenML), 0 = négatif.
- **Raison :** noms lisibles pour les attributs binaires de M1 (`glucose<=99`…) ; codage 0/1 standard pour les modèles, le décodage vers des libellés lisibles se fera à l'affichage.

## 2026-10-21 — Pima : copie figée et découpages
- **Choix :** téléchargement unique, sauvegarde dans `donnees/pima.csv` versionné ; toute la chaîne lit ce fichier. Découpage stratifié en deux temps (60/40, puis 40 → 20/20), `random_state=42`, indices de lignes sauvegardés dans `donnees/decoupages/` ; la fonction de création refuse d'écraser des fichiers existants.
- **Raison :** reproductibilité indépendante d'OpenML et du réseau ; découpages jamais régénérés (`CLAUDE.md`).

## 2026-10-21 — Pima : source des données
- **Choix :** CSV de l'UCI via le miroir GitHub de J. Brownlee, https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv, téléchargé le 2026-10-21 (SHA-256 du fichier source : 6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af).
- **Raison :** OpenML (id 37) inaccessible depuis notre réseau (délai dépassé sur www.openml.org et api.openml.org). Même jeu UCI, même ordre de colonnes ; conformité vérifiée par les tests (768 objets, 500/268, zéros connus par variable).

## 2026-10-21 — M1 : zéros manquants (option b)
- **Choix :** pour glucose, pression, pli_cutane, insuline et imc, un zéro signifie « valeur inconnue » : les seuils sont calculés sur les valeurs non nulles de l'apprentissage, et un objet de valeur 0 ne reçoit aucun des trois attributs de la variable. grossesses n'est pas concernée (0 est une vraie valeur).
- **Raison :** ni correction ni imputation ; seuils au sens médical (insuline : 0 et 94 avec les zéros, 90 et 165 sans) ; en FCA l'absence d'attribut exprime naturellement l'absence d'information ; pas d'attribut « manquant » qui créerait des facteurs reflétant la collecte des données plutôt que la médecine.
- **Option pour plus tard :** une version de Pima sans les objets à valeur manquante sera créée pour mesurer l'impact de ce choix.

## 2026-10-22 — M3 : définitions des classifieurs factoriels
- **Accuracy d'une règle :** accuracy(f, c) = (tp + tn) / |S|, la règle étant vue comme un classifieur binaire « c si couvert, pas c sinon » (Makhalova & Trnecka 2019, §3.1). Test : sur les 10 personnes, F3 {Normal} → N, accuracy 0,70, précision 0,75.
- **Classe d'un facteur :** toujours la classe majoritaire de son extension sur l'apprentissage ; égalité → facteur écarté. La validation ne sert qu'à recalculer le poids.
- **Poids sur la validation :** une règle qui ne couvre aucun objet de validation est écartée (pour la précision comme pour l'accuracy) et comptée à part.
- **Nombre de facteurs :** par défaut tous les facteurs jusqu'à 100 % de couverture, paramètre k pour en utiliser moins ; le nombre de facteurs est noté pour chaque algorithme.
- **x couvert seulement par des facteurs écartés :** compté comme non couvert (aucune règle ne vote).
- **Égalité de scores :** au moins deux classes au score maximal (égalité exacte) → classe majoritaire de l'apprentissage, x marqué.
- **Tableau par facteur :** intention, taille de l'extension, classe, objets couverts en validation, précision et accuracy sur l'apprentissage et la validation, drapeau « écarté » et raison (égalité de classe, ou aucun objet couvert en validation).
- **Raison :** décisions de l'étudiant, d'après Makhalova & Trnecka (2019).

## 2026-10-22 — Modèles classiques de comparaison
- **Choix :** hyperparamètres par défaut de sklearn (aucun réglage), `random_state=42` ; MLP à une couche cachée de 16 neurones, `max_iter=3000`. Données brutes : `Pipeline(StandardScaler, modèle)`, zéros manquants laissés tels quels. Données binarisées : contextes de M1, sans standardisation.
- **Raison :** comparaison simple et reproductible, sans avantage de réglage par rapport aux classifieurs factoriels (non réglés eux non plus). Le réglage des hyperparamètres et l'imputation des zéros pourront être testés plus tard. Asymétrie assumée : les zéros sont des valeurs pour les modèles bruts, et une absence d'information (option b) pour les données binarisées.

## 2026-10-22 — MLP : plafond d'itérations
- **Choix :** `max_iter=3000` au lieu de 2000.
- **Raison :** sur les données binarisées, le MLP n'avait pas convergé en 2000 itérations (ConvergenceWarning) ; il converge en 2344. Seul le plafond change, les autres hyperparamètres restent par défaut. Accuracy de 0,95 sur l'apprentissage binarisé : surapprentissage probable, à surveiller.

## 2026-10-22 — Tableau de résultats sur le test (étape 9)
- **Choix :** 16 classifieurs factoriels = {GreCon, GreConD} × {all-votes, best-vote} × {poids sur apprentissage, sur validation} × {précision, accuracy} ; 10 modèles classiques (5 × {brutes, binarisées}), entraînés sur l'apprentissage seul. Taux d'accord des classifieurs factoriels avec la forêt aléatoire et le MLP, en version brute et binarisée (4 colonnes). Deux colonnes de diagnostic ajoutées : `accuracy_decides` (accuracy sur les x couverts et sans égalité) et `predits_classe_1`.
- **Raison :** la variante accuracy change fortement les poids (étape 7) ; l'accord avec les modèles bruts prépare M5, celui avec les modèles binarisés compare à information égale ; les colonnes de diagnostic séparent l'effet de la classe par défaut et révèlent le déséquilibre de prédiction, utiles pour la phase 2.

## 2026-10-22 — Zéros dans les modèles classiques « bruts » : limite de la comparaison
- **Constat :** dans les modèles classiques sur données brutes, les zéros de glucose, pression, pli_cutane, insuline et imc ont été traités comme des valeurs ordinaires (aucune imputation, standardisés comme les autres valeurs ; décision du 2026-10-22, « Modèles classiques de comparaison »).
- **Limite :** M3 et les modèles sur données binarisées voient ces zéros comme une absence d'information (option b), alors que les modèles bruts voient insuline = 0 comme une valeur réelle et peuvent l'exploiter comme indicateur de donnée manquante. La comparaison M3 / modèles bruts ne se fait donc pas à traitement des données manquantes égal. La comparaison M3 / modèles binarisés, elle, l'est.

## 2026-10-22 — Phase 2 : hypothèses et critères fixés AVANT toute expérience
Ces hypothèses et critères ne seront plus modifiés. Une hypothèse rejetée est un résultat, rapporté tel quel.

- **H1 — L'échelle interordinale produit des facteurs à intentions plus longues.** Critère : avec GreConD à 100 % de couverture sur l'apprentissage de la phase 1 (460 patientes), échelle interordinale, la part de facteurs à un seul attribut est inférieure à 50 % (92 % en phase 1, échelle nominale).
- **H2 — Des facteurs plus longs améliorent M3 sur Pima.** Configuration principale fixée à l'avance, sans sélection sur les plis : best-vote + poids précision (schéma recommandé par Makhalova & Trnecka). Critère : en validation croisée à 10 plis, l'échelle interordinale a un F1 macro strictement supérieur à l'échelle nominale dans au moins 8 plis sur 10 ; une égalité exacte ne compte pas comme une victoire. Les 5 autres combinaisons vote × poids sont rapportées pli par pli, à titre descriptif.
- **H3 — Le lift corrige le biais vers la classe 0.** Critère, à vérifier pour chacune des 4 combinaisons échelle × vote : (i) la moyenne sur les plis de |prédits en classe 1 − réels en classe 1| est strictement plus petite avec le lift qu'avec la précision, ET (ii) l'accuracy moyenne avec le lift est ≥ l'accuracy moyenne de la règle « toujours la classe majoritaire de la partie d'apprentissage du pli », mesurée sur les mêmes plis. H3 est confirmée si les 4 combinaisons satisfont (i) et (ii). Le seuil de 0,656 (test de la phase 1) est abandonné.
- **H4 — Notre M3 reproduit l'article sur mushroom.** Critère : accuracy ≥ 0,90 avec best-vote pondéré par la précision (moyenne sur les 10 plis). L'article rapporte 0,99 ; c'est un ordre de grandeur, son découpage et sa binarisation diffèrent.

## 2026-10-22 — Phase 2 : protocole
- **Choix :** validation croisée stratifiée à 10 plis (`StratifiedKFold`, `shuffle=True`, `random_state=42`) sur les 614 patientes apprentissage + validation de la phase 1, plis tirés une fois et sauvegardés dans `donnees/decoupages/cv10_pima.csv`. Dans chaque pli, seuils, facteurs, classes et poids, modèles sont réappris sur la partie d'apprentissage du pli ; option (b) conservée pour les valeurs manquantes ; règle majoritaire calculée sur la partie d'apprentissage du pli ; nombre de facteurs noté dans chaque pli. Variante « poids sur la validation » abandonnée. Écart-type : pandas, ddof = 1.
- **Mushroom :** miroir UCI accepté si OpenML reste inaccessible. Vérifications : 8124 lignes, 22 variables + la classe, 4208 comestibles / 3916 vénéneux, 2480 « ? » toutes dans stalk-root ; tout écart est signalé, rien n'est corrigé.
- **Raison :** décisions de l'étudiant (instructions de phase 2 et précisions du 2026-10-22).

## 2026-10-24 — M1 : échelle interordinale
- **Choix :** paramètre `echelle` (`"nominale"` par défaut, ou `"interordinale"`) dans `binariser`, `contextes_pima` et `contextes_pli`. Mêmes seuils (terciles appris sur l'apprentissage, ou sur la partie d'apprentissage du pli). Quatre attributs par variable, dans l'ordre `v<=t1`, `v<=t2`, `v>t1`, `v>t2` ; valeur manquante (option b) : aucun des quatre.
- **Source d'arbitraire connue :** cet ordre des colonnes fixe le choix de GreConD en cas d'égalité (premier attribut dans l'ordre des colonnes). À l'étape 12, le nombre de choix de GreConD faits sur une égalité sera compté grâce à la trace.
- **Raison :** instructions de la phase 2 ; ordre conservé tel quel, sans optimisation.

## 2026-10-24 — Étape 13 : poids lift et « facteurs qui votent effectivement »
- **Lift :** lift(f, c) = précision(f, c) / P(c), P(c) = proportion de c dans la partie d'apprentissage (du pli). Remplace la précision dans all-votes et best-vote, sans autre changement.
- **Facteurs qui votent effectivement :** all-votes : toutes les règles qui couvrent x ; best-vote : la seule règle décisive (celle qui donne le score maximal de la classe prédite). Moyenne sur les couples (x, règle). Si plusieurs règles de la classe prédite ont le même poids maximal, la règle décisive est la première dans l'ordre des facteurs. Un x non couvert ou à égalité de scores n'a aucune règle décisive et n'entre pas dans la moyenne.
- **Indicateurs :** longueur moyenne d'intention (en attributs) et nombre moyen de variables distinctes de ces règles (décision de l'étape 12), avec les mêmes définitions.
- **Références :** Naive Bayes de Bernoulli, régression logistique et forêt aléatoire sur chaque binarisation ; régression logistique et forêt aléatoire sur les données brutes standardisées ; hyperparamètres par défaut, `random_state=42`.
- **Raison :** décisions de l'étudiant (plan de l'étape 13).

## 2026-10-28 — Étape 14 : mushroom
- **Source :** OpenML (`fetch_openml(name="mushroom", version=1)`), à défaut le fichier UCI `agaricus-lepiota.data` ; téléchargement unique, copie figée `donnees/mushroom.csv` (« ? » conservé tel quel). Vérifications : 8124 lignes, 22 variables + la classe, 4208 comestibles / 3916 vénéneux, 2480 « ? » toutes dans stalk-root ; tout écart est signalé, rien n'est corrigé.
- **Classe :** 1 = vénéneux, 0 = comestible.
- **Découpages :** 60/20/20 stratifié figé (`random_state=42`), test scellé ; 10 plis stratifiés (`random_state=42`) sur apprentissage + validation.
- **Binarisation nominale :** un attribut par valeur de chaque variable, valeurs apprises sur la partie d'apprentissage du pli. « ? » ne reçoit aucun attribut (comme l'option b) ; une valeur absente de la partie d'apprentissage ne reçoit aucun attribut.
- **Attributs constants (option a) :** un attribut présent chez tous les objets de la partie d'apprentissage du pli est retiré (appris sur l'apprentissage, donc sans fuite). Raison : il n'apporte aucune information et la fermeture l'ajouterait à chaque intention (cas de veil-type, à valeur unique), ce qui fausserait la part de facteurs à un seul attribut.
- **Mesures :** GreConD seulement ; temps de GreConD mesuré par pli ; arrêt si un pli dépasse 30 minutes. H4 : accuracy moyenne sur les 10 plis, best-vote + précision, ≥ 0,90.

## 2026-10-28 — Choix du jeu de la phase 3 (fixé avant l'étape 14)
- **Règle :** mushroom avec B = 10 sacs si le meilleur M3 reste nettement sous 0,99 et si GreConD prend moins d'environ 30 s par pli ; sinon congressional voting (OpenML « vote » : votes y/n/?, « ? » sans attribut, codage nominal ; vérifier 435 lignes, 16 variables et la répartition des classes).
- Sur un jeu catégoriel, une seule échelle : les mentions « pour chaque échelle » et la comparaison nominale / interordinale sont ignorées. H5, H6 et D1 restent identiques. Le « meilleur M3 » de référence est la meilleure configuration de M3 sur ce jeu, choisie AVANT de lancer la phase 3 et écrite dans ce fichier.
- **Raison :** décisions de l'étudiant.

## 2026-10-28 — Phase 3 : jeu de données
- **Choix :** congressional voting (OpenML « vote », version 1), conformément à la règle fixée avant l'étape 14 (meilleur M3 sur mushroom : 0,998, pas nettement sous 0,99). Le fichier `docs/instructions_phase3_ensembles.md` prévoyait Pima et les deux échelles : il a été écrit avant cette règle, **la règle l'emporte**.
- **Données :** copie figée `donnees/vote.csv` (libellés d'origine, « ? » conservé). Vérifié : 435 lignes, 16 variables, 267 démocrates / 168 républicains, valeurs y / n / ?, 392 « ? » sur 203 lignes. Découpage 60/20/20 stratifié figé (261 / 87 / 87, `random_state=42`), test scellé ; 10 plis stratifiés sur apprentissage + validation (348 représentants, 34 ou 35 évalués par pli).
- **Classe :** 1 = républicain, 0 = démocrate (classe minoritaire codée 1, comme pour Pima et mushroom ; sans effet sur le F1 macro des critères).
- **Binarisation :** nominale (catégorielle) avec `apprendre_modalites` / `binariser_categoriel`, apprise sur la partie d'apprentissage du pli, attributs constants retirés. **« ? » = abstention réelle, sans attribut (option b)** : choix gardé par cohérence avec Pima et mushroom, et noté comme **limite** (une abstention peut porter une information politique).
- Une seule échelle : les mentions « pour chaque échelle » et la comparaison nominale / interordinale du fichier de phase 3 sont ignorées.

## 2026-10-28 — Phase 3 : M3 de référence et hypothèses H5 à H7, fixés AVANT toute expérience
- **M3 de référence, fixé à l'avance sans grille préalable :** best-vote + précision (schéma recommandé par Makhalova & Trnecka, et le meilleur sur mushroom). GreConD à 100 % de couverture, classes et poids sur la partie d'apprentissage du pli.
- **H5 — Le bagging améliore M3.** Critère : le F1 macro de M3 (best-vote + précision) avec bagging dépasse strictement celui de la même configuration sans bagging dans au moins 8 plis sur 10.
- **H6 — Le stacking améliore M3.** Critère : le F1 macro du stacking dépasse strictement celui de M3 seul (best-vote + précision) dans au moins 8 plis sur 10.
- **H7 — Non-infériorité.** Critère, pour le stacking et pour le bagging, séparément : F1 macro moyen sur les 10 plis ≥ (F1 macro moyen de la meilleure des références classiques — régression logistique, forêt aléatoire, Naive Bayes de Bernoulli, sur les mêmes attributs binaires) − 0,02. La forêt aléatoire est ajoutée au tableau de l'étape 18.
- Une égalité exacte sur un pli ne compte pas comme une victoire. **D1** (descriptive, sans critère) : pli par pli, stacking contre régression logistique sur les attributs.
- **Stacking, absence de fuite :** en stacking classique, il faut des prédictions hors pli des classifieurs de base, puisqu'ils ont vu la classe. Ici, la matrice A (objets × facteurs) est calculée **sans la classe** (GreConD sur les attributs seuls) : seule la régression logistique utilise y_app.
- **Raison :** décisions de l'étudiant du 2026-10-28.

## 2026-10-28 — Étape 17 : bagging, choix non fixés par les instructions
- **Binarisation dans les sacs :** la binarisation du pli (modalités apprises sur sa partie d'apprentissage) est réutilisée telle quelle dans chaque sac, sans réapprentissage ; les M3 des sacs travaillent ainsi sur les mêmes attributs que le M3 seul.
- **x non couvert dans un sac :** il reçoit la classe majoritaire de la partie d'apprentissage du pli, comme pour le M3 seul, et il est compté dans les non-couverts de ce sac.
- **Tirage :** sac b = `np.random.default_rng(42 + b).integers(0, n_app, n_app)`, avec remise ; configuration best-vote + précision (M3 de référence) ; agrégation par vote majoritaire, égalité → classe majoritaire du pli (impossible avec B = 25 impair).
- **Étape 18 :** pour H5 et H6, à titre descriptif, victoires, défaites, égalités et p-valeur bilatérale du test du signe (égalités exclues).
- **Raison :** décisions de l'étudiant (plan de l'étape 17).
