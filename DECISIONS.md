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
