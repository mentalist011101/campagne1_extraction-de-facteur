# Journal

## 2026-10-21 — Étape 0 : mise en place
- **Fait :** dépôt git, structure des dossiers, `README.md`, `requirements.txt` (numpy 2.5.0, pandas 3.0.3, scikit-learn 1.9.0, pytest 9.1.1), instructions copiées dans `docs/instructions_phase1.md`, `exemple_10_personnes.csv` placé dans `donnees/`.
- **Résultats :** exemple vérifié : 10 objets, 8 attributs, 30 croix. `pytest` s'exécute (aucun test pour l'instant).
- **Questions ouvertes :** aucune.

## 2026-10-21 — Étape 1 : outils FCA de base
- **Fait :** `m2_facteurs/fca.py` (`extension`, `intention`, `est_concept`, `enumerer_concepts` par Close-by-One), chargeur `donnees/exemple.py`, 9 tests dans `tests/test_fca.py`.
- **Résultats :** 22 concepts sur l'exemple des 10 personnes (haut, bas, 20 non triviaux), tous fermés, identiques à ceux de `fcapy`.
- **Questions ouvertes :** `fcapy` dépend de `skmine`, qui utilise des API numpy dépréciées (avertissements avec numpy 2.5) ; à surveiller si numpy est mis à jour.

## 2026-10-21 — Étape 2 : GreCon
- **Fait :** `m2_facteurs/grecon.py` (GreCon glouton sur tous les concepts, arrêts par couverture cible ou nombre maximal), `m2_facteurs/factorisation.py` (résultat commun, produit booléen), 10 tests dans `tests/test_grecon.py`.
- **Résultats :** F1 à F4 conformes (8, 6, 4, 4 croix ; 22/30 avec 4 facteurs) ; 8 facteurs pour 30/30 ; A ∘ B = I et A ∘ B ≤ I à chaque k. Les règles d'égalité servent dès l'exemple : F6 {Adult, Male, Overweight} est préféré à {Female, Obese} par l'ordre lexicographique ; F8 {Male} par la plus grande extension.
- **Questions ouvertes :** aucune.

## 2026-10-21 — Étape 3 : GreConD
- **Fait :** `m2_facteurs/grecond.py` (construction attribut par attribut, égalité → premier attribut dans l'ordre des colonnes, option `trace`), type `PasTrace` dans `factorisation.py`, 10 tests dans `tests/test_grecond.py`, script `experiences/comparer_pybmf.py`.
- **Résultats :** F1 à F3 identiques à GreCon ; F4 = {P6, P10} × {Elderly, Underweight}, 3 croix, 21/30 avec 4 facteurs ; 8 facteurs pour 30/30 ; reconstruction exacte. La trace montre l'égalité Adult / Overweight (6) au premier pas, résolue en faveur d'Adult. GreConD manque {Adult, Female} (4 croix) en F4 parce qu'aucun des deux attributs seul ne domine : c'est le prix du glouton attribut par attribut.
- **Comparaison PyBMF (commit 03c9b43) :** 7 facteurs, 28/30 croix. Deux écarts de PyBMF : (1) le dernier facteur est perdu (`early_stop(k=k)` au lieu de `k+1`) ; (2) `best_u` est mis à jour au milieu du parcours des attributs, donc les candidats ne sont pas tous évalués depuis le même D, contrairement à l'article (F2 = {Adult, Female} avant {Elderly, Female}, contraire au résultat attendu). *Corrigé à l'étape 6 : les scores non décroissants de PyBMF (8, 4, 6) avaient d'abord été cités comme preuve ; ce n'en est pas une, car un GreConD fidèle à l'article peut aussi produire des scores non décroissants (voir étape 6).*. Règle d'égalité identique à la nôtre ; tous ses facteurs sont des concepts.
- **Questions ouvertes :** mentionner ces écarts de PyBMF dans le mémoire ?

## 2026-10-21 — Étape 4 : Pima Indians Diabetes
- **Fait :** `donnees/pima.py` (téléchargement unique, chargement, découpages figés), copie figée `donnees/pima.csv`, indices dans `donnees/decoupages/`, script `experiences/decrire_pima.py`, 8 tests dans `tests/test_pima.py`.
- **Source :** OpenML inaccessible depuis notre réseau ; CSV UCI via le miroir GitHub de J. Brownlee (voir DECISIONS.md, avec SHA-256).
- **Résultats :** 768 objets, 500 négatifs / 268 positifs (34,9 %). Découpages 460 / 154 / 154, proportion de positifs 35,0 / 35,1 / 34,4 %. Zéros : glucose 5, pression 35, pli_cutane 227 (29,6 %), insuline 374 (48,7 %), imc 11 ; grossesses 111 (vraies valeurs).
- **Questions ouvertes :** traitement des zéros-manquants avant la binarisation (étape 5), surtout insuline et pli_cutane.

## 2026-10-21 — Étape 5 : M1 binarisation
- **Fait :** `m1_binarisation/binarisation.py` (`apprendre_seuils`, `binariser`), constante `COLONNES_ZERO_MANQUANT` dans `donnees/pima.py`, script `experiences/binariser_pima.py`, 7 tests dans `tests/test_binarisation.py`.
- **Résultats :** terciles appris sur l'apprentissage seul, zéros manquants exclus (option b). Seuils : glucose 105 / 129, insuline 90 / 165, pli_cutane 25 / 33, imc 29,5 / 34,6, age 25 / 37… Contextes 460 / 154 / 154 × 24 attributs, densités 0,298 / 0,298 / 0,297 (maximum 1/3 sans valeur manquante). Fichiers dans `resultats/pima_m1_*_2026-10-21.csv`.
- **Questions ouvertes :** version de Pima sans valeurs manquantes, reportée : supprimer les patientes incomplètes (392 restantes) ou les colonnes insuline et pli_cutane ? Restreindre les découpages existants ou en créer de nouveaux ?

## 2026-10-21 — Étape 6 : GreCon et GreConD sur Pima
- **Fait :** `nb_facteurs_pour_couverture` dans `factorisation.py` (+ 2 tests), `experiences/contextes_pima.py` (point d'entrée commun vers M1, réutilisé par `binariser_pima.py`), script `experiences/factoriser_pima.py`.
- **Résultats (apprentissage, 460 × 24, 3292 croix, sans la classe) :** GreCon et GreConD : 17 / 21 / 26 facteurs pour 80 / 90 / 100 %. GreCon 17,8 s (7 212 concepts, énumération 1,5 à 2,2 s), GreConD 0,2 s. Reconstruction exacte, aucune croix hors de I. F1 commun {grossesses<=2, age<=25} ; le F2 de GreCon {grossesses>5, age>37} (188 croix) n'arrive qu'en F14 chez GreConD, avec un score supérieur à F2–F13 : les scores de GreConD ne sont pas forcément décroissants. Les facteurs sont surtout des attributs seuls (terciles mutuellement exclusifs, ~150 objets par colonne). 26 facteurs alors que 24 suffisent (un concept-attribut par colonne) : le glouton n'est pas optimal.
- **Questions ouvertes :** des facteurs à un seul attribut donneront des règles M3 très simples ; à surveiller dans le diagnostic des erreurs (phase 2).

## 2026-10-22 — Étape 7 : M3 facteurs → classifieurs
- **Fait :** `m3_classifieurs/classifieur_factoriel.py` (`construire_regles`, `predire` all-votes / best-vote, `compter`, `decrire_facteurs`), 13 tests dans `tests/test_classifieur_factoriel.py`, script `experiences/regles_pima.py`.
- **Résultats :** valeurs attendues retrouvées (F3 {Normal} → N, précision 0,75, accuracy 0,70 ; F1 {Adult, Male} écarté). Sur Pima, GreCon et GreConD donnent les mêmes 26 facteurs dans un ordre différent : à 100 % de couverture, leurs classifieurs seront identiques. Aucun facteur écarté. 21 règles votent 0, 5 votent 1 ; précision des règles de classe 1 entre 0,52 et 0,67, contre jusqu'à 0,91 pour la classe 0 ; l'accuracy favorise au contraire les règles de classe 1.
- **Questions ouvertes :** all-votes avec précision risque de ne presque jamais prédire la classe 1 (F1 macro faible) ; GreCon et GreConD ne se distingueront qu'avec k < 26.

## 2026-10-22 — Étape 8 : modèles classiques
- **Fait :** `modeles_ml/modeles_classiques.py` (5 modèles, hyperparamètres par défaut, standardisation dans un Pipeline pour les données brutes), script `experiences/modeles_pima.py` (validation seulement), 10 tests dans `tests/test_modeles_classiques.py`.
- **Résultats (validation, 154 patientes dont 54 de classe 1) :** données brutes : accuracy 0,734 à 0,753, F1 macro 0,701 à 0,727 (meilleur : MLP) ; données binarisées : accuracy 0,695 à 0,747, F1 macro 0,663 à 0,710 (meilleur : régression logistique). La binarisation coûte 1 à 4 points. Régression logistique et arbre (bruts) : mêmes scores par coïncidence, 26 désaccords. MLP binarisé : 0,95 sur l'apprentissage, surapprentissage.
- **Incident :** MLP non convergé en 2000 itérations sur les données binarisées ; `max_iter=3000` (converge en 2344), test de non-convergence ajouté. Tests réorganisés (106 s → 24 s).
- **Questions ouvertes :** réglage des hyperparamètres et imputation des zéros, reportés.
