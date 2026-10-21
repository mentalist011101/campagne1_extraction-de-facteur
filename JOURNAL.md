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
- **Comparaison PyBMF (commit 03c9b43) :** 7 facteurs, 28/30 croix. Deux écarts de PyBMF : (1) le dernier facteur est perdu (`early_stop(k=k)` au lieu de `k+1`) ; (2) `best_u` est mis à jour au milieu du parcours des attributs, donc les candidats ne sont pas tous évalués depuis le même D, contrairement à l'article (F2 = {Adult, Female} avant {Elderly, Female}, scores non décroissants 8, 4, 6). Règle d'égalité identique à la nôtre ; tous ses facteurs sont des concepts.
- **Questions ouvertes :** mentionner ces écarts de PyBMF dans le mémoire ?
