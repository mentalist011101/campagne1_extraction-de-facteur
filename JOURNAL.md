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
