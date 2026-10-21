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
