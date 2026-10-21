# Décisions

## 2026-10-21 — Racine du projet
- **Choix :** `PROTOCOLE_1/` est la racine du dépôt, sans sous-dossier `memoire_fca/`.
- **Raison :** `CLAUDE.md` y était déjà placé ; on garde la structure prévue, directement à la racine.

## 2026-10-21 — Énumération des concepts
- **Choix :** Close-by-One codé dans le projet (`m2_facteurs/fca.py`) comme méthode principale ; `fcapy` (0.1.4.5) sert uniquement de référence dans les tests.
- **Raison :** code court, lisible et défendable ligne par ligne ; travaille directement sur des masques numpy ; la comparaison avec une bibliothèque reconnue garantit la justesse. `fcapy` tire `skmine`, qui utilise des API numpy dépréciées : risque de casse futur, limité aux tests.
