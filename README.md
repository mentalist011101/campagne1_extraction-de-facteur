# Mémoire M2 — Validation des prédictions d'un modèle profond par facteurs booléens

Mémoire de M2 Data Science (ENSPY). On extrait des données des facteurs par factorisation booléenne (BMF, cadre de l'analyse formelle de concepts), puis on les confronte aux prédictions d'un modèle profond.

Phase en cours : **phase 1**, M1 (binarisation), M2 (GreCon, GreConD) et M3 (classifieurs factoriels) sur Pima, comparés à des modèles classiques. Voir `docs/instructions_phase1.md`.

## Structure

| Dossier | Contenu |
|---|---|
| `donnees/` | jeux de données, chargeurs ; `decoupages/` : indices apprentissage / validation / test figés |
| `m1_binarisation/` | données → contexte formel binaire |
| `m2_facteurs/` | outils FCA, GreCon, GreConD |
| `m3_classifieurs/` | facteurs → règles pondérées, all-votes, best-vote |
| `modeles_ml/` | modèles classiques de comparaison |
| `experiences/` | scripts qui produisent les tableaux |
| `resultats/` | sorties CSV |
| `tests/` | tests pytest |

## Commandes

```bash
pip install -r requirements.txt
python -m pytest -q
python -m experiences.<script>
```

Suivi : `JOURNAL.md` (une entrée par étape validée), `DECISIONS.md` (chaque choix et sa raison).
