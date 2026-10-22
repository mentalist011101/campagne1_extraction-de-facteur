# Instructions pour Claude Code — Phase 2 : donner de la structure aux facteurs

Le cycle de travail et les règles sont dans `CLAUDE.md` : ils s'appliquent sans changement. Place ce fichier dans `docs/instructions_phase2.md`, et mets à jour la ligne « Phase en cours » de `CLAUDE.md` après mon accord.

## Pourquoi cette phase

La phase 1 a montré que, sur Pima binarisé en 3 intervalles disjoints, **24 facteurs sur 26 sont des attributs isolés**. M3 n'est donc pas un ensemble de règles, mais un vote d'attributs, et il ne bat guère la classe majoritaire (0,656). Ce résultat rejoint celui de Makhalova & Trnecka, qui obtiennent 0,66 sur Pima.

Le mécanisme est le suivant. Avec des intervalles disjoints, une colonne couvre environ un tiers des patientes. Une conjonction de deux attributs ne bat cette colonne que si les deux variables sont très corrélées : c'est le cas de grossesses et âge, et de rien d'autre.

La phase 2 change **une seule chose à la fois** et mesure son effet :
1. **La représentation binaire.** On passe à une échelle interordinale, **avec les mêmes seuils** qu'en phase 1.
2. **Le poids des votes.** On ajoute un poids qui tient compte de la proportion de chaque classe (lift).
3. **Le jeu de données.** Si le temps le permet, on ajoute mushroom, catégoriel par nature, où l'article obtient de très bons résultats. Cela vérifie notre M3 dans un cas où il devrait fonctionner.

Les méthodes d'ensemble ne viennent pas encore : elles n'auraient rien à combiner tant que les facteurs sont des attributs isolés.

## Hypothèses fixées à l'avance

On les écrit **avant** de lancer les expériences, avec le critère qui permettra de conclure. Ensuite, on ne change plus ni les hypothèses ni les critères.

| | Hypothèse | Critère |
|---|---|---|
| H1 | L'échelle interordinale produit des facteurs à intentions plus longues. | La part de facteurs à un seul attribut tombe sous 50 % (elle était de 92 % en phase 1). |
| H2 | Des facteurs plus longs améliorent M3 sur Pima. | Le F1 macro de la meilleure configuration interordinale dépasse celui de la même configuration nominale dans au moins 8 plis sur 10. |
| H3 | Le lift corrige le biais vers la classe 0. | Le nombre de prédictions de classe 1 se rapproche du nombre réel, sans que l'accuracy tombe sous 0,656. |
| H4 | Notre M3 reproduit l'article sur mushroom. | L'accuracy atteint au moins 0,90 avec best-vote pondéré par la précision (l'article rapporte 0,99). |

Si une hypothèse est rejetée, c'est un résultat : on le rapporte tel quel.

## Règle sur les données

- **Le test reste scellé.** Aucune ligne de cette phase ne lit le test.
- Toutes les comparaisons se font en **validation croisée stratifiée à 10 plis** sur la réunion apprentissage + validation de la phase 1, soit 614 patientes. C'est le protocole de l'article, et il est plus stable que nos 154 patientes de validation.
- Les plis sont tirés **une seule fois** (`random_state=42`), sauvegardés dans `donnees/decoupages/cv10_pima.csv`, puis réutilisés partout.
- **Dans chaque pli, tout est réappris sur la partie d'apprentissage du pli** : seuils de binarisation, facteurs, classes et poids des facteurs, modèles. Rien n'est appris sur la partie évaluée.
- Les poids sont calculés sur la partie d'apprentissage du pli. On abandonne la variante « poids sur la validation » : en phase 1, elle changeait l'accuracy de moins d'un point.

## Étape 10 — Protocole et référence

1. Crée les 10 plis et sauvegarde-les.
2. Refais la configuration de la phase 1 en validation croisée : échelle nominale, GreConD, all-votes et best-vote, poids précision et accuracy, plus la règle « toujours la classe majoritaire ».
3. Résultat : un tableau avec la moyenne ± l'écart-type sur les 10 plis (accuracy, F1 macro, nombre de prédictions de classe 1, non-couverts, égalités), et les valeurs pli par pli dans un CSV.

On doit retrouver à peu près les résultats de la phase 1 : une accuracy de 0,65 à 0,69 et un F1 macro faible avec le poids précision. Sinon, arrête-toi et explique.

## Étape 11 — M1 : échelle interordinale

Dans `m1_binarisation/`, ajoute un paramètre `echelle` qui vaut `"nominale"` (la phase 1) ou `"interordinale"`.

Pour une variable avec ses deux seuils t1 < t2, **les mêmes que la phase 1** (terciles appris sur l'apprentissage), l'échelle interordinale crée 4 attributs : `v<=t1`, `v<=t2`, `v>t1`, `v>t2`.
- Chaque objet reçoit exactement 2 attributs par variable.
- L'intervalle du milieu s'exprime comme une conjonction : {`v>t1`, `v<=t2`}.
- Une valeur manquante (option b) ne reçoit aucun des 4 attributs.

**Test**, sur une variable de valeurs [1, 2, 3, 4, 5, 6, manquant], avec t1 = 2 et t2 = 4 :

| valeur | v<=2 | v<=4 | v>2 | v>4 |
|---|---|---|---|---|
| 1 et 2 | 1 | 1 | 0 | 0 |
| 3 et 4 | 0 | 1 | 1 | 0 |
| 5 et 6 | 0 | 0 | 1 | 1 |
| manquant | 0 | 0 | 0 | 0 |

Sur Pima, l'échelle interordinale donne 32 attributs. La densité attendue est d'environ 2/4 × (proportion de valeurs connues), donc un peu moins de 0,5. Calcule-la exactement, comme à la vérification b de la phase 1.

## Étape 12 — Mesurer la structure des facteurs

Sur l'apprentissage complet de la phase 1 (460 patientes), pour chaque échelle :
- la densité, le nombre de facteurs à 100 % de couverture, et le nombre de facteurs pour 80 et 90 % ;
- la **distribution des longueurs d'intention**, et la part de facteurs à un seul attribut : c'est le critère de H1 ;
- les 10 premiers facteurs, lisibles.

Fais-le avec GreConD. Ensuite, tente GreCon sur l'échelle interordinale : plus dense, elle peut avoir **beaucoup** de concepts. Compte d'abord les concepts. Au-delà de 200 000 concepts ou de 10 minutes, abandonne GreCon et note-le. Si GreCon tourne, dis si ses facteurs diffèrent maintenant de ceux de GreConD : en phase 1, ils étaient identiques.

Montre-moi ce tableau avant l'étape 13 : il décide déjà de H1.

## Étape 13 — M3 en validation croisée

**Nouveau poids : le lift.** lift(f, c) = précision(f, c) / P(c), où P(c) est la proportion de la classe c dans la partie d'apprentissage du pli. Un lift supérieur à 1 veut dire que la règle fait mieux que le hasard pour sa classe. Le lift remplace la précision dans all-votes et best-vote, sans autre changement.
Test sur les 10 personnes : F3 {Normal} → N a une précision de 0,75, et P(N) = 0,5, donc un lift de **1,5**.

**Grille, en validation croisée à 10 plis :**
- échelle : nominale ou interordinale ;
- vote : all-votes ou best-vote ;
- poids : précision, accuracy ou lift ;
- facteurs : GreConD, à 100 % de couverture.

**Références sur les mêmes binarisations :**
- Naive Bayes de Bernoulli, qui fait voter les attributs isolés avec des poids probabilistes ;
- régression logistique ;
- forêt aléatoire.

Ajoute la régression logistique et la forêt aléatoire sur les données brutes, comme référence générale.

**Tableau :** moyenne ± écart-type de l'accuracy, du F1 macro, du nombre de prédictions de classe 1 (avec le nombre réel), des non-couverts et des égalités. Ajoute la longueur moyenne d'intention des facteurs **qui votent effectivement**.

Pour H2, compare pli par pli la meilleure configuration interordinale à la même configuration nominale, et donne le nombre de plis où elle gagne.

## Étape 14 — Mushroom (si le temps le permet avant le 28 octobre)

- Source : OpenML, `fetch_openml(name="mushroom", version=1)`. 8 124 objets, 22 variables catégorielles. Vérifie la taille et la répartition des classes.
- Découpage figé 60/20/20 stratifié (`random_state=42`), sauvegardé. Le test reste scellé. Validation croisée à 10 plis sur apprentissage + validation.
- Binarisation nominale : un attribut par valeur de chaque variable. Une valeur manquante, « ? » dans `stalk-root`, ne reçoit aucun attribut, comme l'option b.
- GreConD seulement : GreCon est trop coûteux ici. Note le temps de calcul, et arrête-toi en me prévenant s'il dépasse 30 minutes par pli.
- M3 avec all-votes et best-vote, et les poids précision, accuracy et lift.
- Compare à l'article : 0,93 avec all-votes et 0,99 avec best-vote, pondérés par la précision. Attention : l'article a 88 attributs, son découpage et sa binarisation diffèrent. C'est un ordre de grandeur, pas une cible.
- Donne aussi la part de facteurs à un seul attribut. On s'attend à ce qu'elle soit bien plus faible que sur Pima nominal.

## Étape 15 — Synthèse

Un tableau final, puis une réponse courte à chaque hypothèse : confirmée ou rejetée, avec le chiffre du critère. Donne aussi, en trois phrases au plus, ce que cela implique pour la suite : les méthodes d'ensemble sur les facteurs, et M5.
Ajoute une entrée dans `JOURNAL.md`. Les nouvelles décisions vont dans `DECISIONS.md`.

## Hors périmètre

Le test, les méthodes d'ensemble, l'option (c) pour les valeurs manquantes, MDLGreConD, d'autres seuils ou un autre nombre d'intervalles, un réseau profond, M5.
