# Instructions pour Claude Code — Phase 3 (légère) : premières méthodes d'ensemble sur les facteurs

Le cycle de travail et les règles de `CLAUDE.md` s'appliquent sans changement. Place ce fichier dans `docs/instructions_phase3_ensembles.md`. Mets à jour la ligne « Phase en cours » de `CLAUDE.md` après mon accord.

**Échéance : résultats avant 15 h aujourd'hui.** Si une étape menace l'échéance, arrête-toi et préviens-moi : on réduira (par exemple, moins de sacs).

## Pourquoi cette phase

C'est l'axe proposé par mon encadreuse : combiner les classifieurs factoriels avec des méthodes d'ensemble au lieu de les faire voter directement. On le teste sur un exemple léger, avec tout ce qui existe déjà : Pima, les 10 plis de l'étape 10, les deux échelles. Le test reste scellé.

L'étape 13 a montré que M3 souffre surtout d'un **biais**. Ses facteurs ne sont pas liés à la classe, et ses poids sont des heuristiques : le Naive Bayes de Bernoulli le bat d'environ 0,10 de F1 macro. On compare donc deux stratégies choisies pour une raison :

- **Le bagging**, l'exemple cité par l'encadreuse. Il réduit surtout la **variance**. Notre prédiction : il aidera peu.
- **Le stacking**, une combinaison **apprise**. Il s'attaque au **biais**. Ici, il revient à une régression logistique sur la matrice A (objets × facteurs) : les facteurs deviennent des attributs, avec des poids appris.

## Hypothèses fixées à l'avance

Commite-les dans `DECISIONS.md` **avant** la première expérience, comme pour H1 à H4.

| | Hypothèse | Critère |
|---|---|---|
| H5 | Le bagging améliore M3. | Pour chaque échelle, le F1 macro de M3 avec bagging dépasse celui de la même configuration sans bagging dans au moins 8 plis sur 10. (Notre prédiction : rejetée.) |
| H6 | Le stacking améliore M3. | Pour chaque échelle, le F1 macro du stacking dépasse celui du meilleur M3 de l'étape 13 sur la même échelle (all-votes + lift) dans au moins 8 plis sur 10. |

Une égalité exacte sur un pli ne compte pas comme une victoire.

**Comparaison descriptive D1, sans critère :** pli par pli, stacking contre régression logistique sur les **attributs** de la même échelle. C'est elle qui dit si les facteurs apportent quelque chose **en tant que représentation**. En échelle nominale, 24 facteurs sur 26 sont des attributs isolés : on s'attend à ce que les deux soient très proches. Ce sera un contrôle de cohérence.

## Étape 16 — Stacking

Dans chaque pli, et pour chaque échelle :
1. GreConD à 100 % sur le contexte d'apprentissage du pli, **sans la classe**.
2. On construit la matrice A_app (objets d'apprentissage × facteurs) : 1 si l'objet possède **tous** les attributs de l'intention du facteur. On construit A_eval de la même façon pour la partie évaluée.
3. On garde **tous** les facteurs, y compris ceux que M3 écartait pour égalité de classe : ici, ce n'est plus un vote, c'est la régression qui fixe les poids.
4. Le méta-classifieur est `LogisticRegression(max_iter=1000, random_state=42)` avec les autres paramètres par défaut. Il est entraîné sur (A_app, y_app), puis prédit sur A_eval.

**Pourquoi il n'y a pas de fuite :** en stacking classique, il faut des prédictions hors pli des classifieurs de base, puisqu'ils ont vu la classe. Ici, A est calculée **sans la classe** : seule la régression utilise y_app. Écris ce point dans `DECISIONS.md`.

**Tests :**
- Sur l'exemple des 10 personnes, avec GreConD, chaque colonne de A_app est exactement l'extension du facteur. C'est un concept, donc la règle et l'extension coïncident.
- Les dimensions de A_app et A_eval sont correctes, et les colonnes sont dans l'ordre des facteurs.

**Lisibilité :** sur le pli 0, en échelle interordinale, donne les 5 facteurs aux plus grands coefficients positifs et les 5 aux plus grands coefficients négatifs, avec leurs intentions lisibles. Le stacking garde des règles lisibles, avec des poids appris : c'est un point à montrer.

## Étape 17 — Bagging

Dans chaque pli, et pour chaque échelle :
1. B = 25 sacs. Chaque sac est un tirage **avec remise** de n_app patientes dans la partie d'apprentissage du pli, avec la graine `42 + b` pour le sac b.
2. Dans chaque sac : GreConD à 100 % sur le contexte du sac, sans la classe. Puis M3, avec les classes et les poids des facteurs calculés **sur le sac**.
3. La configuration de M3 est all-votes + lift, le meilleur M3 de l'étape 13. H5 compare donc M3 à lui-même, avec ou sans bagging.
4. L'agrégation se fait par un vote majoritaire des 25 prédictions. En cas d'égalité, on prend la classe majoritaire de la partie d'apprentissage du pli. Sauvegarde aussi, pour chaque x, la **proportion de sacs qui prédisent la classe 1** : ce sera un signal possible pour M5.
5. Note le nombre de x non couverts dans chaque sac.

**Coût :** mesure le temps d'un sac. Si le total prévu dépasse 30 minutes, passe à B = 10 et note-le.

## Étape 18 — Tableau et synthèse

Un tableau en moyenne ± écart-type sur les 10 plis, pour chaque échelle :
- la règle majoritaire ;
- le meilleur M3 seul (all-votes + lift) ;
- M3 avec bagging ;
- le stacking ;
- la régression logistique sur les attributs ;
- le Naive Bayes de Bernoulli.

Colonnes : accuracy, F1 macro, nombre de prédictions de classe 1 (avec le nombre réel).

Ensuite :
- les victoires pli par pli pour H5 et H6, et le verdict selon les critères ;
- D1 pli par pli ;
- les 10 facteurs du stacking (étape 16) ;
- une entrée dans `JOURNAL.md`, avec trois phrases au plus sur ce que cela implique pour l'axe ensembliste.

## Ordre par rapport à mushroom

Si l'étape 14 (mushroom) n'est pas terminée, elle passe **après** cette phase. L'axe ensembliste est prioritaire pour la présentation de demain.

## Hors périmètre

Le boosting, la sélection dynamique, d'autres jeux de données, le test, et le réglage des hyperparamètres de la régression logistique.
