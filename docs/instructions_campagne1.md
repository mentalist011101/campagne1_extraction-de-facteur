# Instructions pour Claude Code — Campagne 1 : méthodes d'ensemble sur les facteurs, à grande échelle

Le cycle de travail et les règles de `CLAUDE.md` s'appliquent. Place ce fichier dans `docs/instructions_campagne1.md`.

## 0. Nouvelle règle Git, à ajouter à `CLAUDE.md` avant toute chose

> **Git : tu ne fais jamais de commit, de push ni de pull request.** Quand une étape est validée, tu me proposes un message de commit, et c'est moi qui commite et qui pousse. N'ajoute jamais de ligne `Co-Authored-By` ni aucune autre mention de toi dans les messages de commit, les fichiers ou le dépôt.

Ajoute aussi ce paragraphe à `CLAUDE.md` :

> **Dépôt GitHub et Kaggle.** Le projet est poussé sur GitHub par moi. Les calculs longs tournent sur Kaggle, qui clone le dépôt. Aucun secret (jeton, mot de passe) n'est écrit dans le code ou dans un notebook : si le dépôt est privé, le jeton passe par les *Kaggle Secrets*.

## 1. Pourquoi cette campagne

Sur congressional voting (phase 3), le stacking a atteint le niveau des modèles classiques, et le bagging l'a presque atteint. Mais avec un seul jeu et des plis de 35 députés, rien n'était significatif.

Cette campagne refait la même comparaison sur **une vingtaine de jeux de données**, ceux des articles de référence, pour pouvoir conclure avec des tests statistiques adaptés : Wilcoxon et Friedman, puis Nemenyi (Demšar, JMLR 2006).

## 2. Les jeux de données

On prend la réunion des jeux de Makhalova & Trnecka (2019) et de CNC-TP (Souissi, Boissier & Meddouri, 2026), plus congressional voting. On retire les jeux de moins de 100 objets : une validation croisée à 10 plis y aurait des plis de moins de 10 objets. Cette règle est fixée **avant** de voir le moindre résultat.

| Jeu | Taille attendue | Source de la liste |
|---|---|---|
| anneal | 898 | Makhalova & Trnecka |
| breast (Wisconsin) | 699 | Makhalova & Trnecka, CNC-TP |
| hepatitis | 155 | Makhalova & Trnecka |
| horse colic | 368 | Makhalova & Trnecka |
| iris | 150 | Makhalova & Trnecka, CNC-TP |
| led7 | 3 200 | Makhalova & Trnecka |
| mushroom | 8 124 | Makhalova & Trnecka |
| nursery | 12 960 (l'article en prend 1 000) | Makhalova & Trnecka |
| page blocks | 5 473 | Makhalova & Trnecka |
| pima | 768 | Makhalova & Trnecka, CNC-TP |
| wine | 178 | Makhalova & Trnecka |
| balance-scale | 625 | CNC-TP |
| breast-cancer (Ljubljana) | 286 | CNC-TP |
| credit-rating (credit-a) | 690 | CNC-TP |
| glass | 214 | CNC-TP |
| heart-statlog | 270 | CNC-TP |
| vowel | 990 | CNC-TP |
| zoo | 101 | CNC-TP |
| congressional voting | 435 | notre phase 3 |

**Exclus, avec leur raison** (à noter dans `DECISIONS.md`) : contact-lenses (24 objets), weather numeric et weather symbolic (14), labor (57), lupus (87), iris-2D (doublon d'iris).

**Pour chaque jeu :**
- Retrouve-le sur OpenML. Note dans `donnees/catalogue.csv` son nom, son identifiant OpenML, sa version, sa taille, le nombre de classes et le nombre de variables numériques et catégorielles.
- Vérifie la taille par rapport au tableau. **Signale tout écart, ne corrige rien.**
- Pour nursery : prends le jeu complet. Note que l'article en utilise un sous-ensemble de 1 000 objets dont on ne connaît pas le tirage.
- Enregistre une copie figée en CSV dans `donnees/brutes/`, et commite-la avec le reste. Ainsi, Kaggle n'a pas besoin d'internet. Ces jeux sont sous licence CC BY 4.0.

## 3. Le protocole, fixé à l'avance

**Validation croisée stratifiée à 10 plis sur le jeu entier**, avec la graine 42. Note dans `DECISIONS.md` pourquoi il n'y a pas de test scellé ici : **aucun hyperparamètre n'est réglé**, tout est fixé dans ce fichier avant le lancement. C'est le protocole des articles qu'on compare, ce qui rend les résultats comparables.

Dans chaque pli, tout est appris sur la partie d'apprentissage du pli seulement :

- **Binarisation :**
  - variables catégorielles : un attribut par valeur ;
  - variables numériques : échelle **interordinale** sur les terciles ;
  - valeur manquante : aucun attribut (option b) ;
  - attributs constants sur la partie d'apprentissage : retirés.
- **Facteurs :** GreConD à 100 % de couverture, sans la classe.
- **Plusieurs classes :** beaucoup de ces jeux ont plus de 2 classes, alors que notre code a été écrit pour 2. Il faut le généraliser :
  - all-votes et best-vote fonctionnent tels quels ;
  - le stacking utilise une régression logistique multinomiale ;
  - en cas d'égalité de vote, on prend la classe majoritaire de la partie d'apprentissage ;
  - les métriques sont l'accuracy et le **F1 macro**.

## 4. Les méthodes comparées (liste fermée)

| Code | Méthode |
|---|---|
| MAJ | Classe majoritaire de la partie d'apprentissage |
| M3-AV | Facteurs → classifieurs, all-votes, poids = précision (Makhalova & Trnecka) |
| M3-BV | Facteurs → classifieurs, best-vote, poids = précision (Makhalova & Trnecka) |
| BAG | Bagging de M3-BV : 25 sacs, graine 42 + b, binarisation du pli réutilisée dans les sacs, vote majoritaire |
| STACK | Régression logistique multinomiale sur la matrice A de **tous** les facteurs (`max_iter=1000`, `random_state=42`, autres paramètres par défaut) |
| LR | Régression logistique sur les attributs binarisés (mêmes paramètres) |
| RF | Forêt aléatoire, 100 arbres, `random_state=42`, sur les attributs binarisés |
| NB | Naive Bayes de Bernoulli sur les attributs binarisés |

Aucune autre méthode, aucun réglage. DNC et CNC-TP (le code Java de Meddouri) viendront dans une campagne séparée.

## 5. Les hypothèses, à committer AVANT le lancement

Comme pour H1 à H7 : écris-les dans `DECISIONS.md`. Je les commiterai avant que la campagne ne tourne. Les comparaisons se font **sur les jeux de données**, pas sur les plis : pour chaque jeu, on prend le F1 macro moyen sur ses 10 plis.

| | Hypothèse | Test et critère |
|---|---|---|
| H8 | Le stacking améliore M3-BV | Test de Wilcoxon signé sur les F1 par jeu (STACK contre M3-BV), bilatéral, p < 0,05, et STACK meilleur en médiane |
| H9 | Le bagging améliore M3-BV | Même test (BAG contre M3-BV). Notre prédiction : non significatif |
| H10 | Le stacking n'est pas inférieur à la forêt aléatoire | L'intervalle de confiance bootstrap à 95 % (10 000 tirages sur les jeux, graine 42) de la différence moyenne de F1 (STACK − RF) a une borne basse > −0,02 |

**En plus, de façon descriptive :**
- un test de Friedman sur les 8 méthodes, puis un test de Nemenyi, avec le diagramme de la différence critique ;
- le rang moyen de chaque méthode ;
- pour iris, mushroom, pima et wine, nos accuracies de M3-AV et M3-BV à côté de celles de la Table 3 de Makhalova & Trnecka. Les binarisations diffèrent : c'est un ordre de grandeur.

## 6. Le script de campagne, robuste pour Kaggle

- **Un seul point d'entrée** : `python -m experiences.campagne1 --config experiences/campagne1.yaml`. Le fichier de configuration liste les jeux, les méthodes et les graines.
- **Reprise :** chaque couple (jeu, pli) écrit ses résultats dans `resultats/campagne1/<jeu>/pli_<k>.csv` dès qu'il est fini. Au redémarrage, un couple déjà calculé est sauté. Une session coupée ne fait donc rien perdre.
- **Limite de temps :** si GreConD dépasse 20 minutes sur un pli, le jeu est marqué « abandonné » avec sa raison, et la campagne continue. Ne le cache pas.
- **Journal :** temps par jeu, par pli et par méthode ; nombre de facteurs ; part de facteurs à un seul attribut.
- **Sorties finales :** un CSV de toutes les valeurs par pli, un CSV des moyennes par jeu, et un script séparé `experiences/analyse_campagne1.py` qui produit les tests H8 à H10, Friedman, Nemenyi et le diagramme. L'analyse ne relance aucun calcul.
- **Notebook Kaggle :** `kaggle/campagne1.ipynb`. Il clone le dépôt, installe `requirements.txt` et lance le script. Il ne contient aucun secret. Si le dépôt est privé, le jeton est lu depuis les Kaggle Secrets.

## 7. Les étapes, avec ta validation entre chacune

1. Les règles Git et Kaggle dans `CLAUDE.md`, puis `docs/instructions_campagne1.md`.
2. Le catalogue et les copies figées des 19 jeux, avec les écarts signalés.
3. Le passage à plusieurs classes dans M3, le bagging et le stacking, avec des tests. Test attendu : sur iris, les 3 classes doivent apparaître dans les prédictions de M3-AV.
4. Les hypothèses H8 à H10 dans `DECISIONS.md`. **Je les commite avant toute exécution.**
5. Le script de campagne et sa reprise. Test : interrompre volontairement après 2 plis, relancer, et vérifier que rien n'est recalculé.
6. Un essai local sur 3 petits jeux (iris, wine, zoo), avec une estimation du temps total par jeu.
7. Le notebook Kaggle. C'est moi qui lance la campagne.
8. Après la campagne : le script d'analyse et le tableau final.

## Hors périmètre

Le réglage d'hyperparamètres, GreCon (trop lent sur les grands jeux), d'autres échelles de binarisation, DNC et CNC-TP, le score, M4 à M7.
