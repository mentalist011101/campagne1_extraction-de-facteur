# Données pour la présentation

Toutes les valeurs viennent de la validation croisée à 10 plis (partie apprentissage + validation de chaque jeu). **Le test n'a jamais été lu.**

## 1. Pima, étape 13 : toutes les méthodes (moyenne ± écart-type sur 10 plis)

Source : `resultats/pima_cv10_grille_resume_2026-10-24.csv`. Chaque pli évalue 61 ou 62 patientes, dont 21,5 de classe 1 en moyenne.

| methode | accuracy | f1_macro | predits_classe_1 | reels_classe_1 |
|---|---|---|---|---|
| majoritaire | 0.65 ± 0.006 | 0.394 ± 0.002 | 0.0 ± 0.0 | 21.5 ± 0.527 |
| nominale all-votes precision | 0.689 ± 0.025 | 0.533 ± 0.06 | 4.8 ± 2.821 | 21.5 ± 0.527 |
| nominale best-vote precision | 0.656 ± 0.015 | 0.432 ± 0.038 | 1.4 ± 1.35 | 21.5 ± 0.527 |
| nominale all-votes accuracy | 0.686 ± 0.067 | 0.605 ± 0.07 | 12.2 ± 3.994 | 21.5 ± 0.527 |
| nominale best-vote accuracy | 0.617 ± 0.07 | 0.615 ± 0.074 | 42.0 ± 4.19 | 21.5 ± 0.527 |
| nominale all-votes lift | 0.687 ± 0.065 | 0.61 ± 0.065 | 12.5 ± 3.866 | 21.5 ± 0.527 |
| nominale best-vote lift | 0.617 ± 0.07 | 0.615 ± 0.074 | 42.0 ± 4.19 | 21.5 ± 0.527 |
| naive_bayes_bernoulli (nominale) | 0.73 ± 0.064 | 0.712 ± 0.067 | 24.5 ± 2.593 | 21.5 ± 0.527 |
| regression_logistique (nominale) | 0.756 ± 0.055 | 0.724 ± 0.064 | 19.3 ± 2.751 | 21.5 ± 0.527 |
| foret_aleatoire (nominale) | 0.728 ± 0.066 | 0.691 ± 0.074 | 18.8 ± 2.53 | 21.5 ± 0.527 |
| interordinale all-votes precision | 0.674 ± 0.023 | 0.488 ± 0.06 | 3.1 ± 2.234 | 21.5 ± 0.527 |
| interordinale best-vote precision | 0.679 ± 0.03 | 0.496 ± 0.072 | 3.2 ± 2.044 | 21.5 ± 0.527 |
| interordinale all-votes accuracy | 0.705 ± 0.041 | 0.597 ± 0.065 | 8.6 ± 4.195 | 21.5 ± 0.527 |
| interordinale best-vote accuracy | 0.604 ± 0.061 | 0.599 ± 0.059 | 42.4 ± 7.26 | 21.5 ± 0.527 |
| interordinale all-votes lift | 0.715 ± 0.042 | 0.617 ± 0.055 | 9.2 ± 3.824 | 21.5 ± 0.527 |
| interordinale best-vote lift | 0.585 ± 0.059 | 0.582 ± 0.061 | 44.4 ± 2.875 | 21.5 ± 0.527 |
| naive_bayes_bernoulli (interordinale) | 0.723 ± 0.062 | 0.711 ± 0.062 | 27.1 ± 2.47 | 21.5 ± 0.527 |
| regression_logistique (interordinale) | 0.756 ± 0.058 | 0.724 ± 0.07 | 19.7 ± 3.093 | 21.5 ± 0.527 |
| foret_aleatoire (interordinale) | 0.731 ± 0.064 | 0.697 ± 0.071 | 19.4 ± 3.169 | 21.5 ± 0.527 |
| regression_logistique (brutes) | 0.782 ± 0.062 | 0.745 ± 0.077 | 17.1 ± 2.025 | 21.5 ± 0.527 |
| foret_aleatoire (brutes) | 0.756 ± 0.064 | 0.724 ± 0.077 | 19.5 ± 2.635 | 21.5 ± 0.527 |

## 2. Structure des facteurs (GreConD à 100 %, moyenne sur les 10 plis)

Calculée sur la partie d'apprentissage de chaque pli. Mushroom : relu depuis `resultats/mushroom_cv10_facteurs_2026-10-28.csv`.

| jeu | attributs | densite | facteurs | part_un_attribut | longueur_moyenne | temps_grecond_s |
|---|---|---|---|---|---|---|
| Pima nominale | 24.0 | 0.298 | 26.0 | 0.923 | 1.08 | 0.09 (min 0.06, max 0.13) |
| Pima interordinale | 32.0 | 0.447 | 45.2 | 0.024 | 2.69 | 0.28 (min 0.20, max 0.53) |
| mushroom | 115.0 | 0.180 | 120.6 | 0.034 | 8.43 | 10.84 (min 6.9, max 16.2) |
| vote | 32.0 | 0.471 | 68.3 | 0.162 | 2.63 | 0.52 (min 0.32, max 1.13) |

**GreCon (apprentissage de la phase 1, 460 patientes, pas de validation croisée) :**

| Jeu | Concepts | Temps |
|---|---|---|
| Pima nominal | 7 212 | 17,7 s (étape 6, énumération comprise) |
| Pima interordinal | 150 555 | énumération 35 s + GreCon 309 s (étape 12) |
| mushroom, vote | non lancé | — |

## 3. Exemples de facteurs, pli 0 (3 meilleurs par classe selon la précision)

Précision mesurée sur la partie d'apprentissage du pli 0. À précision égale, la plus grande extension d'abord.

### Pima (nominale) — classe 1 = diabète

26 facteurs, 0 écarté(s).

| facteur | classe | precision | taille_extension | intention |
|---|---|---|---|---|
| F20 | 0 | 0.909 | 99 | insuline<=94 |
| F3 | 0 | 0.882 | 186 | glucose<=105 |
| F5 | 0 | 0.863 | 183 | imc<=28.9 |
| F6 | 1 | 0.67 | 182 | glucose>129 |
| F14 | 1 | 0.565 | 115 | grossesses>5 ET age>37 |
| F21 | 1 | 0.543 | 94 | insuline>165 |

### mushroom — classe 1 = vénéneux

123 facteurs, 0 écarté(s).

| facteur | classe | precision | taille_extension | intention |
|---|---|---|---|---|
| F21 | 0 | 1.0 | 889 | odor=n ET gill-attachment=f ET gill-size=b ET stalk-shape=t ET veil-color=w ET ring-number=o ET spore-print-color=k |
| F14 | 0 | 1.0 | 632 | cap-surface=f ET bruises=t ET odor=n ET gill-attachment=f ET gill-spacing=c ET gill-size=b ET stalk-shape=t ET stalk-root=b ET stalk-surface-above-ring=s ET stalk-surface-below-ring=s ET veil-color=w ET ring-number=o ET ring-type=p ET habitat=d |
| F4 | 0 | 1.0 | 413 | bruises=t ET odor=n ET gill-attachment=f ET gill-spacing=c ET gill-size=b ET stalk-shape=t ET stalk-root=b ET stalk-surface-above-ring=s ET stalk-surface-below-ring=s ET stalk-color-below-ring=g ET veil-color=w ET ring-number=o ET ring-type=p ET habitat=d |
| F40 | 1 | 1.0 | 1573 | odor=f ET gill-attachment=f ET gill-spacing=c ET veil-color=w ET ring-number=o |
| F1 | 1 | 1.0 | 1230 | bruises=f ET gill-attachment=f ET gill-spacing=c ET gill-size=n ET gill-color=b ET stalk-shape=t ET veil-color=w ET ring-number=o ET ring-type=e ET spore-print-color=w ET population=v |
| F31 | 1 | 1.0 | 1105 | bruises=f ET gill-attachment=f ET gill-spacing=c ET stalk-surface-above-ring=k ET veil-color=w ET ring-number=o ET population=v |

### vote — classe 1 = républicain

64 facteurs, 0 écarté(s).

| facteur | classe | precision | taille_extension | intention |
|---|---|---|---|---|
| F45 | 0 | 1.0 | 142 | adoption-of-the-budget-resolution=y ET physician-fee-freeze=n ET aid-to-nicaraguan-contras=y |
| F22 | 0 | 1.0 | 105 | adoption-of-the-budget-resolution=y ET physician-fee-freeze=n ET el-salvador-aid=n ET crime=n |
| F8 | 0 | 1.0 | 89 | handicapped-infants=y ET adoption-of-the-budget-resolution=y ET physician-fee-freeze=n ET education-spending=n |
| F2 | 1 | 1.0 | 49 | adoption-of-the-budget-resolution=n ET physician-fee-freeze=y ET el-salvador-aid=y ET religious-groups-in-schools=y ET anti-satellite-test-ban=n ET aid-to-nicaraguan-contras=n ET mx-missile=n ET synfuels-corporation-cutback=n ET education-spending=y ET crime=y ET duty-free-exports=n |
| F29 | 1 | 0.966 | 88 | handicapped-infants=n ET adoption-of-the-budget-resolution=n ET physician-fee-freeze=y |
| F11 | 1 | 0.92 | 88 | physician-fee-freeze=y ET el-salvador-aid=y ET superfund-right-to-sue=y ET crime=y ET duty-free-exports=n |
