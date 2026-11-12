# CoPRes

**Composantes communes, propres et résiduelles** pour l'analyse de données multiblocs (plusieurs groupes de variables mesurés sur les mêmes individus) et multigroupes (mêmes variables mesurées sur plusieurs groupes d'individus), et **LODE**, décomposition orthogonale étiquetée optimale de matrices semi-définies positives.

Version 0.5.0. Implémentations R et Python identiques : un même générateur pseudo-aléatoire (Park et Miller) donne les mêmes résultats dans les deux langages.

## Installation

R :
```r
install.packages("remotes")
remotes::install_github("hanafisoft/CoPRes", subdir = "R")
```

Python :
```
pip install "git+https://github.com/hanafisoft/CoPRes#subdirectory=python"
```

## Exemples

Données multiblocs (R) :
```r
library(CoPRes)
X <- pretraiter(list(X1, X2, X3), snv = TRUE)   # blocs : mêmes individus en lignes
theta <- seuils_perm(X)                          # seuils par permutation
cop <- copres(X, theta, R = 4, methode = "CPCA")
cop                                              # motif, bilan, borne de Ky Fan
fiche(NULL, cop)                                 # mélange et dilution des composantes de la CPCA
stabilite_blocs(cop, X, B = 100)
```

Données multigroupes (Python) :
```python
import CoPRes as cp
X, g, variables = cp.lire_notes(1)               # notes d'élèves simulées, trois lycées
P = cp.pretraiter_groupes(X, g)
cop = cp.copres_groupes(P, cp.seuils_groupes(P))
print(cop, cp.discrimination(cop, P), cp.stabilite_groupes(cop, P), sep="\n")
```

## Contenu

| Couche | Fonctions |
|---|---|
| LODE | `lode`, `borne_kf`, `ideal`, `b2bl`, `taro`, `base_concentree`, `saliences`, `etiquettes`, `est_etiquetage` |
| Multibloc | `pretraiter`, `seuils_perm`, `cpca`, `acom`, `accps`, `copres`, `lecture_directe`, `fiche`, `equivalence`, `sensibilite` |
| Multigroupe | `pretraiter_groupes`, `seuils_groupes`, `copres_groupes`, `cpc`, `lecture_groupes`, `discrimination`, `lire_notes` |
| Stabilité | `stabilite_blocs`, `stabilite_groupes` |
| Méthodes concurrentes | `jive` (selon r.jive 2.4), `ajive` (selon mvlearn 0.5), `rang_coude` |
| Graphiques (expérimentaux) | `graph_saliences`, `graph_coherence`, `graph_fiche`, `graph_motif`, `graph_scores`, `graph_scores_parties`, `graph_loadings`, `graph_contributions`, `graph_stabilite`, `graph_sensibilite`, `graph_bilan` ; données : `donnees_*` ; langue `langue("fr")` / `langue("en")` ; thème `theme_copres()` |

## Tests

R : `testthat::test_dir("R/tests/testthat", package = "CoPRes")` Python : `cd python && PYTHONPATH=src python -m pytest tests`. Les graphiques R demandent ggplot2 (`install.packages("ggplot2")`). Les tests R contiennent des valeurs de référence calculées en Python et vérifient la concordance des deux versions.

## Références

- Hanafi M. LODE : décomposition orthogonale étiquetée optimale d'un espace par plusieurs matrices semi-définies positives, avec application à l'analyse multibloc et multigroupe. Prépublication, 2026.
- Hanafi M. Composantes communes, propres et résiduelles : lire une analyse multibloc ou multigroupe avec CoPRes. Article compagnon, en préparation.

## Nouveautés

Voir `python/NEWS.md` et `R/NEWS.md`.

## Licence

GPL-3.0 ou ultérieure. Les fonctions `jive` et `ajive` sont des réécritures des algorithmes de r.jive (O'Connell et Lock, GPL-3) et de mvlearn (Carmichael et al., licence MIT), vérifiées contre ces paquets. Auteur : Mohamed Hanafi (StatSC, Oniris-INRAE, Nantes).

Contact : mohamed.hanafi@oniris-nantes.fr, mohammed44hanafi@gmail.com
