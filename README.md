# CoPRes

**Composantes communes, propres et résiduelles** pour l'analyse de données multiblocs (plusieurs groupes de variables mesurés sur les mêmes individus) et multigroupes (mêmes variables mesurées sur plusieurs groupes d'individus), et **LODE**, décomposition orthogonale étiquetée optimale de matrices semi-définies positives.

Version 0.4.0. Implémentations R et Python identiques : un même générateur pseudo-aléatoire (Park et Miller) donne les mêmes résultats dans les deux langages.

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

## Tests

R : `testthat::test_dir("R/tests/testthat", package = "CoPRes")` (144 tests). Python : `cd python && PYTHONPATH=src python -m pytest tests` (34 tests). Les tests R contiennent des valeurs de référence calculées en Python et vérifient la concordance des deux versions.

## Références

- Hanafi M. LODE : décomposition orthogonale étiquetée optimale d'un espace par plusieurs matrices semi-définies positives, avec application à l'analyse multibloc et multigroupe. Prépublication, 2026.
- Hanafi M. Composantes communes, propres et résiduelles : lire une analyse multibloc ou multigroupe avec CoPRes. Article compagnon, en préparation.

## Licence

GPL-3.0 ou ultérieure. Auteur : Mohamed Hanafi (StatSC, Oniris-INRAE, Nantes).

Contact : mohamed.hanafi@oniris-nantes.fr, mohammed44hanafi@gmail.com
