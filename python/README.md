# CoPRes

Composantes communes, propres et résiduelles (CoPRes) et décomposition orthogonale étiquetée optimale (LODE) de matrices semi-définies positives.

Version 0.2.0 (10 novembre 2026) : module `lode` (solveur général) et couche multibloc (`pretraiter`, `seuils_perm`, `cpca`, `acom`, `accps`, `copres`, `lecture_directe`, `fiche`, `proximite`, `equivalence`, `sensibilite`). Version 0.3.0 (10 novembre 2026) : couche multigroupe (`lire_notes`, `pretraiter_groupes`, `seuils_groupes`, `copres_groupes`, `cpc`, `lecture_groupes`, `discrimination`), avec les notes d'élèves simulées des articles. Les couches stabilité et graphiques suivront.

```python
X = cp.pretraiter([X1, X2, X3], snv=True)        # blocs mesurés sur les mêmes individus
theta = cp.seuils_perm(X)                         # seuils par permutation
c = cp.copres(X, theta, R=4, methode="CPCA")     # factorisation CoPRes
print(c, cp.fiche(None, c), cp.lecture_directe(X, c.base_methode, theta), sep="\n")
```

```python
import CoPRes as cp
res = cp.lode(A, theta, departs=20, graine=1)   # A : liste de matrices n x n, theta : seuils
print(res)                                       # motif, critère, borne de Ky Fan, écart
res.parties                                      # {étiquette: base de la partie}
```

Fonctions : `saliences`, `etiquettes`, `motif`, `critere`, `marge`, `est_etiquetage`, `borne_kf`, `ideal`, `concentration`, `base_concentree`, `taro`, `b2bl`, `lode`.

Tests : `PYTHONPATH=src python -m pytest tests`.

Licence : GPL-3.0 ou ultérieure. Auteur : Mohamed Hanafi (StatSC, Oniris-INRAE).

Version 0.4.0 (10 novembre 2026) : stabilité par bootstrap (`stabilite_groupes`, `stabilite_blocs`), critères strict, par étiquette et par type. Les tests longs se lancent avec `-m lent`.
