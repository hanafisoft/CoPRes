# CoPRes (R)

Version R 0.2.0 (10 novembre 2026) du paquet CoPRes : module lode (solveur général) et couche multibloc, identiques pas à pas à la version Python 0.2.0, avec le même générateur de Park et Miller.

Version R 0.3.0 (10 novembre 2026) : couche multigroupe (lire_notes, pretraiter_groupes, seuils_groupes, copres_groupes, cpc, lecture_groupes, discrimination).

    nt <- lire_notes(1)
    P <- pretraiter_groupes(nt$X, nt$groupes)
    cop <- copres_groupes(P, seuils_groupes(P))
    cop; discrimination(cop, P)

    X <- pretraiter(list(X1, X2, X3), snv = TRUE)
    theta <- seuils_perm(X)
    cop <- copres(X, theta, R = 4, methode = "CPCA")
    cop; fiche(NULL, cop); lecture_directe(X, cop$base_methode, theta)

Installation depuis le dossier source :

    R CMD build CoPRes && R CMD INSTALL CoPRes_0.1.0.tar.gz
    # ou, dans R : devtools::install("CoPRes")

Exemple :

    library(CoPRes)
    res <- lode(A, theta, departs = 20, graine = 1)   # A : liste de matrices n x n
    res                                               # motif, critère, borne, écart
    res$parties                                       # liste (étiquette, base)

Tests : `devtools::test()` ou `R CMD check`. Ils contiennent des valeurs de référence calculées par la version Python ; leur réussite atteste la concordance des deux versions.

Licence : GPL (>= 3). Auteur : Mohamed Hanafi (StatSC, Oniris-INRAE).

Version R 0.4.0 (10 novembre 2026) : stabilité par bootstrap (stabilite_groupes, stabilite_blocs).

    stabilite_groupes(cop, P, B = 100)
