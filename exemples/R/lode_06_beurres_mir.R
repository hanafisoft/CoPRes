# LODE, section 7.4 : beurres et margarines en moyen infrarouge (longueurs d'onde = variables).
source("commun.R")
b <- lire_beurres(); mir <- b$blocs[[3]]
mir <- (mir - rowMeans(mir)) / apply(mir, 1, sd)                     # SNV
P <- pretraiter_groupes(mir, b$type, reduire = "aucune", R = 4)
ev <- eigen(crossprod(P$Xs), symmetric = TRUE, only.values = TRUE)$values
cat(sprintf("variance des quatre premiers axes : %.0f %%\n", 100 * sum(ev[1:4]) / sum(ev)))
th <- seuils_groupes(P, regle = "valeur_propre", graine = 7)
c <- copres_groupes(P, th, departs = 50); print(round(th, 4)); print(c)
g <- CoPRes:::.groupes(c$etiquettes)
for (i in seq_along(g)) { cat(nom_etiquette(attr(g, "etiq")[[i]], c$groupes), ": "); print(round(c$saliences[, g[[i]], drop = FALSE] / th, 2)) }
print(stabilite_groupes(c, P, B = 100))
