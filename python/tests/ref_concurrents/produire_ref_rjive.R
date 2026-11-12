# Sorties de référence de r.jive pour les tests de la fonction jive de CoPRes.
# À lancer une fois dans RStudio, après install.packages("r.jive").
# Écrit, pour chaque jeu (a, b, c), dans ce dossier :
#   <jeu>_rjive_joint.csv, <jeu>_rjive_individuel.csv  (rangs donnés ; individus en lignes)
#   <jeu>_rjive_perm_rangs.csv                          (rangs estimés par permutation)
# Indiquer ci-dessous le chemin de ce dossier (barres obliques / ).

dossier <- "CHEMIN/VERS/CoPRes/python/tests/ref_concurrents"

library(r.jive)
rangs <- list(a = list(J = 2, A = c(2, 2, 2)),
              b = list(J = 1, A = c(2, 2, 2)),
              c = list(J = 2, A = c(1, 1)))

for (j in names(rangs)) {
  X <- as.matrix(read.csv(file.path(dossier, paste0(j, "_donnees.csv")), header = FALSE))
  p <- as.numeric(read.csv(file.path(dossier, paste0(j, "_blocs.csv")), header = FALSE))
  fin <- cumsum(p); debut <- fin - p + 1
  # r.jive attend les variables en lignes et les individus en colonnes
  donnees <- lapply(seq_along(p), function(k) t(X[, debut[k]:fin[k], drop = FALSE]))

  g <- jive(donnees, rankJ = rangs[[j]]$J, rankA = rangs[[j]]$A, method = "given",
            showProgress = FALSE)
  joint <- do.call(cbind, lapply(g$joint, t))
  indiv <- do.call(cbind, lapply(g$individual, t))
  write.table(formatC(joint, digits = 17, format = "g"), file.path(dossier, paste0(j, "_rjive_joint.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(formatC(indiv, digits = 17, format = "g"), file.path(dossier, paste0(j, "_rjive_individuel.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

  set.seed(1)
  q <- jive(donnees, method = "perm", showProgress = FALSE)
  write.table(t(c(q$rankJ, q$rankA)), file.path(dossier, paste0(j, "_rjive_perm_rangs.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
  cat("jeu", j, ": rangs donnés OK ; rangs par permutation", q$rankJ, "/", q$rankA, "\n")
}
cat("Terminé. Renvoyer les fichiers *_rjive_*.csv.\n")
