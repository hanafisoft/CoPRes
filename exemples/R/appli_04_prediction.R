# Article appliqué, section 4.2 : plusieurs blocs, des groupes connus (rates : régime ; beurres :
# type), R = 4. Même calcul que python/appli_04_prediction.py.
# Usage (depuis exemples/R) : VC <- TRUE ; source("appli_04_prediction.R")
# (VC <- FALSE pour sauter la validation croisée, la partie la plus longue)
source("commun.R")
if (!exists("VC")) VC <- TRUE
R <- 4

etude <- function(nom, X, groupes, noms, B = 200) {
  cat("\n", nom, "\n", sep = "")
  th <- seuils_perm(X)
  cop <- copres(X, th, R = R, departs = 200)
  copP <- copres(X, th, U = plsda(X, groupes, R), methode = "PLS-DA", departs = 200)
  for (m in list(list("CPCA", cop), list("PLS-DA", copP))) {
    inf <- information_groupe(m[[2]], groupes)
    cat(sprintf("  %s : motif %s\n    %s ; hors de l'espace %.0f %%\n", m[[1]], texte_motif(m[[2]]$motif, noms),
                paste(sprintf("%s %.0f %%", vapply(inf$etiquettes, nom_etiquette, "", noms = noms), 100 * inf$parts), collapse = " ; "),
                100 * inf$hors))
    cat("    contribution propre", paste(sprintf("%s %.0f %%", noms, 100 * inf$contribution_propre), collapse = ", "),
        "; portée", paste(sprintf("%s %.0f %%", noms, 100 * inf$portee), collapse = ", "), "\n")
  }
  eq <- equivalence(cop, copP); cat("  lectures équivalentes :", eq[[1]], "\n")
  K <- length(X); cles <- vapply(cop$etiquettes, function(L) length(L) == K, TRUE); clesP <- vapply(copP$etiquettes, function(L) length(L) == K, TRUE)
  if (any(cles) && any(clesP)) {
    E1 <- cop$base[, cles, drop = FALSE]; E2 <- copP$base[, clesP, drop = FALSE]
    cat(sprintf("  parties communes (dimensions %d et %d) : contenance %.2f\n", ncol(E1), ncol(E2), min(svd(crossprod(E1, E2))$d)^2))
  }
  print(discrimination_parties(cop, groupes, n_perm = 999))
  cat("  part de la variance due aux groupes :", paste(sprintf("%s %.2f", noms, variance_groupe(X, groupes)), collapse = ", "), "\n")
  print(stabilite_blocs(cop, X, B = B, strates = groupes, noms = noms))
}

d <- lire_rates(); Xr <- pretraiter(d$blocs)
etude("Rates, régime", Xr, d$regime, c("M", "C"))
b <- lire_beurres(); Xb <- pretraiter(b$blocs, snv = TRUE)
etude("Beurres, type", Xb, b$type, c("VIS", "NIR", "MIR"))
if (VC) {
  cat("\nValidation croisée (chaîne refaite dans chaque pli)\n")
  print(validation_croisee(d$blocs, d$regime, R, noms = c("M", "C")))
  bs <- lapply(b$blocs, function(x) { x <- sweep(x, 1, rowMeans(x)); sweep(x, 1, apply(x, 1, sd), "/") })
  print(validation_croisee(bs, b$type, R, noms = c("VIS", "NIR", "MIR")))
}
