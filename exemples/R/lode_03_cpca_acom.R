# LODE, section 6.1, tableau 5 : CPCA et ACOM, mélange, dilution, étiquettes différentes, motif,
# écart à la borne, proximité des sous-espaces. Données simulées 1, 3, 4 et beurres.
source("commun.R")
b <- lire_beurres()
jeux <- list(list("Simulées 1", pretraiter(lire_simulees(1)), 5, NULL), list("Simulées 3", pretraiter(lire_simulees(3)), 7, NULL),
             list("Simulées 4", pretraiter(lire_simulees(4)), 6, NULL), list("Vins", pretraiter(lire_vins()$blocs), 3, NULL),
             list("Beurres", pretraiter(b$blocs, snv = TRUE), 4, c("V", "N", "M")))
for (j in jeux) {
  X <- j[[2]]; R <- j[[3]]; th <- seuils_perm(X)
  cops <- lapply(c(CPCA = "CPCA", ACOM = "ACOM"), function(m) copres(X, th, R = R, methode = m, departs = 100))
  prox <- proximite(cops$CPCA$base_methode, cops$ACOM$base_methode)
  for (m in names(cops)) {
    c <- cops[[m]]; f <- fiche(NULL, c); lect <- lecture_directe(X, c$base_methode, th)
    proche <- lapply(seq_len(R), function(h) c$etiquettes[[which.max(crossprod(c$base, c$base_methode[, h])^2)]])
    ndiff <- sum(!mapply(identical, lect$etiquettes, proche))
    cat(sprintf("%-12s %-5s mél %d dil %d étiq≠ %d  %s  écart %.2f %%  cos² min %.3f\n", j[[1]], m, sum(f$melangee), sum(f$diluee),
                ndiff, texte_motif(c$motif, j[[4]]), 100 * c$ecart_relatif, prox))
  }
  cat("   seuils :", round(th, 4), "\n")
}
