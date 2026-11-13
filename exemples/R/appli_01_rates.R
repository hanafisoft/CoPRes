# Article appliqué, section 2 : rates S1 en lactation, CPCA avec R = 3. Même calcul que
# python/appli_01_rates.py (tableau 1, cohérence, 1 000 jeux tirés au hasard, figure de cohérence,
# bilan, sensibilité, fiche). Usage (depuis exemples/R) : source("appli_01_rates.R")
source("commun.R")
NOMS <- c("M", "C")
d <- lire_rates()
X <- pretraiter(d$blocs)
th <- seuils_perm(X)
R <- 3
U <- cpca(X, R)
A <- lapply(X, function(x) crossprod(U, x) %*% t(x) %*% U)
cat(sprintf("Seuils calculés : M %.1f %%, C %.1f %%\n", 100 * th[1], 100 * th[2]))

lecture <- function(S, s) vapply(seq_len(ncol(S)), function(h) nom_etiquette(which(S[, h] >= s), NOMS), "")
S <- saliences(A, diag(R))
cat("\nTableau 1 : saliences de la CPCA (M ; C)\n  ",
    paste(sprintf("u%d %.3f ; %.3f", 1:R, S[1, ], S[2, ]), collapse = "   "), "\n")
for (s in list(list("seuil fixe 5 %", c(0.05, 0.05)), list("seuil fixe 10 %", c(0.10, 0.10)), list("seuil calculé", th)))
  cat(sprintf("   lecture, %-16s : %s\n", s[[1]], paste(lecture(S, s[[2]]), collapse = ", ")))

lam <- lapply(A, function(a) sort(eigen(a, symmetric = TRUE)$values))
cat(sprintf("\nPartie commune = tout l'espace : salience minimale M %.1f %%, C %.1f %%\n", 100 * lam[[1]][1], 100 * lam[[2]][1]))
P2 <- lapply(A, function(a) min(eigen(a[1:2, 1:2], symmetric = TRUE)$values))
cat(sprintf("Plan (u1, u2) au seuil de 10 %% : salience minimale M %.1f %%, C %.1f %%\n", 100 * P2[[1]], 100 * P2[[2]]))

gen <- pm_gen(11)
haar <- function(n) gram_schmidt(normales_mat(gen, n, n))
motif_txt <- function(Q) texte_motif(motif(etiquettes(A, Q, th)), NOMS)
tous <- integer(0); coh <- list(); nb <- c(MC = 0, "un bloc" = 0, aucun = 0)
for (b in 1:1000) {
  Q <- haar(R); m <- motif_txt(Q)
  tous[m] <- if (m %in% names(tous)) tous[m] + 1 else 1
  for (L in etiquettes(A, Q, th)) {
    k <- if (length(L) == 2) "MC" else if (length(L) == 1) "un bloc" else "aucun"
    nb[k] <- nb[k] + 1
  }
  if (est_etiquetage(A, Q, th)) coh[[m]] <- c(coh[[m]], list(list(phi = critere(A, Q, th), Q = Q)))
}
cat("\nTableau des 1 000 jeux : lecture, fréquence, jeux cohérents, Phi médian, Phi maximal\n")
for (m in names(sort(tous, decreasing = TRUE))) {
  ph <- vapply(coh[[m]], `[[`, 0, "phi")
  cat(sprintf("   %-45s %5.1f %%  %4d  %s\n", m, tous[m] / 10, length(ph),
              if (length(ph)) sprintf("%.3f  %.3f", median(ph), max(ph)) else "--     --"))
}
cat("   total cohérents :", sum(lengths(coh)), "\n")
cat("Composantes tirées (3 000) :", paste(sprintf("%s %.1f %%", names(nb), 100 * nb / 3000), collapse = ", "), "\n")

best <- -Inf
for (m in names(coh)) for (e in coh[[m]]) {
  idx <- which(vapply(etiquettes(A, e$Q, th), length, 1L) == 2)
  if (length(idx) != 2) next
  E <- e$Q[, idx, drop = FALSE]
  lmin <- vapply(A, function(a) min(eigen(t(E) %*% a %*% E, symmetric = TRUE)$values), 0)
  if (min(lmin - th) > best) { best <- min(lmin - th); lb <- lmin; mb <- m }
}
cat(sprintf("\nFigure, panneau b (%s) : salience minimale M %.1f %%, C %.1f %%\n", mb, 100 * lb[1], 100 * lb[2]))

cop <- copres(X, th, R = R, departs = 200)
print(cop)
detail <- function(Q, nom) {
  etq <- etiquettes(A, Q, th); Sq <- saliences(A, Q)
  cat(sprintf("Bilan %s : Phi = %.3f ; motif %s ; marge %.4f\n", nom, critere(A, Q, th), motif_txt(Q), marge(A, Q, th)))
  cles <- unique(vapply(etq, function(L) paste(L, collapse = ","), ""))
  tot <- c(M = 0, C = 0)
  for (cl in cles) {
    idx <- which(vapply(etq, function(L) paste(L, collapse = ","), "") == cl)
    L <- etq[[idx[1]]]
    for (k in L) {
      v <- sum(Sq[k, idx]) - th[k] * length(idx); tot[NOMS[k]] <- tot[NOMS[k]] + v
      cat(sprintf("     partie %-8s bloc %s : %.3f\n", nom_etiquette(L, NOMS), NOMS[k], v))
    }
  }
  cat("     par bloc :", paste(sprintf("%s %.3f", names(tot), tot), collapse = ", "), "\n")
}
plus <- names(coh)[which.max(lengths(coh))]
Q0 <- coh[[plus]][[which.max(vapply(coh[[plus]], `[[`, 0, "phi"))]]$Q
Qo <- CoPRes:::.depuis(A, th, Q0, 1e-12, 30)[[1]]
detail(cop$lode$base, "CoPRes")
detail(Qo, paste(plus, "après rotation"))
cat(sprintf("CPCA (incohérente) : Phi = %.3f ; borne de Ky Fan %.3f\n", critere(A, diag(R), th), borne_kf(A, th)))

print(sensibilite(X, th, R_liste = 3, facteurs = c(0.75, 0.9, 1, 1.25, 1.5, 2), departs = 50))
print(fiche(NULL, cop))
