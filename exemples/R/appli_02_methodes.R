# Article appliqué, section 3 : comparer CPCA, ACOM et ACCPS par leurs lectures CoPRes. Même calcul
# que python/appli_02_methodes.py. Choisir les parties à lancer :
#   PARTIES <- c("reels", "contre", "deux_blocs", "plans") ; N <- 50 ; source("appli_02_methodes.R")
# (deux_blocs et plans : plusieurs heures en R avec N = 50 ; N <- 2 pour un essai)
source("commun.R")
source("simulations.R")
if (!exists("PARTIES")) PARTIES <- c("reels", "contre")
if (!exists("N")) N <- 50
METH <- c("CPCA", "ACOM", "ACCPS")

parties_de <- function(Q, etiq) {
  cles <- vapply(etiq, function(L) paste(L, collapse = ","), "")
  out <- lapply(unique(cles), function(k) Q[, cles == k, drop = FALSE]); names(out) <- unique(cles); out
}
equivalentes <- function(p1, p2, seuil = 0.72) {
  d1 <- sort(paste(names(p1), vapply(p1, ncol, 1L))); d2 <- sort(paste(names(p2), vapply(p2, ncol, 1L)))
  if (!identical(d1, d2)) return(list(eq = FALSE, prox = NA))
  pr <- min(vapply(names(p1), function(k) proximite(p1[[k]], p2[[k]]), 0))
  list(eq = pr >= seuil, prox = pr)
}
nom_lecture <- function(etiq, noms) paste(vapply(etiq, function(L) if (length(L)) nom_etiquette(L, noms) else "--", ""), collapse = ", ")
methode_dans <- function(U, X, nom, R) {
  Z <- lapply(X, function(x) crossprod(U, x))
  if (nom == "CPCA") diag(R) else METHODES[[nom]](Z, R)
}

tableau_reel <- function(X, th, R, noms, departs = 200) {
  Uc <- cpca(X, R); Ac <- lapply(X, function(x) crossprod(Uc, x) %*% t(x) %*% Uc)
  copC <- copres(X, th, U = Uc, departs = departs)
  refC <- parties_de(copC$lode$base, copC$etiquettes)
  cat("  R =", R, "\n")
  for (nom in METH) {
    U <- METHODES[[nom]](X, R)
    ld <- lecture_directe(X, U, th)
    Q <- methode_dans(Uc, X, nom, R)
    a <- equivalentes(parties_de(Q, etiquettes(Ac, Q, th)), refC)
    cop <- copres(X, th, U = U, methode = nom, departs = departs)
    eq <- if (nom == "CPCA") NULL else equivalence(copC, cop)
    cat(sprintf("    %-6s lecture directe %-28s cohérente %-5s (a) %-5s (%s) ; CoPRes %-40s proximité de l'espace %s ; équivalente à la CPCA %s\n",
                nom, nom_lecture(ld$etiquettes, noms), ld$coherente, a$eq, ifelse(is.na(a$prox), "--", sprintf("%.2f", a$prox)),
                texte_motif(cop$motif, noms), if (nom == "CPCA") "--" else sprintf("%.2f", proximite(Uc, U)),
                if (is.null(eq)) "--" else paste(format(eq[[1]]), if (length(eq[[2]])) sprintf("(%.2f)", min(unlist(eq[[2]]))) else "")))
  }
}

if ("reels" %in% PARTIES) {
  d <- lire_rates(); X <- pretraiter(d$blocs); th <- seuils_perm(X)
  cat("Rates (M, C)\n"); for (R in 3:4) tableau_reel(X, th, R, c("M", "C"))
  b <- lire_beurres(); Xb <- pretraiter(b$blocs, snv = TRUE); thb <- seuils_perm(Xb)
  cat("Beurres (V, N, I)\n"); tableau_reel(Xb, thb, 4, c("V", "N", "I"))
}

if ("contre" %in% PARTIES) {
  a <- 0.5; bb <- 0.5; cc <- 0.01; phi <- 30 * pi / 180
  v <- c(cos(phi), sin(phi)); w <- c(-sin(phi), cos(phi))
  A <- list(a * diag(c(1, 0)), bb * outer(v, v) + cc * outer(w, w)); th <- c(0.05, 0.05)
  t <- seq(0, pi, length.out = 400001)
  f <- (a * cos(t)^2)^2 + (bb * cos(t - phi)^2 + cc * sin(t - phi)^2)^2; t0 <- t[which.max(f)]
  Qa <- matrix(c(cos(t0), sin(t0), -sin(t0), cos(t0)), 2)
  res <- lode(A, th, departs = 200)
  for (q in list(list("ACCPS", Qa), list("CoPRes", res$base)))
    cat(sprintf("  %-6s : saliences %s ; bilan %.3f\n", q[[1]], paste(round(saliences(A, q[[2]]), 3), collapse = " "), critere(A, q[[2]], th)))
  cat(sprintf("  borne de Ky Fan %.3f\n", borne_kf(A, th)))
}

jeu_deux_blocs <- function(graine, phi_deg, a2, n = 60, J = 30, sigma = 0.5, a11 = 3, a12 = 1.5, a3 = 1.6) {
  gen <- pm_gen(graine)
  T <- normales_mat(gen, n, 3)
  T <- gram_schmidt(sweep(T, 2, colMeans(T)))
  T <- gram_schmidt(sweep(T, 2, colMeans(T))) * sqrt(n)
  p <- phi_deg * pi / 180
  u <- cos(p) * T[, 1] + sin(p) * T[, 2]; w <- -sin(p) * T[, 1] + cos(p) * T[, 2]
  X1 <- sigma * normales_mat(gen, n, J)
  X1 <- X1 + a11 * outer(u, normales_mat(gen, J, 1)[, 1]); X1 <- X1 + a12 * outer(w, normales_mat(gen, J, 1)[, 1])
  X2 <- sigma * normales_mat(gen, n, J)
  X2 <- X2 + a2 * outer(T[, 2], normales_mat(gen, J, 1)[, 1]); X2 <- X2 + a3 * outer(T[, 3], normales_mat(gen, J, 1)[, 1])
  fin <- function(x) { x <- sweep(x, 2, colMeans(x)); x / sqrt(sum(x^2)) }
  list(X = list(fin(X1), fin(X2)),
       verite = list(list(t = T[, 1] / sqrt(n), L = "1"), list(t = T[, 2] / sqrt(n), L = "1,2"), list(t = T[, 3] / sqrt(n), L = "2")))
}
retrouvees <- function(p, verite) mean(vapply(verite, function(v) !is.null(p[[v$L]]) && sum((t(p[[v$L]]) %*% v$t)^2) >= 0.9, TRUE))

if ("deux_blocs" %in% PARTIES) {
  for (a2 in c(0.5, 0.35)) {
    cat("Simulation à deux blocs, intensité de t2 :", a2, "\n")
    for (deg in c(0, 15, 30, 45)) {
      r <- array(0, c(3, 3, N), dimnames = list(c("eq", "rc", "rq"), METH, NULL))
      for (g in seq_len(N)) {
        jeu <- jeu_deux_blocs(g, deg, a2); th <- seuils_perm(jeu$X)
        for (nom in METH) {
          U <- METHODES[[nom]](jeu$X, 3); ld <- lecture_directe(jeu$X, U, th)
          cop <- copres(jeu$X, th, U = U, methode = nom, departs = 50)
          pc <- parties_de(U, ld$etiquettes); pq <- parties_de(cop$base, cop$etiquettes)
          r[, nom, g] <- c(equivalentes(pc, pq)$eq, retrouvees(pc, jeu$verite), retrouvees(pq, jeu$verite))
        }
      }
      m <- apply(r, 1:2, mean)
      cat(sprintf("  %3d°  %s   %s   %s\n", deg, paste(sprintf("%4.0f %%", 100 * m["eq", ]), collapse = " "),
                  paste(sprintf("%.2f", m["rc", ]), collapse = " "), paste(sprintf("%.2f", m["rq", ]), collapse = " ")))
    }
  }
}

if ("plans" %in% PARTIES) {
  RP <- c("1" = 5, "3" = 7, "4" = 6)
  for (p in c(1, 3, 4)) for (R in RP[as.character(p)] + (-1:1)) {
    eq <- matrix(FALSE, N, 2, dimnames = list(NULL, c("ACOM", "ACCPS")))
    for (g in seq_len(N)) {
      X <- jeu_plan(p, g)$X; th <- seuils_perm(X)
      copC <- copres(X, th, R = R, methode = "CPCA", departs = 100)
      for (nom in colnames(eq)) eq[g, nom] <- equivalence(copC, copres(X, th, R = R, methode = nom, departs = 100))[[1]]
    }
    cat(sprintf("  plan %d, R = %d : ACOM %.0f %%, ACCPS %.0f %%\n", p, R, 100 * mean(eq[, 1]), 100 * mean(eq[, 2])))
  }
}
