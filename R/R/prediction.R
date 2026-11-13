# Groupes connus et lecture CoPRes : par quelles parties, donc par quels blocs, passe ce qui
# distingue les groupes. Même calcul que prediction.py (version Python).
#  - information_groupe : part de l'information de groupe (indicatrice centrée des groupes) portée
#    par chaque partie et hors du sous-espace ; contribution propre et portée de chaque bloc ;
#  - plsda : base du sous-espace des scores d'une PLS-DA (PLS2) ;
#  - lda_loo : analyse discriminante linéaire en validation croisée un individu à la fois
#    (a priori = proportions de l'apprentissage, comme scikit-learn) ;
#  - discrimination_parties : analyse discriminante par partie, test par permutation des groupes ;
#  - variance_groupe : part de la variance de chaque bloc due aux groupes ;
#  - validation_croisee : validation croisée un individu à la fois, chaîne refaite dans chaque pli.

.niveaux <- function(g) sort(unique(as.character(g)), method = "radix")

indicatrice <- function(g) {
  g <- as.character(g)
  G <- vapply(.niveaux(g), function(v) as.numeric(g == v), numeric(length(g)))
  G <- matrix(G, nrow = length(g))
  sweep(G, 2, colMeans(G))
}

.parties_cop <- function(cop) {
  gr <- .groupes(cop$etiquettes)
  list(idx = unname(gr), etiq = unname(attr(gr, "etiq")))
}

# ----------------------------------------------------------------------------- information
information_groupe <- function(cop, groupes) {
  G <- indicatrice(groupes)
  tot <- sum(G^2)
  V <- cop$base; U <- cop$base_methode
  P <- .parties_cop(cop)
  parts <- vapply(P$idx, function(idx) sum((t(V[, idx, drop = FALSE]) %*% G)^2) / tot, 0)
  dims <- vapply(P$idx, length, 1L)
  hors <- sum((G - U %*% (t(U) %*% G))^2) / tot
  K <- length(cop$seuils)
  propre <- vapply(seq_len(K), function(k)
    sum(parts[vapply(P$etiq, function(L) length(L) == 1 && L[1] == k, TRUE)]), 0)
  portee <- vapply(seq_len(K), function(k) sum(parts[vapply(P$etiq, function(L) k %in% L, TRUE)]), 0)
  structure(list(parts = parts, etiquettes = P$etiq, noms = vapply(P$etiq, nom_etiquette, ""),
                 dimensions = dims, hors = hors, contribution_propre = propre, portee = portee),
            class = "information_groupe")
}

print.information_groupe <- function(x, ...) {
  cat("Information de groupe\n")
  for (i in seq_along(x$parts))
    cat(sprintf("  %-12s dim %d : %5.1f %%\n", x$noms[i], x$dimensions[i], 100 * x$parts[i]))
  cat(sprintf("  %-12s       : %5.1f %%\n", "hors de S", 100 * x$hors))
  cat("  contribution propre :", paste(sprintf("%.1f %%", 100 * x$contribution_propre), collapse = ", "), "\n")
  cat("  portée              :", paste(sprintf("%.1f %%", 100 * x$portee), collapse = ", "), "\n")
  invisible(x)
}

# ----------------------------------------------------------------------------- PLS-DA
plsda <- function(X, groupes, R) {
  Xh <- do.call(cbind, lapply(X, as.matrix))
  Xh <- sweep(Xh, 2, colMeans(Xh))
  Y <- indicatrice(groupes)
  T <- matrix(0, nrow(Xh), R)
  for (h in seq_len(R)) {
    w <- svd(t(Xh) %*% Y, nu = 1, nv = 0)$u[, 1]
    tt <- as.numeric(Xh %*% w)
    T[, h] <- tt
    Xh <- Xh - outer(tt, as.numeric(tt %*% Xh)) / sum(tt^2)
  }
  Q <- T
  for (j in seq_len(R)) {
    for (passe in 1:2) for (i in seq_len(j - 1)) Q[, j] <- Q[, j] - sum(Q[, i] * Q[, j]) * Q[, i]
    Q[, j] <- Q[, j] / sqrt(sum(Q[, j]^2))
  }
  Q
}

# ----------------------------------------------------------------------------- discrimination
.pinv <- function(M) {
  s <- svd(M)
  tol <- max(dim(M)) * max(s$d) * .Machine$double.eps
  d <- ifelse(s$d > tol, 1 / s$d, 0)
  s$v %*% (d * t(s$u))
}

.lda_predire <- function(Ztr, gtr, Zte, niveaux) {
  Ztr <- as.matrix(Ztr); Zte <- matrix(Zte, ncol = ncol(Ztr))
  presents <- niveaux[niveaux %in% gtr]
  n <- nrow(Ztr); q <- ncol(Ztr)
  mu <- lapply(presents, function(v) colMeans(Ztr[gtr == v, , drop = FALSE]))
  W <- matrix(0, q, q)
  for (i in seq_along(presents)) {
    D <- sweep(Ztr[gtr == presents[i], , drop = FALSE], 2, mu[[i]])
    W <- W + t(D) %*% D
  }
  W <- W / (n - length(presents))
  Wi <- .pinv(W)
  sc <- vapply(seq_along(presents), function(i)
    as.numeric(Zte %*% Wi %*% mu[[i]]) - 0.5 * as.numeric(t(mu[[i]]) %*% Wi %*% mu[[i]]) +
      log(mean(gtr == presents[i])), numeric(nrow(Zte)))
  sc <- matrix(sc, nrow = nrow(Zte))
  presents[max.col(sc, ties.method = "first")]
}

lda_loo <- function(Z, groupes) {
  Z <- as.matrix(Z); g <- as.character(groupes); niv <- .niveaux(g)
  n <- length(g); ok <- 0
  for (i in seq_len(n)) ok <- ok + (.lda_predire(Z[-i, , drop = FALSE], g[-i], Z[i, , drop = FALSE], niv) == g[i])
  ok / n
}

discrimination_parties <- function(cop, groupes, n_perm = 999, graine = 1) {
  g <- as.character(groupes)
  P <- .parties_cop(cop)
  cibles <- c(lapply(P$idx, function(idx) cop$base[, idx, drop = FALSE]), list(cop$base_methode))
  noms <- c(vapply(P$etiq, nom_etiquette, ""), "S")
  gen <- pm_gen(graine)
  perms <- lapply(seq_len(n_perm), function(b) pm_permutation(gen, length(g)))
  taux <- p <- numeric(length(cibles))
  for (j in seq_along(cibles)) {
    t0 <- lda_loo(cibles[[j]], g)
    nul <- sum(vapply(perms, function(pi) lda_loo(cibles[[j]], g[pi]) >= t0 - 1e-12, TRUE))
    taux[j] <- t0; p[j] <- (1 + nul) / (n_perm + 1)
  }
  structure(list(taux = setNames(taux, noms), p = setNames(p, noms), n_perm = n_perm),
            class = "discrimination_parties")
}

print.discrimination_parties <- function(x, ...) {
  cat("Analyse discriminante par partie (LOO)\n")
  for (i in seq_along(x$taux))
    cat(sprintf("  %-12s : %5.1f %%  (p = %.3f)\n", ifelse(names(x$taux)[i] == "S", "sous-espace", names(x$taux)[i]),
                100 * x$taux[i], x$p[i]))
  invisible(x)
}

variance_groupe <- function(X, groupes) {
  g <- as.character(groupes)
  vapply(X, function(x) {
    x <- as.matrix(x); m <- colMeans(x)
    inter <- sum(vapply(.niveaux(g), function(v)
      sum(g == v) * sum((colMeans(x[g == v, , drop = FALSE]) - m)^2), 0))
    inter / sum(sweep(x, 2, m)^2)
  }, 0)
}

# ----------------------------------------------------------------------------- validation croisée
validation_croisee <- function(X, groupes, R, methode = "CPCA", nperm = 50, ndir = 100, departs = 20,
                               graine = 1, blocs_seuls = TRUE, noms = NULL) {
  X <- lapply(X, as.matrix)
  g <- as.character(groupes); niv <- .niveaux(g)
  n <- length(g); K <- length(X)
  if (is.null(noms)) noms <- as.character(seq_len(K))
  bons <- c("toutes les composantes" = 0, "partie commune" = 0, "parties choisies" = 0)
  if (blocs_seuls) for (k in seq_len(K)) bons[paste0("bloc ", noms[k], " seul")] <- 0
  choix <- integer(0)
  for (i in seq_len(n)) {
    tr <- setdiff(seq_len(n), i)
    app <- test <- vector("list", K)
    for (k in seq_len(K)) {
      m <- colMeans(X[[k]][tr, , drop = FALSE])
      s <- sqrt(sum(sweep(X[[k]][tr, , drop = FALSE], 2, m)^2))
      app[[k]] <- sweep(X[[k]][tr, , drop = FALSE], 2, m) / s
      test[[k]] <- matrix((X[[k]][i, ] - m) / s, nrow = 1)
    }
    Xa <- do.call(cbind, app); Xt <- do.call(cbind, test)
    Xp <- .pinv(Xa)
    th <- seuils_perm(app, nperm = nperm, ndir = ndir, graine = 1000 * graine + i)
    cop <- copres(app, th, R = R, methode = methode, departs = departs)
    P <- .parties_cop(cop)
    Pb <- lapply(P$idx, function(idx) cop$base[, idx, drop = FALSE])
    scores <- function(V) { w <- Xp %*% V; list(a = Xa %*% w, b = Xt %*% w) }
    predit <- function(V) { s <- scores(V); .lda_predire(s$a, g[tr], s$b, niv) == g[i] }
    bons["toutes les composantes"] <- bons["toutes les composantes"] + predit(cop$base_methode)
    com <- which(vapply(P$etiq, function(L) length(L) == K, TRUE))
    if (length(com)) bons["partie commune"] <- bons["partie commune"] + predit(Pb[[com[1]]])
    meilleur <- -1; ens <- NULL
    for (r in seq_along(Pb)) {
      for (sub in utils::combn(seq_along(Pb), r, simplify = FALSE)) {
        a <- scores(do.call(cbind, Pb[sub]))$a
        t0 <- lda_loo(a, g[tr])
        if (t0 > meilleur + 1e-12) { meilleur <- t0; ens <- sub }
      }
    }
    Le <- P$etiq[ens]
    o <- order(-vapply(Le, length, 1L), vapply(Le, function(L) paste(sprintf("%05d", L), collapse = ","), ""))
    nom <- paste(vapply(Le[o], nom_etiquette, "", noms = noms), collapse = " + ")
    choix[nom] <- if (nom %in% names(choix)) choix[nom] + 1 else 1
    bons["parties choisies"] <- bons["parties choisies"] + predit(do.call(cbind, Pb[ens]))
    if (blocs_seuls) for (k in seq_len(K)) {
      w <- svd(app[[k]], nu = 0, nv = R)$v
      cle <- paste0("bloc ", noms[k], " seul")
      bons[cle] <- bons[cle] + (.lda_predire(app[[k]] %*% w, g[tr], test[[k]] %*% w, niv) == g[i])
    }
  }
  choix <- choix[order(-choix)]
  structure(list(taux = bons / n, choix = choix, plis = n), class = "validation_croisee")
}

print.validation_croisee <- function(x, ...) {
  cat(sprintf("Validation croisée un individu à la fois (%d plis)\n", x$plis))
  for (k in names(x$taux)) cat(sprintf("  %-28s : %5.1f %%\n", k, 100 * x$taux[k]))
  cat("  parties choisies :", paste(sprintf("%s (%d)", names(x$choix), x$choix), collapse = "; "), "\n")
  invisible(x)
}
