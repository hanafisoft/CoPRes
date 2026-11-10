# Couche CoPRes pour des données structurées par groupes d'individus (mêmes variables mesurées
# sur K groupes). L'espace étiqueté est celui des variables ; A_k est la covariance du groupe k,
# éventuellement restreinte aux R premiers axes de l'ACP de l'ensemble des individus.
# Traduction pas à pas de la version Python (même ordre des tirages aléatoires).

# ------------------------------------------------------------------ données
# Notes d'élèves simulées : 1 500 élèves de trois lycées A, B, C, huit modules.
# Situation 1 : mêmes moyennes ; 2 : spécificité et discrimination ; 3 : discrimination ailleurs.
lire_notes <- function(situation = 1) {
  f <- system.file("extdata", sprintf("notes_situation%d.csv", situation), package = "CoPRes")
  d <- utils::read.csv(f, check.names = FALSE, fileEncoding = "UTF-8", stringsAsFactors = FALSE)
  list(X = as.matrix(d[, -1]), groupes = d[, 1], variables = names(d)[-1])
}

# ------------------------------------------------------------------ prétraitement
pretraiter_groupes <- function(X, g, reduire = c("intra", "aucune"), R = NULL) {
  reduire <- match.arg(reduire)
  X <- as.matrix(X); storage.mode(X) <- "double"; g <- as.character(g)
  groupes <- unique(g); N <- nrow(X); p <- ncol(X); K <- length(groupes)
  if (reduire == "intra") {
    ss <- Reduce(`+`, lapply(groupes, function(k) { x <- X[g == k, , drop = FALSE]; colSums(sweep(x, 2, colMeans(x))^2) }))
    sw <- sqrt(ss / (N - K))
  } else sw <- rep(1, p)
  Xs <- sweep(sweep(X, 2, colMeans(X)), 2, sw, "/")
  if (is.null(R)) U <- diag(p) else {
    U <- eigen(crossprod(Xs), symmetric = TRUE)$vectors[, 1:R, drop = FALSE]
    for (h in seq_len(R)) if (U[which.max(abs(U[, h])), h] < 0) U[, h] <- -U[, h]
  }
  Xk <- lapply(groupes, function(k) { x <- Xs[g == k, , drop = FALSE]; sweep(x, 2, colMeans(x)) })
  A <- lapply(Xk, function(x) t(U) %*% (crossprod(x) / (nrow(x) - 1)) %*% U)
  structure(list(groupes = groupes, g = g, Xs = Xs, Xk = Xk, covariances = A, U = U, ecarts_types = sw),
            class = "prep_groupes")
}

# ------------------------------------------------------------------ seuils
seuils_groupes <- function(prep, nperm = 200, ndir = 100, niveau = 0.95, regle = c("auto", "direction", "valeur_propre"), graine = 1) {
  regle <- match.arg(regle)
  gen <- pm_gen(graine); U <- prep$U; m <- ncol(U)
  vapply(prep$Xk, function(x) {
    n <- nrow(x); p <- ncol(x)
    r <- if (regle == "auto") (if (n - 1 < m) "valeur_propre" else "direction") else regle
    vals <- vector("list", nperm)
    for (q in seq_len(nperm)) {
      u <- matrix(pm_uniformes(gen, n * p), n, p)
      x0 <- matrix(sapply(seq_len(p), function(j) x[order(u[, j]), j]), n, p)
      x0 <- sweep(x0, 2, colMeans(x0))
      C0 <- t(U) %*% (crossprod(x0) / (n - 1)) %*% U
      if (r == "direction") {
        Z <- matrix(pm_normales(gen, m * ndir), m, ndir)
        Z <- sweep(Z, 2, sqrt(colSums(Z^2)), "/")
        vals[[q]] <- colSums(Z * (C0 %*% Z))
      } else vals[[q]] <- max(eigen(C0, symmetric = TRUE, only.values = TRUE)$values)
    }
    unname(quantile(unlist(vals), niveau, type = 7))
  }, 0)
}

# ------------------------------------------------------------------ CoPRes multigroupe
copres_groupes <- function(prep, theta, departs = 20, graine = 1) {
  res <- lode(prep$covariances, theta, departs = departs, graine = graine)
  structure(list(groupes = prep$groupes, base = prep$U %*% res$base, base_analyse = res$base,
                 etiquettes = res$etiquettes, saliences = res$saliences, seuils = theta, motif = res$motif,
                 bilan = res$critere, borne = res$borne, ecart_relatif = res$ecart_relatif, lode = res),
            class = "copres_groupes")
}

print.copres_groupes <- function(x, ...) {
  cat(sprintf("CoPRes multigroupe (%d groupes) : motif %s\n", length(x$groupes), texte_motif(x$motif, x$groupes)))
  cat(sprintf("bilan %.6g, borne de Ky Fan %.6g, écart relatif %.3g %%\n", x$bilan, x$borne, 100 * x$ecart_relatif))
  invisible(x)
}

# scores des individus de chaque groupe sur les co-variations
scores_groupes <- function(cop, prep) lapply(prep$Xk, function(x) x %*% cop$base)

# ------------------------------------------------------------------ CPC
cpc <- function(prep, it = 200, tol = 1e-10) {
  S <- prep$covariances; nk <- vapply(prep$Xk, nrow, 1L); p <- nrow(S[[1]])
  B <- eigen(Reduce(`+`, Map(`*`, nk, S)), symmetric = TRUE)$vectors
  for (r in seq_len(it)) {
    Bold <- B
    for (i in 1:(p - 1)) for (j in (i + 1):p) {
      H <- B[, c(i, j)]
      Tk <- lapply(S, function(s) t(H) %*% s %*% H)
      Q <- diag(2)
      for (gg in 1:50) {
        d1 <- vapply(Tk, function(t2) sum(Q[, 1] * (t2 %*% Q[, 1])), 0)
        d2 <- vapply(Tk, function(t2) sum(Q[, 2] * (t2 %*% Q[, 2])), 0)
        Tm <- Reduce(`+`, lapply(seq_along(Tk), function(k) nk[k] * (d1[k] - d2[k]) / (d1[k] * d2[k]) * Tk[[k]]))
        Qn <- eigen(Tm, symmetric = TRUE)$vectors
        fini <- max(abs(abs(crossprod(Qn, Q)) - diag(2))) < 1e-12
        Q <- Qn
        if (fini) break
      }
      B[, c(i, j)] <- H %*% Q
    }
    if (max(abs(abs(crossprod(Bold, B)) - diag(p))) < tol) break
  }
  for (h in seq_len(p)) if (B[which.max(abs(B[, h])), h] < 0) B[, h] <- -B[, h]
  B
}

lecture_groupes <- function(prep, B, theta) {
  A <- prep$covariances; etiq <- etiquettes(A, B, theta); m <- marge(A, B, theta, etiq)
  structure(list(base = B, saliences = saliences(A, B), etiquettes = etiq, motif = motif(etiq),
                 coherente = m >= -1e-9, marge = m, bilan = critere(A, B, theta), groupes = prep$groupes),
            class = "lecture_groupes")
}

print.lecture_groupes <- function(x, ...) {
  cat(sprintf("Lecture directe : motif %s ; %s (marge %.3g) ; bilan %.6g\n", texte_motif(x$motif, x$groupes),
              if (x$coherente) "cohérente" else "non cohérente", x$marge, x$bilan))
  invisible(x)
}

# ------------------------------------------------------------------ discrimination
.loo <- function(Y, g, groupes, quadratique) {
  N <- nrow(Y); K <- length(groupes)
  idx <- lapply(groupes, function(k) which(g == k)); names(idx) <- groupes
  mu <- lapply(idx, function(i) colMeans(Y[i, , drop = FALSE]))
  SS <- lapply(groupes, function(k) { z <- sweep(Y[idx[[k]], , drop = FALSE], 2, mu[[k]]); crossprod(z) }); names(SS) <- groupes
  ok <- 0
  for (i in seq_len(N)) {
    gi <- g[i]; sc <- list()
    for (k in groupes) {
      m <- mu[[k]]; S <- SS[[k]]; nk <- length(idx[[k]])
      if (k == gi) {
        m2 <- (m * nk - Y[i, ]) / (nk - 1)
        S <- S - tcrossprod(Y[i, ] - m) * nk / (nk - 1)
        m <- m2; nk <- nk - 1
      }
      sc[[k]] <- list(m = m, S = S, nk = nk)
    }
    Wp <- Reduce(`+`, lapply(sc, function(v) v$S)) / (N - 1 - K)
    r <- vapply(groupes, function(k) {
      v <- sc[[k]]; C <- if (quadratique) v$S / (v$nk - 1) else Wp; d <- Y[i, ] - v$m
      -0.5 * sum(d * solve(C, d)) - (if (quadratique) 0.5 * as.numeric(determinant(C, logarithm = TRUE)$modulus) else 0)
    }, 0)
    ok <- ok + (groupes[which.max(r)] == gi)
  }
  ok / N
}

discrimination <- function(cop, prep, classement = TRUE) {
  Y <- prep$Xs %*% prep$U; K <- length(prep$groupes); N <- nrow(Y)
  W <- Reduce(`+`, Map(function(a, x) a * (nrow(x) - 1), prep$covariances, prep$Xk)) / (N - K)
  m <- colMeans(Y)
  D <- sapply(prep$groupes, function(k) colMeans(Y[prep$g == k, , drop = FALSE]) - m)
  Ld <- svd(solve(W, D))$u[, 1:(K - 1), drop = FALSE]
  g <- .groupes(cop$etiquettes); Ls <- attr(g, "etiq")
  parts <- vapply(g, function(idx) sum((crossprod(Ld, cop$base_analyse[, idx, drop = FALSE]))^2), 0)
  names(parts) <- vapply(Ls, nom_etiquette, "", noms = prep$groupes)
  out <- list(parts_plan_discriminant = parts)
  if (classement) {
    out$classement_lineaire <- .loo(Y, prep$g, prep$groupes, FALSE)
    out$classement_quadratique <- .loo(Y, prep$g, prep$groupes, TRUE)
  }
  out
}
