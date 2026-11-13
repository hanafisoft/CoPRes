# Méthodes concurrentes, pour comparer CoPRes à JIVE et à AJIVE sur les mêmes données.
#
# Les blocs ont les individus en lignes (n x J_k), comme dans le reste du paquet.
# - jive suit le paquet R r.jive 2.4 (Lock et al. 2013 ; O'Connell et Lock 2016) : centrage des
#   variables, mise à l'échelle des blocs, algorithme alterné avec individuels orthogonaux entre
#   eux (orthIndiv), rangs donnés ou estimés par permutation (jive.perm).
# - ajive suit l'implémentation de mvlearn 0.5 (Carmichael ; Feng et al. 2018) : extraction du
#   signal par bloc, rang joint par la borne de Wedin et la borne des directions aléatoires,
#   réexamen des composantes jointes, rangs individuels par le seuil des valeurs singulières.
# Les tirages aléatoires utilisent le générateur de Park et Miller du paquet : les versions R et
# Python donnent les mêmes résultats, mais pas les mêmes tirages que r.jive ou mvlearn.

.svd_c <- function(X) {
  s <- svd(X)
  list(u = s$u, d = s$d, v = s$v)
}

.quantile7 <- function(x, p) unname(stats::quantile(x, p, type = 7))

.base_orthonormee <- function(Z) svd(Z)$u[, seq_len(ncol(Z)), drop = FALSE]

.proj_hors <- function(V, n) diag(n) - V %*% t(V)

# ============================================================================= JIVE
.pretraiter_jive <- function(blocs, centrer, echelle) {
  X <- lapply(blocs, function(x) {
    x <- as.matrix(x)
    storage.mode(x) <- "double"
    if (centrer) x <- sweep(x, 2, colMeans(x))
    x
  })
  if (echelle) {
    total <- sum(vapply(X, length, 1))
    X <- lapply(X, function(x) x / (sqrt(sum(x^2)) * sqrt(total)))
  }
  X
}

.jive_iter <- function(X, rJ, rA, conv, maxit, orth_indiv) {
  K <- length(X)
  n <- nrow(X[[1]])
  Xtot <- do.call(cbind, X)
  p <- vapply(X, ncol, 1)
  fin <- cumsum(p); debut <- fin - p + 1
  Jtot <- matrix(-1, n, ncol(Xtot))
  Atot <- matrix(-1, n, ncol(Xtot))
  Vind <- lapply(rA, function(r) matrix(0, n, r))
  V <- matrix(0, n, rJ)
  nrun <- 0
  converge <- FALSE
  A <- NULL
  while (nrun < maxit && !converge) {
    Jlast <- Jtot; Alast <- Atot
    if (rJ > 0) {
      s <- .svd_c(Xtot - Atot)
      V <- s$u[, 1:rJ, drop = FALSE]
      Jtot <- s$u[, 1:rJ, drop = FALSE] %*% diag(s$d[1:rJ], rJ) %*% t(s$v[, 1:rJ, drop = FALSE])
    } else {
      Jtot <- matrix(0, n, ncol(Xtot))
      V <- matrix(0, n, 0)
    }
    J <- lapply(seq_len(K), function(i) Jtot[, debut[i]:fin[i], drop = FALSE])
    A <- vector("list", K)
    for (i in seq_len(K)) {
      if (rA[i] > 0) {
        T <- .proj_hors(V, n) %*% (X[[i]] - J[[i]])
        if (orth_indiv && nrun > 0) {
          for (j in seq_len(K)) if (j != i) T <- .proj_hors(Vind[[j]], n) %*% T
        }
        s <- .svd_c(T)
        if (orth_indiv) Vind[[i]] <- s$u[, 1:rA[i], drop = FALSE]
        A[[i]] <- s$u[, 1:rA[i], drop = FALSE] %*% diag(s$d[1:rA[i]], rA[i]) %*%
          t(s$v[, 1:rA[i], drop = FALSE])
      } else {
        A[[i]] <- matrix(0, n, ncol(X[[i]]))
        if (orth_indiv) Vind[[i]] <- matrix(0, n, 0)
      }
    }
    if (orth_indiv && nrun == 0) {
      for (i in seq_len(K)) for (j in seq_len(K)) if (j != i) A[[i]] <- .proj_hors(Vind[[j]], n) %*% A[[i]]
      for (i in seq_len(K)) if (rA[i] > 0) Vind[[i]] <- svd(A[[i]])$u[, 1:rA[i], drop = FALSE]
    }
    Atot <- do.call(cbind, A)
    if (sqrt(sum((Jtot - Jlast)^2)) <= conv && sqrt(sum((Atot - Alast)^2)) <= conv) converge <- TRUE
    nrun <- nrun + 1
  }
  scores_ind <- lapply(seq_len(K), function(i)
    if (rA[i] > 0) svd(A[[i]])$u[, 1:rA[i], drop = FALSE] else matrix(0, n, 0))
  list(J = J, A = A, V = V[, seq_len(rJ), drop = FALSE], Si = scores_ind, converge = converge,
       iterations = nrun)
}

.compter_rang <- function(actuel, perms, alpha) {
  r <- 0
  for (i in seq_along(actuel)) {
    if (actuel[i] > .quantile7(perms[, i], 1 - alpha)) r <- r + 1 else break
  }
  r
}

.rangs_perm <- function(X, Jperp, Aperp, last, nperm, alpha, gen) {
  n <- nrow(X[[1]])
  K <- length(X)
  full <- lapply(seq_len(K), function(i) X[[i]] - Aperp[[i]])
  actuel <- svd(do.call(cbind, full), nu = 0, nv = 0)$d
  perms <- matrix(NA_real_, nperm, length(actuel))
  for (b in seq_len(nperm)) {
    perms[b, ] <- svd(do.call(cbind, lapply(full, function(f) f[pm_permutation(gen, n), , drop = FALSE])),
                      nu = 0, nv = 0)$d
  }
  rJ <- max(.compter_rang(actuel, perms, alpha), last[1])
  rA <- integer(K)
  for (k in seq_len(K)) {
    ind <- X[[k]] - Jperp[[k]]
    actuel <- svd(ind, nu = 0, nv = 0)$d
    perms <- matrix(NA_real_, nperm, length(actuel))
    for (b in seq_len(nperm)) {
      P <- ind
      for (j in seq_len(ncol(ind))) P[, j] <- ind[pm_permutation(gen, n), j]
      perms[b, ] <- svd(P, nu = 0, nv = 0)$d
    }
    rA[k] <- .compter_rang(actuel, perms, alpha)
  }
  c(rJ, rA)
}

# JIVE selon r.jive 2.4.
# methode = "given" (défaut si rang_joint est donné) : rangs fixés ;
# methode = "perm" (défaut sinon) : rangs estimés par permutation (jive.perm).
# Les sorties sont exprimées sur les données centrées et mises à l'échelle (comme r.jive).
jive <- function(blocs, rang_joint = NULL, rangs_individuels = NULL, methode = NULL, nperm = 100,
                 alpha = 0.05, centrer = TRUE, echelle = TRUE, orth_indiv = TRUE, conv = NULL,
                 maxit = 1000, graine = 1) {
  K <- length(blocs)
  if (is.null(methode)) methode <- if (!is.null(rang_joint)) "given" else "perm"
  X <- .pretraiter_jive(blocs, centrer, echelle)
  if (is.null(conv)) conv <- 1e-6 * sqrt(sum(do.call(cbind, X)^2))
  emballer <- function(res, rJ, rA, nrun_ok, meth) {
    structure(list(joint = res$J, individuel = res$A, scores_joint = res$V,
                   scores_individuels = res$Si, rang_joint = rJ, rangs_individuels = rA,
                   donnees = X, converge = nrun_ok, iterations = res$iterations, methode = meth),
              class = "jive_copres")
  }
  if (methode == "given") {
    rA <- if (is.null(rangs_individuels)) rep(1L, K) else as.integer(rangs_individuels)
    res <- .jive_iter(X, as.integer(rang_joint), rA, conv, maxit, orth_indiv)
    return(emballer(res, as.integer(rang_joint), rA, res$converge, "given"))
  }
  if (methode != "perm") stop("methode doit valoir 'given' ou 'perm'")
  gen <- pm_gen(graine)
  Jperp <- lapply(X, function(x) x * 0)
  Aperp <- Jperp
  last <- rep(-2, K + 1); current <- rep(-1, K + 1)
  nrun <- 0
  res <- NULL
  while (!identical(as.numeric(last), as.numeric(current)) && nrun < 10) {
    last <- current
    current <- .rangs_perm(X, Jperp, Aperp, last, nperm, alpha, gen)
    if (!identical(as.numeric(last), as.numeric(current))) {
      res <- .jive_iter(X, current[1], current[-1], conv, maxit, orth_indiv)
      Jperp <- res$J; Aperp <- res$A
    }
    nrun <- nrun + 1
  }
  emballer(res, as.integer(current[1]), as.integer(current[-1]), nrun < 10, "perm")
}

print.jive_copres <- function(x, ...) {
  cat("JIVE (", x$methode, ") : rang joint ", x$rang_joint, ", rangs individuels ",
      paste(x$rangs_individuels, collapse = ", "), "\n", sep = "")
  invisible(x)
}

# ============================================================================= AJIVE
.log_vraisemblance <- function(arr) {
  n <- length(arr)
  out <- numeric(n)
  for (idx in seq_len(n)) {
    s1 <- arr[seq_len(idx)]
    s2 <- if (idx < n) arr[(idx + 1):n] else numeric(0)
    if (length(s1) == 1 && length(s2) == 1) { out[idx] <- -Inf; next }
    m1 <- mean(s1)
    m2 <- if (length(s2)) mean(s2) else -Inf
    v <- (sum((s1 - m1)^2) + (if (length(s2)) sum((s2 - m2)^2) else 0)) / (n - 1 - (idx < n))
    sd <- sqrt(v)
    ll <- sum(-0.5 * ((s1 - m1) / sd)^2 - log(sd) - 0.5 * log(2 * pi))
    if (length(s2)) ll <- ll + sum(-0.5 * ((s2 - m2) / sd)^2 - log(sd) - 0.5 * log(2 * pi))
    out[idx] <- ll
  }
  out
}

# Comme numpy.argmax : une valeur NaN compte comme le maximum (première rencontrée) ; sinon,
# premier indice du maximum.
.argmax_numpy <- function(x) {
  nan <- which(is.nan(x))
  if (length(nan)) return(nan[1])
  which.max(x)
}

# Rang de signal par le coude du profil de vraisemblance (Zhu et Ghodsi 2006), comme mvlearn.
rang_coude <- function(X, n_coudes = 2) {
  X <- as.matrix(X)
  k <- ceiling(log2(min(dim(X))))
  D <- svd(X, nu = 0, nv = 0)$d[seq_len(k)]
  idx <- 0
  for (i in seq_len(n_coudes)) idx <- idx + .argmax_numpy(.log_vraisemblance(D[(idx + 1):length(D)]))
  as.integer(idx)
}

.echantillon_hasard <- function(n, rangs, gen) {
  M <- do.call(cbind, lapply(rangs, function(r) .base_orthonormee(matrix(pm_normales(gen, n * r), n, r))))
  svd(M, nu = 0, nv = 0)$d[1]^2
}

.norme_bruit <- function(X, base, gen) {
  p <- nrow(base); r <- ncol(base)
  Z <- matrix(pm_normales(gen, p * r), p, r)
  Z <- Z - base %*% (t(base) %*% Z)
  Z <- .base_orthonormee(Z)
  norm(X %*% Z, "2")
}

# AJIVE selon mvlearn 0.5.
# rangs_signal : vecteur des rangs de signal initiaux, ou "coude" (estimés par rang_coude sur les
# blocs non centrés, comme mvlearn). rang_joint : donné, ou NULL pour l'estimer (bornes de Wedin
# et des directions aléatoires). rangs_individuels : donnés, ou NULL (seuil des valeurs singulières).
# centrer = TRUE (défaut, comme mvlearn) centre les colonnes ; centrer = FALSE prend les blocs tels quels.
ajive <- function(blocs, rangs_signal = "coude", rang_joint = NULL, rangs_individuels = NULL,
                  reconsiderer = TRUE, n_wedin = 1000, n_hasard = 1000, centile_wedin = 5,
                  centile_hasard = 95, n_coudes = 2, graine = 1, centrer = TRUE) {
  X0 <- lapply(blocs, function(x) { x <- as.matrix(x); storage.mode(x) <- "double"; x })
  K <- length(X0)
  n <- nrow(X0[[1]])
  if (is.character(rangs_signal)) {
    if (rangs_signal != "coude") stop("rangs_signal doit être un vecteur ou 'coude'")
    rs <- vapply(X0, rang_coude, 1L, n_coudes = n_coudes)
  } else rs <- as.integer(rangs_signal)
  X <- if (centrer) lapply(X0, function(x) sweep(x, 2, colMeans(x))) else X0
  seuils <- numeric(K); U_s <- V_s <- D_s <- vector("list", K)
  for (i in seq_len(K)) {
    if (rs[i] >= min(dim(X[[i]]))) rs[i] <- min(dim(X[[i]])) - 1L
    s <- .svd_c(X[[i]])
    seuils[i] <- (s$d[rs[i]] + s$d[rs[i] + 1]) / 2
    U_s[[i]] <- s$u[, 1:rs[i], drop = FALSE]
    D_s[[i]] <- s$d[1:rs[i]]
    V_s[[i]] <- s$v[, 1:rs[i], drop = FALSE]
  }
  sj <- .svd_c(do.call(cbind, U_s))
  Uj <- sj$u; dj <- sj$d
  seuil_w <- seuil_h <- NA_real_
  if (is.null(rang_joint)) {
    gen <- pm_gen(graine)
    hasard <- vapply(seq_len(n_hasard), function(b) .echantillon_hasard(n, rs, gen), 1)
    gen <- pm_gen(graine)
    wedin <- matrix(NA_real_, K, n_wedin)
    for (i in seq_len(K)) {
      un <- vapply(seq_len(n_wedin), function(b) .norme_bruit(t(X[[i]]), U_s[[i]], gen), 1)
      vn <- vapply(seq_len(n_wedin), function(b) .norme_bruit(X[[i]], V_s[[i]], gen), 1)
      wedin[i, ] <- pmin(pmax(un, vn) / D_s[[i]][rs[i]], 1)
    }
    echant_w <- K - colSums(wedin^2)
    seuil_w <- .quantile7(echant_w, centile_wedin / 100)
    seuil_h <- .quantile7(hasard, centile_hasard / 100)
    rJ <- sum(dj^2 > max(seuil_w, seuil_h))
  } else rJ <- as.integer(rang_joint)
  rJ_estime <- rJ
  garder <- seq_len(rJ)
  if (reconsiderer) {
    for (i in seq_len(K)) {
      for (j in garder) {
        if (sqrt(sum((t(X[[i]]) %*% Uj[, j])^2)) < seuils[i]) { garder <- setdiff(garder, j); break }
      }
    }
  }
  Sj <- Uj[, garder, drop = FALSE]
  rJ <- length(garder)
  J <- Ind <- Si <- vector("list", K); rI <- integer(K)
  for (i in seq_len(K)) {
    x <- X[[i]]
    Ji <- if (rJ > 0) Sj %*% (t(Sj) %*% x) else x * 0
    s <- .svd_c(x - Ji)
    r <- if (is.null(rangs_individuels)) {
      m <- max(min(dim(x)) - rJ, 0)
      sum(s$d[seq_len(m)] > seuils[i])
    } else as.integer(rangs_individuels[i])
    J[[i]] <- Ji
    Ind[[i]] <- if (r > 0) s$u[, 1:r, drop = FALSE] %*% diag(s$d[1:r], r) %*% t(s$v[, 1:r, drop = FALSE]) else x * 0
    Si[[i]] <- s$u[, seq_len(r), drop = FALSE]
    rI[i] <- as.integer(r)
  }
  structure(list(joint = J, individuel = Ind, scores_joint = Sj, scores_individuels = Si,
                 rang_joint = rJ, rangs_individuels = rI, rangs_signal = rs, seuils_vs = seuils,
                 vs_joint = dj, seuil_wedin = seuil_w, seuil_hasard = seuil_h,
                 rang_joint_estime = rJ_estime, donnees = X),
            class = "ajive_copres")
}

print.ajive_copres <- function(x, ...) {
  cat("AJIVE : rangs de signal ", paste(x$rangs_signal, collapse = ", "), " ; rang joint ",
      x$rang_joint, " ; rangs individuels ", paste(x$rangs_individuels, collapse = ", "), "\n", sep = "")
  invisible(x)
}
