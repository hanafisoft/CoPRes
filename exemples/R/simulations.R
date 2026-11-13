# Générateurs des données simulées des exemples (modèle de JIVE, situations A, B, C, T ; plans des
# données simulées 1 à 4 de LODE). Mêmes tirages que les versions Python (lode_08_jive.py).
# À sourcer après commun.R.
normales <- function(gen, nl, nc) matrix(pm_normales(gen, nl * nc), nl, nc)
finir_bloc <- function(x) { x <- sweep(x, 2, colMeans(x)); x / sqrt(sum(x^2)) }
cle <- function(L) paste(L, collapse = ",")

# ----------------------------------------------------------------------------- modèle de JIVE
SITUATIONS <- list(list("A. canonique", list()),
                   list("B. axes proches (60°)", list(angle = 60)),
                   list("C. commun 1-2, égal", list(c12 = c(0.9, 0.9))),
                   list("C. commun 1-2, inégal", list(c12 = c(0.9, 0.5))),
                   list("T. propres corrélées, rho = 0,5", list(rho = 0.5)),
                   list("T. propres corrélées, rho = 0,9", list(rho = 0.9)))

jeu_jive <- function(opts, graine, n = 100, p = 50, K = 3, sigma = 0.35) {
  gen <- pm_gen(graine)
  supl <- if (!is.null(opts$c12) || !is.null(opts$rho)) 1 else 0
  m <- 2 + 2 * K + supl
  Z <- normales(gen, n, m)
  T <- gram_schmidt(sweep(Z, 2, colMeans(Z)))
  TJ <- T[, 1:2]
  TI <- lapply(seq_len(K), function(k) T[, 2 + 2 * (k - 1) + 1:2])
  verite <- lapply(1:2, function(j) list(t = TJ[, j], L = seq_len(K)))
  if (!is.null(opts$rho)) TI[[2]][, 1] <- opts$rho * TI[[1]][, 1] + sqrt(1 - opts$rho^2) * T[, m]
  for (k in seq_len(K)) for (j in 1:2) verite[[length(verite) + 1]] <- list(t = TI[[k]][, j], L = k)
  if (!is.null(opts$c12)) verite[[length(verite) + 1]] <- list(t = T[, m], L = 1:2)
  DJ <- diag(c(1, 0.8)); DI <- diag(c(0.8, 0.6)); f <- sqrt(n / p)
  X <- vector("list", K)
  for (k in seq_len(K)) {
    P <- normales(gen, p, 2) * f
    Q <- normales(gen, p, 2) * f
    if (!is.null(opts$angle)) {
      a <- opts$angle * pi / 180
      Q0 <- normales(gen, p, 2) * f
      Q <- cos(a) * P + sin(a) * Q0
    }
    x <- TJ %*% DJ %*% t(P) + TI[[k]] %*% DI %*% t(Q)
    if (!is.null(opts$c12) && k <= 2) {
      b <- normales(gen, p, 1) * f
      x <- x + opts$c12[k] * outer(T[, m], b[, 1])
    }
    x <- x + sigma * normales(gen, n, p)
    X[[k]] <- finir_bloc(x)
  }
  rA <- vapply(seq_len(K), function(k) 2L + as.integer(!is.null(opts$c12) && k <= 2), 1L)
  list(X = X, verite = verite, rJ = 2L, rA = rA)
}

# ----------------------------------------------------------------------------- plans 1 à 4
PLANS <- list(
  list(nom = "1. commun partiel, c faible dans le bloc 3", n = 30, J = c(20, 20, 20), sigma = 0.5,
       struct = list(c = 1:3, c12 = 1:2, s1 = 1, s2 = 2, s3 = 3),
       a = list(c(c = 1.0, c12 = 0.8, s1 = 0.7), c(c = 1.0, c12 = 0.8, s2 = 0.7), c(c = 0.3, s3 = 3.0))),
  list(nom = "2. six blocs, trois communs par paires", n = 40, J = c(10, 15, 20, 25, 30, 40), sigma = 0.4,
       struct = list(c = 1:6, c12 = 1:2, c34 = 3:4, c56 = 5:6, s1 = 1, s2 = 2, s3 = 3, s4 = 4, s5 = 5, s6 = 6),
       a = list(c(c = 1.0, c12 = 0.8, s1 = 0.5), c(c = 1.0, c12 = 0.8, s2 = 0.6),
                c(c = 1.0, c34 = 0.8, s3 = 0.7), c(c = 1.0, c34 = 0.8, s4 = 0.8),
                c(c = 1.0, c56 = 0.8, s5 = 0.9), c(c = 1.0, c56 = 0.8, s6 = 1.0))),
  list(nom = "3. quatre structures propres au bloc 3", n = 30, J = c(20, 20, 20), sigma = 0.4,
       struct = list(c = 1:3, c12 = 1:2, s1 = 1, s2 = 2, s3a = 3, s3b = 3, s3c = 3, s3d = 3),
       a = list(c(c = 1.0, c12 = 0.8, s1 = 0.7), c(c = 1.0, c12 = 0.8, s2 = 0.7),
                c(c = 0.8, s3a = 1.0, s3b = 0.8, s3c = 0.8, s3d = 0.8))),
  list(nom = "4. structures propres en surnombre", n = 30, J = c(20, 20, 20), sigma = 0.4,
       struct = list(c = 1:3, c12 = 1:2, s1 = 1, s2 = 2, e21 = 2, e22 = 2, e23 = 2, e24 = 2, s3 = 3,
                     e31 = 3, e32 = 3, e33 = 3, e34 = 3),
       a = list(c(c = 1.0, c12 = 0.8, s1 = 0.7),
                c(c = 1.0, c12 = 0.8, s2 = 0.7, e21 = 0.7, e22 = 0.7, e23 = 0.7, e24 = 0.7),
                c(c = 0.8, s3 = 1.0, e31 = 0.7, e32 = 0.7, e33 = 0.7, e34 = 0.7))))

jeu_plan <- function(num, graine) {
  P <- PLANS[[num]]
  gen <- pm_gen(graine)
  n <- P$n; noms <- names(P$struct)
  T <- gram_schmidt(normales(gen, n, length(noms)))
  T <- gram_schmidt(sweep(T, 2, colMeans(T))) * sqrt(n)
  K <- length(P$J)
  X <- vector("list", K)
  for (k in seq_len(K)) {
    x <- P$sigma * normales(gen, n, P$J[k])
    for (i in seq_along(noms)) {
      s <- noms[i]
      if (s %in% names(P$a[[k]])) x <- x + P$a[[k]][[s]] * outer(T[, i], normales(gen, P$J[k], 1)[, 1])
    }
    X[[k]] <- finir_bloc(x)
  }
  verite <- lapply(seq_along(noms), function(i) list(t = T[, i] / sqrt(n), L = P$struct[[noms[i]]]))
  tous <- seq_len(K)
  est_tous <- vapply(verite, function(v) identical(as.integer(v$L), tous), TRUE)
  rJ <- sum(est_tous)
  rA <- vapply(seq_len(K), function(k) sum(vapply(verite, function(v) k %in% v$L, TRUE) & !est_tous), 1L)
  list(X = X, verite = verite, rJ = as.integer(rJ), rA = as.integer(rA))
}
