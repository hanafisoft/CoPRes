# Générateur pseudo-aléatoire commun aux versions R et Python de CoPRes :
# générateur multiplicatif de Park et Miller (a = 16 807, m = 2^31 - 1).
# Les calculs sont faits en double précision, exacte jusqu'à 2^53 > a * m.

.PM_A <- 16807
.PM_M <- 2147483647

#' Générateur de Park et Miller
#' @param graine entier ; ramené dans [1, m - 1]
#' @return un environnement qui porte l'état du générateur
pm_gen <- function(graine = 1) {
  e <- new.env()
  g <- graine %% .PM_M
  e$etat <- if (g > 0) g else 1
  e
}

pm_uniforme <- function(gen) {
  gen$etat <- (.PM_A * gen$etat) %% .PM_M
  gen$etat / .PM_M
}

.cache <- new.env()

# a^k mod m pour k = 1, ..., 65536 (calculées une fois)
.puissances <- function() {
  if (is.null(.cache$p)) {
    p <- numeric(65536); x <- 1
    for (k in 1:65536) { x <- (.PM_A * x) %% .PM_M; p[k] <- x }
    .cache$p <- p
  }
  .cache$p
}

# (p * s) mod m sans perte de précision : p est coupé en deux parties de 16 bits
.mulmod <- function(p, s) {
  h <- floor(p / 65536); l <- p - h * 65536
  x1 <- (((h * s) %% .PM_M) * 65536) %% .PM_M
  (x1 + (l * s) %% .PM_M) %% .PM_M
}

# nb tirages uniformes successifs (même suite que nb appels à pm_uniforme), vectorisés
pm_uniformes <- function(gen, nb) {
  out <- numeric(nb); P <- .puissances(); i <- 0
  while (i < nb) {
    b <- min(65536, nb - i)
    out[(i + 1):(i + b)] <- .mulmod(P[1:b], gen$etat)
    gen$etat <- out[i + b]
    i <- i + b
  }
  out / .PM_M
}

# Tirage gaussien centré réduit (Box-Muller, premier terme)
pm_normale <- function(gen) {
  u1 <- pm_uniforme(gen)
  u2 <- pm_uniforme(gen)
  sqrt(-2 * log(u1)) * cos(2 * pi * u2)
}

pm_normales <- function(gen, nb) {
  u <- pm_uniformes(gen, 2 * nb)
  sqrt(-2 * log(u[c(TRUE, FALSE)])) * cos(2 * pi * u[c(FALSE, TRUE)])
}

# Permutation de 1, ..., n : rangs de n tirages uniformes (même suite que la version Python)
pm_permutation <- function(gen, n) order(pm_uniformes(gen, n))

# Matrice orthogonale n x n : produit de rotations planes d'angles uniformes sur [0, 2 pi[,
# appliquées à toutes les paires (i, j), i < j, dans l'ordre lexicographique.
rotation_aleatoire <- function(n, gen) {
  Q <- diag(n)
  if (n < 2) return(Q)
  for (i in 1:(n - 1)) for (j in (i + 1):n) {
    a <- 2 * pi * pm_uniforme(gen)
    c <- cos(a); s <- sin(a)
    qi <- Q[, i]; qj <- Q[, j]
    Q[, i] <- c * qi + s * qj
    Q[, j] <- -s * qi + c * qj
  }
  Q
}
