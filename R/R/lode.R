# Module lode : décomposition orthogonale étiquetée optimale de matrices semi-définies positives.
# A : liste de K matrices n x n semi-définies positives ; theta : seuils ; V : base orthonormée
# (composantes en colonnes). Une étiquette est un vecteur entier trié des matrices qui portent la
# composante (numérotées à partir de 1) ; integer(0) désigne la part résiduelle.
# Les calculs suivent pas à pas la version Python (mêmes ordres de parcours, valeurs propres
# rangées par ordre croissant), pour des résultats identiques aux arrondis près.

.EPS <- 1e-12

.verifier <- function(A, theta) {
  if (!is.list(A)) stop("A doit être une liste de matrices")
  A <- lapply(A, function(a) { a <- as.matrix(a); storage.mode(a) <- "double"; a })
  n <- nrow(A[[1]])
  for (a in A) {
    if (!all(dim(a) == c(n, n))) stop("les matrices doivent toutes être carrées et de même taille")
    if (max(abs(a - t(a))) > 1e-10 * max(1, max(abs(a)))) stop("les matrices doivent être symétriques")
  }
  theta <- as.numeric(theta)
  if (length(theta) != length(A)) stop("il faut un seuil par matrice")
  if (any(theta <= 0)) stop("les seuils doivent être strictement positifs")
  list(A = A, theta = theta)
}

.cle <- function(L) paste(L, collapse = ",")

nom_etiquette <- function(L, noms = NULL) {
  if (!length(L)) return("résiduelle")
  paste0("{", paste(if (is.null(noms)) L else noms[L], collapse = ","), "}")
}

# valeurs et vecteurs propres rangés par valeurs croissantes (comme numpy.linalg.eigh)
.eigh <- function(M) {
  e <- eigen((M + t(M)) / 2, symmetric = TRUE)
  k <- ncol(e$vectors)
  list(values = rev(e$values), vectors = e$vectors[, k:1, drop = FALSE])
}

saliences <- function(A, V) {
  V <- as.matrix(V)
  do.call(rbind, lapply(A, function(a) colSums(V * (a %*% V))))
}

etiquettes <- function(A, V, theta) {
  S <- saliences(A, V)
  lapply(seq_len(ncol(S)), function(h) which(S[, h] >= theta))
}

# groupes de composantes de même étiquette, dans l'ordre de première apparition
.groupes <- function(etiq) {
  cles <- vapply(etiq, .cle, "")
  u <- unique(cles)
  g <- lapply(u, function(k) which(cles == k))
  names(g) <- u
  attr(g, "etiq") <- lapply(g, function(idx) etiq[[idx[1]]])
  g
}

.ordre_etiquettes <- function(Ls) {
  vide <- vapply(Ls, function(L) length(L) == 0, TRUE)
  long <- vapply(Ls, length, 1L)
  lex <- vapply(Ls, function(L) paste(sprintf("%05d", L), collapse = ","), "")
  order(vide, -long, lex)
}

motif <- function(etiq) {
  g <- .groupes(etiq)
  Ls <- attr(g, "etiq")
  o <- .ordre_etiquettes(Ls)
  data.frame(etiquette = I(Ls[o]), nom = vapply(Ls[o], nom_etiquette, ""),
             multiplicite = vapply(g[o], length, 1L), row.names = NULL)
}

texte_motif <- function(m, noms = NULL) {
  n <- if (is.null(noms)) m$nom else vapply(m$etiquette, nom_etiquette, "", noms = noms)
  paste(sprintf("(%s, %d)", n, m$multiplicite), collapse = ", ")
}

critere <- function(A, V, theta) {
  S <- saliences(A, V)
  sum(pmax(S - theta, 0))
}

critere_etiquetage <- function(A, parties, theta) {
  phi <- 0
  for (p in parties) for (k in p$etiquette)
    phi <- phi + sum(diag(t(p$base) %*% A[[k]] %*% p$base)) - ncol(p$base) * theta[k]
  phi
}

marge <- function(A, V, theta, etiq = NULL) {
  v <- .verifier(A, theta); A <- v$A; theta <- v$theta
  if (is.null(etiq)) etiq <- etiquettes(A, V, theta)
  g <- .groupes(etiq); Ls <- attr(g, "etiq")
  m <- Inf
  for (i in seq_along(g)) {
    idx <- g[[i]]; L <- Ls[[i]]
    if (length(idx) < 2) next
    VL <- V[, idx, drop = FALSE]
    for (k in seq_along(A)) {
      ev <- .eigh(t(VL) %*% A[[k]] %*% VL)$values
      m <- min(m, if (k %in% L) ev[1] - theta[k] else theta[k] - ev[length(ev)])
    }
  }
  m
}

est_etiquetage <- function(A, V, theta, eps = 1e-9) marge(A, V, theta) >= -eps

borne_kf <- function(A, theta) {
  v <- .verifier(A, theta)
  sum(mapply(function(a, t) sum(pmax(.eigh(a)$values - t, 0)), v$A, v$theta))
}

# Étiquetage idéal : renvoie list(ideal = TRUE/FALSE, parties = list(list(etiquette, base), ...))
ideal <- function(A, theta, tol = 1e-9) {
  v <- .verifier(A, theta); A <- v$A; theta <- v$theta
  n <- nrow(A[[1]]); K <- length(A)
  P <- lapply(seq_len(K), function(k) {
    e <- .eigh(A[[k]]); U <- e$vectors[, e$values > theta[k], drop = FALSE]; U %*% t(U)
  })
  if (K > 1) for (k in 1:(K - 1)) for (l in (k + 1):K)
    if (max(abs(P[[k]] %*% P[[l]] - P[[l]] %*% P[[k]])) > tol) return(list(ideal = FALSE, parties = NULL))
  I <- diag(n); parties <- list()
  for (code in 0:(2^K - 1)) {
    L <- which(bitwAnd(code, 2^(0:(K - 1))) > 0)
    PL <- I
    for (k in seq_len(K)) PL <- PL %*% (if (k %in% L) P[[k]] else I - P[[k]])
    e <- .eigh(PL)
    VL <- e$vectors[, e$values > 0.5, drop = FALSE]
    if (ncol(VL)) parties[[length(parties) + 1]] <- list(etiquette = L, base = VL)
  }
  list(ideal = TRUE, parties = parties)
}

.polaire <- function(G) { s <- svd(G); s$u %*% t(s$v) }

concentration <- function(A, V0 = NULL, it = 1000, tol = .EPS) {
  n <- nrow(A[[1]])
  Q <- if (is.null(V0)) diag(n) else as.matrix(V0)
  c_old <- -Inf
  for (r in seq_len(it)) {
    S <- saliences(A, Q)
    G <- sapply(seq_len(n), function(h) Reduce(`+`, lapply(seq_along(A), function(k) S[k, h] * (A[[k]] %*% Q[, h]))))
    Q <- .polaire(G)
    cc <- sum(saliences(A, Q)^2)
    if (cc - c_old <= tol * max(1, abs(cc))) break
    c_old <- cc
  }
  Q
}

base_concentree <- function(A, graine = 1, nb_aleatoires = 4) {
  n <- nrow(A[[1]]); gen <- pm_gen(graine)
  departs <- c(list(diag(n)), lapply(seq_len(nb_aleatoires), function(i) rotation_aleatoire(n, gen)))
  best <- NULL; cbest <- -Inf
  for (V0 in departs) {
    Q <- concentration(A, V0); cc <- sum(saliences(A, Q)^2)
    if (cc > cbest + 1e-14) { best <- Q; cbest <- cc }
  }
  best
}

.somme <- function(A, L, n) if (length(L)) Reduce(`+`, A[L]) else matrix(0, n, n)

taro <- function(V, cible, A, eps = .EPS, max_balayages = 10000) {
  V <- as.matrix(V); n <- nrow(V)
  if (n < 2) return(V)
  AL <- lapply(cible, function(L) .somme(A, L, n))
  cles <- vapply(cible, .cle, "")
  for (b in seq_len(max_balayages)) {
    gain <- 0
    for (i in 1:(n - 1)) for (j in (i + 1):n) {
      if (cles[i] == cles[j]) next
      D <- AL[[i]] - AL[[j]]
      vi <- V[, i]; vj <- V[, j]
      be <- 0.5 * (sum(vi * (D %*% vi)) - sum(vj * (D %*% vj)))
      ga <- sum(vi * (D %*% vj))
      phi <- 0.5 * atan2(ga, be); c <- cos(phi); s <- sin(phi)
      V[, i] <- c * vi + s * vj
      V[, j] <- -s * vi + c * vj
      gain <- gain + sqrt(be^2 + ga^2) - be
    }
    J <- sum(vapply(seq_len(n), function(h) sum(V[, h] * (AL[[h]] %*% V[, h])), 0))
    if (gain <= eps * max(1, abs(J))) break
  }
  V
}

# Construction d'une base d'étiquetage ; renvoie list(base, remplacements)
b2bl <- function(U, A, theta, eps = .EPS, max_remplacements = NULL) {
  v <- .verifier(A, theta); A <- v$A; theta <- v$theta
  V <- as.matrix(U); K <- length(A)
  if (is.null(max_remplacements)) max_remplacements <- floor(sum(vapply(A, function(a) sum(diag(a)), 0)) / eps) + 1
  nb <- 0
  while (nb < max_remplacements) {
    trouve <- FALSE
    g <- .groupes(etiquettes(A, V, theta)); Ls <- attr(g, "etiq")
    for (i in seq_along(g)) {
      idx <- g[[i]]; L <- Ls[[i]]
      if (length(idx) < 2) next
      VL <- V[, idx, drop = FALSE]
      for (k in seq_len(K)) {
        e <- .eigh(t(VL) %*% A[[k]] %*% VL); ev <- e$values
        if ((k %in% L && ev[1] < theta[k] - eps) || (!(k %in% L) && ev[length(ev)] > theta[k] + eps)) {
          V[, idx] <- VL %*% e$vectors; nb <- nb + 1; trouve <- TRUE; break
        }
      }
      if (trouve) break
    }
    if (!trouve) break
  }
  list(base = V, remplacements = nb)
}

.depuis <- function(A, theta, V0, eps, maxit) {
  V <- b2bl(V0, A, theta, eps)$base
  p_old <- critere(A, V, theta)
  for (t in seq_len(maxit)) {
    cible <- etiquettes(A, V, theta)
    V <- taro(V, cible, A, eps)
    r <- b2bl(V, A, theta, eps); V <- r$base
    p <- critere(A, V, theta)
    if (r$remplacements == 0 && p <= (1 + eps) * p_old) return(list(V = V, p = p, it = t, converge = TRUE))
    p_old <- p
  }
  list(V = V, p = critere(A, V, theta), it = maxit, converge = FALSE)
}

.orienter <- function(A, V, etiq) {
  K <- length(A)
  g <- .groupes(etiq); Ls <- attr(g, "etiq")
  for (i in seq_along(g)) {
    idx <- g[[i]]; L <- Ls[[i]]
    if (length(idx) < 2) next
    VL <- V[, idx, drop = FALSE]
    M <- Reduce(`+`, lapply(if (length(L)) L else seq_len(K), function(k) t(VL) %*% A[[k]] %*% VL))
    P <- .eigh(M)$vectors
    V[, idx] <- VL %*% P[, ncol(P):1, drop = FALSE]
  }
  for (h in seq_len(ncol(V))) if (V[which.max(abs(V[, h])), h] < 0) V[, h] <- -V[, h]
  o <- order(-colSums(saliences(A, V)), method = "radix")
  list(V = V[, o, drop = FALSE], etiq = etiq[o])
}

#' Algorithme LODE avec plusieurs départs
#' Départs : base concentrée, puis departs - 1 rotations tirées au hasard (Park et Miller, graine) ;
#' si depart_propre, la base propre de la somme des A_k est ajoutée en deuxième position.
lode <- function(A, theta, departs = 20, graine = 1, depart_propre = FALSE, eps = .EPS, maxit = 30, tol_meilleur = 1e-9) {
  v <- .verifier(A, theta); A <- v$A; theta <- v$theta
  n <- nrow(A[[1]]); gen <- pm_gen(graine)
  liste <- list(base_concentree(A, graine = graine))
  if (depart_propre) { e <- .eigh(Reduce(`+`, A)); liste[[length(liste) + 1]] <- e$vectors[, n:1] }
  for (i in seq_len(max(0, departs - 1))) liste[[length(liste) + 1]] <- rotation_aleatoire(n, gen)
  res <- lapply(liste, function(V0) .depuis(A, theta, V0, eps, maxit))
  crits <- vapply(res, function(r) r$p, 0)
  i <- which.max(crits)
  V <- res[[i]]$V
  etiq <- etiquettes(A, V, theta)
  m <- marge(A, V, theta, etiq)
  o <- .orienter(A, V, etiq); V <- o$V; etiq <- o$etiq
  g <- .groupes(etiq); Ls <- attr(g, "etiq")
  parties <- lapply(seq_along(g), function(j) list(etiquette = Ls[[j]], base = V[, g[[j]], drop = FALSE]))
  parties <- parties[.ordre_etiquettes(Ls)]
  phi <- critere_etiquetage(A, parties, theta)
  eta <- borne_kf(A, theta)
  structure(list(base = V, etiquettes = etiq, saliences = saliences(A, V), seuils = theta,
                 parties = parties, motif = motif(etiq), critere = phi, borne = eta,
                 ecart_relatif = if (eta > 0) 1 - phi / eta else 0, marge = m,
                 criteres_departs = crits,
                 departs_au_meilleur = sum(crits >= crits[i] - tol_meilleur * max(1, abs(crits[i]))),
                 converge = all(vapply(res, function(r) r$converge, TRUE)), matrices = A),
            class = "lode")
}

print.lode <- function(x, ...) {
  cat("Étiquetage LODE : motif", texte_motif(x$motif), "\n")
  cat(sprintf("critère %.6g, borne de Ky Fan %.6g, écart relatif %.3g %%\n", x$critere, x$borne, 100 * x$ecart_relatif))
  cat(sprintf("meilleur critère atteint par %d départ(s) sur %d\n", x$departs_au_meilleur, length(x$criteres_departs)))
  if (!x$converge) cat("attention : nombre maximal d'itérations atteint pour au moins un départ\n")
  invisible(x)
}
