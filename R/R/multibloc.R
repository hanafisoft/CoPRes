# Couche CoPRes pour des données structurées par groupes de variables (blocs mesurés sur les
# mêmes individus). X : liste de K matrices n x J_k. Les matrices étudiées par LODE sont les
# restrictions A_k = U' X_k X_k' U au sous-espace de la base U d'une méthode de la factorisation
# canonique. Traduction pas à pas de la version Python (même ordre des tirages aléatoires).

# ------------------------------------------------------------------ prétraitement
pretraiter <- function(X, snv = FALSE, reduire = FALSE, inertie = TRUE) {
  lapply(X, function(x) {
    x <- as.matrix(x); storage.mode(x) <- "double"
    if (snv) x <- (x - rowMeans(x)) / apply(x, 1, sd)
    x <- sweep(x, 2, colMeans(x))
    if (reduire) { s <- apply(x, 2, sd); s[s == 0] <- 1; x <- sweep(x, 2, s, "/") }
    if (inertie) x <- x / sqrt(sum(x^2))
    x
  })
}

# ------------------------------------------------------------------ seuils
seuils_perm <- function(X, nperm = 200, ndir = 100, niveau = 0.95, regle = c("direction", "valeur_propre"), graine = 1) {
  regle <- match.arg(regle)
  gen <- pm_gen(graine)
  vapply(X, function(x) {
    x <- as.matrix(x); n <- nrow(x); J <- ncol(x)
    vals <- vector("list", nperm)
    for (p in seq_len(nperm)) {
      u <- matrix(pm_uniformes(gen, n * J), n, J)
      X0 <- sapply(seq_len(J), function(j) x[order(u[, j]), j])
      X0 <- sweep(matrix(X0, n, J), 2, colMeans(matrix(X0, n, J)))
      if (regle == "direction") {
        Z <- matrix(pm_normales(gen, n * ndir), n, ndir)
        Z <- sweep(Z, 2, colMeans(Z))
        Z <- sweep(Z, 2, sqrt(colSums(Z^2)), "/")
        vals[[p]] <- colSums((t(X0) %*% Z)^2)
      } else {
        M <- if (J <= n) crossprod(X0) else tcrossprod(X0)
        vals[[p]] <- max(eigen(M, symmetric = TRUE, only.values = TRUE)$values)
      }
    }
    unname(quantile(unlist(vals), niveau, type = 7))
  }, 0)
}

# ------------------------------------------------------------------ méthodes
.signes <- function(U) {
  for (h in seq_len(ncol(U))) if (U[which.max(abs(U[, h])), h] < 0) U[, h] <- -U[, h]
  U
}

.gram_schmidt <- function(M) {
  Q <- matrix(0, nrow(M), ncol(M))
  for (h in seq_len(ncol(M))) {
    v <- M[, h]
    if (h > 1) {
      Qh <- Q[, 1:(h - 1), drop = FALSE]
      v <- v - Qh %*% crossprod(Qh, M[, h])
      v <- v - Qh %*% crossprod(Qh, v)
    }
    Q[, h] <- v / sqrt(sum(v^2))
  }
  Q
}

.vp_dominant <- function(W) eigen(W, symmetric = TRUE)$vectors[, 1]

cpca <- function(X, R) {
  W <- Reduce(`+`, lapply(X, tcrossprod))
  .signes(eigen(W, symmetric = TRUE)$vectors[, 1:R, drop = FALSE])
}

acom <- function(X, R) {
  Xh <- lapply(X, as.matrix); U <- matrix(0, nrow(Xh[[1]]), R)
  for (h in seq_len(R)) {
    u <- .vp_dominant(Reduce(`+`, lapply(Xh, tcrossprod))); U[, h] <- u
    Xh <- lapply(Xh, function(x) { a <- crossprod(x, u); a <- a / sqrt(sum(a^2)); x - (x %*% a) %*% t(a) })
  }
  .signes(.gram_schmidt(U))
}

accps <- function(X, R, it = 3000, tol = 1e-12, nb_aleatoires = 10, graine = 1) {
  W <- lapply(X, tcrossprod); n <- nrow(W[[1]]); P <- diag(n)
  gen <- pm_gen(graine); U <- matrix(0, n, R)
  for (h in seq_len(R)) {
    Wd <- lapply(W, function(w) P %*% w %*% P)
    departs <- c(list(.vp_dominant(Reduce(`+`, Wd))), lapply(Wd, .vp_dominant))
    for (i in seq_len(nb_aleatoires)) departs[[length(departs) + 1]] <- as.vector(P %*% pm_normales(gen, n))
    best_f <- -Inf; best_u <- NULL
    for (u in departs) {
      u <- u / sqrt(sum(u^2)); f0 <- -1
      for (r in seq_len(it)) {
        g <- Reduce(`+`, lapply(Wd, function(w) { wu <- as.vector(w %*% u); sum(u * wu) * wu }))
        u <- g / sqrt(sum(g^2))
        f <- sum(vapply(Wd, function(w) sum(u * (w %*% u))^2, 0))
        if (abs(f - f0) < tol) break
        f0 <- f
      }
      if (is.null(best_u) || f > best_f + 1e-14) { best_f <- f; best_u <- u }
    }
    U[, h] <- best_u
    P <- P - best_u %o% best_u
  }
  .signes(.gram_schmidt(U))
}

METHODES <- list(CPCA = cpca, ACOM = acom, ACCPS = accps)

# ------------------------------------------------------------------ lecture et CoPRes
.restrictions <- function(X, U) lapply(X, function(x) { y <- crossprod(U, x); tcrossprod(y) })

lecture_directe <- function(X, U, theta) {
  U <- as.matrix(U); A <- .restrictions(X, U); I <- diag(ncol(U))
  etiq <- etiquettes(A, I, theta); m <- marge(A, I, theta, etiq)
  structure(list(base = U, saliences = saliences(A, I), seuils = theta, etiquettes = etiq,
                 motif = motif(etiq), coherente = m >= -1e-9, marge = m, bilan = critere(A, I, theta)),
            class = "lecture_directe")
}

print.lecture_directe <- function(x, ...) {
  cat(sprintf("Lecture directe : motif %s ; %s (marge %.3g) ; bilan %.6g\n", texte_motif(x$motif),
              if (x$coherente) "cohérente" else "non cohérente", x$marge, x$bilan))
  invisible(x)
}

copres <- function(X, theta, R = NULL, methode = "CPCA", U = NULL, departs = 20, graine = 1) {
  if (is.null(U)) {
    if (is.null(R)) stop("donner R ou U")
    U <- METHODES[[methode]](X, R)
  } else U <- as.matrix(U)
  A <- .restrictions(X, U)
  res <- lode(A, theta, departs = departs, graine = graine)
  V <- U %*% res$base
  poids <- do.call(rbind, lapply(X, function(x) sqrt(colSums(crossprod(x, V)^2))))
  H <- lapply(seq_along(X), function(k) { p <- poids[k, ]; p[p == 0] <- 1; sweep(crossprod(X[[k]], V), 2, p, "/") })
  structure(list(methode = methode, base_methode = U, base = V, etiquettes = res$etiquettes,
                 saliences = res$saliences, poids = poids, loadings = H, seuils = theta, motif = res$motif,
                 bilan = res$critere, borne = res$borne, ecart_relatif = res$ecart_relatif,
                 parts_sous_espace = vapply(A, function(a) sum(diag(a)), 0), lode = res),
            class = "copres")
}

print.copres <- function(x, ...) {
  cat(sprintf("CoPRes (%s, R = %d) : motif %s\n", x$methode, ncol(x$base), texte_motif(x$motif)))
  cat(sprintf("bilan %.6g, borne de Ky Fan %.6g, écart relatif %.3g %%\n", x$bilan, x$borne, 100 * x$ecart_relatif))
  invisible(x)
}

# ------------------------------------------------------------------ fiche, équivalence
fiche <- function(U = NULL, cop, seuil_melange = 0.10, bornes_dilution = c(0.10, 0.90)) {
  if (is.null(U)) U <- cop$base_methode
  G <- crossprod(cop$base, as.matrix(U))^2
  cles <- vapply(cop$etiquettes, .cle, "")
  vide <- vapply(cop$etiquettes, length, 1L) == 0
  composition <- list(); principale <- list(); mu <- r <- numeric(ncol(G))
  for (h in seq_len(ncol(G))) {
    parts <- tapply(G[, h], factor(cles, levels = unique(cles)), sum)
    composition[[h]] <- sort(parts, decreasing = TRUE)
    r[h] <- sum(G[vide, h])
    sig <- parts[names(parts) != ""]
    if (length(sig)) {
      principale[[h]] <- names(sig)[which.max(sig)]
      mu[h] <- sum(sig) - max(sig)
    } else principale[[h]] <- NA
  }
  structure(list(composition = composition, principale = unlist(principale), melange = mu, residuel = r,
                 melangee = mu > seuil_melange, diluee = r > bornes_dilution[1] & r < bornes_dilution[2]),
            class = "fiche")
}

print.fiche <- function(x, ...) {
  for (h in seq_along(x$melange)) {
    c <- x$composition[[h]]; c <- c[c >= 0.005]
    noms <- ifelse(names(c) == "", "résiduelle", paste0("{", names(c), "}"))
    cat(sprintf("u%d : %s ; mu* = %.2f, r = %.2f%s%s\n", h, paste(sprintf("%s %.0f %%", noms, 100 * c), collapse = ", "),
                x$melange[h], x$residuel[h], if (x$melangee[h]) " ; mélangée" else "", if (x$diluee[h]) " ; diluée" else ""))
  }
  invisible(x)
}

proximite <- function(V1, V2) {
  if (ncol(V1) != ncol(V2)) return(NA_real_)
  min(svd(crossprod(V1, V2))$d)^2
}

equivalence <- function(cop1, cop2, seuil = 0.72) {
  if (!identical(texte_motif(cop1$motif), texte_motif(cop2$motif))) return(list(equivalentes = FALSE, proximites = NULL))
  c1 <- vapply(cop1$etiquettes, .cle, ""); c2 <- vapply(cop2$etiquettes, .cle, "")
  prox <- vapply(unique(c1), function(k) proximite(cop1$base[, c1 == k, drop = FALSE], cop2$base[, c2 == k, drop = FALSE]), 0)
  list(equivalentes = all(prox >= seuil), proximites = prox)
}

sensibilite <- function(X, theta, R_liste = 2:5, facteurs = c(0.75, 1, 1.5, 2), methode = "CPCA", departs = 20, graine = 1) {
  out <- expand.grid(facteur = facteurs, R = R_liste)[, c("R", "facteur")]
  out$motif <- NA_character_
  for (i in seq_len(nrow(out))) {
    U <- METHODES[[methode]](X, out$R[i])
    out$motif[i] <- texte_motif(copres(X, theta * out$facteur[i], U = U, methode = methode, departs = departs, graine = graine)$motif)
  }
  out
}
