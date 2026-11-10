# Stabilité de l'étiquetage par rééchantillonnage bootstrap (traduction de la version Python).
# Pour chaque partie de référence (étiquette L, dimension d, base E), fréquence des échantillons
# où elle est retrouvée :
#  - « strict » : partie de même étiquette et de même dimension, proximité >= seuil ;
#  - « étiquette » : partie de même étiquette, de dimension >= d, qui contient E à seuil près ;
#  - « type » : E contenue dans la réunion des parties de même type (commune, partagée, propre,
#    résiduelle).

type_etiquette <- function(L, K) {
  if (length(L) == 0) "résiduelle" else if (length(L) == 1) "propre" else if (length(L) == K) "commune" else "partagée"
}

.inclusion <- function(E, F) {
  if (ncol(F) < ncol(E)) return(-1)
  s <- svd(crossprod(E, F))$d[seq_len(ncol(E))]
  min(s)^2
}

.parties_base <- function(base, etiq) {
  g <- .groupes(etiq); Ls <- attr(g, "etiq")
  out <- lapply(g, function(idx) base[, idx, drop = FALSE])
  attr(out, "etiq") <- Ls
  out
}

.comparer <- function(ref, boot, K, seuil) {
  Lr <- attr(ref, "etiq"); Lb <- attr(boot, "etiq")
  tb <- vapply(Lb, type_etiquette, "", K = K)
  out <- lapply(seq_along(ref), function(i) {
    E <- ref[[i]]; k <- names(ref)[i]
    j <- match(k, names(boot))          # les noms "" (partie résiduelle) ne s'indexent pas par [[ ]]
    F <- if (!is.na(j)) boot[[j]] else NULL
    strict <- !is.null(F) && ncol(F) == ncol(E) && .inclusion(E, F) >= seuil
    etiq <- !is.null(F) && .inclusion(E, F) >= seuil
    t <- type_etiquette(Lr[[i]], K)
    typ <- any(tb == t) && .inclusion(E, do.call(cbind, boot[tb == t])) >= seuil
    c(strict, etiq, typ)
  })
  out
}

.resume <- function(ref, res, motifs, motif_ref, B, noms) {
  Lr <- attr(ref, "etiq")
  f <- function(i) vapply(seq_along(ref), function(j) sum(vapply(res, function(r) r[[j]][i], TRUE)) / B, 0)
  nm <- vapply(Lr, nom_etiquette, "", noms = noms)
  dims <- vapply(ref, ncol, 1L)
  tab <- table(motifs)
  structure(list(nb = B, noms = noms, etiquettes = Lr, parties = nm, dimensions = setNames(dims, nm),
                 strict = setNames(f(1), nm), etiquette = setNames(f(2), nm), type = setNames(f(3), nm),
                 motif_retrouve = sum(motifs == motif_ref) / B,
                 motifs = sort(setNames(as.integer(tab), names(tab)), decreasing = TRUE)),
            class = "stabilite")
}

print.stabilite <- function(x, ...) {
  cat(sprintf("Stabilité sur %d échantillons bootstrap ; motif complet retrouvé : %.0f %%\n", x$nb, 100 * x$motif_retrouve))
  cat("partie (dimension) : strict / étiquette / type\n")
  for (i in seq_along(x$parties))
    cat(sprintf("  %s (%d) : %.0f / %.0f / %.0f %%\n", x$parties[i], x$dimensions[i], 100 * x$strict[i],
                100 * x$etiquette[i], 100 * x$type[i]))
  invisible(x)
}

.indices <- function(gen, strates) {
  unlist(lapply(unique(strates), function(s) {
    membres <- which(strates == s)
    u <- pm_uniformes(gen, length(membres))
    membres[floor(u * length(membres)) + 1]
  }), use.names = FALSE)
}

.pinv <- function(M) {
  s <- svd(M); tol <- 1e-15 * max(s$d)   # même règle que numpy.linalg.pinv(rcond = 1e-15)
  keep <- s$d > tol
  s$v[, keep, drop = FALSE] %*% (t(s$u[, keep, drop = FALSE]) / s$d[keep])
}

# ------------------------------------------------------------------ multigroupe
stabilite_groupes <- function(cop, prep, B = 100, graine = 1, departs = 20, seuil = 0.72) {
  gen <- pm_gen(graine); K <- length(prep$groupes)
  ref <- .parties_base(cop$base_analyse, cop$etiquettes)
  motif_ref <- texte_motif(cop$motif)
  res <- vector("list", B); motifs <- character(B)
  for (b in seq_len(B)) {
    A <- lapply(prep$Xk, function(x) {
      u <- pm_uniformes(gen, nrow(x))
      xb <- x[floor(u * nrow(x)) + 1, , drop = FALSE]
      xb <- sweep(xb, 2, colMeans(xb))
      t(prep$U) %*% (crossprod(xb) / (nrow(xb) - 1)) %*% prep$U
    })
    p2 <- prep; p2$covariances <- A
    cb <- copres_groupes(p2, cop$seuils, departs = departs, graine = b)
    res[[b]] <- .comparer(ref, .parties_base(cb$base_analyse, cb$etiquettes), K, seuil)
    motifs[b] <- texte_motif(cb$motif)
  }
  .resume(ref, res, motifs, motif_ref, B, prep$groupes)
}

# ------------------------------------------------------------------ multibloc
stabilite_blocs <- function(cop, X, B = 100, strates = NULL, graine = 1, departs = 20, seuil = 0.72, nperm = 50, ndir = 100,
                            recalculer_seuils = TRUE, noms = NULL) {
  X <- lapply(X, as.matrix); n <- nrow(X[[1]]); K <- length(X)
  if (is.null(strates)) strates <- rep(0L, n)
  R <- ncol(cop$base)
  methode <- if (!is.null(cop$methode) && cop$methode %in% names(METHODES)) cop$methode else "CPCA"
  gen <- pm_gen(graine)
  ref <- .parties_base(cop$base, cop$etiquettes)
  motif_ref <- texte_motif(cop$motif)
  Xcat <- do.call(cbind, X)
  res <- vector("list", B); motifs <- character(B)
  for (b in seq_len(B)) {
    idx <- .indices(gen, strates)
    Xb <- lapply(X, function(x) { z <- sweep(x[idx, , drop = FALSE], 2, colMeans(x[idx, , drop = FALSE])); z / sqrt(sum(z^2)) })
    th <- if (recalculer_seuils) seuils_perm(Xb, nperm = nperm, ndir = ndir, graine = 1000 * graine + b) else cop$seuils
    cb <- copres(Xb, th, U = METHODES[[methode]](Xb, R), methode = methode, departs = departs, graine = b)
    P <- .pinv(do.call(cbind, Xb))
    pb <- .parties_base(cb$base, cb$etiquettes)
    boot <- lapply(pb, function(E) {
      S <- Xcat %*% (P %*% E); S <- sweep(S, 2, colMeans(S))
      s <- svd(S); s$u[, s$d > 1e-10 * max(1, max(s$d)), drop = FALSE]
    })
    attr(boot, "etiq") <- attr(pb, "etiq")
    res[[b]] <- .comparer(ref, boot, K, seuil)
    motifs[b] <- texte_motif(cb$motif)
  }
  .resume(ref, res, motifs, motif_ref, B, noms)
}
