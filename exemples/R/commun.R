# Lecture des données des exemples (dossier ../donnees). À sourcer depuis le dossier exemples/R :
# setwd(".../exemples/R"); source("commun.R")
library(CoPRes)
DONNEES <- file.path("..", "donnees")
chemin <- function(nom) file.path(DONNEES, nom)
lire <- function(nom, ...) utils::read.csv(chemin(nom), check.names = FALSE, fileEncoding = "UTF-8", stringsAsFactors = FALSE, ...)

lire_beurres <- function() {
  blocs <- list(); axes <- list()
  for (n in c("VIS", "NIR", "MIR")) {
    d <- lire(paste0("beurres_", n, ".csv"))
    ids <- d[[1]]
    blocs[[n]] <- as.matrix(d[, -1]); axes[[n]] <- as.numeric(names(d)[-1])
  }
  list(blocs = unname(blocs), type = ifelse(substr(ids, 1, 1) == "B", "beurre", "margarine"), ids = ids, axes = axes)
}

lire_simulees <- function(plan) {
  d <- lire(sprintf("simulees%d.csv", plan))
  lapply(sort(unique(d$bloc)), function(b) t(as.matrix(d[d$bloc == b, -1])))
}

symetrique <- function(i, j, v, p) { A <- matrix(0, p, p); A[cbind(i, j)] <- v; A[cbind(j, i)] <- v; A }

# Quatre juges (blocs) décrivant l'apparence de 8 vins (profil libre, notes de 0 à 10) ;
# Williams et Langron (1984), données de Hanafi et Qannari (2008, annexe 6).
lire_vins <- function() {
  d <- utils::read.csv(chemin("vins.csv"), check.names = FALSE)
  juge <- sub("_.*", "", names(d)[-1])
  blocs <- lapply(paste0("J", 1:4), function(j) as.matrix(d[, -1][, juge == j]))
  list(blocs = blocs, ids = d$vin)
}

# Rates S1 en lactation (projet GDM-MILK, ANR-22-CE17-0039) : métabolome (M, 514) et indicateurs
# (C, 18). Métabolites : log10 (valeur nulle remplacée par la moitié du plus petit positif de la
# variable), puis échelle de Pareto ; indicateurs réduits. Centrage et inertie 1 : pretraiter().
lire_rates <- function() {
  d <- lire("rates_S1_L12.csv")
  X <- as.matrix(d[, -(1:2)])
  M <- X[, 1:514]; C <- X[, 515:532]
  for (j in seq_len(ncol(M))) {
    pos <- M[M[, j] > 0, j]
    M[M[, j] <= 0, j] <- if (length(pos)) min(pos) / 2 else 1
  }
  M <- log10(M)
  M <- sweep(M, 2, sqrt(apply(M, 2, sd)), "/")
  C <- sweep(C, 2, apply(C, 2, sd), "/")
  list(blocs = list(M, C), regime = d$regime, ids = d$rate)
}

# Matrice gaussienne nl x nc remplie colonne par colonne, et orthonormalisation de Gram-Schmidt
# (deux passages) : mêmes tirages et mêmes bases qu'en Python.
normales_mat <- function(gen, nl, nc) matrix(pm_normales(gen, nl * nc), nl, nc)
gram_schmidt <- function(Z) {
  Q <- as.matrix(Z)
  for (j in seq_len(ncol(Q))) {
    for (passe in 1:2) for (i in seq_len(j - 1)) Q[, j] <- Q[, j] - sum(Q[, i] * Q[, j]) * Q[, i]
    Q[, j] <- Q[, j] / sqrt(sum(Q[, j]^2))
  }
  Q
}
