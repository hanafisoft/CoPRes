# LODE, section 6.3, tableaux 7 et 8 : CoPRes, JIVE et AJIVE dans le vrai sous-espace de signal.
# Même calcul que python/lode_08_jive.py (voir son en-tête) : mêmes générateurs, mêmes graines,
# mêmes réglages, donc mêmes résultats.
#   CoPRes : seuils_perm par défaut, copres(X, seuils, U = base de S).
#   JIVE  : jive (r.jive 2.4), orth_indiv = FALSE ; rangs vrais, ou estimés par permutation.
#   AJIVE : ajive (mvlearn 0.5) ; rangs vrais, ou estimés (coude, bornes de Wedin et des directions
#           aléatoires, seuil des valeurs singulières).
# Usage (depuis exemples/R) : N <- 50 ; source("lode_08_jive.R")   (N <- 1 pour un essai)
source("commun.R")
if (!exists("N")) N <- 50
SEUIL_COS2 <- 0.9

source("simulations.R")

# ----------------------------------------------------------------------------- analyse
retrouvees <- function(espaces, verite) {
  ok <- 0
  for (v in verite) {
    B <- espaces[[cle(v$L)]]
    if (!is.null(B) && ncol(B) > 0) {
      u <- v$t / sqrt(sum(v$t^2))
      if (sum((t(B) %*% u)^2) >= SEUIL_COS2) ok <- ok + 1
    }
  }
  ok / length(verite)
}

espaces_jive <- function(res, K) {
  e <- list()
  if (ncol(res$scores_joint) > 0) e[[cle(seq_len(K))]] <- res$scores_joint
  for (k in seq_len(K)) if (ncol(res$scores_individuels[[k]]) > 0) e[[cle(k)]] <- res$scores_individuels[[k]]
  e
}

analyser <- function(jeu) {
  X <- jeu$X; verite <- jeu$verite; rJ <- jeu$rJ; rA <- jeu$rA; K <- length(X)
  S <- gram_schmidt(do.call(cbind, lapply(verite, `[[`, "t")))
  th <- seuils_perm(X)
  lecture <- lapply(verite, function(v) {
    u <- v$t / sqrt(sum(v$t^2))
    list(t = v$t, L = which(vapply(seq_len(K), function(k) sum((t(X[[k]]) %*% u)^2), 0) >= th))
  })
  cop <- copres(X, th, U = S)
  esp_c <- list()
  for (h in seq_along(cop$etiquettes)) {
    k <- cle(cop$etiquettes[[h]])
    esp_c[[k]] <- cbind(esp_c[[k]], cop$base[, h])
  }
  Y <- lapply(X, function(x) S %*% (t(S) %*% x))
  jv <- jive(Y, rJ, rA, orth_indiv = FALSE)
  je <- jive(Y, nperm = 100, orth_indiv = FALSE)
  av <- ajive(Y, rJ + rA, rang_joint = rJ, rangs_individuels = rA)
  ae <- suppressWarnings(ajive(Y, "coude"))
  esp <- list(copres = esp_c, jive_vrais = espaces_jive(jv, K), jive_estimes = espaces_jive(je, K),
              ajive_vrais = espaces_jive(av, K), ajive_estimes = espaces_jive(ae, K))
  out <- list()
  for (m in names(esp)) {
    out[[paste0(m, "_plan")]] <- retrouvees(esp[[m]], verite)
    out[[paste0(m, "_seuils")]] <- retrouvees(esp[[m]], lecture)
  }
  out$rangs_jive_estimes <- paste(c(je$rang_joint, je$rangs_individuels), collapse = "/")
  out$rangs_ajive_estimes <- paste(c(ae$rang_joint, ae$rangs_individuels), collapse = "/")
  out$signal_ajive_estime <- paste(ae$rangs_signal, collapse = "/")
  out
}

# ----------------------------------------------------------------------------- programme
COLS <- c("copres", "jive_vrais", "jive_estimes", "ajive_vrais", "ajive_estimes")
taches <- c(lapply(SITUATIONS, function(s) list(type = "jive", nom = s[[1]], arg = s[[2]])),
            lapply(seq_along(PLANS), function(p) list(type = "plan", nom = PLANS[[p]]$nom, arg = p)))
debut <- Sys.time()
resultats <- list()
for (tache in taches) {
  res <- lapply(seq_len(N), function(g) {
    jeu <- if (tache$type == "jive") jeu_jive(tache$arg, g) else jeu_plan(tache$arg, g)
    analyser(jeu)
  })
  moy <- sapply(c(paste0(COLS, "_plan"), paste0(COLS, "_seuils")), function(k) mean(vapply(res, `[[`, 0, k)))
  resultats[[tache$nom]] <- list(type = tache$type, moy = moy, res = res)
  cat(sprintf("%-42s %s   (%.0f s)\n", tache$nom, paste(sprintf("%.2f", moy[paste0(COLS, "_plan")]), collapse = "  "),
              as.numeric(difftime(Sys.time(), debut, units = "secs"))))
}
cat("\nTableau 8 : vérité du plan / lecture par les seuils\n")
for (nom in names(resultats)) if (resultats[[nom]]$type == "plan") {
  m <- resultats[[nom]]$moy
  cat(sprintf("%-45s %s\n", nom, paste(sprintf("%.2f / %.2f", m[paste0(COLS, "_plan")], m[paste0(COLS, "_seuils")]), collapse = "  ")))
}
