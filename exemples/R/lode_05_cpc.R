# LODE, section 7.3 : LODE et lecture directe de la base CPC (covariances et seuils d'origine).
source("commun.R")
cov <- lire("cpc_covariances.csv"); se <- lire("cpc_seuils.csv"); ve <- lire("cpc_verites.csv", colClasses = c(etiquette = "character"))
evaluer <- function(V, etiq, vraies, p) {
  cles <- vapply(etiq, function(L) paste(L, collapse = ","), "")
  trouve <- mean(vapply(vraies, function(v) { idx <- which(cles == v$cle); length(idx) > 0 && sum(crossprod(V[, idx, drop = FALSE], v$t)^2) >= 0.9 }, TRUE))
  att <- c(table(vapply(vraies, function(v) v$cle, "")), setNames(p - length(vraies), ""))
  obs <- table(cles)
  list(trouve, setequal(names(att), names(obs)) && all(att[names(obs)] == obs))
}
for (plan in c("modele_CPC", "ideal_phi0", "ideal_phi15", "ideal_phi30", "ideal_phi45")) {
  lignes <- t(sapply(1:50, function(r) {
    s <- se[se$plan == plan & se$simulation == r, ]; K <- nrow(s)
    A <- lapply(seq_len(K), function(k) { e <- cov[cov$plan == plan & cov$simulation == r & cov$groupe == k, ]; symetrique(e$i, e$j, e$valeur, max(e$j)) })
    p <- nrow(A[[1]]); th <- s$seuil
    v <- if (plan == "modele_CPC") ve[ve$plan == plan & ve$simulation == r, ] else ve[ve$plan == "ideal", ]
    vraies <- lapply(seq_len(nrow(v)), function(i) list(cle = paste(which(strsplit(sprintf("%s", v$etiquette[i]), "")[[1]] == "1"), collapse = ","),
                                                       t = as.numeric(v[i, paste0("x", 1:p)])))
    res <- lode(A, th, departs = 20, graine = r)
    prep <- list(covariances = A, Xk = lapply(s$effectif, function(n) matrix(0, n, p)))
    B <- cpc(prep); eB <- etiquettes(A, B, th)
    eL <- evaluer(res$base, res$etiquettes, vraies, p); eC <- evaluer(B, eB, vraies, p)
    f <- fiche(B, list(base = res$base, etiquettes = res$etiquettes, base_methode = B))
    c(eL[[1]], eC[[1]], eL[[2]], eC[[2]], est_etiquetage(A, B, th), sum(f$melangee))
  }))
  m <- colMeans(lignes)
  cat(sprintf("%-12s structures %.2f / %.2f ; motif %.0f / %.0f %% ; lecture CPC = étiquetage %.0f %% ; CPC mélangées %.2f\n",
              plan, m[1], m[2], 100 * m[3], 100 * m[4], 100 * m[5], m[6]))
}
