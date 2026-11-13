# Article appliqué, section 4.1 : les mêmes variables dans plusieurs groupes. Même calcul que
# python/appli_03_groupes.py. Lycées simulés (données du paquet) ; HELIX si les fichiers
# ../donnees/helix_chimie.csv et ../donnees/helix_urine.csv ont été produits par helix_extraire.R.
# Usage (depuis exemples/R) : source("appli_03_groupes.R")
source("commun.R")

anova_modules <- function(X, g) sum(apply(X, 2, function(x) anova(lm(x ~ factor(g)))[["Pr(>F)"]][1] < 0.05))

cat("Lycées simulés\n")
for (s in 1:3) {
  d <- lire_notes(s)
  P <- pretraiter_groupes(d$X, d$groupes)
  th <- seuils_groupes(P, graine = 7)
  cop <- copres_groupes(P, th, departs = 50)
  di <- discrimination(cop, P)
  pp <- di$parts_plan_discriminant
  et <- attr(CoPRes:::.groupes(cop$etiquettes), "etiq")
  sig <- sum(pp[vapply(et, function(L) length(L) >= 1 && length(L) < length(P$groupes), TRUE)])
  res <- sum(pp[vapply(et, length, 1L) == 0])
  cat(sprintf("  situation %d : motif %s\n     modules qui diffèrent : %d ; classement %.0f / %.0f %% ; partagée et propres %.2f, résiduelle %.2f\n",
              s, texte_motif(cop$motif, P$groupes), anova_modules(d$X, d$groupes), 100 * di$classement_lineaire,
              100 * di$classement_quadratique, sig, res))
  if (s == 1) print(stabilite_groupes(cop, P, B = 100, seuil = 0.72))
}
v <- lire("notes_80eleves.csv")
Pv <- pretraiter_groupes(as.matrix(v[, -1]), v[[1]])
cv <- copres_groupes(Pv, seuils_groupes(Pv, graine = 7), departs = 50)
cat("  80 élèves par lycée : motif de l'échantillon", texte_motif(cv$motif, cv$groupes), "\n")
print(stabilite_groupes(cv, Pv, B = 100, seuil = 0.72))

type_partie <- function(L, K) if (length(L) == K) "commune" else if (length(L) > 1) "partagée" else if (length(L) == 1) "propre" else "résiduelle"

helix_jeu <- function(nom, fichier, B = 100) {
  d <- lire(fichier)
  d <- d[order(d$cohorte, method = "radix"), ]
  X <- as.matrix(d[, -(1:2)]); g <- as.character(d$cohorte); var <- names(d)[-(1:2)]
  P <- pretraiter_groupes(X, g); K <- length(P$groupes)
  th <- seuils_groupes(P, graine = 2026)
  cat(sprintf("\nHELIX, %s : %d enfants, %d variables ; seuils %.2f à %.2f\n", nom, nrow(X), ncol(X), min(th), max(th)))
  cop <- copres_groupes(P, th, departs = 50); print(cop)
  gr <- CoPRes:::.groupes(cop$etiquettes); et <- attr(gr, "etiq")
  dims <- tapply(vapply(gr, length, 1L), vapply(et, type_partie, "", K = K), sum); print(dims)
  P0 <- eigen(crossprod(P$Xs) / (nrow(X) - 1), symmetric = TRUE)$vectors
  for (m in list(list("ACP", P0), list("CPC", cpc(P)))) {
    f <- fiche(m[[2]], list(base = cop$base_analyse, etiquettes = cop$etiquettes, base_methode = m[[2]]))
    ld <- lecture_groupes(P, m[[2]], th)
    cat(sprintf("  %s : cohérente %s ; mélangées %d, diluées %d\n", m[[1]], ld$coherente, sum(f$melangee), sum(f$diluee)))
  }
  sc <- P$Xs %*% cop$base
  for (i in seq_along(gr)) {
    L <- et[[i]]; if (!length(L)) next
    idx <- gr[[i]]; V <- cop$base[, idx, drop = FALSE]
    contrib <- rowSums(V^2) / length(idx); top <- order(-contrib)[1:5]
    eta <- vapply(idx, function(h) { y <- sc[, h]; sum(vapply(P$groupes, function(k) sum(g == k) * (mean(y[g == k]) - mean(y))^2, 0)) / sum((y - mean(y))^2) }, 0)
    cat(sprintf("  %-10s %-14s dim %d : %s ; part de variance due à la cohorte %s\n", type_partie(L, K), nom_etiquette(L, P$groupes),
                length(idx), paste(sprintf("%s %.0f %%", var[top], 100 * contrib[top]), collapse = ", "),
                paste(sprintf("%.2f", eta), collapse = ", ")))
  }
  print(stabilite_groupes(cop, P, B = B, departs = 5, seuil = 0.72))
}
for (h in list(c("expositions chimiques", "helix_chimie.csv"), c("métabolome urinaire", "helix_urine.csv"))) {
  if (file.exists(chemin(h[2]))) helix_jeu(h[1], h[2]) else cat("HELIX :", h[2], "absent (voir helix_extraire.R)\n")
}
