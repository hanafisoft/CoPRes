# LODE, sections 2.4 et 4.2 : bases de R^4 et étiquetage idéal de deux matrices qui ne commutent pas.
library(CoPRes)
c40 <- cos(40 * pi / 180); s40 <- sin(40 * pi / 180); t4 <- c(c40, s40, 0, 0)
A <- list(diag(c(0.50, 0, 0.45, 0.05)), 0.50 * t4 %o% t4 + diag(c(0, 0, 0.05, 0.45)))
theta <- c(0.10, 0.10); r <- sqrt(0.5)
bases <- list(U1 = diag(4),
              U2 = rbind(c(c40, -s40, 0, 0), c(s40, c40, 0, 0), c(0, 0, 1, 0), c(0, 0, 0, 1)),
              U3 = rbind(c(1, 0, 0, 0), c(0, 1, 0, 0), c(0, 0, r, r), c(0, 0, r, -r)))
for (nom in names(bases)) {
  U <- bases[[nom]]; S <- saliences(A, U)
  cat(nom, ":", texte_motif(motif(etiquettes(A, U, theta))), "; étiquetage :", est_etiquetage(A, U, theta), "; ",
      paste(sprintf("(%.2f ; %.2f)", S[1, ], S[2, ]), collapse = ", "), "\n")
}
print(lode(A, theta))
A1 <- rbind(c(0.45, 0.15, 0, 0), c(0.15, 0.35, 0, 0), c(0, 0, 0.10, 0), c(0, 0, 0, 0.10))
A2 <- rbind(c(0.40, 0, -0.15, 0), c(0, 0.10, 0, 0.02), c(-0.15, 0, 0.40, 0), c(0, 0.02, 0, 0.10))
id <- ideal(list(A1, A2), c(0.15, 0.15))
cat("\nLes analyses à seuil commutent :", id$ideal, "\n")
for (p in id$parties) cat(" ", nom_etiquette(p$etiquette), ":", round(p$base, 3), "\n")
cat("borne de Ky Fan :", borne_kf(list(A1, A2), c(0.15, 0.15)), "\n")
print(lode(list(A1, A2), c(0.15, 0.15)))
