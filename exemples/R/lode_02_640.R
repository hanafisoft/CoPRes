# LODE, section 5.5, tableau 4 : algorithme LODE sur les 640 jeux simulés (matrices et seuils
# d'origine). Long en R : choisir nb (jeux par couple (n, K), 40 au maximum).
source("commun.R")
nb <- 5
mat <- lire("sim640_matrices.csv"); se <- lire("sim640_seuils.csv")
res <- list()
for (j in sort(unique(se$jeu))) {
  s <- se[se$jeu == j, ]; K <- s$K[1]; n <- s$n[1]
  if (sum(unique(se$jeu[se$K == K & se$n == n]) <= j) > nb) next
  A <- lapply(seq_len(K), function(k) { e <- mat[mat$jeu == j & mat$k == k, ]; symetrique(e$i, e$j, e$valeur, n) })
  t0 <- Sys.time(); lode(A, s$seuil, departs = 20, graine = j); t20 <- as.numeric(Sys.time() - t0, units = "secs")
  r <- lode(A, s$seuil, departs = 50, depart_propre = TRUE, graine = j)
  c <- r$criteres_departs; best <- max(c); hit <- c >= best - 1e-9 * max(1, abs(best)); ordre <- c(1, 3:51)
  res[[length(res) + 1]] <- data.frame(n = n, conc = hit[1], gap = 1 - best / r$borne, des5 = any(hit[ordre[1:5]]),
                                       des20 = any(hit[ordre[1:20]]), t20 = t20)
}
res <- do.call(rbind, res)
print(do.call(rbind, lapply(split(res, res$n), function(d) data.frame(n = d$n[1], concentree = mean(d$conc),
      ecart_med = median(d$gap), ecart_max = max(d$gap), des5 = mean(d$des5), des20 = mean(d$des20), t20 = median(d$t20)))))
