# Tests des données des graphiques et, si ggplot2 est installé, de la construction des figures.

nt <- lire_notes(1)
P <- pretraiter_groupes(nt$X, nt$groupes)
CG <- copres_groupes(P, seuils_groupes(P, nperm = 20, ndir = 20, graine = 7), departs = 10)
mat <- function(g, r, c) matrix(pm_normales(g, r * c), r, c)
g <- pm_gen(2026); n <- 30; T4 <- mat(g, n, 4)
X <- pretraiter(list(T4[, 1:2] %*% mat(g, 2, 15) + 0.5 * mat(g, n, 15),
                     T4[, c(1, 3)] %*% mat(g, 2, 15) + 0.5 * mat(g, n, 15),
                     0.4 * T4[, 1, drop = FALSE] %*% mat(g, 1, 15) + T4[, 4, drop = FALSE] %*% mat(g, 1, 15) + 0.5 * mat(g, n, 15)))
TH <- seuils_perm(X, nperm = 50, ndir = 50, graine = 3)
C <- copres(X, TH, R = 4, departs = 10)

test_that("données des graphiques et langue", {
  d <- donnees_saliences(CG)
  expect_equal(nrow(d), 24)
  expect_true("résiduelle" %in% d$etiquette)
  expect_true("residual" %in% donnees_saliences(CG, langue = "en")$etiquette)
  langue("en"); expect_true("residual" %in% donnees_motif(CG)$partie); langue("fr")
  f <- donnees_fiche(fiche(NULL, C))
  expect_equal(sum(f$part[f$composante == 1]), 1, tolerance = 1e-9)
  expect_equal(nrow(donnees_coherence(C, c(1, 2), nb_points = 11)), 3 * 2 * 11)
  expect_equal(theme_copres(taille = 10)$taille, 10)
  expect_equal(theme_copres(defaut = TRUE)$taille, 9)
  s <- sensibilite(X, TH, R_liste = 3:4, facteurs = c(1, 1.5), departs = 3)
  expect_false(any(donnees_sensibilite(s)$different[donnees_sensibilite(s)$facteur == 1]))
})

test_that("figures ggplot", {
  skip_if_not_installed("ggplot2")
  figs <- list(graph_saliences(C), graph_saliences(CG), graph_coherence(C, c(1, 2), bloc = 1),
               graph_fiche(fiche(NULL, C)), graph_motif(CG), graph_scores(C), graph_scores(CG, prep = P),
               graph_scores_parties(CG, prep = P, partie = 3L), graph_loadings(C, X = X), graph_loadings(C, type = "spectre"),
               graph_loadings(CG, noms_variables = nt$variables), graph_contributions(CG, 3, noms_variables = nt$variables),
               graph_stabilite(stabilite_groupes(CG, P, B = 3)),
               graph_sensibilite(sensibilite(X, TH, R_liste = 3:4, facteurs = c(1, 1.5), departs = 3)),
               graph_bilan(X, TH, 4, departs = 3))
  for (p in figs) {
    expect_s3_class(p, "ggplot")
    expect_no_error(ggplot2::ggplot_build(p))
  }
  expect_length(graph_coherence(C, c(1, 2)), 3)
  expect_type(graph_scores_parties(CG, prep = P), "list")
})
