# Tests du module lode : exemples de l'article LODE et concordance avec la version Python
# (valeurs de référence calculées par CoPRes 0.1.0 en Python, le 9 novembre 2026).

c40 <- cos(40 * pi / 180); s40 <- sin(40 * pi / 180)
tt <- c(c40, s40, 0, 0)
A_BASES <- list(diag(c(0.50, 0, 0.45, 0.05)), 0.50 * tt %o% tt + diag(c(0, 0, 0.05, 0.45)))
TH <- c(0.10, 0.10)
U1 <- diag(4)
U2 <- rbind(c(c40, -s40, 0, 0), c(s40, c40, 0, 0), c(0, 0, 1, 0), c(0, 0, 0, 1))
r <- sqrt(0.5)
U3 <- rbind(c(1, 0, 0, 0), c(0, 1, 0, 0), c(0, 0, r, r), c(0, 0, r, -r))

test_that("générateur de Park et Miller : valeurs de référence", {
  g <- pm_gen(1)
  for (i in 1:10000) pm_uniforme(g)
  expect_equal(g$etat, 1043618065)
  g <- pm_gen(42)
  expect_equal(pm_uniformes(g, 3), c(0.00032870750889587566, 0.5245871020129822, 0.7354235321913956), tolerance = 1e-15)
  expect_equal(pm_normale(pm_gen(42)), -3.9573852295840655, tolerance = 1e-12)
})

test_that("rotation aléatoire orthogonale", {
  Q <- rotation_aleatoire(6, pm_gen(3))
  expect_equal(crossprod(Q), diag(6), tolerance = 1e-12)
})

test_that("tableau des trois bases de R^4", {
  expect_true(est_etiquetage(A_BASES, U1, TH))
  expect_true(est_etiquetage(A_BASES, U2, TH))
  expect_false(est_etiquetage(A_BASES, U3, TH))
  expect_equal(etiquettes(A_BASES, U3, TH), list(1:2, 2L, 1:2, 1:2))
  expect_equal(texte_motif(motif(etiquettes(A_BASES, U1, TH))), "({1,2}, 1), ({1}, 1), ({2}, 2)")
  expect_equal(texte_motif(motif(etiquettes(A_BASES, U2, TH))), "({1,2}, 1), ({1}, 2), ({2}, 1)")
})

test_that("B2BL rend un étiquetage et fait croître le critère", {
  r <- b2bl(U3, A_BASES, TH)
  expect_gte(r$remplacements, 1)
  expect_true(est_etiquetage(A_BASES, r$base, TH))
  expect_gt(critere(A_BASES, r$base, TH), critere(A_BASES, U3, TH))
})

test_that("LODE sur le tableau des bases", {
  res <- lode(A_BASES, TH, departs = 20)
  expect_equal(res$critere, 1.4, tolerance = 1e-10)
  expect_equal(res$borne, 1.5, tolerance = 1e-10)
  expect_equal(texte_motif(res$motif), "({1,2}, 1), ({1}, 1), ({2}, 2)")
  expect_gte(res$marge, -1e-9)
})

test_that("étiquetage idéal sans commutation", {
  A1 <- rbind(c(0.45, 0.15, 0, 0), c(0.15, 0.35, 0, 0), c(0, 0, 0.10, 0), c(0, 0, 0, 0.10))
  A2 <- rbind(c(0.40, 0, -0.15, 0), c(0, 0.10, 0, 0.02), c(-0.15, 0, 0.40, 0), c(0, 0.02, 0, 0.10))
  A <- list(A1, A2); th <- c(0.15, 0.15)
  id <- ideal(A, th)
  expect_true(id$ideal)
  expect_length(id$parties, 4)
  expect_equal(borne_kf(A, th), 1, tolerance = 1e-12)
  res <- lode(A, th)
  expect_equal(res$critere, 1, tolerance = 1e-9)
  expect_equal(texte_motif(res$motif), "({1,2}, 1), ({1}, 1), ({2}, 1), (résiduelle, 1)")
})

test_that("concordance avec la version Python sur des matrices aléatoires", {
  ref <- list(
    list(crit = 0.5865517738919718, borne = 0.6769492511618349, motif = "({1}, 1), ({2}, 2), ({3}, 1), (résiduelle, 2)"),
    list(crit = 0.4867764174646842, borne = 0.5531258107895524, motif = "({1}, 1), ({2}, 2), ({3}, 1), (résiduelle, 2)"),
    list(crit = 0.385927680718359, borne = 0.496860176494826, motif = "({1,2}, 1), ({2}, 1), ({3}, 2), (résiduelle, 2)"),
    list(crit = 0.49790411081336644, borne = 0.6219090221496192, motif = "({1,3}, 1), ({2}, 2), (résiduelle, 3)"),
    list(crit = 0.5811333060157389, borne = 0.6445234792622477, motif = "({1,3}, 1), ({1}, 1), ({2}, 1), (résiduelle, 3)"))
  for (g in 1:5) {
    gen <- pm_gen(g); n <- 6
    A <- lapply(1:3, function(k) {
      X <- sapply(1:8, function(j) pm_normales(gen, n))
      M <- X %*% t(X); M / sum(diag(M))
    })
    res <- lode(A, rep(1.5 / n, 3), departs = 10, graine = g)
    expect_equal(res$critere, ref[[g]]$crit, tolerance = 1e-8)
    expect_equal(res$borne, ref[[g]]$borne, tolerance = 1e-10)
    expect_equal(texte_motif(res$motif), ref[[g]]$motif)
    expect_lte(res$critere, res$borne + 1e-10)
    expect_true(est_etiquetage(A, res$base, rep(1.5 / n, 3)))
  }
})
