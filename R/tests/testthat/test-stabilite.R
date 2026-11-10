# Tests de la stabilité par bootstrap (valeurs de référence calculées par CoPRes 0.4.0 en Python).

test_that("stabilité multigroupe : notes, 10 échantillons", {
  nt <- lire_notes(1); P <- pretraiter_groupes(nt$X, nt$groupes)
  th <- seuils_groupes(P, nperm = 50, ndir = 50, graine = 7)
  c <- copres_groupes(P, th, departs = 20)
  s <- stabilite_groupes(c, P, B = 10)
  expect_equal(s$motif_retrouve, 1)
  expect_true(all(s$strict == 1))
})

test_that("stabilité multibloc : concordance avec Python", {
  mat <- function(g, r, c) matrix(pm_normales(g, r * c), r, c)
  g <- pm_gen(2026); n <- 30; T4 <- mat(g, n, 4)
  X <- pretraiter(list(T4[, 1:2] %*% mat(g, 2, 15) + 0.5 * mat(g, n, 15),
                       T4[, c(1, 3)] %*% mat(g, 2, 15) + 0.5 * mat(g, n, 15),
                       0.4 * T4[, 1, drop = FALSE] %*% mat(g, 1, 15) + T4[, 4, drop = FALSE] %*% mat(g, 1, 15) + 0.5 * mat(g, n, 15)))
  TH <- seuils_perm(X, nperm = 50, ndir = 50, graine = 3)
  c <- copres(X, TH, R = 4, methode = "CPCA", departs = 20)
  s <- stabilite_blocs(c, X, B = 5, nperm = 20, ndir = 50, departs = 10)
  expect_equal(unname(s$strict[c("{1,2,3}", "{1,3}", "{1}", "{2}")]), c(0.8, 0.4, 0.4, 1.0))
  expect_equal(s$motif_retrouve, 0.2)
  expect_true(all(s$strict <= s$etiquette & s$etiquette <= s$type))
})
