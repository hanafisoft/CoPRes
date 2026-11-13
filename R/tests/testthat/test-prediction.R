# Tests du module prediction : mêmes résultats que la version Python (tests/testthat/ref_prediction).
REF <- test_path("ref_prediction")
lire <- function(nom) as.matrix(utils::read.csv(file.path(REF, nom), header = FALSE))
donnees <- function() {
  X <- lire("donnees.csv"); p <- as.integer(lire("blocs.csv"))
  g <- readLines(file.path(REF, "groupes.csv"))
  list(B = list(X[, 1:p[1]], X[, p[1] + 1:p[2]]), g = g)
}
d <- donnees()
X <- pretraiter(d$B)
th <- seuils_perm(X, nperm = 50)
cop <- copres(X, th, R = 3)

test_that("seuils comme Python", {
  expect_lt(max(abs(th - as.numeric(lire("seuils.csv")))), 1e-12)
})

test_that("information de groupe : somme à 1 et valeurs de Python", {
  inf <- information_groupe(cop, d$g)
  expect_lt(abs(sum(inf$parts) + inf$hors - 1), 1e-12)
  val <- c(inf$hors, inf$contribution_propre, inf$portee, sort(inf$parts))
  expect_lt(max(abs(val - as.numeric(lire("information.csv")))), 1e-10)
})

test_that("PLS-DA comme Python", {
  U <- plsda(X, d$g, 3)
  expect_lt(max(abs(crossprod(U) - diag(3))), 1e-12)
  expect_lt(max(abs(U %*% t(U) - unname(lire("plsda_projecteur.csv")))), 1e-10)
})

test_that("analyse discriminante comme Python", {
  ref <- as.numeric(lire("lda.csv"))
  expect_equal(c(lda_loo(cop$base_methode, d$g), lda_loo(cop$base[, 1, drop = FALSE], d$g)), ref)
  g <- rep(c("a", "b"), each = 5)
  expect_equal(lda_loo(matrix(c(0:4, 100:104)), g), 1)
})

test_that("discrimination par partie et variance due aux groupes comme Python", {
  dp <- discrimination_parties(cop, d$g, n_perm = 49)
  expect_equal(unname(c(dp$taux["S"], dp$p["S"])), as.numeric(lire("discrimination.csv")))
  expect_lt(max(abs(variance_groupe(X, d$g) - as.numeric(lire("variance.csv")))), 1e-12)
})

test_that("validation croisée comme Python", {
  v <- validation_croisee(d$B, d$g, 3, nperm = 10, ndir = 50, departs = 5)
  expect_equal(unname(v$taux), as.numeric(lire("validation.csv")))
})
