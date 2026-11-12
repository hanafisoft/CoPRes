# Tests de concurrents.R : AJIVE contre mvlearn 0.5, JIVE contre r.jive 2.4, et égalité avec la
# version Python (sorties de référence dans tests/testthat/ref_concurrents).
REF <- test_path("ref_concurrents")
RANGS <- list(a = list(J = 2, A = c(2, 2, 2), S = c(4, 4, 4)),
              b = list(J = 1, A = c(2, 2, 2), S = c(3, 3, 3)),
              c = list(J = 2, A = c(1, 1), S = c(3, 3)))

lire <- function(nom) as.matrix(utils::read.csv(file.path(REF, nom), header = FALSE))
ref <- function(nom) {
  if (!file.exists(file.path(REF, nom))) skip(paste("référence absente :", nom))
  lire(nom)
}
blocs <- function(j) {
  X <- lire(paste0(j, "_donnees.csv"))
  p <- as.integer(lire(paste0(j, "_blocs.csv")))
  fin <- cumsum(p); debut <- fin - p + 1
  lapply(seq_along(p), function(k) X[, debut[k]:fin[k], drop = FALSE])
}
ecart <- function(A, B) max(abs(unname(A) - unname(B)))

for (j in names(RANGS)) {
  r <- RANGS[[j]]
  test_that(paste("AJIVE à rangs donnés comme mvlearn, jeu", j), {
    a <- ajive(blocs(j), r$S, rang_joint = r$J)
    expect_lt(ecart(a$scores_joint %*% t(a$scores_joint), lire(paste0(j, "_ajive_scores_joint.csv"))), 1e-10)
    expect_lt(ecart(do.call(cbind, a$individuel), lire(paste0(j, "_ajive_individuel.csv"))), 1e-10)
    expect_equal(c(a$rang_joint, a$rangs_individuels), as.integer(lire(paste0(j, "_ajive_rangs.csv"))))
  })
  test_that(paste("AJIVE à rangs estimés comme la version Python, jeu", j), {
    a <- ajive(blocs(j), r$S, n_wedin = 200, n_hasard = 200)
    py <- as.numeric(lire(paste0(j, "_py_ajive_estime.csv")))
    expect_lt(max(abs(c(a$seuil_wedin, a$seuil_hasard) - py[1:2])), 1e-10)
    expect_equal(c(a$rang_joint, a$rangs_individuels), as.integer(py[-(1:2)]))
    c <- ajive(blocs(j), n_wedin = 200, n_hasard = 200)
    expect_equal(c(c$rangs_signal, c$rang_joint, c$rangs_individuels),
                 as.integer(lire(paste0(j, "_py_ajive_coude.csv"))))
  })
  test_that(paste("JIVE à rangs donnés comme la version Python, jeu", j), {
    g <- jive(blocs(j), r$J, r$A)
    expect_lt(ecart(do.call(cbind, g$joint), lire(paste0(j, "_py_jive_joint.csv"))), 1e-9)
    expect_lt(ecart(do.call(cbind, g$individuel), lire(paste0(j, "_py_jive_individuel.csv"))), 1e-9)
  })
  test_that(paste("JIVE par permutation comme la version Python, jeu", j), {
    g <- jive(blocs(j), nperm = 100)
    expect_equal(c(g$rang_joint, g$rangs_individuels), as.integer(lire(paste0(j, "_py_jive_perm_rangs.csv"))))
  })
  test_that(paste("JIVE à rangs donnés comme r.jive, jeu", j), {
    Jr <- ref(paste0(j, "_rjive_joint.csv"))
    Ar <- ref(paste0(j, "_rjive_individuel.csv"))
    g <- jive(blocs(j), r$J, r$A)
    e <- max(abs(do.call(cbind, g$donnees)))
    expect_lt(ecart(do.call(cbind, g$joint), Jr), 1e-6 * e)
    expect_lt(ecart(do.call(cbind, g$individuel), Ar), 1e-6 * e)
  })
}

test_that("AJIVE sans centrage", {
  X <- lapply(blocs("a"), function(x) sweep(x, 2, colMeans(x)))
  r1 <- ajive(X, c(4, 4, 4), rang_joint = 2)
  r2 <- ajive(X, c(4, 4, 4), rang_joint = 2, centrer = FALSE)
  expect_lt(ecart(r1$scores_joint %*% t(r1$scores_joint), r2$scores_joint %*% t(r2$scores_joint)), 1e-12)
  Y <- lapply(X, function(x) x + 3)
  r3 <- ajive(Y, c(4, 4, 4), rang_joint = 2, centrer = FALSE)
  expect_lt(ecart(r3$joint[[1]], r3$scores_joint %*% t(r3$scores_joint) %*% Y[[1]]), 1e-10)
})
