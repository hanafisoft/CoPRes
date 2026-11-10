# Tests de la couche multibloc : propriétés de la factorisation CoPRes et concordance avec la
# version Python (valeurs de référence calculées par CoPRes 0.2.0 en Python, le 10 novembre 2026).

mat <- function(g, r, c) matrix(pm_normales(g, r * c), r, c)
donnees <- function() {
  g <- pm_gen(2026); n <- 30
  T4 <- mat(g, n, 4)
  X <- list(T4[, 1:2] %*% mat(g, 2, 15) + 0.5 * mat(g, n, 15),
            T4[, c(1, 3)] %*% mat(g, 2, 15) + 0.5 * mat(g, n, 15),
            0.4 * T4[, 1, drop = FALSE] %*% mat(g, 1, 15) + T4[, 4, drop = FALSE] %*% mat(g, 1, 15) + 0.5 * mat(g, n, 15))
  pretraiter(X)
}
X <- donnees()
TH <- seuils_perm(X, nperm = 50, ndir = 50, graine = 3)

REF <- list(
  seuils = c(0.062442126839843876, 0.07729766455175517, 0.06506439774960371),
  seuils_vp = c(0.25184971313176985, 0.45915685700218134, 0.30128052236855263),
  CPCA = list(2.0707660961275867, 2.147789149363547, "({1,2,3}, 1), ({1,3}, 1), ({1}, 1), ({2}, 1)", 4, 1.8361411015679512),
  ACOM = list(1.7656387359955377, 1.855227726329899, "({1,2,3}, 1), ({1,3}, 1), ({1}, 1), ({2}, 1)", 4, 1.6146947550864703),
  ACCPS = list(2.068164141051449, 2.1454787679474414, "({1,2,3}, 1), ({1,3}, 1), ({1}, 1), ({2}, 1)", 4, 1.8892729546517912))

test_that("générateur vectorisé = générateur pas à pas", {
  a <- pm_gen(5); b <- pm_gen(5)
  u <- pm_uniformes(a, 70000)
  v <- numeric(70000); for (i in 1:70000) v[i] <- pm_uniforme(b)
  expect_identical(u, v)
  expect_identical(a$etat, b$etat)
})

test_that("prétraitement", {
  for (x in X) {
    expect_equal(unname(colMeans(x)), rep(0, ncol(x)), tolerance = 1e-12)
    expect_equal(sqrt(sum(x^2)), 1, tolerance = 1e-12)
  }
})

test_that("seuils : concordance avec Python", {
  expect_equal(TH, REF$seuils, tolerance = 1e-10)
  expect_equal(seuils_perm(X, nperm = 50, regle = "valeur_propre", graine = 3), REF$seuils_vp, tolerance = 1e-10)
})

for (m in c("CPCA", "ACOM", "ACCPS")) {
  test_that(paste("méthode", m, ": concordance avec Python et propriétés"), {
    c <- copres(X, TH, R = 4, methode = m, departs = 20)
    ref <- REF[[m]]
    expect_equal(c$bilan, ref[[1]], tolerance = 1e-8)
    expect_equal(c$borne, ref[[2]], tolerance = 1e-10)
    expect_equal(texte_motif(c$motif), ref[[3]])
    expect_equal(sum(fiche(NULL, c)$melangee), ref[[4]])
    expect_equal(lecture_directe(X, c$base_methode, TH)$bilan, ref[[5]], tolerance = 1e-8)
    U <- c$base_methode; V <- c$base
    expect_equal(crossprod(V), diag(4), tolerance = 1e-10)
    expect_equal(V %*% t(V), U %*% t(U), tolerance = 1e-10)
    expect_equal(c$poids^2, c$saliences, tolerance = 1e-10)
    for (k in seq_along(X)) expect_equal(V %*% diag(c$poids[k, ]) %*% t(c$loadings[[k]]), V %*% t(V) %*% X[[k]], tolerance = 1e-10)
    ld <- lecture_directe(X, V, TH)
    expect_true(ld$coherente)
    expect_equal(ld$bilan, c$bilan, tolerance = 1e-9)
    f <- fiche(V, c)
    expect_false(any(f$melangee)); expect_false(any(f$diluee))
    expect_true(equivalence(c, c)$equivalentes)
  })
}

test_that("sensibilité", {
  s <- sensibilite(X, TH, R_liste = 3:4, facteurs = c(1, 1.5), departs = 5)
  expect_equal(nrow(s), 4)
  expect_true(all(!is.na(s$motif)))
})
