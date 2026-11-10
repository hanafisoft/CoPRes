# Tests de la couche multigroupe sur les notes d'élèves (valeurs de référence calculées par
# CoPRes 0.3.0 en Python, le 10 novembre 2026).

PLAN <- "({A,B,C}, 1), ({A,B}, 1), ({A}, 1), ({C}, 1), (résiduelle, 4)"
nt <- lire_notes(1)
P <- pretraiter_groupes(nt$X, nt$groupes)
TH <- seuils_groupes(P, nperm = 50, ndir = 50, graine = 7)

test_that("données et prétraitement", {
  expect_equal(dim(nt$X), c(1500L, 8L))
  expect_equal(P$groupes, c("A", "B", "C"))
  expect_equal(nt$variables[5], "Français")
  W <- Reduce(`+`, Map(function(a, x) a * (nrow(x) - 1), P$covariances, P$Xk)) / (1500 - 3)
  expect_equal(diag(W), rep(1, 8), tolerance = 1e-12)
})

test_that("concordance avec Python", {
  expect_equal(TH, c(1.410596466850003, 1.153675180242619, 1.3706957140934921), tolerance = 1e-10)
  c <- copres_groupes(P, TH, departs = 20)
  expect_equal(c$bilan, 7.077536447057212, tolerance = 1e-8)
  expect_equal(c$borne, 7.505642793966811, tolerance = 1e-10)
  expect_equal(texte_motif(c$motif, c$groupes), PLAN)
  d <- discrimination(c, P, classement = FALSE)$parts_plan_discriminant
  expect_lt(abs(unname(d["résiduelle"]) - 1.6453169938919017), 1e-6)
  expect_lt(abs(unname(d["{A,B}"]) - 0.20019027752751048), 1e-6)
  expect_equal(sum(d), 2, tolerance = 1e-10)
  l <- lecture_groupes(P, cpc(P), TH)
  expect_equal(l$bilan, 7.068454780874832, tolerance = 1e-8)
  expect_equal(texte_motif(l$motif, P$groupes), PLAN)
  expect_true(l$coherente)
})

test_that("restriction et règle de la plus grande valeur propre", {
  P2 <- pretraiter_groupes(nt$X, nt$groupes, R = 4)
  th2 <- seuils_groupes(P2, nperm = 50, regle = "valeur_propre", graine = 7)
  expect_equal(th2, c(1.791081693655434, 1.3705786471785817, 1.797839649285621), tolerance = 1e-10)
  c2 <- copres_groupes(P2, th2)
  expect_equal(c2$bilan, 5.3362522109239325, tolerance = 1e-8)
  expect_equal(texte_motif(c2$motif, c2$groupes), "({A,B}, 1), ({A}, 1), ({C}, 1), (résiduelle, 1)")
})

test_that("article : trois situations (classement et part du plan discriminant)", {
  attendu <- list(c(0.343, 0.583, 0.30, 1.65), c(0.693, 0.734, 1.81, 0.16), c(0.676, 0.753, 0.04, 1.92))
  for (s in 1:3) {
    nts <- lire_notes(s); Ps <- pretraiter_groupes(nts$X, nts$groupes)
    th <- seuils_groupes(Ps, graine = 7); c <- copres_groupes(Ps, th, departs = 50)
    expect_equal(texte_motif(c$motif, c$groupes), PLAN)
    d <- discrimination(c, Ps); pp <- d$parts_plan_discriminant
    expect_equal(round(d$classement_lineaire, 3), attendu[[s]][1])
    expect_equal(round(d$classement_quadratique, 3), attendu[[s]][2])
    expect_lt(abs(unname(pp["{A,B}"] + pp["{A}"] + pp["{C}"]) - attendu[[s]][3]), 0.006)
    expect_lt(abs(unname(pp["résiduelle"]) - attendu[[s]][4]), 0.006)
  }
})
