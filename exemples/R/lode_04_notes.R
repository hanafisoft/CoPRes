# LODE, section 7.2 : notes d'élèves ; étiquetage, stabilité, variantes 80 élèves et intensité 1,6.
source("commun.R")
PLAN <- "({A,B,C}, 1), ({A,B}, 1), ({A}, 1), ({C}, 1), (résiduelle, 4)"
nt <- lire_notes(1); P <- pretraiter_groupes(nt$X, nt$groupes)
th <- seuils_groupes(P, graine = 7); c <- copres_groupes(P, th, departs = 50)
print(round(th, 3)); print(c)
print(stabilite_groupes(c, P, B = 100, seuil = 0.8))
for (v in list(c("80 élèves par lycée", "notes_80eleves.csv"), c("intensité 1,6", "notes_intensite16.csv"))) {
  d <- lire(v[2]); Pv <- pretraiter_groupes(as.matrix(d[, -1]), d[[1]])
  cv <- copres_groupes(Pv, seuils_groupes(Pv, graine = 7), departs = 50)
  sv <- stabilite_groupes(cv, Pv, B = 100, seuil = 0.8)
  plan_num <- "({1,2,3}, 1), ({1,2}, 1), ({1}, 1), ({3}, 1), (résiduelle, 4)"
  cat(v[1], ": motif de l'échantillon = plan :", texte_motif(cv$motif, cv$groupes) == PLAN,
      "; motif du plan dans", if (plan_num %in% names(sv$motifs)) sv$motifs[[plan_num]] else 0, "rééchantillonnages sur 100\n")
}
