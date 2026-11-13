"""LODE, section 7.2 : notes d'élèves simulées (trois lycées, mêmes moyennes). Étiquetage,
stabilité sur 100 rééchantillonnages bootstrap, et deux variantes (80 élèves par lycée ;
co-variations scientifique et de projet d'intensité 1,6)."""
import numpy as np
import CoPRes as cp
from commun import lire_csv

def lire(nom):
    tete, lignes = lire_csv(nom)
    return np.array([[float(v) for v in l[1:]] for l in lignes]), np.array([l[0] for l in lignes])

PLAN = "({A,B,C}, 1), ({A,B}, 1), ({A}, 1), ({C}, 1), (résiduelle, 4)"
X, g, var = cp.lire_notes(1)
P = cp.pretraiter_groupes(X, g)
th = cp.seuils_groupes(P, graine=7)
c = cp.copres_groupes(P, th, departs=50)
print("seuils :", np.round(th, 3))
print(c)
for L, idx in c.parties().items():
    v = c.base[:, idx[0]]
    print(f"  {cp.nom_etiquette(L, c.groupes):12s}", ", ".join(f"{var[j]} {v[j]:+.2f}" for j in np.argsort(-np.abs(v))[:4]))
s = cp.stabilite_groupes(c, P, B=100, seuil=0.8)
print(s)
for nom, f in (("80 élèves par lycée", "notes_80eleves.csv"), ("intensité 1,6", "notes_intensite16.csv")):
    Xv, gv = lire(f)
    Pv = cp.pretraiter_groupes(Xv, gv)
    cv = cp.copres_groupes(Pv, cp.seuils_groupes(Pv, graine=7), departs=50)
    sv = cp.stabilite_groupes(cv, Pv, B=100, seuil=0.8)
    plan_boot = sv.motifs.get(PLAN.replace("{", "{").replace("A,B,C", "1,2,3").replace("A,B", "1,2")
                              .replace("({A}", "({1}").replace("({C}", "({3}"), 0)
    print(f"{nom} : motif de l'échantillon = plan : {cp.texte_motif(cv.motif, cv.groupes) == PLAN} ; "
          f"motif du plan dans {plan_boot} rééchantillonnages sur 100")
