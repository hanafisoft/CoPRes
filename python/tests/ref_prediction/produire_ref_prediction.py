"""Jeu de test et sorties de référence de la version Python du module prediction, pour vérifier
que la version R donne les mêmes résultats."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import CoPRes as cp
from CoPRes.prediction import (information_groupe, plsda, lda_loo, discrimination_parties,
                               variance_groupe, validation_croisee)

REF = os.path.dirname(os.path.abspath(__file__))


def jeu():
    """Deux blocs de 24 individus (3 groupes de 8) : une structure commune liée aux groupes, une
    propre au bloc 1 liée aux groupes, une propre au bloc 2 sans lien, et du bruit."""
    gen = cp.ParkMiller(11)
    n = 24
    g = np.repeat(["a", "b", "c"], 8)
    eff = {"a": -1.0, "b": 0.0, "c": 1.0}
    z = np.array([eff[v] for v in g])
    t1 = z + gen.normales(n)
    t2 = np.array([0.8 * (v == "b") for v in g]) + gen.normales(n)
    t3 = gen.normales(n)
    X1 = np.outer(t1, gen.normales(12)) + np.outer(t2, gen.normales(12)) + 0.8 * gen.normales(n * 12).reshape(12, n).T
    X2 = np.outer(t1, gen.normales(9)) + np.outer(t3, gen.normales(9)) + 0.8 * gen.normales(n * 9).reshape(9, n).T
    return [X1, X2], g


def ecrire(nom, M):
    np.savetxt(os.path.join(REF, nom), np.atleast_2d(M), delimiter=",", fmt="%.17g")


if __name__ == "__main__":
    B, g = jeu()
    ecrire("donnees.csv", np.hstack(B))
    ecrire("blocs.csv", [B[0].shape[1], B[1].shape[1]])
    with open(os.path.join(REF, "groupes.csv"), "w") as f:
        f.write("\n".join(g) + "\n")
    X = cp.pretraiter(B)
    th = cp.seuils_perm(X, nperm=50)
    c = cp.copres(X, th, R=3)
    inf = information_groupe(c, g)
    ecrire("seuils.csv", th)
    ecrire("information.csv", [inf.hors] + list(inf.contribution_propre) + list(inf.portee) + sorted(inf.parts.values()))
    U = plsda(X, g, 3)
    ecrire("plsda_projecteur.csv", U @ U.T)
    ecrire("lda.csv", [lda_loo(c.base_methode, g), lda_loo(c.base[:, :1], g)])
    d = discrimination_parties(c, g, n_perm=49)
    ecrire("discrimination.csv", [d.taux["S"], d.p["S"]])
    ecrire("variance.csv", variance_groupe(X, g))
    v = validation_croisee(B, g, 3, nperm=10, ndir=50, departs=5)
    ecrire("validation.csv", list(v.taux.values()))
    print(c, inf, d, v, sep="\n")
