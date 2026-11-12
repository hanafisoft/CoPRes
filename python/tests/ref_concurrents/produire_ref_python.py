"""Sorties de la version Python de CoPRes, pour vérifier que la version R donne les mêmes."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from CoPRes.concurrents import jive, ajive

REF = os.path.dirname(os.path.abspath(__file__))
RANGS = {"a": (2, [2, 2, 2], [4, 4, 4]), "b": (1, [2, 2, 2], [3, 3, 3]), "c": (2, [1, 1], [3, 3])}


def lire(nom):
    return np.loadtxt(os.path.join(REF, nom), delimiter=",", ndmin=2)


def ecrire(nom, M):
    np.savetxt(os.path.join(REF, nom), np.atleast_2d(M), delimiter=",", fmt="%.17g")


for j, (rJ, rA, rS) in RANGS.items():
    X = lire(f"{j}_donnees.csv")
    p = lire(f"{j}_blocs.csv").ravel().astype(int)
    b = np.cumsum(np.r_[0, p])
    B = [X[:, b[k]:b[k + 1]] for k in range(len(p))]
    g = jive(B, rJ, rA)
    ecrire(f"{j}_py_jive_joint.csv", np.hstack(g.joint))
    ecrire(f"{j}_py_jive_individuel.csv", np.hstack(g.individuel))
    q = jive(B, nperm=100)
    ecrire(f"{j}_py_jive_perm_rangs.csv", [q.rang_joint] + q.rangs_individuels)
    a = ajive(B, rS, n_wedin=200, n_hasard=200)
    ecrire(f"{j}_py_ajive_estime.csv", [a.seuil_wedin, a.seuil_hasard, a.rang_joint] + a.rangs_individuels)
    c = ajive(B, n_wedin=200, n_hasard=200)
    ecrire(f"{j}_py_ajive_coude.csv", c.rangs_signal + [c.rang_joint] + c.rangs_individuels)
    print(j, "jive perm", q.rang_joint, q.rangs_individuels, "| ajive", a.rang_joint, a.rangs_individuels,
          "| coude", c.rangs_signal, c.rang_joint, c.rangs_individuels)
