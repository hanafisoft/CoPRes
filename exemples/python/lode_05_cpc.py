"""LODE, section 7.3, tableau CPC : LODE et lecture directe de la base CPC sur deux plans de 50
simulations (covariances et seuils d'origine, fichiers cpc_*.csv). Structures retrouvées (carré du
cosinus >= 0,9 avec une partie de même étiquette), motif du plan retrouvé, lecture CPC = étiquetage,
composantes CPC mélangées."""
import types
import numpy as np
import CoPRes as cp
from commun import lire_csv

_, cov = lire_csv("cpc_covariances.csv")
_, se = lire_csv("cpc_seuils.csv")
_, ve = lire_csv("cpc_verites.csv")
COV = {}
for l in cov:
    k = (l[0], int(l[1]), int(l[2]))
    COV.setdefault(k, []).append((int(l[3]) - 1, int(l[4]) - 1, float(l[5])))
SEUIL = {(l[0], int(l[1]), int(l[2])): (float(l[4]), int(l[3])) for l in se}

def matrice(entrees):
    p = max(i for i, _, _ in entrees) + 1
    A = np.zeros((p, p))
    for i, j, v in entrees:
        A[i, j] = A[j, i] = v
    return A

def verites(plan, r):
    cle = ("modele_CPC", r) if plan == "modele_CPC" else ("ideal", 0)
    out = []
    for l in ve:
        if (l[0], int(l[1])) == cle:
            L = tuple(k for k, c in enumerate(l[3]) if c == "1")
            out.append((L, np.array([float(v) for v in l[4:] if v != ""])))
    return out

def evaluer(V, etiq, vraies, K, p):
    d = {}
    for h, L in enumerate(etiq):
        d.setdefault(L, []).append(h)
    trouve = sum(1 for L, t in vraies if L in d and np.sum((V[:, d[L]].T @ t) ** 2) >= 0.9)
    attendu = {L: 1 for L, _ in vraies}
    attendu[()] = p - len(vraies)
    return trouve / len(vraies), {L: len(i) for L, i in d.items()} == attendu

print(f"{'plan':22s} structures LODE / CPC   motif LODE / CPC   lecture CPC = étiquetage   CPC mélangées")
for plan in ["modele_CPC", "ideal_phi0", "ideal_phi15", "ideal_phi30", "ideal_phi45"]:
    lignes = []
    for r in range(1, 51):
        K = 3 if plan == "modele_CPC" else 2
        A = [matrice(COV[(plan, r, k)]) for k in range(1, K + 1)]
        th = [SEUIL[(plan, r, k)][0] for k in range(1, K + 1)]
        nk = [SEUIL[(plan, r, k)][1] for k in range(1, K + 1)]
        p = A[0].shape[0]
        prep = types.SimpleNamespace(covariances=A, Xk=[np.zeros((n, p)) for n in nk])
        res = cp.lode(A, th, departs=20, graine=r)
        B = cp.cpc(prep)
        eB = cp.etiquettes(A, B, th)
        vraies = verites(plan, r)
        fL, mL = evaluer(res.base, res.etiquettes, vraies, K, p)
        fC, mC = evaluer(B, eB, vraies, K, p)
        cop = types.SimpleNamespace(base=res.base, etiquettes=res.etiquettes, base_methode=B)
        f = cp.fiche(B, cop)
        lignes.append((fL, fC, mL, mC, cp.est_etiquetage(A, B, th), f.melangee.sum()))
    m = np.mean(np.array(lignes, dtype=float), axis=0)
    print(f"{plan:22s} {m[0]:.2f} / {m[1]:.2f}            {100 * m[2]:3.0f} / {100 * m[3]:3.0f} %"
          f"          {100 * m[4]:3.0f} %                    {m[5]:.2f}")
