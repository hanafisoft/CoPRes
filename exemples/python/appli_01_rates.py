"""Article appliqué, section 2 : rates S1 en lactation, CPCA avec R = 3.

Tableau 1 (saliences et lectures selon le seuil), seuils calculés, cohérence des lectures
(directions extrêmes), tableau des 1 000 jeux tirés au hasard (fréquence des lectures, jeux
cohérents, bilan), figure de cohérence (plan commun le plus loin des seuils), tableau du bilan
de CoPRes et de la lecture concurrente après rotation dans son motif, sensibilité au seuil, fiche
de la CPCA. Les jeux tirés au hasard sont des bases orthonormées de Haar (Gram-Schmidt de
matrices gaussiennes, générateur du paquet, graine 11)."""
import numpy as np
import CoPRes as cp
from CoPRes.lode import _depuis, EPS
from commun import lire_rates

NOMS = ["M", "C"]
B, regime, _, _ = lire_rates()
X = cp.pretraiter(B)
th = cp.seuils_perm(X)
R = 3
U = cp.cpca(X, R)
A = [U.T @ x @ x.T @ U for x in X]
print(f"Seuils calculés : M {100 * th[0]:.1f} %, C {100 * th[1]:.1f} %")


def lecture(S, seuils):
    return [cp.nom_etiquette(tuple(k for k in range(2) if S[k, h] >= seuils[k]), NOMS) for h in range(S.shape[1])]


# --- tableau 1
S = cp.saliences(A, np.eye(R))
print("\nTableau 1 : saliences de la CPCA (M ; C)")
print("   " + "   ".join(f"u{h + 1} {S[0, h]:.3f} ; {S[1, h]:.3f}" for h in range(R)))
for nom, s in (("seuil fixe 5 %", [0.05, 0.05]), ("seuil fixe 10 %", [0.10, 0.10]), ("seuil calculé", th)):
    print(f"   lecture, {nom:16s} : {lecture(S, s)}")

# --- cohérence des lectures directes : directions extrêmes de la partie commune
lam = [np.linalg.eigvalsh(a) for a in A]
print("\nPartie commune = tout l'espace (seuil calculé) : salience minimale M "
      f"{100 * lam[0][0]:.1f} %, C {100 * lam[1][0]:.1f} %")
P2 = [np.linalg.eigvalsh(a[:2, :2]) for a in A]
print(f"Plan (u1, u2) au seuil de 10 % : salience minimale M {100 * P2[0][0]:.1f} %, C {100 * P2[1][0]:.1f} %")

# --- 1 000 jeux tirés au hasard
gen = cp.ParkMiller(11)


def haar(n):
    Z = gen.normales(n * n).reshape(n, n).T
    Q = Z.copy()
    for j in range(n):
        for _ in range(2):
            for i in range(j):
                Q[:, j] -= (Q[:, i] @ Q[:, j]) * Q[:, i]
        Q[:, j] /= np.linalg.norm(Q[:, j])
    return Q


def motif_txt(Q):
    return cp.texte_motif(cp.motif(cp.etiquettes(A, Q, th)), NOMS)


tous, coh, nb_comp = {}, {}, {"MC": 0, "un bloc": 0, "aucun": 0}
for _ in range(1000):
    Q = haar(R)
    m = motif_txt(Q)
    tous[m] = tous.get(m, 0) + 1
    for L in cp.etiquettes(A, Q, th):
        nb_comp["MC" if len(L) == 2 else ("un bloc" if len(L) == 1 else "aucun")] += 1
    if cp.est_etiquetage(A, Q, th):
        coh.setdefault(m, []).append((cp.critere(A, Q, th), Q))
print("\nTableau des 1 000 jeux : lecture, fréquence, jeux cohérents, Phi médian, Phi maximal")
for m, nb in sorted(tous.items(), key=lambda kv: -kv[1]):
    c = coh.get(m, [])
    phis = [p for p, _ in c]
    print(f"   {m:45s} {nb / 10:5.1f} %  {len(c):4d}  " +
          (f"{np.median(phis):.3f}  {max(phis):.3f}" if c else "--     --"))
print(f"   total cohérents : {sum(len(v) for v in coh.values())}")
print("Composantes tirées (3 000) : " + ", ".join(f"{k} {100 * v / (1000 * R):.1f} %" for k, v in nb_comp.items()))

# --- figure de cohérence, panneau b : parmi les jeux cohérents dont la partie commune est un plan,
# celui dont le plan commun est le plus loin des seuils
best, Qb, Pb = -np.inf, None, None
for m, liste in coh.items():
    for _, Q in liste:
        idx = [h for h, L in enumerate(cp.etiquettes(A, Q, th)) if len(L) == 2]
        if len(idx) != 2:
            continue
        E = Q[:, idx]
        lmin = [np.linalg.eigvalsh(E.T @ a @ E)[0] for a in A]
        marge_plan = min(l - t for l, t in zip(lmin, th))
        if marge_plan > best:
            best, Qb, Pb, lb, mb = marge_plan, Q, E, lmin, m
print(f"\nFigure, panneau b ({mb}) : plan commun le plus loin des seuils, salience minimale "
      f"M {100 * lb[0]:.1f} %, C {100 * lb[1]:.1f} %")
np.savetxt("appli_01_plan_commun.csv", Pb, delimiter=",")

# --- bilan : CoPRes et lecture cohérente la plus fréquente, tournée dans son motif
cop = cp.copres(X, th, R=R, departs=200)
print(f"\n{cop}")


def detail(Q):
    etq = cp.etiquettes(A, Q, th)
    Sq = cp.saliences(A, Q)
    out = {}
    for L in dict.fromkeys(etq):
        idx = [h for h, l in enumerate(etq) if l == L]
        for k in L:
            out[(cp.nom_etiquette(L, NOMS), NOMS[k])] = float(Sq[k, idx].sum() - th[k] * len(idx))
    return out


plus_coh = max(coh, key=lambda m: len(coh[m]))
Q0 = max(coh[plus_coh], key=lambda t: t[0])[1]
Qo = _depuis(A, np.asarray(th), Q0, EPS, 30)[0]
for nom, Q in (("CoPRes", cop.lode.base), (f"{plus_coh}, après rotation", Qo)):
    d = detail(Q)
    print(f"Bilan {nom} : Phi = {cp.critere(A, Q, th):.3f} ; motif {motif_txt(Q)} ; marge {cp.marge(A, Q, th):.4f}")
    for (L, k), v in d.items():
        print(f"     partie {L:8s} bloc {k} : {v:.3f}")
    print("     par bloc : " + ", ".join(f"{b} {sum(v for (L, k), v in d.items() if k == b):.3f}" for b in NOMS))
print(f"CPCA (incohérente) : Phi = {cp.critere(A, np.eye(R), th):.3f} ; borne de Ky Fan {cp.borne_kf(A, th):.3f}")

# --- sensibilité au seuil
sens = cp.sensibilite(X, th, R_liste=(3,), facteurs=(0.75, 0.9, 1.0, 1.25, 1.5, 2.0), departs=50)
print("\nSensibilité (R = 3) : " + " ; ".join(f"x{f}: {m}" for (r, f), m in sens.items()))

# --- fiche de la CPCA
f = cp.fiche(None, cop)
print("\nFiche de la CPCA (R = 3)\n" + repr(f).replace("{1,2}", "{M,C}").replace("{1}", "{M}").replace("{2}", "{C}"))
print("Lecture directe et composante CoPRes la plus proche : saliences de u3 :", np.round(S[:, 2], 3))
