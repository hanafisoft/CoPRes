"""Article appliqué, section 3 : comparer trois méthodes (CPCA, ACOM, ACCPS) par leurs lectures CoPRes.

Parties (argument) :
  reels       tableaux des rates (R = 3 et 4) et des beurres (R = 4) ;
  deux_blocs  simulation à deux blocs (angle phi, 50 jeux par angle, intensité de t2 dans le bloc 2 :
              0,5 et variante 0,35) ;
  plans       plans 1, 3 et 4 de LODE, R du plan, R - 1 et R + 1, 50 jeux : équivalence des lectures
              CoPRes de l'ACOM et de l'ACCPS à celle de la CPCA ;
  contre      contre-exemple de l'ACCPS dans un plan (figure).
Sans argument : toutes les parties. Données simulées : générateur du paquet (graines 1 à 50)."""
import sys
import math
import numpy as np
import CoPRes as cp
from commun import lire_rates, lire_beurres
from lode_08_jive import gram_schmidt, normales, finir_bloc, jeu_plan, PLANS

METHODES = ("CPCA", "ACOM", "ACCPS")


def parties(Q, etiq):
    g = {}
    for h, L in enumerate(etiq):
        g.setdefault(tuple(L), []).append(h)
    return {L: Q[:, idx] for L, idx in g.items()}


def equivalentes(p1, p2, seuil=0.72):
    """Même motif et parties de même étiquette à proximité >= seuil ; renvoie (bool, proximité min)."""
    if sorted((L, v.shape[1]) for L, v in p1.items()) != sorted((L, v.shape[1]) for L, v in p2.items()):
        return False, None
    pr = min(cp.proximite(p1[L], p2[L]) for L in p1)
    return pr >= seuil, pr


def nom_lecture(etiq, noms):
    return ", ".join(cp.nom_etiquette(L, noms) if L else "--" for L in etiq)


def methode_dans(U, X, nom, R):
    """Base de la méthode construite dans le sous-espace U (blocs réduits Z_k = U' X_k)."""
    Z = [U.T @ x for x in X]
    if nom == "CPCA":
        return np.eye(R)
    return cp.METHODES[nom](Z, R)


def tableau_reel(X, th, R, noms, departs=200):
    Uc = cp.cpca(X, R)
    Ac = [Uc.T @ x @ x.T @ Uc for x in X]
    copC = cp.copres(X, th, U=Uc, departs=departs)
    refC = parties(copC.lode.base, copC.etiquettes)
    print(f"  R = {R}")
    for nom in METHODES:
        U = cp.METHODES[nom](X, R)
        ld = cp.lecture_directe(X, U, th)
        Q = methode_dans(Uc, X, nom, R)
        eq_a, pr_a = equivalentes(parties(Q, cp.etiquettes(Ac, Q, th)), refC)
        cop = cp.copres(X, th, U=U, methode=nom, departs=departs)
        prox_esp = cp.proximite(Uc, U)
        eq, pr = (None, None) if nom == "CPCA" else cp.equivalence(copC, cop)
        prm = None if pr is None else (min(pr.values()) if pr else None)
        print(f"    {nom:6s} lecture directe {nom_lecture(ld.etiquettes, noms):28s} cohérente {ld.coherente!s:5s} "
              f"(a) {eq_a!s:5s} ({'--' if pr_a is None else f'{pr_a:.2f}'}) ; CoPRes {cp.texte_motif(cop.motif, noms):40s} "
              f"proximité de l'espace {'--' if nom == 'CPCA' else f'{prox_esp:.2f}'} ; équivalente à la CPCA "
              f"{'--' if nom == 'CPCA' else eq} ({'--' if prm is None else f'{prm:.2f}'})")


def reels():
    B, _, _, _ = lire_rates()
    X = cp.pretraiter(B)
    th = cp.seuils_perm(X)
    print("Rates (M, C)")
    for R in (3, 4):
        tableau_reel(X, th, R, ["M", "C"])
    b, _, _, _ = lire_beurres()
    Xb = cp.pretraiter(b, snv=True)
    thb = cp.seuils_perm(Xb)
    print("Beurres (V, N, I)")
    tableau_reel(Xb, thb, 4, ["V", "N", "I"])


# ----------------------------------------------------------------------------- deux blocs
def jeu_deux_blocs(graine, phi_deg, a2, n=60, J=30, sigma=0.5, a11=3.0, a12=1.5, a3=1.6):
    """Tirages, dans l'ordre : structures (n x 3, centrées, orthonormées, centrées, orthonormées,
    multipliées par sqrt(n)) ; bloc 1 : bruit (n x J), axes de u puis de w ; bloc 2 : bruit, axes de
    t2 puis de t3."""
    gen = cp.ParkMiller(graine)
    T = normales(gen, n, 3)
    T = gram_schmidt(T - T.mean(0))
    T = gram_schmidt(T - T.mean(0)) * math.sqrt(n)
    p = math.radians(phi_deg)
    u = math.cos(p) * T[:, 0] + math.sin(p) * T[:, 1]
    w = -math.sin(p) * T[:, 0] + math.cos(p) * T[:, 1]
    X1 = sigma * normales(gen, n, J)
    X1 = X1 + a11 * np.outer(u, normales(gen, J, 1)[:, 0]) + a12 * np.outer(w, normales(gen, J, 1)[:, 0])
    X2 = sigma * normales(gen, n, J)
    X2 = X2 + a2 * np.outer(T[:, 1], normales(gen, J, 1)[:, 0]) + a3 * np.outer(T[:, 2], normales(gen, J, 1)[:, 0])
    verite = [(T[:, 0] / math.sqrt(n), (0,)), (T[:, 1] / math.sqrt(n), (0, 1)), (T[:, 2] / math.sqrt(n), (1,))]
    return [finir_bloc(X1), finir_bloc(X2)], verite


def retrouvees(p, verite):
    return np.mean([L in p and float(np.sum((p[L].T @ t) ** 2)) >= 0.9 for t, L in verite])


def deux_blocs(N=50):
    R = 3
    for a2 in (0.5, 0.35):
        print(f"Simulation à deux blocs, intensité de t2 dans le bloc 2 : {a2}")
        print("  phi   équivalentes à CoPRes (CPCA ACOM ACCPS)   retrouvées, composantes   retrouvées, CoPRes")
        for deg in (0, 15, 30, 45):
            eq = {m: [] for m in METHODES}
            rc = {m: [] for m in METHODES}
            rq = {m: [] for m in METHODES}
            for g in range(1, N + 1):
                X, verite = jeu_deux_blocs(g, deg, a2)
                th = cp.seuils_perm(X)
                for nom in METHODES:
                    U = cp.METHODES[nom](X, R)
                    ld = cp.lecture_directe(X, U, th)
                    cop = cp.copres(X, th, U=U, methode=nom, departs=50)
                    pc, pq = parties(U, ld.etiquettes), parties(cop.base, cop.etiquettes)
                    eq[nom].append(equivalentes(pc, pq)[0])
                    rc[nom].append(retrouvees(pc, verite))
                    rq[nom].append(retrouvees(pq, verite))
            print(f"  {deg:3d}°  " + "  ".join(f"{100 * np.mean(eq[m]):4.0f} %" for m in METHODES) + "        " +
                  "  ".join(f"{np.mean(rc[m]):.2f}" for m in METHODES) + "        " +
                  "  ".join(f"{np.mean(rq[m]):.2f}" for m in METHODES), flush=True)


# ----------------------------------------------------------------------------- plans de LODE
R_PLANS = {1: 5, 3: 7, 4: 6}


def plans(N=50, departs=100):
    print("Plans de LODE : lecture CoPRes de l'ACOM et de l'ACCPS équivalente à celle de la CPCA")
    for p in (1, 3, 4):
        for R in (R_PLANS[p] - 1, R_PLANS[p], R_PLANS[p] + 1):
            eq = {"ACOM": [], "ACCPS": []}
            for g in range(1, N + 1):
                X = jeu_plan(p, g)[0]
                th = cp.seuils_perm(X)
                copC = cp.copres(X, th, R=R, methode="CPCA", departs=departs)
                for nom in eq:
                    cop = cp.copres(X, th, R=R, methode=nom, departs=departs)
                    eq[nom].append(cp.equivalence(copC, cop)[0])
            print(f"  plan {p}, R = {R} : ACOM {100 * np.mean(eq['ACOM']):.0f} %, ACCPS {100 * np.mean(eq['ACCPS']):.0f} %",
                  flush=True)


# ----------------------------------------------------------------------------- contre-exemple
def contre():
    a, b, c, phi = 0.5, 0.5, 0.01, math.radians(30)
    v = np.array([math.cos(phi), math.sin(phi)])
    w = np.array([-math.sin(phi), math.cos(phi)])
    A = [a * np.diag([1.0, 0.0]), b * np.outer(v, v) + c * np.outer(w, w)]
    th = np.array([0.05, 0.05])
    # ACCPS dans le plan : maximiser sum_k (u' A_k u)^2 sur l'angle (grille fine), puis l'orthogonal
    t = np.linspace(0, np.pi, 400001)
    f = (a * np.cos(t) ** 2) ** 2 + (b * np.cos(t - phi) ** 2 + c * np.sin(t - phi) ** 2) ** 2
    t0 = t[f.argmax()]
    Qa = np.array([[math.cos(t0), -math.sin(t0)], [math.sin(t0), math.cos(t0)]])
    res = cp.lode(A, th, departs=200)
    for nom, Q in (("ACCPS", Qa), ("CoPRes", res.base)):
        S = cp.saliences(A, Q)
        print(f"  {nom:6s} : saliences {np.round(S, 3).tolist()} ; étiquettes {cp.etiquettes(A, Q, th)} ; "
              f"bilan {cp.critere(A, Q, th):.3f}")
    print(f"  borne de Ky Fan {cp.borne_kf(A, th):.3f}")


if __name__ == "__main__":
    choix = sys.argv[1:] or ["reels", "contre", "deux_blocs", "plans"]
    for c in choix:
        {"reels": reels, "deux_blocs": deux_blocs, "plans": plans, "contre": contre}[c]()
        print()
