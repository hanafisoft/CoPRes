"""LODE, section 6.3, tableaux 7 et 8 : CoPRes, JIVE et AJIVE dans le vrai sous-espace de signal.

Pour comparer les définitions et non les sous-espaces, les trois méthodes sont appliquées dans le
sous-espace S engendré par les vraies structures (scores) des données simulées : CoPRes par LODE
sur les restrictions des blocs à S, JIVE et AJIVE sur les blocs projetés P_S X_k.

Méthodes (paquet CoPRes 0.6.0) :
- CoPRes : seuils par défaut (`seuils_perm`, règle « direction »), `copres(X, seuils, U = base de S)`.
- JIVE (`jive`, selon r.jive 2.4) : vrais rangs, ou rangs estimés par permutation (`method = "perm"`,
  la règle par défaut de r.jive, proposée par Lock et al. 2013). Option `orthIndiv = FALSE` : JIVE de
  Lock et al. (2013), dont les espaces individuels de deux blocs peuvent se recouvrir. Avec
  `orthIndiv = TRUE` (défaut de r.jive 2.4), la première itération de r.jive part d'une partie
  individuelle remplie de -1 au lieu de 0, et l'algorithme s'arrête souvent sur une solution qui
  mêle les espaces individuels des blocs.
- AJIVE (`ajive`, selon mvlearn 0.5) : vrais rangs, ou rangs estimés selon la pratique de la
  méthode : rangs de signal initiaux lus sur l'éboulis des valeurs singulières (Feng et al. 2018,
  section 2.2.1), automatisé par le coude de Zhu et Ghodsi comme dans mvlearn ; rang joint par la
  borne de Wedin et la borne des directions aléatoires ; rangs individuels par le seuil des
  valeurs singulières.

Une structure (vecteur de scores t, étiquette L) est retrouvée si un sous-espace estimé
d'étiquette L la contient avec un carré du cosinus d'au moins 0,9. JIVE et AJIVE : l'espace joint
porte l'étiquette commune à tous les blocs, l'espace individuel du bloc k l'étiquette {k}.

Données : régénérées par le générateur du paquet (Park et Miller), donc identiques en R et en
Python ; graines 1 à N par situation (N = 50 par défaut). Ordre des tirages : voir `jeu_jive` et
`jeu_plan`.

Usage : python lode_08_jive.py [N]      (N = 1 pour un essai rapide)
Sorties : lode_08_jive_resultats.txt (tableaux) et lode_08_jive_detail.csv (une ligne par jeu).
"""
import os
import sys
import time
import math
import warnings
import numpy as np
import CoPRes as cp

warnings.filterwarnings("ignore", category=RuntimeWarning)  # coude : variance nulle (comme mvlearn)

ICI = os.path.dirname(os.path.abspath(__file__))
SEUIL_COS2 = 0.9


# ----------------------------------------------------------------------------- outils
def gram_schmidt(Z):
    """Orthonormalisation de Gram-Schmidt modifiée, deux passages (même résultat en R)."""
    Q = np.array(Z, dtype=float)
    for j in range(Q.shape[1]):
        for _ in range(2):
            for i in range(j):
                Q[:, j] -= (Q[:, i] @ Q[:, j]) * Q[:, i]
        Q[:, j] /= np.linalg.norm(Q[:, j])
    return Q


def normales(gen, nl, nc):
    """Matrice nl x nc de tirages gaussiens, remplie colonne par colonne (comme matrix() en R)."""
    return gen.normales(nl * nc).reshape(nc, nl).T


def finir_bloc(x):
    x = x - x.mean(0)
    return x / np.linalg.norm(x)


# ----------------------------------------------------------------------------- modèle de JIVE
SITUATIONS = [("A. canonique", dict()),
              ("B. axes proches (60°)", dict(angle=60.0)),
              ("C. commun 1-2, égal", dict(c12=(0.9, 0.9))),
              ("C. commun 1-2, inégal", dict(c12=(0.9, 0.5))),
              ("T. propres corrélées, rho = 0,5", dict(rho=0.5)),
              ("T. propres corrélées, rho = 0,9", dict(rho=0.9))]


def jeu_jive(opts, graine, n=100, p=50, K=3, sigma=0.35):
    """Modèle de JIVE : X_k = T_J D_J P_k' + T_k D_I Q_k' (+ t12 a_k b_k') + sigma E_k.
    Tirages, dans l'ordre : scores (n x m, centrés puis orthonormés) ; puis pour chaque bloc :
    P_k (p x 2), Q_k (p x 2), Q0_k (p x 2, situation B), b_k (p, situation C, blocs 1 et 2),
    E_k (n x p). Axes multipliés par sqrt(n / p). Blocs centrés, inertie 1."""
    gen = cp.ParkMiller(graine)
    supl = 1 if ("c12" in opts or "rho" in opts) else 0
    m = 2 + 2 * K + supl
    Z = normales(gen, n, m)
    T = gram_schmidt(Z - Z.mean(0))
    TJ = T[:, :2]
    TI = [T[:, 2 + 2 * k:4 + 2 * k].copy() for k in range(K)]
    verite = [(TJ[:, j], tuple(range(K))) for j in range(2)]
    if "rho" in opts:
        r = opts["rho"]
        TI[1][:, 0] = r * TI[0][:, 0] + math.sqrt(1 - r * r) * T[:, m - 1]
    for k in range(K):
        verite += [(TI[k][:, j], (k,)) for j in range(2)]
    if "c12" in opts:
        verite.append((T[:, m - 1], (0, 1)))
    DJ, DI = np.diag([1.0, 0.8]), np.diag([0.8, 0.6])
    f = math.sqrt(n / p)
    X = []
    for k in range(K):
        P = normales(gen, p, 2) * f
        Q = normales(gen, p, 2) * f
        if "angle" in opts:
            a = math.radians(opts["angle"])
            Q0 = normales(gen, p, 2) * f
            Q = math.cos(a) * P + math.sin(a) * Q0
        x = TJ @ DJ @ P.T + TI[k] @ DI @ Q.T
        if "c12" in opts and k < 2:
            b = normales(gen, p, 1) * f
            x = x + opts["c12"][k] * np.outer(T[:, m - 1], b[:, 0])
        x = x + sigma * normales(gen, n, p)
        X.append(finir_bloc(x))
    rJ = 2
    rA = [2 + (1 if ("c12" in opts and k < 2) else 0) for k in range(K)]
    return X, verite, rJ, rA


# ----------------------------------------------------------------------------- plans 1 à 4
PLANS = {
    1: dict(nom="1. commun partiel, c faible dans le bloc 3", n=30, J=[20, 20, 20], sigma=0.5,
            struct={"c": (0, 1, 2), "c12": (0, 1), "s1": (0,), "s2": (1,), "s3": (2,)},
            a=[{"c": 1.0, "c12": 0.8, "s1": 0.7}, {"c": 1.0, "c12": 0.8, "s2": 0.7},
               {"c": 0.3, "s3": 3.0}]),
    2: dict(nom="2. six blocs, trois communs par paires", n=40, J=[10, 15, 20, 25, 30, 40], sigma=0.4,
            struct={"c": (0, 1, 2, 3, 4, 5), "c12": (0, 1), "c34": (2, 3), "c56": (4, 5),
                    "s1": (0,), "s2": (1,), "s3": (2,), "s4": (3,), "s5": (4,), "s6": (5,)},
            a=[{"c": 1.0, "c12": 0.8, "s1": 0.5}, {"c": 1.0, "c12": 0.8, "s2": 0.6},
               {"c": 1.0, "c34": 0.8, "s3": 0.7}, {"c": 1.0, "c34": 0.8, "s4": 0.8},
               {"c": 1.0, "c56": 0.8, "s5": 0.9}, {"c": 1.0, "c56": 0.8, "s6": 1.0}]),
    3: dict(nom="3. quatre structures propres au bloc 3", n=30, J=[20, 20, 20], sigma=0.4,
            struct={"c": (0, 1, 2), "c12": (0, 1), "s1": (0,), "s2": (1,), "s3a": (2,), "s3b": (2,),
                    "s3c": (2,), "s3d": (2,)},
            a=[{"c": 1.0, "c12": 0.8, "s1": 0.7}, {"c": 1.0, "c12": 0.8, "s2": 0.7},
               {"c": 0.8, "s3a": 1.0, "s3b": 0.8, "s3c": 0.8, "s3d": 0.8}]),
    4: dict(nom="4. structures propres en surnombre", n=30, J=[20, 20, 20], sigma=0.4,
            struct={"c": (0, 1, 2), "c12": (0, 1), "s1": (0,), "s2": (1,), "e21": (1,), "e22": (1,),
                    "e23": (1,), "e24": (1,), "s3": (2,), "e31": (2,), "e32": (2,), "e33": (2,),
                    "e34": (2,)},
            a=[{"c": 1.0, "c12": 0.8, "s1": 0.7},
               {"c": 1.0, "c12": 0.8, "s2": 0.7, "e21": 0.7, "e22": 0.7, "e23": 0.7, "e24": 0.7},
               {"c": 0.8, "s3": 1.0, "e31": 0.7, "e32": 0.7, "e33": 0.7, "e34": 0.7}]),
}


def jeu_plan(num, graine):
    """Plans des données simulées 1 à 4 (annexe de LODE). Tirages, dans l'ordre : structures
    (n x m : orthonormées, centrées, orthonormées de nouveau, multipliées par sqrt(n)) ; puis pour
    chaque bloc : bruit (n x J_k), puis un vecteur d'axes (J_k) par structure portée, dans l'ordre
    des structures. Blocs centrés, inertie 1."""
    P = PLANS[num]
    gen = cp.ParkMiller(graine)
    n, noms = P["n"], list(P["struct"])
    T = gram_schmidt(normales(gen, n, len(noms)))
    T = gram_schmidt(T - T.mean(0)) * math.sqrt(n)
    X = []
    for k, J in enumerate(P["J"]):
        x = P["sigma"] * normales(gen, n, J)
        for i, s in enumerate(noms):
            if s in P["a"][k]:
                x = x + P["a"][k][s] * np.outer(T[:, i], normales(gen, J, 1)[:, 0])
        X.append(finir_bloc(x))
    K = len(P["J"])
    verite = [(T[:, i] / math.sqrt(n), P["struct"][s]) for i, s in enumerate(noms)]
    tous = tuple(range(K))
    rJ = sum(1 for _, L in verite if L == tous)
    rA = [sum(1 for _, L in verite if k in L and L != tous) for k in range(K)]
    return X, verite, rJ, rA


# ----------------------------------------------------------------------------- analyse d'un jeu
def retrouvees(espaces, verite):
    ok = 0
    for t, L in verite:
        B = espaces.get(tuple(L))
        if B is not None and B.shape[1] > 0:
            u = t / np.linalg.norm(t)
            if float(np.sum((B.T @ u) ** 2)) >= SEUIL_COS2:
                ok += 1
    return ok / len(verite)


def espaces_jive(res, K):
    e = {}
    if res.scores_joint.shape[1] > 0:
        e[tuple(range(K))] = res.scores_joint
    for k in range(K):
        if res.scores_individuels[k].shape[1] > 0:
            e[(k,)] = res.scores_individuels[k]
    return e


def analyser(X, verite, rJ, rA):
    K = len(X)
    S = gram_schmidt(np.column_stack([t for t, _ in verite]))
    th = cp.seuils_perm(X)
    # vérité lue par les seuils : blocs dont la salience de la structure atteint le seuil
    lecture = []
    for t, _ in verite:
        u = t / np.linalg.norm(t)
        lecture.append((t, tuple(k for k in range(K) if float(np.sum((X[k].T @ u) ** 2)) >= th[k])))
    c = cp.copres(X, th, U=S)
    esp_c = {L: c.base[:, idx] for L, idx in c.parties().items()}
    Y = [S @ (S.T @ x) for x in X]
    rS = [rJ + r for r in rA]
    t0 = time.time()
    jv = cp.jive(Y, rJ, rA, orth_indiv=False)
    je = cp.jive(Y, nperm=100, orth_indiv=False)
    av = cp.ajive(Y, rS, rang_joint=rJ, rangs_individuels=rA)
    ae = cp.ajive(Y, "coude")
    out = {}
    for nom, esp in (("copres", esp_c), ("jive_vrais", espaces_jive(jv, K)),
                     ("jive_estimes", espaces_jive(je, K)), ("ajive_vrais", espaces_jive(av, K)),
                     ("ajive_estimes", espaces_jive(ae, K))):
        out[nom + "_plan"] = retrouvees(esp, verite)
        out[nom + "_seuils"] = retrouvees(esp, lecture)
    out["rangs_jive_estimes"] = "/".join(map(str, [je.rang_joint] + je.rangs_individuels))
    out["rangs_ajive_estimes"] = "/".join(map(str, [ae.rang_joint] + ae.rangs_individuels))
    out["signal_ajive_estime"] = "/".join(map(str, ae.rangs_signal))
    out["motif_copres"] = cp.texte_motif(c.motif)
    out["seuils"] = "/".join(f"{v:.4f}" for v in th)
    return out


# ----------------------------------------------------------------------------- programme
COLS = ["copres", "jive_vrais", "jive_estimes", "ajive_vrais", "ajive_estimes"]


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    lignes, detail = [], []
    t0 = time.time()
    taches = [("jive", nom, opts) for nom, opts in SITUATIONS] + \
             [("plan", PLANS[p]["nom"], p) for p in PLANS]
    for type_, nom, arg in taches:
        res = []
        for g in range(1, N + 1):
            X, verite, rJ, rA = jeu_jive(arg, g) if type_ == "jive" else jeu_plan(arg, g)
            r = analyser(X, verite, rJ, rA)
            r.update(tableau=7 if type_ == "jive" else 8, situation=nom, graine=g)
            res.append(r)
            detail.append(r)
        moy = {k: np.mean([r[k] for r in res]) for k in res[0] if k.endswith(("_plan", "_seuils"))}
        lignes.append((type_, nom, moy, res))
        print(f"{nom:40s} " + "  ".join(f"{moy[c + '_plan']:.2f}" for c in COLS)
              + f"   ({time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(ICI, "lode_08_jive_resultats.txt"), "w", encoding="utf-8") as f:
        f.write(f"CoPRes {cp.__version__}, {N} jeux par situation, graines 1 à {N}\n\n")
        f.write("Tableau 7 (modèle de JIVE) : part des structures retrouvées avec leur étiquette\n")
        f.write(f"{'situation':40s} CoPRes  JIVE vrais  JIVE estimés  AJIVE vrais  AJIVE estimés\n")
        for type_, nom, moy, res in lignes:
            if type_ == "jive":
                f.write(f"{nom:40s} " + "  ".join(f"{moy[c + '_plan']:10.2f}" for c in COLS) + "\n")
        f.write("\nTableau 8 (plans 1 à 4) : vérité du plan / lecture par les seuils\n")
        f.write(f"{'plan':45s} CoPRes       JIVE vrais   JIVE estimés  AJIVE vrais  AJIVE estimés\n")
        for type_, nom, moy, res in lignes:
            if type_ == "plan":
                f.write(f"{nom:45s} " + "  ".join(f"{moy[c + '_plan']:.2f} / {moy[c + '_seuils']:.2f}"
                                                  for c in COLS) + "\n")
        f.write("\nRangs estimés (joint/individuels), effectifs sur les jeux\n")
        for type_, nom, moy, res in lignes:
            for cle in ("rangs_jive_estimes", "rangs_ajive_estimes", "signal_ajive_estime"):
                vals, cnt = np.unique([r[cle] for r in res], return_counts=True)
                f.write(f"{nom:45s} {cle:20s} " + ", ".join(f"{v} ({c})" for v, c in zip(vals, cnt)) + "\n")
    cles = ["tableau", "situation", "graine"] + [c + s for c in COLS for s in ("_plan", "_seuils")] + \
           ["rangs_jive_estimes", "rangs_ajive_estimes", "signal_ajive_estime", "motif_copres", "seuils"]
    with open(os.path.join(ICI, "lode_08_jive_detail.csv"), "w", encoding="utf-8") as f:
        f.write(";".join(cles) + "\n")
        for r in detail:
            f.write(";".join(str(r[k]) for k in cles) + "\n")
    print(f"Terminé en {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
