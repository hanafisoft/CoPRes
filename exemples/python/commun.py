"""Lecture des données des exemples (dossier ../donnees)."""
import os
import numpy as np

DONNEES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "donnees")


def chemin(nom):
    return os.path.join(DONNEES, nom)


def lire_csv(nom, sep=","):
    with open(chemin(nom), encoding="utf-8") as f:
        lignes = [l.rstrip("\n").split(sep) for l in f if l.strip()]
    return lignes[0], lignes[1:]


def lire_beurres():
    """Trois blocs (VIS, NIR, MIR) de 21 produits ; type (beurre, margarine) ; axes spectraux."""
    blocs, axes = [], []
    for n in ("VIS", "NIR", "MIR"):
        tete, lignes = lire_csv(f"beurres_{n}.csv")
        ids = [l[0] for l in lignes]
        blocs.append(np.array([[float(v) for v in l[1:]] for l in lignes]))
        axes.append(np.array([float(v) for v in tete[1:]]))
    typ = np.array(["beurre" if i[0] == "B" else "margarine" for i in ids])
    return blocs, typ, ids, axes


def lire_simulees(plan):
    """Blocs des données simulées 1, 3 ou 4 de LODE (individus en lignes)."""
    tete, lignes = lire_csv(f"simulees{plan}.csv")
    k = np.array([int(l[0]) for l in lignes])
    M = np.array([[float(v) for v in l[1:]] for l in lignes])
    return [M[k == b].T for b in sorted(set(k))]


def matrices_symetriques(lignes, cle, n_col_i, n_col_j, n_col_v, taille):
    """Reconstruit des matrices symétriques à partir de lignes (clé, i, j, valeur), triangle supérieur."""
    out = {}
    for l in lignes:
        c = cle(l)
        A = out.setdefault(c, np.zeros((taille(l), taille(l))))
        i, j, v = int(l[n_col_i]) - 1, int(l[n_col_j]) - 1, float(l[n_col_v])
        A[i, j] = A[j, i] = v
    return out


def lire_vins():
    """Quatre juges (blocs) décrivant l'apparence de 8 vins (profil libre, notes de 0 à 10) ;
    Williams et Langron (1984), données de Hanafi et Qannari (2008, annexe 6)."""
    tete, lignes = lire_csv("vins.csv")
    ids = [l[0] for l in lignes]
    M = np.array([[float(v) for v in l[1:]] for l in lignes])
    juges = [t.split("_")[0] for t in tete[1:]]
    blocs = [M[:, [j for j, t in enumerate(juges) if t == f"J{k}"]] for k in range(1, 5)]
    noms = [[t.split("_")[1] for t in tete[1:] if t.startswith(f"J{k}_")] for k in range(1, 5)]
    return blocs, ids, noms


def lire_rates():
    """Rates S1 en lactation (projet GDM-MILK, ANR-22-CE17-0039) : 22 rates, métabolome du plasma
    (bloc M, 514 métabolites) et indicateurs de la régulation du glucose et de l'insuline (bloc C,
    18). Transformations par variable : métabolites passés au log10 (une valeur nulle est remplacée
    par la moitié de la plus petite valeur positive de sa variable) puis mis à l'échelle de Pareto
    (division par la racine de l'écart type) ; indicateurs réduits. Le centrage et l'inertie 1 se
    font ensuite par `pretraiter`. Renvoie ([M, C], régime, identifiants, noms des variables)."""
    tete, lignes = lire_csv("rates_S1_L12.csv")
    ids = [l[0] for l in lignes]
    regime = np.array([l[1] for l in lignes])
    X = np.array([[float(v) for v in l[2:]] for l in lignes])
    M, C = X[:, :514].copy(), X[:, 514:].copy()
    for j in range(M.shape[1]):
        pos = M[:, j][M[:, j] > 0]
        M[M[:, j] <= 0, j] = pos.min() / 2 if len(pos) else 1.0
    M = np.log10(M)
    M = M / np.sqrt(M.std(0, ddof=1))
    C = C / C.std(0, ddof=1)
    return [M, C], regime, ids, [tete[2:516], tete[516:]]
