"""Méthodes concurrentes, pour comparer CoPRes à JIVE et à AJIVE sur les mêmes données.

Les blocs ont les individus en lignes (n x J_k), comme dans le reste du paquet.

- `jive` suit le paquet R r.jive 2.4 (Lock et al. 2013 ; O'Connell et Lock 2016) : centrage des
  variables, mise à l'échelle des blocs, algorithme alterné avec individuels orthogonaux entre eux
  (orthIndiv), rangs donnés ou estimés par permutation (jive.perm).
- `ajive` suit l'implémentation de mvlearn 0.5 (Carmichael ; Feng et al. 2018) : extraction du
  signal par bloc, rang joint par la borne de Wedin et la borne des directions aléatoires,
  réexamen des composantes jointes, rangs individuels par le seuil des valeurs singulières.

Les tirages aléatoires (permutations, directions) utilisent le générateur de Park et Miller du
paquet : les versions R et Python donnent les mêmes résultats, mais pas les mêmes tirages que
r.jive ou mvlearn. À rangs donnés, aucune des deux méthodes n'est aléatoire.
"""
from dataclasses import dataclass, field
import math
import numpy as np

from .alea import ParkMiller


# ----------------------------------------------------------------------------- outils
def _svd(X):
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    return U, d, Vt.T


def _quantile(x, p):
    """Quantile de type 7 (défaut de R, `numpy.quantile` linéaire)."""
    return float(np.quantile(np.asarray(x, dtype=float), p))


def _base_orthonormee(Z):
    """Base orthonormée de l'espace engendré par les colonnes de Z (vecteurs singuliers à
    gauche ; même espace que la QR utilisée par mvlearn)."""
    U, d, _ = _svd(Z)
    return U[:, :Z.shape[1]]


# ============================================================================= JIVE
@dataclass
class ResultatJIVE:
    joint: list              # J_k, n x J_k (données centrées et mises à l'échelle)
    individuel: list         # A_k, n x J_k
    scores_joint: np.ndarray  # n x r_J, base orthonormée de l'espace joint des individus
    scores_individuels: list  # n x r_k, une base par bloc
    rang_joint: int
    rangs_individuels: list
    donnees: list            # blocs centrés et mis à l'échelle
    converge: bool = True
    iterations: int = 0
    methode: str = "given"


def _pretraiter_jive(blocs, centrer, echelle):
    X = [np.array(x, dtype=float) for x in blocs]
    if centrer:
        X = [x - x.mean(0) for x in X]
    if echelle:
        total = sum(x.size for x in X)
        X = [x / (np.linalg.norm(x) * math.sqrt(total)) for x in X]
    return X


def _jive_iter(X, rJ, rA, conv, maxit, orth_indiv):
    """Boucle alternée de r.jive (jive.iter), individus en lignes : J = rang rJ de [X - A],
    puis A_k = rang r_k de (I - V V')(X_k - J_k), V scores joints, et, avec orth_indiv, projeté
    hors des scores individuels des autres blocs."""
    K = len(X)
    n = X[0].shape[0]
    Xtot = np.hstack(X)
    bornes = np.cumsum([0] + [x.shape[1] for x in X])
    Jtot = -np.ones_like(Xtot)
    Atot = -np.ones_like(Xtot)
    Vind = [np.zeros((n, r)) for r in rA]
    I = np.eye(n)
    V = np.zeros((n, rJ))
    nrun, converge = 0, False
    while nrun < maxit and not converge:
        Jlast, Alast = Jtot, Atot
        if rJ > 0:
            U, d, W = _svd(Xtot - Atot)
            V = U[:, :rJ]
            Jtot = (U[:, :rJ] * d[:rJ]) @ W[:, :rJ].T
        else:
            Jtot = np.zeros_like(Xtot)
            V = np.zeros((n, 0))
        J = [Jtot[:, bornes[i]:bornes[i + 1]] for i in range(K)]
        A = []
        for i in range(K):
            if rA[i] > 0:
                T = (I - V @ V.T) @ (X[i] - J[i])
                if orth_indiv and nrun > 0:
                    for j in range(K):
                        if j != i:
                            T = (I - Vind[j] @ Vind[j].T) @ T
                U, d, W = _svd(T)
                if orth_indiv:
                    Vind[i] = U[:, :rA[i]]
                A.append((U[:, :rA[i]] * d[:rA[i]]) @ W[:, :rA[i]].T)
            else:
                A.append(np.zeros_like(X[i]))
                if orth_indiv:
                    Vind[i] = np.zeros((n, 0))
        if orth_indiv and nrun == 0:
            for i in range(K):
                for j in range(K):
                    if j != i:
                        A[i] = (I - Vind[j] @ Vind[j].T) @ A[i]
            for i in range(K):
                if rA[i] > 0:
                    Vind[i] = _svd(A[i])[0][:, :rA[i]]
        Atot = np.hstack(A)
        if (np.linalg.norm(Jtot - Jlast) <= conv and np.linalg.norm(Atot - Alast) <= conv):
            converge = True
        nrun += 1
    scores_ind = []
    for i in range(K):
        scores_ind.append(_svd(A[i])[0][:, :rA[i]] if rA[i] > 0 else np.zeros((n, 0)))
    return J, A, V[:, :rJ].copy(), scores_ind, converge, nrun


def _rangs_perm(X, Jperp, Aperp, last, nperm, alpha, gen):
    """Un tour de jive.perm : rang joint par permutation des individus, bloc par bloc ;
    rangs individuels par permutation des individus, variable par variable."""
    n = X[0].shape[0]
    full = [X[i] - Aperp[i] for i in range(len(X))]
    actuel = np.linalg.svd(np.hstack(full), compute_uv=False)
    perms = np.empty((nperm, actuel.size))
    for b in range(nperm):
        perms[b] = np.linalg.svd(np.hstack([f[gen.permutation(n)] for f in full]),
                                 compute_uv=False)
    rJ = 0
    for i in range(actuel.size):
        if actuel[i] > _quantile(perms[:, i], 1 - alpha):
            rJ += 1
        else:
            break
    rJ = max(rJ, last[0])
    rA = []
    for k in range(len(X)):
        ind = X[k] - Jperp[k]
        actuel = np.linalg.svd(ind, compute_uv=False)
        perms = np.empty((nperm, actuel.size))
        for b in range(nperm):
            P = np.empty_like(ind)
            for j in range(ind.shape[1]):
                P[:, j] = ind[gen.permutation(n), j]
            perms[b] = np.linalg.svd(P, compute_uv=False)
        r = 0
        for i in range(actuel.size):
            if actuel[i] > _quantile(perms[:, i], 1 - alpha):
                r += 1
            else:
                break
        rA.append(r)
    return [rJ] + rA


def jive(blocs, rang_joint=None, rangs_individuels=None, methode=None, nperm=100, alpha=0.05,
         centrer=True, echelle=True, orth_indiv=True, conv=None, maxit=1000, graine=1):
    """JIVE selon r.jive 2.4.

    methode="given" (défaut si les rangs sont donnés) : rangs fixés.
    methode="perm" (défaut sinon) : rangs estimés par permutation (jive.perm, nperm permutations,
    niveau alpha, au plus dix tours, rang joint non décroissant d'un tour à l'autre).
    Les sorties sont exprimées sur les données centrées et mises à l'échelle (comme r.jive)."""
    K = len(blocs)
    if methode is None:
        methode = "given" if rang_joint is not None else "perm"
    X = _pretraiter_jive(blocs, centrer, echelle)
    if conv is None:
        conv = 1e-6 * np.linalg.norm(np.hstack(X))
    if methode == "given":
        rA = list(rangs_individuels) if rangs_individuels is not None else [1] * K
        J, A, V, Si, ok, it = _jive_iter(X, int(rang_joint), [int(r) for r in rA], conv, maxit,
                                         orth_indiv)
        return ResultatJIVE(J, A, V, Si, int(rang_joint), [int(r) for r in rA], X, ok, it, "given")
    if methode != "perm":
        raise ValueError("methode doit valoir 'given' ou 'perm'")
    gen = ParkMiller(graine)
    n = X[0].shape[0]
    Jperp = [np.zeros_like(x) for x in X]
    Aperp = [np.zeros_like(x) for x in X]
    last, current = [-2] * (K + 1), [-1] * (K + 1)
    nrun = 0
    res = None
    while last != current and nrun < 10:
        last = current
        current = _rangs_perm(X, Jperp, Aperp, last, nperm, alpha, gen)
        if current != last:
            res = _jive_iter(X, current[0], current[1:], conv, maxit, orth_indiv)
            Jperp, Aperp = res[0], res[1]
        nrun += 1
    J, A, V, Si, ok, it = res
    return ResultatJIVE(J, A, V, Si, current[0], current[1:], X, nrun < 10, it, "perm")


# ============================================================================= AJIVE
@dataclass
class ResultatAJIVE:
    joint: list               # J_k = P_J X_k, n x J_k (blocs centrés si centrer=True)
    individuel: list          # I_k, n x J_k
    scores_joint: np.ndarray  # n x r_J
    scores_individuels: list  # n x r_k
    rang_joint: int
    rangs_individuels: list
    rangs_signal: list
    seuils_vs: list           # seuils des valeurs singulières, bloc par bloc
    vs_joint: np.ndarray      # valeurs singulières de la concaténation des bases de signal
    seuil_wedin: float = float("nan")
    seuil_hasard: float = float("nan")
    rang_joint_estime: int = -1   # avant réexamen des composantes jointes
    donnees: list = field(default_factory=list)


def _log_vraisemblance(arr):
    """Profil de vraisemblance de Zhu et Ghodsi (2006), tel que dans mvlearn."""
    n = len(arr)
    out = np.zeros(n)
    for idx in range(1, n + 1):
        s1, s2 = arr[:idx], arr[idx:]
        if s1.size == 1 and s2.size == 1:
            out[idx - 1] = -np.inf
            continue
        m1 = s1.mean()
        m2 = s2.mean() if s2.size else -np.inf
        var = (np.sum((s1 - m1) ** 2) + (np.sum((s2 - m2) ** 2) if s2.size else 0.0)) / \
            (n - 1 - (idx < n))
        sd = math.sqrt(var)
        ll = np.sum(-0.5 * ((s1 - m1) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2 * math.pi))
        if s2.size:
            ll += np.sum(-0.5 * ((s2 - m2) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2 * math.pi))
        out[idx - 1] = ll
    return out


def rang_coude(X, n_coudes=2):
    """Rang de signal par le coude du profil de vraisemblance (Zhu et Ghodsi 2006), comme
    mvlearn : sur les ceil(log2(min(n, J))) plus grandes valeurs singulières de X, rang donné
    par le dernier des `n_coudes` coudes successifs."""
    X = np.asarray(X, dtype=float)
    k = int(math.ceil(math.log2(min(X.shape))))
    D = np.linalg.svd(X, compute_uv=False)[:k]
    idx = 0
    for _ in range(n_coudes):
        idx += int(np.argmax(_log_vraisemblance(D[idx:]))) + 1
    return idx


def _echantillon_hasard(n, rangs, gen):
    """Carré de la plus grande valeur singulière de la concaténation de bases aléatoires."""
    M = np.hstack([_base_orthonormee(gen.normales(n * r).reshape(r, n).T) for r in rangs])
    return np.linalg.svd(M, compute_uv=False)[0] ** 2


def _norme_bruit(X, base, gen):
    """|| X Z ||_2 pour Z base orthonormée de vecteurs gaussiens projetés hors de `base`."""
    p, r = base.shape
    Z = gen.normales(p * r).reshape(r, p).T
    Z = Z - base @ (base.T @ Z)
    Z = _base_orthonormee(Z)
    return np.linalg.norm(X @ Z, 2)


def ajive(blocs, rangs_signal="coude", rang_joint=None, rangs_individuels=None, reconsiderer=True,
          n_wedin=1000, n_hasard=1000, centile_wedin=5, centile_hasard=95, n_coudes=2, graine=1,
          centrer=True):
    """AJIVE selon mvlearn 0.5.

    rangs_signal : liste des rangs de signal initiaux, ou "coude" (estimés par `rang_coude` sur
    les blocs non centrés, comme mvlearn). Les blocs sont ensuite centrés par colonne si
    centrer=True (défaut, comme mvlearn) ; avec centrer=False, ils sont pris tels quels.
    rang_joint : rang joint donné, ou None pour l'estimer (borne de Wedin, n_wedin tirages,
    centile_wedin ; borne des directions aléatoires, n_hasard tirages, centile_hasard).
    rangs_individuels : rangs individuels donnés, ou None (valeurs singulières de (I - P_J) X_k
    au-dessus du seuil du bloc, à mi-chemin des valeurs singulières de rang r et r + 1)."""
    X0 = [np.array(x, dtype=float) for x in blocs]
    K = len(X0)
    n = X0[0].shape[0]
    if isinstance(rangs_signal, str):
        if rangs_signal != "coude":
            raise ValueError("rangs_signal doit être une liste ou 'coude'")
        rs = [rang_coude(x, n_coudes) for x in X0]
    else:
        rs = [int(r) for r in rangs_signal]
    X = [x - x.mean(0) for x in X0] if centrer else [x.copy() for x in X0]
    seuils, U_s, D_s, V_s = [], [], [], []
    for i, x in enumerate(X):
        if rs[i] >= min(x.shape):
            rs[i] = min(x.shape) - 1
        U, d, V = _svd(x)
        seuils.append((d[rs[i] - 1] + d[rs[i]]) / 2)
        U_s.append(U[:, :rs[i]])
        D_s.append(d[:rs[i]])
        V_s.append(V[:, :rs[i]])
    Uj, dj, _ = _svd(np.hstack(U_s))
    seuil_w = seuil_h = float("nan")
    if rang_joint is None:
        gen = ParkMiller(graine)
        hasard = np.array([_echantillon_hasard(n, rs, gen) for _ in range(n_hasard)])
        gen = ParkMiller(graine)
        wedin = []
        for i in range(K):
            un = np.array([_norme_bruit(X[i].T, U_s[i], gen) for _ in range(n_wedin)])
            vn = np.array([_norme_bruit(X[i], V_s[i], gen) for _ in range(n_wedin)])
            wedin.append(np.minimum(np.maximum(un, vn) / D_s[i][-1], 1.0))
        echant_w = K - np.sum(np.array(wedin) ** 2, axis=0)
        seuil_w = _quantile(echant_w, centile_wedin / 100)
        seuil_h = _quantile(hasard, centile_hasard / 100)
        rJ = int(np.sum(dj ** 2 > max(seuil_w, seuil_h)))
    else:
        rJ = int(rang_joint)
    rJ_estime = rJ
    garder = list(range(rJ))
    if reconsiderer:
        for i, x in enumerate(X):
            for j in list(garder):
                if np.linalg.norm(x.T @ Uj[:, j]) < seuils[i]:
                    garder.remove(j)
                    break
    Sj = Uj[:, garder]
    rJ = len(garder)
    J, Ind, Si, rI = [], [], [], []
    for i, x in enumerate(X):
        Ji = Sj @ (Sj.T @ x) if rJ > 0 else np.zeros_like(x)
        Xo = x - Ji
        U, d, V = _svd(Xo)
        if rangs_individuels is None:
            r = int(np.sum(d[:max(min(x.shape) - rJ, 0)] > seuils[i]))
        else:
            r = int(rangs_individuels[i])
        J.append(Ji)
        Ind.append((U[:, :r] * d[:r]) @ V[:, :r].T)
        Si.append(U[:, :r])
        rI.append(r)
    return ResultatAJIVE(J, Ind, Sj, Si, rJ, rI, rs, seuils, dj, seuil_w, seuil_h, rJ_estime, X)
