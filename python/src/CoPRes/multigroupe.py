"""Couche CoPRes pour des données structurées par groupes d'individus (mêmes variables mesurées
sur K groupes). L'espace étiqueté est celui des variables ; A_k est la covariance du groupe k,
éventuellement restreinte aux R premiers axes de l'ACP de l'ensemble des individus.
"""
from dataclasses import dataclass, field
import os
import numpy as np

from .alea import ParkMiller
from .lode import lode, saliences, etiquettes, motif, texte_motif, critere, marge


# ----------------------------------------------------------------------------- données
def lire_notes(situation=1):
    """Notes d'élèves simulées (LODE, section 7 ; article appliqué, section 4) : 1 500 élèves de
    trois lycées A, B, C, huit modules. Situation 1 : mêmes moyennes ; 2 : spécificité et
    discrimination ; 3 : discrimination ailleurs. Renvoie (X, groupes, variables)."""
    f = os.path.join(os.path.dirname(__file__), "donnees", f"notes_situation{situation}.csv")
    with open(f, encoding="utf-8") as fh:
        lignes = [l.rstrip("\n").split(",") for l in fh if l.strip()]
    variables = lignes[0][1:]
    g = np.array([l[0] for l in lignes[1:]])
    X = np.array([[float(v) for v in l[1:]] for l in lignes[1:]])
    return X, g, variables


# ----------------------------------------------------------------------------- prétraitement
@dataclass
class PrepGroupes:
    groupes: list            # noms des groupes, dans l'ordre de première apparition
    g: np.ndarray            # groupe de chaque individu
    Xs: np.ndarray           # données centrées sur l'ensemble et réduites
    Xk: list                 # données de chaque groupe, centrées sur la moyenne du groupe
    covariances: list        # A_k (dans l'espace d'analyse)
    U: np.ndarray            # base de l'espace d'analyse (p x R), identité si pas de restriction
    ecarts_types: np.ndarray


def pretraiter_groupes(X, g, reduire="intra", R=None):
    """Centrage sur l'ensemble des individus ; réduction par l'écart-type intra-groupe commun
    (reduire = "intra") ou aucune (None) ; centrage de chaque groupe sur sa moyenne ; covariance
    de chaque groupe (diviseur n_k - 1). Si R est donné, les covariances sont restreintes aux R
    premiers axes de l'ACP de l'ensemble des individus (cas p grand devant les effectifs)."""
    X = np.asarray(X, dtype=float)
    g = np.asarray(g)
    groupes = list(dict.fromkeys(g.tolist()))
    N, p = X.shape
    K = len(groupes)
    if reduire == "intra":
        ss = sum(((X[g == k] - X[g == k].mean(0)) ** 2).sum(0) for k in groupes)
        sw = np.sqrt(ss / (N - K))
    elif reduire is None:
        sw = np.ones(p)
    else:
        raise ValueError("reduire doit valoir 'intra' ou None")
    Xs = (X - X.mean(0)) / sw
    if R is None:
        U = np.eye(p)
    else:
        ev, W = np.linalg.eigh(Xs.T @ Xs)
        U = W[:, ::-1][:, :R]
        for h in range(R):
            if U[np.argmax(np.abs(U[:, h])), h] < 0:
                U[:, h] = -U[:, h]
    Xk = [Xs[g == k] - Xs[g == k].mean(0) for k in groupes]
    A = [U.T @ (x.T @ x / (len(x) - 1)) @ U for x in Xk]
    return PrepGroupes(groupes=groupes, g=g, Xs=Xs, Xk=Xk, covariances=A, U=U, ecarts_types=sw)


# ----------------------------------------------------------------------------- seuils
def seuils_groupes(prep, nperm=200, ndir=100, niveau=0.95, regle="auto", graine=1):
    """Seuils par permutation de chaque variable à l'intérieur de chaque groupe.

    Règle "direction" : salience d'une direction tirée uniformément sur la sphère de l'espace
    d'analyse pour la covariance permutée ; règle "valeur_propre" : plus grande valeur propre de
    la covariance permutée restreinte à l'espace d'analyse ; règle "auto" : "valeur_propre" pour
    un groupe tel que n_k - 1 < dimension de l'espace d'analyse, "direction" sinon.
    Ordre des tirages : groupes, puis permutations, puis variables, puis directions."""
    gen = ParkMiller(graine)
    U = prep.U
    m = U.shape[1]
    th = []
    for x in prep.Xk:
        n, p = x.shape
        r = regle
        if regle == "auto":
            r = "valeur_propre" if n - 1 < m else "direction"
        vals = []
        for _ in range(nperm):
            u = gen.uniformes(n * p).reshape(p, n)
            idx = np.argsort(u, axis=1, kind="stable")
            x0 = np.take_along_axis(x.T, idx, axis=1).T
            x0 = x0 - x0.mean(0)
            C0 = U.T @ (x0.T @ x0 / (n - 1)) @ U
            if r == "direction":
                Z = gen.normales(m * ndir).reshape(ndir, m).T
                Z = Z / np.linalg.norm(Z, axis=0)
                vals.append(np.einsum("ij,ij->j", Z, C0 @ Z))
            elif r == "valeur_propre":
                vals.append([np.linalg.eigvalsh(C0)[-1]])
            else:
                raise ValueError("regle doit valoir 'auto', 'direction' ou 'valeur_propre'")
        th.append(float(np.quantile(np.concatenate([np.ravel(v) for v in vals]), niveau)))
    return np.array(th)


# ----------------------------------------------------------------------------- CoPRes multigroupe
@dataclass
class FactorisationGroupes:
    groupes: list
    base: np.ndarray            # co-variations dans l'espace des variables (p x R)
    base_analyse: np.ndarray    # mêmes co-variations dans l'espace d'analyse (R x R)
    etiquettes: list
    saliences: np.ndarray       # K x R : variance de chaque groupe sur chaque co-variation
    seuils: np.ndarray
    motif: dict
    bilan: float
    borne: float
    ecart_relatif: float
    lode: object

    def __repr__(self):
        return (f"CoPRes multigroupe ({len(self.groupes)} groupes) : motif {texte_motif(self.motif, self.groupes)}\n"
                f"bilan {self.bilan:.6g}, borne de Ky Fan {self.borne:.6g}, écart relatif {100 * self.ecart_relatif:.3g} %")

    def parties(self):
        d = {}
        for h, L in enumerate(self.etiquettes):
            d.setdefault(L, []).append(h)
        return d

    def scores(self, prep):
        """Scores des individus de chaque groupe sur les co-variations (données du groupe centrées)."""
        return [x @ self.base for x in prep.Xk]


def copres_groupes(prep, theta, departs=20, graine=1):
    """Étiquetage LODE des co-variations : communes à tous les groupes, partagées par une partie
    d'entre eux, propres à un seul, ou résiduelles."""
    res = lode(prep.covariances, theta, departs=departs, graine=graine)
    return FactorisationGroupes(groupes=prep.groupes, base=prep.U @ res.base, base_analyse=res.base,
                                etiquettes=res.etiquettes, saliences=res.saliences,
                                seuils=np.asarray(theta, float), motif=res.motif, bilan=res.critere,
                                borne=res.borne, ecart_relatif=res.ecart_relatif, lode=res)


# ----------------------------------------------------------------------------- CPC
def cpc(prep, it=200, tol=1e-10):
    """Base des composantes principales communes (algorithme de Flury et Gautschi, pondéré par
    les effectifs), dans l'espace d'analyse."""
    S = prep.covariances
    nk = [len(x) for x in prep.Xk]
    p = S[0].shape[0]
    B = np.linalg.eigh(sum(w * s for w, s in zip(nk, S)))[1][:, ::-1].copy()
    for _ in range(it):
        Bold = B.copy()
        for i in range(p - 1):
            for j in range(i + 1, p):
                H = B[:, [i, j]]
                Tk = [H.T @ s @ H for s in S]
                Q = np.eye(2)
                for _g in range(50):
                    d1 = np.array([Q[:, 0] @ t @ Q[:, 0] for t in Tk])
                    d2 = np.array([Q[:, 1] @ t @ Q[:, 1] for t in Tk])
                    T = sum(w * (a - b) / (a * b) * t for w, a, b, t in zip(nk, d1, d2, Tk))
                    Qn = np.linalg.eigh(T)[1][:, ::-1]
                    fini = np.abs(np.abs(Qn.T @ Q) - np.eye(2)).max() < 1e-12
                    Q = Qn
                    if fini:
                        break
                B[:, [i, j]] = H @ Q
        if np.abs(np.abs(Bold.T @ B) - np.eye(p)).max() < tol:
            break
    for h in range(p):
        if B[np.argmax(np.abs(B[:, h])), h] < 0:
            B[:, h] = -B[:, h]
    return B


@dataclass
class LectureGroupes:
    base: np.ndarray
    saliences: np.ndarray
    etiquettes: list
    motif: dict
    coherente: bool
    marge: float
    bilan: float
    groupes: list

    def __repr__(self):
        return (f"Lecture directe : motif {texte_motif(self.motif, self.groupes)} ; "
                f"{'cohérente' if self.coherente else 'non cohérente'} (marge {self.marge:.3g}) ; bilan {self.bilan:.6g}")


def lecture_groupes(prep, B, theta):
    """Lecture directe d'une base B de l'espace d'analyse (par exemple la base CPC) :
    étiquettes, cohérence et bilan."""
    A = prep.covariances
    etiq = etiquettes(A, B, theta)
    m = marge(A, B, theta, etiq)
    return LectureGroupes(base=B, saliences=saliences(A, B), etiquettes=etiq, motif=motif(etiq),
                          coherente=bool(m >= -1e-9), marge=m, bilan=critere(A, B, theta), groupes=prep.groupes)


# ----------------------------------------------------------------------------- discrimination
def _loo(Y, g, groupes, quadratique):
    """Taux de bon classement de l'analyse discriminante (linéaire ou quadratique) en validation
    croisée un individu à la fois, avec mises à jour exactes des moyennes et des covariances."""
    N = len(Y)
    K = len(groupes)
    idx = {k: np.flatnonzero(g == k) for k in groupes}
    mu = {k: Y[idx[k]].mean(0) for k in groupes}
    SS = {k: (Y[idx[k]] - mu[k]).T @ (Y[idx[k]] - mu[k]) for k in groupes}
    ok = 0
    for i in range(N):
        gi = g[i]
        sc = {}
        for k in groupes:
            m, S, nk = mu[k], SS[k], len(idx[k])
            if k == gi:
                m2 = (m * nk - Y[i]) / (nk - 1)
                S = S - np.outer(Y[i] - m, Y[i] - m) * nk / (nk - 1)
                m, nk = m2, nk - 1
            sc[k] = (m, S, nk)
        Wp = sum(v[1] for v in sc.values()) / (N - 1 - K)
        r = {}
        for k, (m, S, nk) in sc.items():
            C = S / (nk - 1) if quadratique else Wp
            d = Y[i] - m
            r[k] = -0.5 * d @ np.linalg.solve(C, d) - (0.5 * np.linalg.slogdet(C)[1] if quadratique else 0.0)
        ok += max(r, key=r.get) == gi
    return ok / N


def discrimination(cop, prep, classement=True):
    """Confronte l'étiquetage à la discrimination des groupes.

    Plan discriminant : sous-espace engendré par W^{-1} D, où W est la covariance intra-groupe
    commune et D la matrice des écarts des moyennes de groupe à la moyenne générale (dans l'espace
    d'analyse) ; sa dimension est K - 1. Pour chaque partie de l'étiquetage, la part du plan
    discriminant est la somme des carrés des cosinus entre une base orthonormée du plan et la base
    de la partie (entre 0 et K - 1 ; la somme sur les parties vaut K - 1). Si classement, ajoute
    les taux de bon classement linéaire et quadratique en validation croisée un individu à la fois."""
    Y = prep.Xs @ prep.U
    K = len(prep.groupes)
    N = len(Y)
    W = sum(a * (len(x) - 1) for a, x in zip(prep.covariances, prep.Xk)) / (N - K)
    m = Y.mean(0)
    D = np.column_stack([Y[prep.g == k].mean(0) - m for k in prep.groupes])
    Ld = np.linalg.svd(np.linalg.solve(W, D), full_matrices=False)[0][:, :K - 1]
    parts = {L: float(np.sum((Ld.T @ cop.base_analyse[:, idx]) ** 2)) for L, idx in cop.parties().items()}
    out = {"parts_plan_discriminant": parts}
    if classement:
        out["classement_lineaire"] = _loo(Y, prep.g, prep.groupes, False)
        out["classement_quadratique"] = _loo(Y, prep.g, prep.groupes, True)
    return out
