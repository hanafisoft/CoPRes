"""Groupes connus et lecture CoPRes : par quelles parties, donc par quels blocs, passe ce qui
distingue les groupes.

- `information_groupe` : part de l'information de groupe (indicatrice centrée des groupes) portée
  par chaque partie de la lecture CoPRes et hors du sous-espace ; contribution propre et portée
  de chaque bloc. Décomposition exacte, indépendante de la base choisie dans chaque partie.
- `plsda` : base du sous-espace des scores d'une PLS-DA (PLS2 sur le tableau concaténé des blocs
  et l'indicatrice centrée des groupes), à lire ensuite par `copres(X, seuils, U=...)`.
- `lda_loo` : taux de bon classement de l'analyse discriminante linéaire en validation croisée
  un individu à la fois (covariance intra-groupe commune, probabilités a priori égales aux
  proportions de l'échantillon d'apprentissage, comme LinearDiscriminantAnalysis de scikit-learn).
- `discrimination_parties` : analyse discriminante sur chaque partie et sur tout le sous-espace,
  avec un test par permutation des groupes.
- `variance_groupe` : part de la variance de chaque bloc due aux groupes.
- `validation_croisee` : validation croisée un individu à la fois, toute la chaîne étant refaite
  dans chaque pli (centrage, inertie, seuils, méthode, CoPRes) ; prédiction par toutes les
  composantes, par la partie commune, par des parties choisies par validation croisée interne,
  et par chaque bloc seul.
"""
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np

from .alea import ParkMiller
from .lode import nom_etiquette
from .multibloc import seuils_perm, copres, METHODES


def _groupes(g):
    g = np.asarray(g)
    niveaux = sorted(set(g.tolist()))
    return g, niveaux


def indicatrice(g):
    """Indicatrice des groupes, centrée (n x nombre de groupes), colonnes dans l'ordre trié."""
    g, niv = _groupes(g)
    G = np.column_stack([(g == v).astype(float) for v in niv])
    return G - G.mean(0)


# ----------------------------------------------------------------------------- information
@dataclass
class InformationGroupe:
    parts: dict              # {étiquette: part de l'information de groupe}
    hors: float              # part hors du sous-espace retenu
    contribution_propre: np.ndarray   # par bloc : parts des parties propres au bloc
    portee: np.ndarray       # par bloc : parts de toutes les parties que le bloc porte
    dimensions: dict         # {étiquette: dimension de la partie}

    def __repr__(self):
        lignes = [f"  {nom_etiquette(L):12s} dim {self.dimensions[L]} : {100 * p:5.1f} %"
                  for L, p in self.parts.items()]
        lignes.append(f"  {'hors de S':12s}       : {100 * self.hors:5.1f} %")
        lignes.append("  contribution propre : " + ", ".join(f"{100 * c:.1f} %" for c in self.contribution_propre))
        lignes.append("  portée              : " + ", ".join(f"{100 * c:.1f} %" for c in self.portee))
        return "Information de groupe\n" + "\n".join(lignes)


def information_groupe(cop, groupes):
    """Part de l'information de groupe portée par chaque partie de la factorisation CoPRes `cop`
    et hors de son sous-espace. Les parts somment à 1."""
    G = indicatrice(groupes)
    tot = float(np.sum(G ** 2))
    V, U = cop.base, cop.base_methode
    parts, dims = {}, {}
    for L, idx in cop.parties().items():
        E = V[:, idx]
        parts[L] = float(np.sum((E.T @ G) ** 2) / tot)
        dims[L] = len(idx)
    hors = float(np.sum((G - U @ (U.T @ G)) ** 2) / tot)
    K = len(cop.seuils)
    propre = np.array([sum(p for L, p in parts.items() if tuple(L) == (k,)) for k in range(K)])
    portee = np.array([sum(p for L, p in parts.items() if k in L) for k in range(K)])
    return InformationGroupe(parts, hors, propre, portee, dims)


# ----------------------------------------------------------------------------- PLS-DA
def plsda(X, groupes, R):
    """Base orthonormée (n x R) du sous-espace des scores de la PLS-DA : PLS2 sur le tableau
    concaténé des blocs (pris tels quels, en général prétraités par `pretraiter`) et
    l'indicatrice centrée des groupes. À chaque étape, le poids est le premier vecteur singulier
    à gauche de X_h' Y, le score t = X_h w, puis X est déflaté par t."""
    Xh = np.hstack([np.asarray(x, dtype=float) for x in X])
    Xh = Xh - Xh.mean(0)
    Y = indicatrice(groupes)
    T = []
    for _ in range(R):
        u, s, vt = np.linalg.svd(Xh.T @ Y, full_matrices=False)
        w = u[:, 0]
        t = Xh @ w
        T.append(t)
        Xh = Xh - np.outer(t, t @ Xh) / float(t @ t)
    T = np.column_stack(T)
    # les scores sont orthogonaux ; on les normalise (Gram-Schmidt pour la précision)
    Q = T.copy()
    for j in range(R):
        for _ in range(2):
            for i in range(j):
                Q[:, j] -= (Q[:, i] @ Q[:, j]) * Q[:, i]
        Q[:, j] /= np.linalg.norm(Q[:, j])
    return Q


# ----------------------------------------------------------------------------- discrimination
def _pinv(M):
    """Pseudo-inverse, valeurs singulières sous max(dim) * eps * sigma_max tenues pour nulles
    (même règle que la version R)."""
    u, d, vt = np.linalg.svd(M, full_matrices=False)
    tol = max(M.shape) * np.finfo(float).eps * (d.max() if d.size else 0.0)
    dinv = np.where(d > tol, 1.0 / np.where(d > tol, d, 1.0), 0.0)
    return (vt.T * dinv) @ u.T


def _lda_predire(Ztr, gtr, Zte, niveaux):
    """Analyse discriminante linéaire (covariance intra-groupe commune, dénominateur n - nombre de
    groupes, a priori = proportions de l'apprentissage) ; renvoie les groupes prédits."""
    Ztr = np.atleast_2d(Ztr)
    Zte = np.atleast_2d(Zte)
    presents = [v for v in niveaux if np.any(gtr == v)]
    n, q = Ztr.shape
    mu = {v: Ztr[gtr == v].mean(0) for v in presents}
    W = np.zeros((q, q))
    for v in presents:
        D = Ztr[gtr == v] - mu[v]
        W += D.T @ D
    W /= (n - len(presents))
    Winv = _pinv(W)
    sc = np.column_stack([Zte @ Winv @ mu[v] - 0.5 * mu[v] @ Winv @ mu[v] + np.log(np.mean(gtr == v))
                          for v in presents])
    return np.array(presents, dtype=object)[np.argmax(sc, axis=1)]


def lda_loo(Z, groupes):
    """Taux de bon classement de l'analyse discriminante linéaire, validation croisée un individu
    à la fois."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    g, niv = _groupes(groupes)
    n = len(g)
    ok = 0
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        ok += _lda_predire(Z[m], g[m], Z[i:i + 1], niv)[0] == g[i]
    return ok / n


@dataclass
class DiscriminationParties:
    taux: dict          # {étiquette ou "S": taux de bon classement LOO}
    p: dict             # {étiquette ou "S": p-valeur par permutation}
    n_perm: int

    def __repr__(self):
        return "Analyse discriminante par partie (LOO)\n" + "\n".join(
            f"  {('sous-espace' if L == 'S' else nom_etiquette(L)):12s} : {100 * self.taux[L]:5.1f} %"
            f"  (p = {self.p[L]:.3f})" for L in self.taux)


def discrimination_parties(cop, groupes, n_perm=999, graine=1):
    """Analyse discriminante linéaire (LOO) sur les scores de chaque partie de la factorisation
    CoPRes et sur tout le sous-espace ("S"). p-valeur : (1 + nombre de permutations des groupes
    donnant un taux au moins égal) / (n_perm + 1) ; les mêmes permutations servent à toutes les
    parties."""
    g, _ = _groupes(groupes)
    V, U = cop.base, cop.base_methode
    cibles = {L: V[:, idx] for L, idx in cop.parties().items()}
    cibles["S"] = U
    gen = ParkMiller(graine)
    perms = [gen.permutation(len(g)) for _ in range(n_perm)]
    taux, p = {}, {}
    for L, Z in cibles.items():
        t = lda_loo(Z, g)
        nul = sum(lda_loo(Z, g[pi]) >= t - 1e-12 for pi in perms)
        taux[L], p[L] = t, (1 + nul) / (n_perm + 1)
    return DiscriminationParties(taux, p, n_perm)


def variance_groupe(X, groupes):
    """Part de la variance de chaque bloc due aux groupes : somme des carrés inter-groupes sur
    somme des carrés totale, toutes variables du bloc réunies."""
    g, niv = _groupes(groupes)
    out = []
    for x in X:
        x = np.asarray(x, dtype=float)
        xc = x - x.mean(0)
        inter = sum(np.sum(g == v) * np.sum((x[g == v].mean(0) - x.mean(0)) ** 2) for v in niv)
        out.append(float(inter / np.sum(xc ** 2)))
    return np.array(out)


# ----------------------------------------------------------------------------- validation croisée
@dataclass
class ValidationCroisee:
    taux: dict          # {prédicteur: taux de bon classement}
    choix: dict         # {parties choisies: nombre de plis}
    plis: int

    def __repr__(self):
        lignes = [f"  {k:28s} : {100 * v:5.1f} %" for k, v in self.taux.items()]
        lignes.append("  parties choisies : " + "; ".join(f"{k} ({v})" for k, v in self.choix.items()))
        return f"Validation croisée un individu à la fois ({self.plis} plis)\n" + "\n".join(lignes)


def validation_croisee(X, groupes, R, methode="CPCA", nperm=50, ndir=100, departs=20, graine=1,
                       blocs_seuls=True, noms=None):
    """Validation croisée un individu à la fois. Dans chaque pli, les blocs d'apprentissage sont
    centrés et ramenés à l'inertie 1 (l'individu test est transformé de même), les seuils
    recalculés (nperm permutations, graine 1000 * graine + pli), la base de la méthode et la
    factorisation CoPRes recalculées. Le score de l'individu test sur une partie s'obtient par
    les poids w = X_app^+ v (pseudo-inverse du tableau concaténé d'apprentissage). Prédicteurs :
    toutes les composantes ; la partie commune à tous les blocs (si elle existe dans le pli) ;
    les parties choisies par validation croisée interne (meilleur taux, à égalité le premier
    ensemble dans l'ordre des tailles croissantes) ; chaque bloc seul (R premières composantes
    principales du bloc)."""
    X = [np.asarray(x, dtype=float) for x in X]
    g, niv = _groupes(groupes)
    n, K = len(g), len(X)
    noms = noms or [str(k + 1) for k in range(K)]
    tous = tuple(range(K))
    bons = {"toutes les composantes": 0, "partie commune": 0, "parties choisies": 0}
    if blocs_seuls:
        for k in range(K):
            bons[f"bloc {noms[k]} seul"] = 0
    choix = {}
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        app, test = [], []
        for x in X:
            m = x[tr].mean(0)
            s = np.linalg.norm(x[tr] - m)
            app.append((x[tr] - m) / s)
            test.append((x[i:i + 1] - m) / s)
        Xa, Xt = np.hstack(app), np.hstack(test)
        Xp = _pinv(Xa)
        th = seuils_perm(app, nperm=nperm, ndir=ndir, graine=1000 * graine + i + 1)
        cop = copres(app, th, R=R, methode=methode, departs=departs)
        P = {L: cop.base[:, idx] for L, idx in cop.parties().items()}

        def scores(V):
            w = Xp @ V
            return Xa @ w, Xt @ w

        def predit(V):
            a, b = scores(V)
            return _lda_predire(a, g[tr], b, niv)[0] == g[i]

        bons["toutes les composantes"] += predit(cop.base_methode)
        if tous in P:
            bons["partie commune"] += predit(P[tous])
        meilleur, ens = -1.0, None
        cles = list(P)
        for r in range(1, len(cles) + 1):
            for sub in combinations(cles, r):
                a, _ = scores(np.hstack([P[L] for L in sub]))
                t = lda_loo(a, g[tr])
                if t > meilleur + 1e-12:
                    meilleur, ens = t, sub
        nom = " + ".join(nom_etiquette(L, noms) for L in sorted(ens, key=lambda L: (-len(L), L)))
        choix[nom] = choix.get(nom, 0) + 1
        bons["parties choisies"] += predit(np.hstack([P[L] for L in ens]))
        if blocs_seuls:
            for k in range(K):
                _, _, vt = np.linalg.svd(app[k], full_matrices=False)
                w = vt[:R].T
                bons[f"bloc {noms[k]} seul"] += _lda_predire(app[k] @ w, g[tr], test[k] @ w, niv)[0] == g[i]
    taux = {k: v / n for k, v in bons.items()}
    choix = dict(sorted(choix.items(), key=lambda kv: -kv[1]))
    return ValidationCroisee(taux, choix, n)
