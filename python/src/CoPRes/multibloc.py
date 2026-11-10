"""Couche CoPRes pour des données structurées par groupes de variables (blocs mesurés sur les
mêmes individus).

X : liste de K tableaux n x J_k (mêmes individus en lignes). Les matrices étudiées par LODE sont
les restrictions A_k = U' X_k X_k' U au sous-espace engendré par la base U d'une méthode de la
factorisation canonique (CPCA, ACOM, ACCPS ou toute autre base orthonormée de l'espace des
individus centrés).
"""
from dataclasses import dataclass
import numpy as np

from .alea import ParkMiller
from .lode import (lode, saliences, etiquettes, motif, texte_motif, nom_etiquette, critere,
                   est_etiquetage, borne_kf, marge)


# ----------------------------------------------------------------------------- prétraitement
def pretraiter(X, snv=False, reduire=False, inertie=True):
    """Prétraitement bloc par bloc : SNV éventuelle (chaque individu centré et réduit sur les
    variables du bloc), centrage par colonne, réduction éventuelle des colonnes, puis inertie
    totale ramenée à 1."""
    out = []
    for x in X:
        x = np.array(x, dtype=float)
        if snv:
            x = (x - x.mean(1, keepdims=True)) / x.std(1, ddof=1, keepdims=True)
        x = x - x.mean(0)
        if reduire:
            s = x.std(0, ddof=1)
            x = x / np.where(s > 0, s, 1.0)
        if inertie:
            x = x / np.linalg.norm(x)
        out.append(x)
    return out


# ----------------------------------------------------------------------------- seuils
def seuils_perm(X, nperm=200, ndir=100, niveau=0.95, regle="direction", graine=1):
    """Seuils par permutation, bloc par bloc.

    Pour chaque bloc et chaque permutation, chaque colonne est permutée indépendamment des
    autres (rangs de n tirages uniformes), puis recentrée. Règle "direction" : salience
    u' X0 X0' u de ndir directions tirées uniformément sur la sphère des individus centrés ;
    règle "valeur_propre" : plus grande valeur propre de X0' X0. Le seuil est le quantile
    `niveau` (définition linéaire, type 7 de R) des valeurs obtenues.
    Ordre des tirages : blocs, puis permutations, puis colonnes, puis directions."""
    gen = ParkMiller(graine)
    th = []
    for x in X:
        x = np.asarray(x, dtype=float)
        n, J = x.shape
        vals = []
        for _ in range(nperm):
            u = gen.uniformes(n * J).reshape(J, n)
            idx = np.argsort(u, axis=1, kind="stable")
            X0 = np.take_along_axis(x.T, idx, axis=1).T
            X0 = X0 - X0.mean(0)
            if regle == "direction":
                Z = gen.normales(n * ndir).reshape(ndir, n).T
                Z = Z - Z.mean(0)
                Z = Z / np.linalg.norm(Z, axis=0)
                vals.append(((X0.T @ Z) ** 2).sum(0))
            elif regle == "valeur_propre":
                vals.append([np.linalg.eigvalsh(X0.T @ X0 if J <= n else X0 @ X0.T)[-1]])
            else:
                raise ValueError("regle doit valoir 'direction' ou 'valeur_propre'")
        th.append(float(np.quantile(np.concatenate([np.ravel(v) for v in vals]), niveau)))
    return np.array(th)


# ----------------------------------------------------------------------------- méthodes
def _signes(U):
    U = U.copy()
    for h in range(U.shape[1]):
        if U[np.argmax(np.abs(U[:, h])), h] < 0:
            U[:, h] = -U[:, h]
    return U


def cpca(X, R):
    """Composantes globales de la CPCA : R vecteurs propres dominants de la somme des X_k X_k'."""
    W = sum(x @ x.T for x in X)
    return _signes(np.linalg.eigh(W)[1][:, ::-1][:, :R])


def acom(X, R):
    """Composantes globales de l'ACOM : même première composante que la CPCA, puis déflation de
    chaque bloc par son axe ; les composantes sont ensuite orthonormalisées (Gram-Schmidt)."""
    Xh = [np.array(x, dtype=float) for x in X]
    U = []
    for _ in range(R):
        u = np.linalg.eigh(sum(x @ x.T for x in Xh))[1][:, -1]
        U.append(u)
        for i, x in enumerate(Xh):
            a = x.T @ u
            a = a / np.linalg.norm(a)
            Xh[i] = x - np.outer(x @ a, a)
    return _signes(_gram_schmidt(np.column_stack(U)))


def _gram_schmidt(M):
    Q = np.zeros_like(M)
    for h in range(M.shape[1]):
        v = M[:, h] - Q[:, :h] @ (Q[:, :h].T @ M[:, h])
        v = v - Q[:, :h] @ (Q[:, :h].T @ v)
        Q[:, h] = v / np.linalg.norm(v)
    return Q


def accps(X, R, it=3000, tol=1e-12, nb_aleatoires=10, graine=1):
    """Composantes globales de l'ACCPS (version séquentielle d'INDORT) : chaque composante
    maximise la somme des carrés des saliences, orthogonalement aux précédentes. Départs : vecteur
    propre dominant de la somme, de chaque bloc, et nb_aleatoires directions tirées au hasard."""
    W = [x @ x.T for x in X]
    n = W[0].shape[0]
    P = np.eye(n)
    gen = ParkMiller(graine)
    U = []
    for _ in range(R):
        Wd = [P @ w @ P for w in W]
        departs = [np.linalg.eigh(sum(Wd))[1][:, -1]] + [np.linalg.eigh(w)[1][:, -1] for w in Wd]
        departs += [P @ gen.normales(n) for _ in range(nb_aleatoires)]
        best = None
        for u in departs:
            u = u / np.linalg.norm(u)
            f0 = -1.0
            for _ in range(it):
                g = sum((u @ w @ u) * (w @ u) for w in Wd)
                u = g / np.linalg.norm(g)
                f = sum((u @ w @ u) ** 2 for w in Wd)
                if abs(f - f0) < tol:
                    break
                f0 = f
            if best is None or f > best[0] + 1e-14:
                best = (f, u)
        U.append(best[1])
        P = P - np.outer(best[1], best[1])
    return _signes(_gram_schmidt(np.column_stack(U)))


METHODES = {"CPCA": cpca, "ACOM": acom, "ACCPS": accps}


# ----------------------------------------------------------------------------- lecture et CoPRes
def _restrictions(X, U):
    return [U.T @ x @ x.T @ U for x in X]


@dataclass
class LectureDirecte:
    base: np.ndarray
    saliences: np.ndarray
    seuils: np.ndarray
    etiquettes: list
    motif: dict
    coherente: bool
    marge: float
    bilan: float

    def __repr__(self):
        return (f"Lecture directe : motif {texte_motif(self.motif)} ; "
                f"{'cohérente' if self.coherente else 'non cohérente'} (marge {self.marge:.3g}) ; bilan {self.bilan:.6g}")


def lecture_directe(X, U, theta):
    """Lecture d'une base U composante par composante : étiquettes, cohérence (la décomposition
    engendrée est-elle un étiquetage ?) et bilan Phi(U)."""
    U = np.asarray(U, dtype=float)
    A = _restrictions(X, U)
    I = np.eye(U.shape[1])
    etiq = etiquettes(A, I, theta)
    m = marge(A, I, theta, etiq)
    return LectureDirecte(base=U, saliences=saliences(A, I), seuils=np.asarray(theta, float),
                          etiquettes=etiq, motif=motif(etiq), coherente=bool(m >= -1e-9), marge=m,
                          bilan=critere(A, I, theta))


@dataclass
class FactorisationCoPRes:
    """Factorisation CoPRes du sous-espace d'une méthode : X_k ~ V diag(poids_k) H_k'."""
    methode: str
    base_methode: np.ndarray
    base: np.ndarray            # V = U Q, n x R
    etiquettes: list
    saliences: np.ndarray       # K x R, carrés des poids
    poids: np.ndarray           # K x R
    loadings: list              # H_k, J_k x R
    seuils: np.ndarray
    motif: dict
    bilan: float
    borne: float
    ecart_relatif: float
    parts_sous_espace: np.ndarray   # tau_k = tr(A_k)
    lode: object

    def __repr__(self):
        return (f"CoPRes ({self.methode}, R = {self.base.shape[1]}) : motif {texte_motif(self.motif)}\n"
                f"bilan {self.bilan:.6g}, borne de Ky Fan {self.borne:.6g}, écart relatif {100 * self.ecart_relatif:.3g} %")

    def parties(self):
        """{étiquette: indices des composantes CoPRes de cette étiquette}."""
        g = {}
        for h, L in enumerate(self.etiquettes):
            g.setdefault(L, []).append(h)
        return g


def copres(X, theta, R=None, methode="CPCA", U=None, departs=20, graine=1):
    """Factorisation CoPRes : applique LODE aux restrictions A_k = U' X_k X_k' U et réécrit la
    factorisation canonique de la méthode dans la base V = U Q de l'étiquetage obtenu, à
    sous-espace et ajustement égaux. U est fourni, ou calculé par `methode` avec R composantes."""
    if U is None:
        if R is None:
            raise ValueError("donner R ou U")
        U = METHODES[methode](X, R)
        nom = methode
    else:
        U = np.asarray(U, dtype=float)
        nom = methode if methode else "base fournie"
    A = _restrictions(X, U)
    res = lode(A, theta, departs=departs, graine=graine)
    V = U @ res.base
    poids = np.array([np.linalg.norm(x.T @ V, axis=0) for x in X])
    H = [x.T @ V / np.where(p > 0, p, 1.0) for x, p in zip(X, poids)]
    return FactorisationCoPRes(methode=nom, base_methode=U, base=V, etiquettes=res.etiquettes,
                  saliences=res.saliences, poids=poids, loadings=H, seuils=np.asarray(theta, float),
                  motif=res.motif, bilan=res.critere, borne=res.borne, ecart_relatif=res.ecart_relatif,
                  parts_sous_espace=np.array([np.trace(a) for a in A]), lode=res)


# ----------------------------------------------------------------------------- fiche, équivalence
@dataclass
class Fiche:
    composition: list       # pour chaque u_h : {étiquette: part}
    principale: list        # étiquette principale de chaque u_h (None si aucune part de signal)
    melange: np.ndarray     # mu*_h
    residuel: np.ndarray    # r_h
    melangee: np.ndarray    # mu*_h > 0,1
    diluee: np.ndarray      # 0,1 < r_h < 0,9

    def __repr__(self):
        lignes = []
        for h, (c, L) in enumerate(zip(self.composition, self.principale)):
            parts = ", ".join(f"{nom_etiquette(k)} {100 * v:.0f} %" for k, v in c.items() if v >= 0.005)
            lignes.append(f"u{h + 1} : {parts} ; mu* = {self.melange[h]:.2f}, r = {self.residuel[h]:.2f}"
                          + (" ; mélangée" if self.melangee[h] else "") + (" ; diluée" if self.diluee[h] else ""))
        return "\n".join(lignes)


def fiche(U, cop, seuil_melange=0.10, bornes_dilution=(0.10, 0.90)):
    """Composition de chaque composante u_h (colonnes de U, dans le sous-espace de `cop`) dans la
    base CoPRes : u_h = somme des gamma_jh v_j ; parts gamma_jh^2 regroupées par étiquette.
    Étiquette principale : étiquette non vide de plus grande part ; mu*_h : part prise aux autres
    étiquettes non vides ; r_h : part résiduelle."""
    U = cop.base_methode if U is None else np.asarray(U, dtype=float)
    G = (cop.base.T @ U) ** 2
    comp, princ, mu, r = [], [], [], []
    for h in range(U.shape[1]):
        c = {}
        for j, L in enumerate(cop.etiquettes):
            c[L] = c.get(L, 0.0) + G[j, h]
        sig = {L: v for L, v in c.items() if L}
        rh = c.get((), 0.0)
        if sig:
            Lp = max(sig, key=sig.get)
            mh = sum(sig.values()) - sig[Lp]
        else:
            Lp, mh = None, 0.0
        comp.append(dict(sorted(c.items(), key=lambda kv: -kv[1])))
        princ.append(Lp)
        mu.append(mh)
        r.append(rh)
    mu, r = np.array(mu), np.array(r)
    return Fiche(composition=comp, principale=princ, melange=mu, residuel=r,
                 melangee=mu > seuil_melange, diluee=(r > bornes_dilution[0]) & (r < bornes_dilution[1]))


def proximite(V1, V2):
    """Plus petit carré du cosinus des angles principaux entre deux sous-espaces de même
    dimension (None si les dimensions diffèrent)."""
    if V1.shape[1] != V2.shape[1]:
        return None
    return float(np.linalg.svd(V1.T @ V2, compute_uv=False).min() ** 2)


def equivalence(cop1, cop2, seuil=0.72):
    """Deux factorisations CoPRes sont équivalentes si elles ont le même motif et si chaque
    partie de l'une est proche (proximité >= seuil) de la partie de même étiquette de l'autre.
    Renvoie (équivalentes, {étiquette: proximité})."""
    if cop1.motif != cop2.motif:
        return False, {}
    p1, p2 = cop1.parties(), cop2.parties()
    prox = {L: proximite(cop1.base[:, p1[L]], cop2.base[:, p2[L]]) for L in p1}
    return all(v >= seuil for v in prox.values()), prox


def sensibilite(X, theta, R_liste=(2, 3, 4, 5), facteurs=(0.75, 1.0, 1.5, 2.0), methode="CPCA",
                departs=20, graine=1):
    """Motif CoPRes en fonction du nombre de composantes R et d'un facteur multiplicatif des
    seuils. Renvoie {(R, facteur): texte du motif}."""
    out = {}
    for R in R_liste:
        U = METHODES[methode](X, R)
        for c in facteurs:
            cop = copres(X, np.asarray(theta) * c, U=U, methode=methode, departs=departs, graine=graine)
            out[(R, c)] = texte_motif(cop.motif)
    return out
