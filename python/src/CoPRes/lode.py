"""Module lode : décomposition orthogonale étiquetée optimale de matrices semi-définies positives.

Notations de l'article LODE : A = [A_1, ..., A_K], matrices n x n semi-définies positives ;
theta = seuils ; V = base orthonormée de R^n (colonnes) ; s_kh = v_h' A_k v_h ;
l'étiquette d'une composante est l'ensemble des matrices k telles que s_kh >= theta_k.
Les étiquettes sont des tuples d'indices de matrices, comptés à partir de 0 ; l'étiquette
vide () désigne la part résiduelle. Pour l'affichage, `nom_etiquette` les numérote à partir de 1.
"""
from dataclasses import dataclass, field
import numpy as np

from .alea import ParkMiller, rotation_aleatoire

EPS = 1e-12


# ----------------------------------------------------------------------------- outils
def _verifier(A, theta):
    A = [np.asarray(a, dtype=float) for a in A]
    n = A[0].shape[0]
    for a in A:
        if a.shape != (n, n):
            raise ValueError("les matrices doivent toutes être carrées et de même taille")
        if not np.allclose(a, a.T, atol=1e-10 * max(1.0, np.abs(a).max())):
            raise ValueError("les matrices doivent être symétriques")
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.size != len(A):
        raise ValueError("il faut un seuil par matrice")
    if np.any(theta <= 0):
        raise ValueError("les seuils doivent être strictement positifs")
    return A, theta


def nom_etiquette(L, noms=None):
    """Nom lisible d'une étiquette : (0, 2) -> '{1,3}' (ou '{A,C}' avec noms = ['A', 'B', 'C']),
    () -> 'résiduelle'."""
    if not L:
        return "résiduelle"
    return "{" + ",".join(str(noms[k]) if noms is not None else str(k + 1) for k in L) + "}"


def saliences(A, V):
    """Matrice K x d des saliences s_kh = v_h' A_k v_h."""
    V = np.asarray(V, dtype=float)
    return np.array([np.einsum("ih,ij,jh->h", V, a, V) for a in A])


def etiquettes(A, V, theta):
    """Étiquette de chaque colonne de V (tuple des indices k tels que s_kh >= theta_k)."""
    S = saliences(A, V)
    return [tuple(int(k) for k in np.flatnonzero(S[:, h] >= theta)) for h in range(S.shape[1])]


def motif(etiq):
    """Motif d'une liste d'étiquettes : dictionnaire {étiquette: multiplicité}, trié."""
    m = {}
    for L in etiq:
        m[L] = m.get(L, 0) + 1
    return dict(sorted(m.items(), key=lambda kv: (len(kv[0]) == 0, -len(kv[0]), kv[0])))


def texte_motif(m, noms=None):
    return ", ".join(f"({nom_etiquette(L, noms)}, {d})" for L, d in m.items())


def _groupes(etiq):
    g = {}
    for h, L in enumerate(etiq):
        g.setdefault(L, []).append(h)
    return g


def critere(A, V, theta):
    """Critère Psi_theta(V) = somme des (s_kh - theta_k)_+ ; égal à Phi(E(V))."""
    S = saliences(A, V)
    return float(np.clip(S - np.asarray(theta)[:, None], 0, None).sum())


def critere_etiquetage(A, parties, theta):
    """Phi(E) = somme sur les parties L et les k de L de tr(V_L' A_k V_L) - d_L theta_k."""
    phi = 0.0
    for L, VL in parties.items():
        for k in L:
            phi += float(np.trace(VL.T @ A[k] @ VL)) - VL.shape[1] * theta[k]
    return phi


def marge(A, V, theta, etiq=None):
    """Marge d'étiquetage de la base V : minimum, sur les groupes de même étiquette de
    multiplicité au moins 2, de lambda_min - theta_k (k dans L) et de theta_k - lambda_max
    (k hors de L). Une marge positive ou nulle signifie que V est une base d'étiquetage.
    Renvoie +inf s'il n'y a aucun groupe de multiplicité au moins 2."""
    A, theta = _verifier(A, theta)
    etiq = etiquettes(A, V, theta) if etiq is None else etiq
    m = np.inf
    for L, idx in _groupes(etiq).items():
        if len(idx) < 2:
            continue
        VL = V[:, idx]
        for k in range(len(A)):
            ev = np.linalg.eigvalsh(VL.T @ A[k] @ VL)
            m = min(m, ev[0] - theta[k] if k in L else theta[k] - ev[-1])
    return float(m)


def est_etiquetage(A, V, theta, eps=1e-9):
    """Vrai si la décomposition engendrée par V est un étiquetage (à eps près)."""
    return marge(A, V, theta) >= -eps


def borne_kf(A, theta):
    """Borne de Ky Fan : somme des tr(A_k - theta_k I)_+."""
    A, theta = _verifier(A, theta)
    return float(sum(np.clip(np.linalg.eigvalsh(a) - t, 0, None).sum() for a, t in zip(A, theta)))


# ----------------------------------------------------------------------------- étiquetage idéal
def ideal(A, theta, tol=1e-9):
    """Teste si les projecteurs des analyses à seuil commutent deux à deux. Si oui, renvoie
    (True, parties) où parties est le dictionnaire {étiquette: base orthonormée de E_L} de
    l'étiquetage idéal ; sinon (False, None)."""
    A, theta = _verifier(A, theta)
    n, K = A[0].shape[0], len(A)
    P = []
    for a, t in zip(A, theta):
        ev, U = np.linalg.eigh(a)
        Uk = U[:, ev > t]
        P.append(Uk @ Uk.T)
    for k in range(K):
        for l in range(k + 1, K):
            if np.abs(P[k] @ P[l] - P[l] @ P[k]).max() > tol:
                return False, None
    I = np.eye(n)
    parties = {}
    for code in range(2 ** K):
        L = tuple(k for k in range(K) if code >> k & 1)
        PL = I.copy()
        for k in range(K):
            PL = PL @ (P[k] if k in L else I - P[k])
        ev, U = np.linalg.eigh((PL + PL.T) / 2)
        VL = U[:, ev > 0.5]
        if VL.shape[1]:
            parties[L] = VL
    return True, parties


# ----------------------------------------------------------------------------- concentration
def _polaire(G):
    U, _, Vt = np.linalg.svd(G)
    return U @ Vt


def concentration(A, V0=None, it=1000, tol=EPS):
    """Maximise le critère d'INDORT C(V) = somme des s_kh^2 par l'algorithme polaire,
    à partir de V0 (identité par défaut). Le critère ne décroît pas d'une itération à l'autre."""
    n = A[0].shape[0]
    Q = np.eye(n) if V0 is None else np.array(V0, dtype=float)
    c_old = -np.inf
    for _ in range(it):
        S = saliences(A, Q)
        G = np.column_stack([sum(S[k, h] * (A[k] @ Q[:, h]) for k in range(len(A))) for h in range(n)])
        Q = _polaire(G)
        c = float((saliences(A, Q) ** 2).sum())
        if c - c_old <= tol * max(1.0, abs(c)):
            break
        c_old = c
    return Q


def base_concentree(A, graine=1, nb_aleatoires=4):
    """Base concentrée : meilleure des solutions de l'algorithme polaire partant de l'identité
    puis de nb_aleatoires rotations tirées au hasard (annexe B de LODE)."""
    n = A[0].shape[0]
    gen = ParkMiller(graine)
    departs = [np.eye(n)] + [rotation_aleatoire(n, gen) for _ in range(nb_aleatoires)]
    best, cbest = None, -np.inf
    for V0 in departs:
        Q = concentration(A, V0)
        c = float((saliences(A, Q) ** 2).sum())
        if c > cbest + 1e-14:
            best, cbest = Q, c
    return best


# ----------------------------------------------------------------------------- TARO et B2BL
def _somme(A, L, n):
    return sum(A[k] for k in L) if L else np.zeros((n, n))


def taro(V, cible, A, eps=EPS, max_balayages=10000):
    """Rotation vers la cible : à étiquettes fixées (liste `cible`, une étiquette par colonne),
    maximise J_T(V) = somme des v_h' A_{L_h} v_h par rotations planes exactes des paires
    d'étiquettes différentes. Arrêt quand un balayage augmente J_T de moins de eps en valeur
    relative."""
    V = np.array(V, dtype=float)
    n = V.shape[0]
    AL = [_somme(A, L, n) for L in cible]
    for _ in range(max_balayages):
        gain = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if cible[i] == cible[j]:
                    continue
                D = AL[i] - AL[j]
                vi, vj = V[:, i].copy(), V[:, j].copy()
                b = 0.5 * (vi @ D @ vi - vj @ D @ vj)
                g = vi @ D @ vj
                phi = 0.5 * np.arctan2(g, b)
                c, s = np.cos(phi), np.sin(phi)
                V[:, i], V[:, j] = c * vi + s * vj, -s * vi + c * vj
                gain += np.hypot(b, g) - b
        J = sum(V[:, h] @ AL[h] @ V[:, h] for h in range(n))
        if gain <= eps * max(1.0, abs(J)):
            break
    return V


def b2bl(U, A, theta, eps=EPS, max_remplacements=None):
    """Construction d'une base d'étiquetage à partir de la base U (algorithme B2BL).
    Renvoie (V, nombre de remplacements effectués)."""
    A, theta = _verifier(A, theta)
    V = np.array(U, dtype=float)
    K = len(A)
    if max_remplacements is None:
        max_remplacements = int(sum(np.trace(a) for a in A) / eps) + 1
    nb = 0
    while nb < max_remplacements:
        trouve = False
        for L, idx in _groupes(etiquettes(A, V, theta)).items():
            if len(idx) < 2:
                continue
            VL = V[:, idx]
            for k in range(K):
                ev, Q = np.linalg.eigh(VL.T @ A[k] @ VL)
                if (k in L and ev[0] < theta[k] - eps) or (k not in L and ev[-1] > theta[k] + eps):
                    V[:, idx] = VL @ Q
                    nb += 1
                    trouve = True
                    break
            if trouve:
                break
        if not trouve:
            break
    return V, nb


# ----------------------------------------------------------------------------- algorithme LODE
def _depuis(A, theta, V0, eps, maxit):
    """Boucle (b)-(c) de l'algorithme LODE depuis la base V0. Renvoie (V, critère, itérations, convergé)."""
    V, _ = b2bl(V0, A, theta, eps)
    p_old = critere(A, V, theta)
    for t in range(1, maxit + 1):
        cible = etiquettes(A, V, theta)
        V = taro(V, cible, A, eps)
        V, nb = b2bl(V, A, theta, eps)
        p = critere(A, V, theta)
        if nb == 0 and p <= (1 + eps) * p_old:
            return V, p, t, True
        p_old = p
    return V, critere(A, V, theta), maxit, False


def _orienter(A, V, etiq):
    """Étape (d) : dans chaque partie de dimension au moins 2, axes principaux de la somme des
    matrices de son étiquette (toutes les matrices pour la partie résiduelle), puis signe
    (plus grand coefficient positif) et rangement par somme décroissante des saliences."""
    V = V.copy()
    K = len(A)
    for L, idx in _groupes(etiq).items():
        if len(idx) < 2:
            continue
        VL = V[:, idx]
        M = sum(VL.T @ A[k] @ VL for k in (L if L else range(K)))
        ev, P = np.linalg.eigh(M)
        V[:, idx] = VL @ P[:, ::-1]
    for h in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, h])), h] < 0:
            V[:, h] = -V[:, h]
    ordre = np.argsort(-saliences(A, V).sum(0), kind="stable")
    return V[:, ordre], [etiq[h] for h in ordre]


@dataclass
class ResultatLODE:
    """Résultat de l'algorithme LODE."""
    base: np.ndarray                      # n x n, composantes en colonnes
    etiquettes: list                      # étiquette de chaque composante
    saliences: np.ndarray                 # K x n
    seuils: np.ndarray
    parties: dict                         # {étiquette: base orthonormée de E_L}
    motif: dict                           # {étiquette: multiplicité}
    critere: float                        # Phi(E)
    borne: float                          # eta_KF
    ecart_relatif: float                  # 1 - Phi / eta_KF
    marge: float
    criteres_departs: list = field(default_factory=list)
    departs_au_meilleur: int = 0
    converge: bool = True
    matrices: list = field(default_factory=list)   # les matrices A_k analysées

    def __repr__(self):
        return (f"Étiquetage LODE : motif {texte_motif(self.motif)}\n"
                f"critère {self.critere:.6g}, borne de Ky Fan {self.borne:.6g}, "
                f"écart relatif {100 * self.ecart_relatif:.3g} %\n"
                f"meilleur critère atteint par {self.departs_au_meilleur} départ(s) sur {len(self.criteres_departs)}"
                + ("" if self.converge else "\nattention : nombre maximal d'itérations atteint pour au moins un départ"))


def lode(A, theta, departs=20, graine=1, depart_propre=False, eps=EPS, maxit=30, tol_meilleur=1e-9):
    """Algorithme LODE avec plusieurs départs.

    Départs : la base concentrée, puis departs - 1 rotations tirées au hasard (générateur de
    Park et Miller, graine `graine`) ; si depart_propre, la base propre de la somme des A_k est
    ajoutée. On retient l'étiquetage de plus grand critère, puis on l'oriente (étape (d))."""
    A, theta = _verifier(A, theta)
    n = A[0].shape[0]
    gen = ParkMiller(graine)
    liste = [base_concentree(A, graine=graine)]
    if depart_propre:
        liste.append(np.linalg.eigh(sum(A))[1][:, ::-1])
    liste += [rotation_aleatoire(n, gen) for _ in range(max(0, departs - 1))]
    res = [_depuis(A, theta, V0, eps, maxit) for V0 in liste]
    crits = [r[1] for r in res]
    i = int(np.argmax(crits))
    V = res[i][0]
    etiq = etiquettes(A, V, theta)
    m = marge(A, V, theta, etiq)
    V, etiq = _orienter(A, V, etiq)
    parties = {L: V[:, idx] for L, idx in _groupes(etiq).items()}
    phi = critere_etiquetage(A, parties, theta)
    eta = borne_kf(A, theta)
    au_meilleur = int(sum(c >= crits[i] - tol_meilleur * max(1.0, abs(crits[i])) for c in crits))
    return ResultatLODE(base=V, etiquettes=etiq, saliences=saliences(A, V), seuils=theta,
                        parties=dict(sorted(parties.items(), key=lambda kv: (len(kv[0]) == 0, -len(kv[0]), kv[0]))),
                        motif=motif(etiq), critere=phi, borne=eta,
                        ecart_relatif=(1 - phi / eta) if eta > 0 else 0.0, marge=m,
                        criteres_departs=crits, departs_au_meilleur=au_meilleur,
                        converge=all(r[3] for r in res), matrices=A)
