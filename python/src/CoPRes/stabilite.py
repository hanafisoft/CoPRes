"""Stabilité de l'étiquetage par rééchantillonnage bootstrap.

Pour chaque partie de l'étiquetage de référence (étiquette L, dimension d, base E), on compte les
échantillons bootstrap où elle est retrouvée, selon trois critères de plus en plus larges :
- « strict » : l'échantillon a une partie de même étiquette et de même dimension, proche de E
  (proximité, plus petit carré du cosinus des angles principaux, au moins égale à `seuil`) ;
- « étiquette » : l'échantillon a une partie de même étiquette, de dimension au moins d, qui
  contient E à `seuil` près (les d plus grands carrés des cosinus sont au moins égaux à `seuil`) ;
- « type » : E est contenue, au même sens, dans la réunion des parties de même type (commune à
  tous les groupes ou blocs, partagée par une partie d'entre eux, propre à un seul, résiduelle).
On compte aussi les échantillons qui redonnent le motif complet.
"""
from dataclasses import dataclass, field
import numpy as np

from .alea import ParkMiller
from .lode import texte_motif, nom_etiquette
from .multibloc import METHODES, seuils_perm, copres
from .multigroupe import copres_groupes


def type_etiquette(L, K):
    if len(L) == 0:
        return "résiduelle"
    if len(L) == 1:
        return "propre"
    return "commune" if len(L) == K else "partagée"


def _inclusion(E, F):
    """Plus petit des d carrés des cosinus des angles principaux entre E (dimension d) et F
    (dimension au moins d) ; -1 si F est trop petit."""
    if F.shape[1] < E.shape[1]:
        return -1.0
    s = np.linalg.svd(E.T @ F, compute_uv=False)[:E.shape[1]]
    return float(s.min() ** 2)


def _comparer(ref, boot, K, seuil):
    """ref, boot : {étiquette: base orthonormée}. Renvoie {étiquette de ref: (strict, étiquette, type)}."""
    out = {}
    for L, E in ref.items():
        F = boot.get(L)
        strict = F is not None and F.shape[1] == E.shape[1] and _inclusion(E, F) >= seuil
        etiq = F is not None and _inclusion(E, F) >= seuil
        t = type_etiquette(L, K)
        Fs = [B for M, B in boot.items() if type_etiquette(M, K) == t]
        typ = bool(Fs) and _inclusion(E, np.column_stack(Fs)) >= seuil
        out[L] = (bool(strict), bool(etiq), bool(typ))
    return out


@dataclass
class Stabilite:
    nb: int
    noms: list
    dimensions: dict          # {étiquette: dimension de la partie de référence}
    strict: dict              # {étiquette: fréquence}
    etiquette: dict
    type: dict
    motif_retrouve: float
    motifs: dict = field(default_factory=dict)   # {texte du motif: nombre d'échantillons}

    def __repr__(self):
        l = [f"Stabilité sur {self.nb} échantillons bootstrap ; motif complet retrouvé : {100 * self.motif_retrouve:.0f} %",
             "partie (dimension) : strict / étiquette / type"]
        for L, d in self.dimensions.items():
            l.append(f"  {nom_etiquette(L, self.noms)} ({d}) : {100 * self.strict[L]:.0f} / "
                     f"{100 * self.etiquette[L]:.0f} / {100 * self.type[L]:.0f} %")
        return "\n".join(l)


def _resume(ref, res, motifs, motif_ref, B, noms):
    dims = {L: E.shape[1] for L, E in ref.items()}
    f = lambda i: {L: sum(r[L][i] for r in res) / B for L in ref}
    return Stabilite(nb=B, noms=noms, dimensions=dims, strict=f(0), etiquette=f(1), type=f(2),
                     motif_retrouve=sum(m == motif_ref for m in motifs) / B,
                     motifs=dict(sorted({m: motifs.count(m) for m in set(motifs)}.items(), key=lambda kv: -kv[1])))


def _parties(base, etiquettes):
    d = {}
    for h, L in enumerate(etiquettes):
        d.setdefault(L, []).append(h)
    return {L: base[:, idx] for L, idx in d.items()}


def _indices(gen, strates):
    """Indices bootstrap, tirés avec remise à l'intérieur de chaque strate (ordre de première
    apparition des strates) : floor(u * n_s) pour n_s tirages uniformes."""
    idx = []
    for s in dict.fromkeys(strates.tolist()):
        membres = np.flatnonzero(strates == s)
        u = gen.uniformes(len(membres))
        idx.append(membres[np.floor(u * len(membres)).astype(int)])
    return np.concatenate(idx)


# ----------------------------------------------------------------------------- multigroupe
def stabilite_groupes(cop, prep, B=100, graine=1, departs=20, seuil=0.72):
    """Bootstrap des individus à l'intérieur de chaque groupe, seuils de référence conservés ;
    les covariances sont recalculées dans l'espace d'analyse de `prep`, et l'étiquetage est
    recalculé par LODE (graine b + 1 pour l'échantillon b)."""
    gen = ParkMiller(graine)
    K = len(prep.groupes)
    ref = _parties(cop.base_analyse, cop.etiquettes)
    motif_ref = texte_motif(cop.motif)
    res, motifs = [], []
    for b in range(B):
        A = []
        for x in prep.Xk:
            u = gen.uniformes(len(x))
            xb = x[np.floor(u * len(x)).astype(int)]
            xb = xb - xb.mean(0)
            A.append(prep.U.T @ (xb.T @ xb / (len(xb) - 1)) @ prep.U)
        p2 = type(prep)(groupes=prep.groupes, g=prep.g, Xs=prep.Xs, Xk=prep.Xk, covariances=A, U=prep.U,
                        ecarts_types=prep.ecarts_types)
        cb = copres_groupes(p2, cop.seuils, departs=departs, graine=b + 1)
        res.append(_comparer(ref, _parties(cb.base_analyse, cb.etiquettes), K, seuil))
        motifs.append(texte_motif(cb.motif))
    return _resume(ref, res, motifs, motif_ref, B, prep.groupes)


# ----------------------------------------------------------------------------- multibloc
def stabilite_blocs(cop, X, B=100, strates=None, graine=1, departs=20, seuil=0.72, nperm=50, ndir=100,
                    recalculer_seuils=True, noms=None):
    """Bootstrap des individus (à l'intérieur de chaque strate si `strates` est donné, à effectif
    constant) ; sur chaque échantillon, les blocs sont recentrés et ramenés à l'inertie 1, les
    seuils recalculés (nperm permutations, graine 1000 * graine + b + 1) sauf si
    recalculer_seuils = False, la base de la méthode et la factorisation CoPRes recalculées.
    Les parties de l'échantillon sont ramenées aux individus d'origine : les poids des variables
    qui donnent une partie sur l'échantillon (pseudo-inverse du tableau concaténé) sont appliqués
    au tableau d'origine, puis les scores sont centrés et orthonormalisés."""
    X = [np.asarray(x, dtype=float) for x in X]
    n = X[0].shape[0]
    K = len(X)
    strates = np.zeros(n, dtype=int) if strates is None else np.asarray(strates)
    R = cop.base.shape[1]
    methode = cop.methode if cop.methode in METHODES else "CPCA"
    gen = ParkMiller(graine)
    ref = _parties(cop.base, cop.etiquettes)
    motif_ref = texte_motif(cop.motif)
    Xcat = np.hstack(X)
    res, motifs = [], []
    for b in range(B):
        idx = _indices(gen, strates)
        Xb = []
        for x in X:
            z = x[idx] - x[idx].mean(0)
            Xb.append(z / np.linalg.norm(z))
        th = seuils_perm(Xb, nperm=nperm, ndir=ndir, graine=1000 * graine + b + 1) if recalculer_seuils else cop.seuils
        cb = copres(Xb, th, U=METHODES[methode](Xb, R), methode=methode, departs=departs, graine=b + 1)
        P = np.linalg.pinv(np.hstack(Xb), rcond=1e-15)
        boot = {}
        for L, E in _parties(cb.base, cb.etiquettes).items():
            S = Xcat @ (P @ E)
            S = S - S.mean(0)
            Us, sv, _ = np.linalg.svd(S, full_matrices=False)
            boot[L] = Us[:, sv > 1e-10 * max(1.0, sv.max())]
        res.append(_comparer(ref, boot, K, seuil))
        motifs.append(texte_motif(cb.motif))
    return _resume(ref, res, motifs, motif_ref, B, noms)
