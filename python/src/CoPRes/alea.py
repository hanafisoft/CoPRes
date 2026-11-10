"""Générateur pseudo-aléatoire commun aux versions R et Python de CoPRes.

Générateur multiplicatif de Park et Miller (a = 16 807, m = 2^31 - 1). Les mêmes graines
donnent les mêmes suites dans les deux langages, donc les mêmes départs et les mêmes
permutations. Les tirages en série (`uniformes`, `normales`) sont vectorisés : l'état k pas
plus loin est a^k * etat mod m.
"""
import math
import numpy as np

_A = 16807
_M = 2147483647
_BLOC = 65536
_PUISSANCES = None


def _puissances():
    """a^k mod m pour k = 1, ..., _BLOC (calculées une fois)."""
    global _PUISSANCES
    if _PUISSANCES is None:
        p = np.empty(_BLOC, dtype=np.int64)
        x = 1
        for k in range(_BLOC):
            x = (_A * x) % _M
            p[k] = x
        _PUISSANCES = p
    return _PUISSANCES


class ParkMiller:
    """Générateur de Park et Miller ; graine entière dans [1, m - 1]."""

    def __init__(self, graine=1):
        g = int(graine) % _M
        self.etat = g if g > 0 else 1

    def uniforme(self):
        """Tirage uniforme dans ]0, 1[."""
        self.etat = (_A * self.etat) % _M
        return self.etat / _M

    def uniformes(self, nb):
        """nb tirages uniformes successifs (même suite que nb appels à `uniforme`)."""
        nb = int(nb)
        out = np.empty(nb, dtype=np.int64)
        P = _puissances()
        i = 0
        while i < nb:
            b = min(_BLOC, nb - i)
            out[i:i + b] = (P[:b] * self.etat) % _M      # produit < 2^62 : exact en int64
            self.etat = int(out[i + b - 1])
            i += b
        return out / _M

    def normale(self):
        """Tirage gaussien centré réduit (Box-Muller, premier terme)."""
        u1, u2 = self.uniforme(), self.uniforme()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def normales(self, nb):
        """nb tirages gaussiens successifs (même suite que nb appels à `normale`)."""
        u = self.uniformes(2 * int(nb))
        return np.sqrt(-2.0 * np.log(u[0::2])) * np.cos(2.0 * np.pi * u[1::2])

    def permutation(self, n):
        """Permutation de 0, ..., n-1 : rangs de n tirages uniformes."""
        return np.argsort(self.uniformes(n), kind="stable")


def rotation_aleatoire(n, gen):
    """Matrice orthogonale n x n, produit de rotations planes d'angles uniformes sur [0, 2 pi[,
    appliquées à toutes les paires (i, j), i < j, dans l'ordre lexicographique."""
    Q = np.eye(n)
    for i in range(n - 1):
        for j in range(i + 1, n):
            a = 2.0 * math.pi * gen.uniforme()
            c, s = math.cos(a), math.sin(a)
            qi, qj = Q[:, i].copy(), Q[:, j].copy()
            Q[:, i], Q[:, j] = c * qi + s * qj, -s * qi + c * qj
    return Q
