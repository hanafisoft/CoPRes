"""Tests de la couche multibloc : propriétés de la factorisation CoPRes, valeurs de référence
(partagées avec la version R) et tableau 5 de LODE sur les beurres et margarines."""
import glob
import numpy as np
import pytest
import CoPRes as cp
from CoPRes.alea import ParkMiller


def mat(g, r, c):
    """Matrice r x c de tirages gaussiens, remplie par colonnes (comme matrix() en R)."""
    return g.normales(r * c).reshape(c, r).T


def donnees():
    g = ParkMiller(2026)
    n = 30
    T = mat(g, n, 4)
    X = [T[:, [0, 1]] @ mat(g, 2, 15) + 0.5 * mat(g, n, 15),
         T[:, [0, 2]] @ mat(g, 2, 15) + 0.5 * mat(g, n, 15),
         0.4 * T[:, [0]] @ mat(g, 1, 15) + T[:, [3]] @ mat(g, 1, 15) + 0.5 * mat(g, n, 15)]
    return cp.pretraiter(X)


X = donnees()
TH = cp.seuils_perm(X, nperm=50, ndir=50, graine=3)

REF = {
    "seuils": [0.062442126839843876, 0.07729766455175517, 0.06506439774960371],
    "seuils_vp": [0.25184971313176985, 0.45915685700218134, 0.30128052236855263],
    "CPCA": (2.0707660961275867, 2.147789149363547, "({1,2,3}, 1), ({1,3}, 1), ({1}, 1), ({2}, 1)", 4, 1.8361411015679512),
    "ACOM": (1.7656387359955377, 1.855227726329899, "({1,2,3}, 1), ({1,3}, 1), ({1}, 1), ({2}, 1)", 4, 1.6146947550864703),
    "ACCPS": (2.068164141051449, 2.1454787679474414, "({1,2,3}, 1), ({1,3}, 1), ({1}, 1), ({2}, 1)", 4, 1.8892729546517912),
}


def test_pretraitement():
    for x in X:
        assert np.allclose(x.mean(0), 0)
        assert np.linalg.norm(x) == pytest.approx(1.0)


def test_seuils_reference():
    assert np.allclose(TH, REF["seuils"], rtol=1e-10)
    thv = cp.seuils_perm(X, nperm=50, regle="valeur_propre", graine=3)
    assert np.allclose(thv, REF["seuils_vp"], rtol=1e-10)


@pytest.mark.parametrize("methode", ["CPCA", "ACOM", "ACCPS"])
def test_methodes_reference(methode):
    c = cp.copres(X, TH, R=4, methode=methode, departs=20)
    bilan, borne, motif, nmel, bilan_direct = REF[methode]
    assert c.bilan == pytest.approx(bilan, abs=1e-8)
    assert c.borne == pytest.approx(borne, abs=1e-10)
    assert cp.texte_motif(c.motif) == motif
    assert int(cp.fiche(None, c).melangee.sum()) == nmel
    assert cp.lecture_directe(X, c.base_methode, TH).bilan == pytest.approx(bilan_direct, abs=1e-10)


@pytest.mark.parametrize("methode", ["CPCA", "ACOM", "ACCPS"])
def test_proprietes_factorisation(methode):
    c = cp.copres(X, TH, R=4, methode=methode, departs=10)
    U, V = c.base_methode, c.base
    assert np.allclose(U.T @ U, np.eye(4)) and np.allclose(V.T @ V, np.eye(4))
    # même sous-espace, même ajustement de chaque bloc
    assert np.allclose(V @ V.T, U @ U.T)
    for x in X:
        assert np.linalg.norm(x - V @ V.T @ x) == pytest.approx(np.linalg.norm(x - U @ U.T @ x))
    # carré des poids = salience ; poids et loadings reconstituent la projection
    assert np.allclose(c.poids ** 2, c.saliences)
    for x, p, H in zip(X, c.poids, c.loadings):
        assert np.allclose(V @ np.diag(p) @ H.T, V @ V.T @ x)
    # la lecture directe de V est cohérente et redonne le bilan
    ld = cp.lecture_directe(X, V, TH)
    assert ld.coherente
    assert ld.bilan == pytest.approx(c.bilan, abs=1e-9)
    assert ld.bilan >= cp.lecture_directe(X, U, TH).bilan - 1e-9
    assert c.bilan <= c.borne + 1e-10
    # une base comparée à elle-même : ni mélange, ni dilution ; équivalence avec elle-même
    f = cp.fiche(V, c)
    assert not f.melangee.any() and not f.diluee.any()
    ok, prox = cp.equivalence(c, c)
    assert ok and all(v == pytest.approx(1.0) for v in prox.values())


def test_sensibilite():
    s = cp.sensibilite(X, TH, R_liste=(3, 4), facteurs=(1.0, 1.5), departs=5)
    assert set(s) == {(3, 1.0), (3, 1.5), (4, 1.0), (4, 1.5)}


FICHIERS = {n: glob.glob(f"/mnt/user-data/attachments/*/*-{n}21.csv") for n in ("VIS", "NIR", "MIR")}


@pytest.mark.skipif(not all(FICHIERS.values()), reason="spectres des beurres non disponibles")
def test_tableau5_beurres():
    def lire(f):
        L = [l.strip().split(",") for l in open(f)]
        return np.array([[float(v) for v in r[1:]] for r in L[1:]])
    Xb = cp.pretraiter([lire(FICHIERS[n][0]) for n in ("VIS", "NIR", "MIR")], snv=True)
    th = [0.0616, 0.0559, 0.0563]       # seuils du tableau 5 de LODE
    attendu = {"CPCA": ("({1,2,3}, 1), ({1}, 2), ({3}, 1)", 0.031, 3, 0),
               "ACOM": ("({1,2,3}, 1), ({1}, 2), (résiduelle, 1)", 0.015, 0, 0)}
    for m, (motif, ecart, nmel, ndil) in attendu.items():
        c = cp.copres(Xb, th, R=4, methode=m, departs=50)
        f = cp.fiche(None, c)
        assert cp.texte_motif(c.motif) == motif
        assert c.ecart_relatif == pytest.approx(ecart, abs=0.0006)
        assert int(f.melangee.sum()) == nmel and int(f.diluee.sum()) == ndil
