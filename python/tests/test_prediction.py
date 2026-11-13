"""Tests du module prediction."""
import os
import numpy as np
import pytest

import CoPRes as cp
from CoPRes.prediction import (indicatrice, information_groupe, plsda, lda_loo, discrimination_parties,
                               variance_groupe, validation_croisee)

REF = os.path.join(os.path.dirname(__file__), "ref_prediction")


def lire(nom):
    return np.loadtxt(os.path.join(REF, nom), delimiter=",", ndmin=2)


def donnees():
    X = lire("donnees.csv")
    p = lire("blocs.csv").ravel().astype(int)
    g = np.array(open(os.path.join(REF, "groupes.csv")).read().split())
    return [X[:, :p[0]], X[:, p[0]:p[0] + p[1]]], g


@pytest.fixture(scope="module")
def analyse():
    B, g = donnees()
    X = cp.pretraiter(B)
    th = cp.seuils_perm(X, nperm=50)
    return B, X, g, th, cp.copres(X, th, R=3)


def test_seuils(analyse):
    assert np.allclose(analyse[3], lire("seuils.csv").ravel(), atol=1e-12)


def test_information_somme_a_un_et_reference(analyse):
    B, X, g, th, c = analyse
    inf = information_groupe(c, g)
    assert abs(sum(inf.parts.values()) + inf.hors - 1) < 1e-12
    ref = lire("information.csv").ravel()
    val = [inf.hors] + list(inf.contribution_propre) + list(inf.portee) + sorted(inf.parts.values())
    assert np.allclose(val, ref, atol=1e-10)


def test_information_independante_de_la_base(analyse):
    """Tourner les composantes dans une partie ne change pas sa part."""
    B, X, g, th, c = analyse
    G = indicatrice(g)
    for L, idx in c.parties().items():
        E = c.base[:, idx]
        Q = np.linalg.qr(np.arange(1, len(idx) ** 2 + 1, dtype=float).reshape(len(idx), len(idx)) + np.eye(len(idx)))[0]
        assert abs(np.sum((E.T @ G) ** 2) - np.sum(((E @ Q).T @ G) ** 2)) < 1e-12


def test_plsda(analyse):
    B, X, g, th, c = analyse
    U = plsda(X, g, 3)
    assert np.allclose(U.T @ U, np.eye(3), atol=1e-12)
    assert np.allclose(U @ U.T, lire("plsda_projecteur.csv"), atol=1e-10)
    # la première composante maximise la covariance avec l'indicatrice : |X' Y| plus grand que pour la CPCA
    Y = indicatrice(g)
    Xc = np.hstack(X)
    w = Xc.T @ U[:, 0]
    assert np.linalg.norm(U[:, 0] @ Xc @ Xc.T @ Y) >= np.linalg.norm(c.base_methode[:, 0] @ Xc @ Xc.T @ Y) - 1e-12


def test_lda(analyse):
    B, X, g, th, c = analyse
    ref = lire("lda.csv").ravel()
    assert np.allclose([lda_loo(c.base_methode, g), lda_loo(c.base[:, :1], g)], ref)


def test_lda_cas_separable():
    g = np.array(["a"] * 5 + ["b"] * 5)
    Z = np.r_[np.arange(5), np.arange(5) + 100.0][:, None] + 0.0
    assert lda_loo(Z, g) == 1.0


def test_discrimination_et_variance(analyse):
    B, X, g, th, c = analyse
    d = discrimination_parties(c, g, n_perm=49)
    assert np.allclose([d.taux["S"], d.p["S"]], lire("discrimination.csv").ravel())
    assert np.allclose(variance_groupe(X, g), lire("variance.csv").ravel(), atol=1e-12)


@pytest.mark.lent
def test_validation_croisee(analyse):
    B, X, g, th, c = analyse
    v = validation_croisee(B, g, 3, nperm=10, ndir=50, departs=5)
    assert np.allclose(list(v.taux.values()), lire("validation.csv").ravel())
