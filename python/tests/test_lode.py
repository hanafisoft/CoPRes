"""Tests du module lode : exemples et propriétés démontrées dans l'article LODE."""
import numpy as np
import pytest
import CoPRes as cp
from CoPRes.alea import ParkMiller, rotation_aleatoire

c40, s40 = np.cos(np.radians(40)), np.sin(np.radians(40))
t = np.array([c40, s40, 0, 0])
A_BASES = [np.diag([0.50, 0, 0.45, 0.05]), 0.50 * np.outer(t, t) + np.diag([0, 0, 0.05, 0.45])]
TH_BASES = [0.10, 0.10]
U1 = np.eye(4)
U2 = np.array([[c40, -s40, 0, 0], [s40, c40, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
r = np.sqrt(0.5)
U3 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, r, r], [0, 0, r, -r]])


def test_park_miller_valeur_de_reference():
    # valeur de contrôle classique de Park et Miller : graine 1, 10 000 tirages
    g = ParkMiller(1)
    for _ in range(10000):
        g.uniforme()
    assert g.etat == 1043618065


def test_rotation_aleatoire_orthogonale():
    Q = rotation_aleatoire(6, ParkMiller(3))
    assert np.allclose(Q.T @ Q, np.eye(6))


def test_tableau_des_bases():
    assert cp.est_etiquetage(A_BASES, U1, TH_BASES)
    assert cp.est_etiquetage(A_BASES, U2, TH_BASES)
    assert not cp.est_etiquetage(A_BASES, U3, TH_BASES)
    assert cp.etiquettes(A_BASES, U3, TH_BASES) == [(0, 1), (1,), (0, 1), (0, 1)]
    assert cp.motif(cp.etiquettes(A_BASES, U1, TH_BASES)) == {(0, 1): 1, (0,): 1, (1,): 2}
    assert cp.motif(cp.etiquettes(A_BASES, U2, TH_BASES)) == {(0, 1): 1, (0,): 2, (1,): 1}


def test_b2bl_rend_un_etiquetage_et_fait_croitre_le_critere():
    V, nb = cp.b2bl(U3, A_BASES, TH_BASES)
    assert nb >= 1
    assert cp.est_etiquetage(A_BASES, V, TH_BASES)
    assert cp.critere(A_BASES, V, TH_BASES) > cp.critere(A_BASES, U3, TH_BASES)


def test_lode_tableau_des_bases():
    res = cp.lode(A_BASES, TH_BASES, departs=20)
    assert res.critere >= cp.critere(A_BASES, U1, TH_BASES) - 1e-10
    assert res.critere <= res.borne + 1e-10
    assert res.marge >= -1e-9
    assert np.allclose(res.base.T @ res.base, np.eye(4))


def test_ideal_sans_commutation():
    A1 = np.array([[0.45, 0.15, 0, 0], [0.15, 0.35, 0, 0], [0, 0, 0.10, 0], [0, 0, 0, 0.10]])
    A2 = np.array([[0.40, 0, -0.15, 0], [0, 0.10, 0, 0.02], [-0.15, 0, 0.40, 0], [0, 0.02, 0, 0.10]])
    A, th = [A1, A2], [0.15, 0.15]
    assert not np.allclose(A1 @ A2, A2 @ A1)
    ok, parties = cp.ideal(A, th)
    assert ok
    e = np.eye(4)
    attendu = {(0, 1): e[:, [0]], (0,): e[:, [1]], (1,): e[:, [2]], (): e[:, [3]]}
    assert set(parties) == set(attendu)
    for L, V in attendu.items():
        assert np.allclose(np.abs(parties[L].T @ V), 1)
    assert cp.borne_kf(A, th) == pytest.approx(1.0)
    res = cp.lode(A, th)
    assert res.critere == pytest.approx(1.0, abs=1e-9)
    assert res.motif == {(0, 1): 1, (0,): 1, (1,): 1, (): 1}


def test_matrices_qui_commutent_borne_atteinte():
    gen = ParkMiller(7)
    Q = rotation_aleatoire(6, gen)
    A = [Q @ np.diag(d) @ Q.T for d in ([5, 4, 1, 1, .2, 0], [4, .5, 3, .2, 1, 0], [.3, 3, 3, .1, 2, 1])]
    th = [0.8, 0.8, 0.8]
    ok, _ = cp.ideal(A, th)
    assert ok
    res = cp.lode(A, th)
    assert res.critere == pytest.approx(res.borne, rel=1e-9)


@pytest.mark.parametrize("graine", [1, 2, 3, 4, 5])
def test_proprietes_sur_matrices_aleatoires(graine):
    gen = ParkMiller(graine)
    n, K = 6, 3
    A = []
    for _ in range(K):
        X = np.array([gen.normales(n) for _ in range(8)]).T
        A.append(X @ X.T / np.trace(X @ X.T))
    th = [1.5 / n] * K
    res = cp.lode(A, th, departs=10, graine=graine)
    # le critère ne dépasse pas la borne ; l'étiquetage rendu est un étiquetage
    assert res.critere <= res.borne + 1e-10
    assert res.marge >= -1e-9
    assert cp.est_etiquetage(A, res.base, th)
    # l'orientation ne change pas le critère
    assert res.critere == pytest.approx(cp.critere(A, res.base, th), abs=1e-9)


def test_monotonie_de_la_boucle():
    from CoPRes.lode import _depuis
    gen = ParkMiller(11)
    n = 5
    A = []
    for _ in range(3):
        X = np.array([gen.normales(n) for _ in range(6)]).T
        A.append(X @ X.T / np.trace(X @ X.T))
    th = np.array([0.3, 0.3, 0.3])
    V0 = rotation_aleatoire(n, gen)
    V1, _ = cp.b2bl(V0, A, th)
    V, p, _, _ = _depuis(A, th, V0, 1e-12, 30)
    assert p >= cp.critere(A, V1, th) - 1e-12


def test_concentration_monotone():
    gen = ParkMiller(5)
    A = []
    for _ in range(3):
        X = np.array([gen.normales(5) for _ in range(7)]).T
        A.append(X @ X.T)
    V0 = rotation_aleatoire(5, gen)
    c0 = (cp.saliences(A, V0) ** 2).sum()
    V = cp.concentration(A, V0)
    assert (cp.saliences(A, V) ** 2).sum() >= c0 - 1e-12
