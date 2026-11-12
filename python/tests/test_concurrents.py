"""Tests de concurrents.py : AJIVE contre mvlearn 0.5, JIVE contre r.jive 2.4 (sorties de
référence dans tests/ref_concurrents ; les tests JIVE sont sautés tant qu'elles manquent)."""
import os
import numpy as np
import pytest

from CoPRes.concurrents import jive, ajive, rang_coude

REF = os.path.join(os.path.dirname(__file__), "ref_concurrents")
RANGS = {"a": (2, [2, 2, 2], [4, 4, 4]), "b": (1, [2, 2, 2], [3, 3, 3]), "c": (2, [1, 1], [3, 3])}


def lire(nom):
    return np.loadtxt(os.path.join(REF, nom), delimiter=",", ndmin=2)


def blocs(j):
    X = lire(f"{j}_donnees.csv")
    p = lire(f"{j}_blocs.csv").ravel().astype(int)
    b = np.cumsum(np.r_[0, p])
    return [X[:, b[k]:b[k + 1]] for k in range(len(p))]


def ref(nom):
    chemin = os.path.join(REF, nom)
    if not os.path.exists(chemin):
        pytest.skip(f"référence absente : {nom} (lancer produire_ref_rjive.R)")
    return lire(nom)


# ----------------------------------------------------------------------------- AJIVE
@pytest.mark.parametrize("j", ["a", "b", "c"])
def test_ajive_rangs_donnes_comme_mvlearn(j):
    rJ, _, rS = RANGS[j]
    r = ajive(blocs(j), rS, rang_joint=rJ)
    P = r.scores_joint @ r.scores_joint.T
    assert np.abs(P - lire(f"{j}_ajive_scores_joint.csv")).max() < 1e-10
    assert np.abs(np.hstack(r.individuel) - lire(f"{j}_ajive_individuel.csv")).max() < 1e-10
    assert [r.rang_joint] + r.rangs_individuels == list(lire(f"{j}_ajive_rangs.csv").ravel().astype(int))


@pytest.mark.parametrize("j", ["a", "b", "c"])
def test_ajive_rangs_estimes_comme_mvlearn(j):
    _, _, rS = RANGS[j]
    r = ajive(blocs(j), rS, n_wedin=300, n_hasard=300)
    assert [r.rang_joint] + r.rangs_individuels == list(lire(f"{j}_ajive_rangs.csv").ravel().astype(int))


def test_ajive_decomposition():
    X = blocs("a")
    r = ajive(X, [4, 4, 4], rang_joint=2)
    for k in range(3):
        xc = X[k] - X[k].mean(0)
        assert np.allclose(r.scores_joint.T @ r.individuel[k], 0, atol=1e-12)
        assert np.allclose(r.joint[k], r.scores_joint @ r.scores_joint.T @ xc)


def test_rang_coude():
    """Mêmes rangs que select_dimension de mvlearn 0.5."""
    assert [rang_coude(x) for x in blocs("a")] == [4, 4, 4]
    assert [rang_coude(x) for x in blocs("b")] == [3, 2, 3]
    assert [rang_coude(x) for x in blocs("c")] == [5, 5]


# ----------------------------------------------------------------------------- JIVE
@pytest.mark.parametrize("j", ["a", "b", "c"])
def test_jive_rangs_donnes_comme_rjive(j):
    J_ref = ref(f"{j}_rjive_joint.csv")
    A_ref = ref(f"{j}_rjive_individuel.csv")
    rJ, rA, _ = RANGS[j]
    r = jive(blocs(j), rJ, rA)
    echelle = np.abs(np.hstack(r.donnees)).max()
    assert np.abs(np.hstack(r.joint) - J_ref).max() < 1e-6 * echelle
    assert np.abs(np.hstack(r.individuel) - A_ref).max() < 1e-6 * echelle


@pytest.mark.parametrize("j", ["a", "b", "c"])
def test_jive_perm_comme_rjive(j):
    rangs = ref(f"{j}_rjive_perm_rangs.csv").ravel().astype(int)
    r = jive(blocs(j), nperm=100)
    assert [r.rang_joint] + r.rangs_individuels == list(rangs)


def test_jive_proprietes():
    r = jive(blocs("a"), 2, [2, 2, 2])
    assert r.converge
    S = r.scores_individuels
    for i in range(3):
        assert np.abs(r.scores_joint.T @ S[i]).max() < 1e-5
        for k in range(i + 1, 3):
            assert np.abs(S[i].T @ S[k]).max() < 1e-10


def test_ajive_sans_centrage():
    """centrer=False : les blocs sont pris tels quels ; sur des blocs déjà centrés, même résultat."""
    X = [x - x.mean(0) for x in blocs("a")]
    r1 = ajive(X, [4, 4, 4], rang_joint=2)
    r2 = ajive(X, [4, 4, 4], rang_joint=2, centrer=False)
    assert np.allclose(r1.scores_joint @ r1.scores_joint.T, r2.scores_joint @ r2.scores_joint.T, atol=1e-12)
    Y = [x + 3.0 for x in X]
    r3 = ajive(Y, [4, 4, 4], rang_joint=2, centrer=False)
    assert np.allclose(r3.joint[0], r3.scores_joint @ r3.scores_joint.T @ Y[0])
