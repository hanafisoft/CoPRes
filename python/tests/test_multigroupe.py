"""Tests de la couche multigroupe sur les notes d'élèves : motif du plan, chiffres de LODE et de
l'article appliqué (discrimination), valeurs de référence partagées avec la version R."""
import numpy as np
import pytest
import CoPRes as cp

PLAN = "({A,B,C}, 1), ({A,B}, 1), ({A}, 1), ({C}, 1), (résiduelle, 4)"
X, G, VARS = cp.lire_notes(1)
P = cp.pretraiter_groupes(X, G)
TH = cp.seuils_groupes(P, nperm=50, ndir=50, graine=7)


def test_donnees():
    assert X.shape == (1500, 8) and P.groupes == ["A", "B", "C"]
    assert VARS[0] == "Maths" and VARS[4] == "Français"


def test_pretraitement():
    assert np.allclose(P.Xs.mean(0), 0)
    for x in P.Xk:
        assert np.allclose(x.mean(0), 0)
    W = sum(a * (len(x) - 1) for a, x in zip(P.covariances, P.Xk)) / (len(X) - 3)
    assert np.allclose(np.diag(W), 1)      # réduction par l'écart-type intra-groupe commun


def test_reference():
    assert np.allclose(TH, [1.410596466850003, 1.153675180242619, 1.3706957140934921], rtol=1e-10)
    c = cp.copres_groupes(P, TH, departs=20)
    assert c.bilan == pytest.approx(7.077536447057212, abs=1e-8)
    assert c.borne == pytest.approx(7.505642793966811, abs=1e-10)
    assert cp.texte_motif(c.motif, c.groupes) == PLAN
    d = cp.discrimination(c, P, classement=False)["parts_plan_discriminant"]
    assert d[()] == pytest.approx(1.6453169938919017, abs=1e-6)
    assert d[(0, 1)] == pytest.approx(0.20019027752751048, abs=1e-6)
    assert sum(d.values()) == pytest.approx(2.0)
    l = cp.lecture_groupes(P, cp.cpc(P), TH)
    assert l.bilan == pytest.approx(7.068454780874832, abs=1e-8)
    assert cp.texte_motif(l.motif, P.groupes) == PLAN and l.coherente


def test_restriction_valeur_propre():
    P2 = cp.pretraiter_groupes(X, G, R=4)
    th2 = cp.seuils_groupes(P2, nperm=50, regle="valeur_propre", graine=7)
    assert np.allclose(th2, [1.791081693655434, 1.3705786471785817, 1.797839649285621], rtol=1e-10)
    c2 = cp.copres_groupes(P2, th2)
    assert c2.bilan == pytest.approx(5.3362522109239325, abs=1e-8)
    assert cp.texte_motif(c2.motif, c2.groupes) == "({A,B}, 1), ({A}, 1), ({C}, 1), (résiduelle, 1)"
    assert c2.base.shape == (8, 4) and np.allclose(c2.base.T @ c2.base, np.eye(4))


@pytest.mark.parametrize("situation, lin, quad, propres, residuel", [
    (1, 0.343, 0.583, 0.30, 1.65), (2, 0.693, 0.734, 1.81, 0.16), (3, 0.676, 0.753, 0.04, 1.92)])
def test_article_trois_situations(situation, lin, quad, propres, residuel):
    """Tableau des notes de l'article appliqué : classement en validation croisée et part du plan
    discriminant dans les parties partagée et propres, et dans la partie résiduelle."""
    Xs, Gs, _ = cp.lire_notes(situation)
    Ps = cp.pretraiter_groupes(Xs, Gs)
    th = cp.seuils_groupes(Ps, graine=7)
    c = cp.copres_groupes(Ps, th, departs=50)
    assert cp.texte_motif(c.motif, c.groupes) == PLAN
    d = cp.discrimination(c, Ps)
    parts = d["parts_plan_discriminant"]
    assert round(d["classement_lineaire"], 3) == lin
    assert round(d["classement_quadratique"], 3) == quad
    assert parts[(0, 1)] + parts[(0,)] + parts[(2,)] == pytest.approx(propres, abs=0.006)
    assert parts[()] == pytest.approx(residuel, abs=0.006)
