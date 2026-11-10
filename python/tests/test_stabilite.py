"""Tests de la stabilité par bootstrap : valeurs de référence partagées avec la version R et
ordre de grandeur des fréquences de l'article sur les notes d'élèves."""
import numpy as np
import pytest
import CoPRes as cp
from test_multibloc import X as XB, TH as THB


def test_reference_groupes():
    X, g, _ = cp.lire_notes(1)
    P = cp.pretraiter_groupes(X, g)
    th = cp.seuils_groupes(P, nperm=50, ndir=50, graine=7)
    c = cp.copres_groupes(P, th, departs=20)
    s = cp.stabilite_groupes(c, P, B=10)
    assert s.motif_retrouve == 1.0
    assert all(v == 1.0 for v in s.strict.values())


def test_reference_blocs():
    c = cp.copres(XB, THB, R=4, methode="CPCA", departs=20)
    s = cp.stabilite_blocs(c, XB, B=5, nperm=20, ndir=50, departs=10)
    assert s.strict == {(0, 1, 2): 0.8, (0, 2): 0.4, (0,): 0.4, (1,): 1.0}
    assert s.motif_retrouve == pytest.approx(0.2)
    # les critères sont emboîtés : strict <= étiquette <= type
    for L in s.strict:
        assert s.strict[L] <= s.etiquette[L] <= s.type[L]


@pytest.mark.lent
def test_article_notes():
    """Article : avec 500 élèves par lycée, chaque partie est retrouvée dans 91 à 100 % de 100
    rééchantillonnages, et le motif complet dans 91 %."""
    X, g, _ = cp.lire_notes(1)
    P = cp.pretraiter_groupes(X, g)
    c = cp.copres_groupes(P, cp.seuils_groupes(P, graine=7), departs=50)
    s = cp.stabilite_groupes(c, P, B=100, seuil=0.8)
    assert 0.85 <= s.motif_retrouve <= 0.97
    assert all(0.85 <= v <= 1.0 for v in s.etiquette.values())
