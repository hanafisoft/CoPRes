"""Tests de fumée des graphiques : chaque fonction rend une figure sans erreur."""
import numpy as np
import pytest
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import CoPRes as cp
from test_multibloc import X, TH

C = cp.copres(X, TH, R=4, departs=10)
NT = cp.lire_notes(1)
P = cp.pretraiter_groupes(NT[0], NT[1])
CG = cp.copres_groupes(P, cp.seuils_groupes(P, nperm=20, ndir=20, graine=7), departs=10)


@pytest.mark.parametrize("fabrique", [
    lambda: cp.graph_saliences(C), lambda: cp.graph_saliences(CG), lambda: cp.graph_coherence(C, (0, 1)),
    lambda: cp.graph_fiche(cp.fiche(None, C)), lambda: cp.graph_motif(CG), lambda: cp.graph_scores(C),
    lambda: cp.graph_scores(CG, prep=P), lambda: cp.graph_scores_parties(CG, prep=P),
    lambda: cp.graph_loadings(C, X=X), lambda: cp.graph_loadings(C, type="spectre"),
    lambda: cp.graph_loadings(CG, noms_variables=NT[2]), lambda: cp.graph_contributions(CG, 2, noms_variables=NT[2]),
    lambda: cp.graph_stabilite(cp.stabilite_groupes(CG, P, B=3)),
    lambda: cp.graph_sensibilite(cp.sensibilite(X, TH, R_liste=(3, 4), facteurs=(1.0, 1.5), departs=3)),
    lambda: cp.graph_bilan(X, TH, 4, departs=3)])
def test_figure(fabrique):
    fig = fabrique()
    assert hasattr(fig, "savefig")
    matplotlib.pyplot.close(fig)


def test_donnees_et_langue():
    d = cp.donnees_saliences(CG)
    assert len(d["salience"]) == 3 * 8 and set(d) >= {"bloc", "seuil", "porteur", "type"}
    assert "résiduelle" in d["etiquette"]
    assert "residual" in cp.donnees_saliences(CG, langue="en")["etiquette"]
    cp.langue("en")
    assert "residual" in cp.donnees_motif(CG)["partie"]
    cp.langue("fr")
    f = cp.donnees_fiche(cp.fiche(None, C))
    assert abs(sum(p for p, c in zip(f["part"], f["composante"]) if c == 1) - 1) < 1e-9
    assert cp.theme(taille=10)["taille"] == 10
    assert cp.theme(defaut=True)["taille"] == 9


def test_un_seul_bloc():
    fig = cp.graph_coherence(C, (0, 1), bloc=1)
    assert len(fig.axes) == 1
