"""Article appliqué, section 4.2 : plusieurs blocs, des groupes connus (rates : régime ; beurres :
type de produit), R = 4.

Part de l'information de groupe par partie dans l'espace de la CPCA et dans celui de la PLS-DA,
proximité des parties des deux lectures, analyse discriminante par partie (test par permutation,
999 permutations), part de la variance de chaque bloc due aux groupes, stabilité (200
rééchantillonnages à effectif constant dans chaque groupe), contribution propre et portée de chaque
bloc, validation croisée un individu à la fois avec choix des parties."""
import sys
import numpy as np
import CoPRes as cp
from commun import lire_rates, lire_beurres

R = 4


def etude(nom, X, groupes, noms, B=200, vc=True):
    print(f"\n{nom}")
    th = cp.seuils_perm(X)
    c = cp.copres(X, th, R=R, departs=200)
    cP = cp.copres(X, th, U=cp.plsda(X, groupes, R), methode="PLS-DA", departs=200)
    for nm, cop in (("CPCA", c), ("PLS-DA", cP)):
        inf = cp.information_groupe(cop, groupes)
        print(f"  {nm} : motif {cp.texte_motif(cop.motif, noms)}")
        print("    " + " ; ".join(f"{cp.nom_etiquette(L, noms)} {100 * v:.0f} %" for L, v in inf.parts.items())
              + f" ; hors de l'espace {100 * inf.hors:.0f} %")
        print("    contribution propre " + ", ".join(f"{n} {100 * v:.0f} %" for n, v in zip(noms, inf.contribution_propre))
              + " ; portée " + ", ".join(f"{n} {100 * v:.0f} %" for n, v in zip(noms, inf.portee)))
    eq, pr = cp.equivalence(c, cP)
    print(f"  lectures équivalentes : {eq} ; proximités des parties de même étiquette : "
          + (", ".join(f"{cp.nom_etiquette(L, noms)} {v:.2f}" for L, v in pr.items()) if pr else "--"))
    tous = tuple(range(len(X)))
    if tous in c.parties() and tous in cP.parties():
        E1, E2 = c.base[:, c.parties()[tous]], cP.base[:, cP.parties()[tous]]
        # dimensions éventuellement différentes : plus petit carré du cosinus des angles principaux
        # (la plus petite partie est-elle contenue dans l'autre ?)
        cont = float(np.min(np.linalg.svd(E1.T @ E2, compute_uv=False)) ** 2)
        print(f"  parties communes (dimensions {E1.shape[1]} et {E2.shape[1]}) : contenance {cont:.2f}")
    d = cp.discrimination_parties(c, groupes, n_perm=999)
    print("  analyse discriminante par partie (LOO, 999 permutations) : " + " ; ".join(
        f"{'sous-espace' if L == 'S' else cp.nom_etiquette(L, noms)} {100 * d.taux[L]:.0f} % (p = {d.p[L]:.3f})" for L in d.taux))
    print("  part de la variance due aux groupes : " + ", ".join(f"{n} {v:.2f}" for n, v in zip(noms, cp.variance_groupe(X, groupes))))
    st = cp.stabilite_blocs(c, X, B=B, strates=groupes, noms=noms)
    print("  motifs des rééchantillonnages : " + "; ".join(f"{m} ({n})" for m, n in list(st.motifs.items())[:4]))
    print("  " + repr(st).replace("\n", "\n  "))
    return th


if __name__ == "__main__":
    vc = "sans_vc" not in sys.argv
    Brt, regime, _, _ = lire_rates()
    Xr = cp.pretraiter(Brt)
    etude("Rates, régime", Xr, regime, ["M", "C"])
    b, typ, _, _ = lire_beurres()
    Xb = cp.pretraiter(b, snv=True)
    etude("Beurres, type", Xb, typ, ["VIS", "NIR", "MIR"])
    if vc:
        print("\nValidation croisée (chaîne refaite dans chaque pli ; seuils 50 x 100, CoPRes 20 départs)")
        print("Rates : " + repr(cp.validation_croisee(Brt, regime, R, noms=["M", "C"])).replace("\n", "\n  "))
        # beurres : la SNV ne dépend que de l'individu ; elle est faite avant la validation croisée
        bs = [(x - x.mean(1, keepdims=True)) / x.std(1, ddof=1, keepdims=True) for x in b]
        print("Beurres : " + repr(cp.validation_croisee(bs, typ, R, noms=["VIS", "NIR", "MIR"])).replace("\n", "\n  "))
