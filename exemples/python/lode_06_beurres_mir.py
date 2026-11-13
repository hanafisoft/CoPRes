"""LODE, section 7.4 : beurres et margarines en moyen infrarouge, les longueurs d'onde étant les
variables et les deux types de produits les groupes. Spectres SNV, centrés sur les 21 produits et
non réduits ; covariances restreintes aux quatre premiers axes de l'ACP ; seuils par la plus grande
valeur propre de la covariance permutée ; stabilité sur 100 rééchantillonnages."""
import numpy as np
import CoPRes as cp
from commun import lire_beurres

blocs, typ, _, _ = lire_beurres()
mir = blocs[2]
mir = (mir - mir.mean(1, keepdims=True)) / mir.std(1, ddof=1, keepdims=True)      # SNV
P = cp.pretraiter_groupes(mir, typ, reduire=None, R=4)
part = np.linalg.eigvalsh(P.Xs.T @ P.Xs)[::-1]
print(f"variance portée par les quatre premiers axes : {100 * part[:4].sum() / part.sum():.0f} %")
th = cp.seuils_groupes(P, regle="valeur_propre", graine=7)
c = cp.copres_groupes(P, th, departs=50)
print("seuils :", np.round(th, 4))
print(c)
print("saliences / seuil :")
for L, idx in c.parties().items():
    print(f"  {cp.nom_etiquette(L, c.groupes):14s}", np.round(c.saliences[:, idx] / th[:, None], 2).tolist())
print(cp.stabilite_groupes(c, P, B=100))
th_dir = cp.seuils_groupes(P, regle="direction", graine=7)
print("avec le seuil par une direction tirée au hasard :",
      cp.texte_motif(cp.copres_groupes(P, th_dir, departs=20).motif, c.groupes))
