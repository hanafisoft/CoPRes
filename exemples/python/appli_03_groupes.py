"""Article appliqué, section 4.1 : les mêmes variables dans plusieurs groupes.

Parties (argument) :
  lycees  trois lycées simulés (données du paquet) : analyse de variance par module, classement,
          part du plan discriminant dans les parties, stabilité (500 et 80 élèves par lycée) ;
  helix   (ou helix_chimie, helix_urine) six cohortes HELIX (Exposome Data Challenge 2021) : expositions chimiques postnatales et
          métabolome urinaire. Les fichiers donnees/helix_chimie.csv et donnees/helix_urine.csv sont
          produits par R/helix_extraire.R à partir des données téléchargées par l'utilisateur
          (elles ne sont pas redistribuées).
Sans argument : les deux parties (HELIX seulement si ses fichiers sont présents)."""
import os
import sys
from types import SimpleNamespace
import numpy as np
from scipy.stats import f as loi_f
import CoPRes as cp
from commun import lire_csv, chemin


def anova_modules(X, g, niveau=0.05):
    niv = sorted(set(g))
    n, K = len(g), len(niv)
    nb = 0
    for j in range(X.shape[1]):
        x = X[:, j]
        inter = sum(np.sum(g == v) * (x[g == v].mean() - x.mean()) ** 2 for v in niv)
        intra = sum(np.sum((x[g == v] - x[g == v].mean()) ** 2) for v in niv)
        F = (inter / (K - 1)) / (intra / (n - K))
        nb += loi_f.sf(F, K - 1, n - K) < niveau
    return nb


def lycees():
    print("Lycées simulés")
    for s in (1, 2, 3):
        X, g, var = cp.lire_notes(s)
        P = cp.pretraiter_groupes(X, g)
        th = cp.seuils_groupes(P, graine=7)
        c = cp.copres_groupes(P, th, departs=50)
        d = cp.discrimination(c, P)
        pp = d["parts_plan_discriminant"]
        sig = sum(v for L, v in pp.items() if L and len(L) < len(c.groupes))
        res = pp.get((), 0.0)
        print(f"  situation {s} : motif {cp.texte_motif(c.motif, c.groupes)}")
        print(f"     modules qui diffèrent (5 %) : {anova_modules(X, g)} ; classement linéaire / quadratique "
              f"{100 * d['classement_lineaire']:.0f} / {100 * d['classement_quadratique']:.0f} % ; "
              f"part du plan discriminant : partagée et propres {sig:.2f}, résiduelle {res:.2f}")
        if s == 1:
            st = cp.stabilite_groupes(c, P, B=100, seuil=0.72)
            print("     stabilité, 500 élèves par lycée :\n" + "\n".join("       " + l for l in repr(st).split("\n")))
    tete, lignes = lire_csv("notes_80eleves.csv")
    Xv = np.array([[float(v) for v in l[1:]] for l in lignes])
    gv = np.array([l[0] for l in lignes])
    Pv = cp.pretraiter_groupes(Xv, gv)
    cv = cp.copres_groupes(Pv, cp.seuils_groupes(Pv, graine=7), departs=50)
    sv = cp.stabilite_groupes(cv, Pv, B=100, seuil=0.72)
    print(f"  80 élèves par lycée : motif de l'échantillon {cp.texte_motif(cv.motif, cv.groupes)}")
    print("     motifs des rééchantillonnages : " + "; ".join(f"{m} ({n})" for m, n in list(sv.motifs.items())[:5]))


def lire_helix(nom):
    tete, lignes = lire_csv(nom)
    X = np.array([[float(v) for v in l[2:]] for l in lignes])
    g = np.array([l[1].strip('"') for l in lignes])
    return X, g, [t.strip('"') for t in tete[2:]]


def type_partie(L, K):
    return "commune" if len(L) == K else "partagée" if len(L) > 1 else "propre" if len(L) == 1 else "résiduelle"


def helix_jeu(nom, fichier, B=100):
    X, g, var = lire_helix(fichier)
    ordre = np.argsort(g, kind="stable")
    X, g = X[ordre], g[ordre]
    P = cp.pretraiter_groupes(X, g)
    K = len(P.groupes)
    th = cp.seuils_groupes(P, graine=2026)
    print(f"\nHELIX, {nom} : {len(g)} enfants, {X.shape[1]} variables ; effectifs "
          + ", ".join(f"{k} {np.sum(g == k)}" for k in P.groupes) + f" ; seuils {th.min():.2f} à {th.max():.2f}")
    c = cp.copres_groupes(P, th, departs=50)
    print("  " + repr(c).replace("\n", "\n  "))
    dims = {}
    for L, idx in c.parties().items():
        dims[type_partie(L, K)] = dims.get(type_partie(L, K), 0) + len(idx)
    print("  dimensions par type : " + ", ".join(f"{k} {v}" for k, v in dims.items()))
    # lectures directes : ACP de l'ensemble et CPC
    ev, P0 = np.linalg.eigh(P.Xs.T @ P.Xs / (len(g) - 1))
    P0 = P0[:, ::-1]
    Bc = cp.cpc(P)
    pseudo = SimpleNamespace(base=c.base_analyse, etiquettes=c.etiquettes)
    for nm, Bm in (("ACP", P0), ("CPC", Bc)):
        f = cp.fiche(Bm, pseudo)
        ld = cp.lecture_groupes(P, Bm, th)
        print(f"  {nm} : cohérente {ld.coherente} ; mélangées {int(f.melangee.sum())}, diluées {int(f.diluee.sum())}")
        if nm == "CPC":
            # composante lue résiduelle et faite pour l'essentiel d'une partie propre
            cand = [(h, L, v) for h, (Ld, comp) in enumerate(zip(ld.etiquettes, f.composition)) if not Ld
                    for L, v in comp.items() if len(L) == 1]
            if cand:
                h, L, v = max(cand, key=lambda t: t[2])
                print(f"    composante {h + 1} de la CPC, lue résiduelle : {100 * v:.0f} % de la partie propre à "
                      f"{cp.nom_etiquette(L, P.groupes)}, {100 * f.residuel[h]:.0f} % résiduelle")
    # parties : variables principales et part de la variance due à la cohorte
    sc = P.Xs @ c.base
    for L, idx in sorted(c.parties().items(), key=lambda kv: (-len(kv[0]), kv[0])):
        if not L:
            continue
        V = c.base[:, idx]
        contrib = (V ** 2).sum(1) / len(idx)
        top = np.argsort(-contrib)[:5]
        eta = []
        for h in idx:
            y = sc[:, h]
            eta.append(sum(np.sum(g == k) * (y[g == k].mean() - y.mean()) ** 2 for k in P.groupes) / np.sum((y - y.mean()) ** 2))
        print(f"  {type_partie(L, K):10s} {cp.nom_etiquette(L, P.groupes):14s} dim {len(idx)} : "
              + ", ".join(f"{var[j]} {100 * contrib[j]:.0f} %" for j in top)
              + " ; part de variance due à la cohorte " + ", ".join(f"{e:.2f}" for e in eta))
    st = cp.stabilite_groupes(c, P, B=B, departs=5, seuil=0.72)
    print("  stabilité (strict / étiquette / type) :\n" + "\n".join("    " + l for l in repr(st).split("\n")))


def helix(jeux=("chimie", "urine")):
    for nom, f in [(n, f) for n, f in (("expositions chimiques", "helix_chimie.csv"), ("métabolome urinaire", "helix_urine.csv"))
                   if f[6:-4] in jeux]:
        if not os.path.exists(chemin(f)):
            print(f"HELIX : {f} absent (voir R/helix_extraire.R)")
            continue
        helix_jeu(nom, f)


if __name__ == "__main__":
    choix = sys.argv[1:] or ["lycees", "helix"]
    for c in choix:
        {"lycees": lycees, "helix": helix, "helix_chimie": lambda: helix(("chimie",)),
         "helix_urine": lambda: helix(("urine",))}[c]()
