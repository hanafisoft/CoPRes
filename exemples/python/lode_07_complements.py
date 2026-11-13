"""LODE, chiffres du texte des sections 6.1 et 7.3 : exemple du mélange (simulées 1), accord des
départs, sous-espaces de l'étiquetage des beurres (tableau 6), et stabilité comparée de LODE et de
la lecture directe de la base CPC sur les notes d'élèves (100 rééchantillonnages)."""
import numpy as np
import CoPRes as cp
from CoPRes.alea import ParkMiller
from commun import lire_simulees, lire_beurres

# --- simulées 1 : composante u1 de la CPCA
X = cp.pretraiter(lire_simulees(1))
th = cp.seuils_perm(X)
c = cp.copres(X, th, R=5, methode="CPCA", departs=100)
ld = cp.lecture_directe(X, c.base_methode, th)
print("Simulées 1, seuils :", np.round(th, 3))
for h in range(5):
    print(f"  u{h + 1} : saliences {np.round(ld.saliences[:, h], 3)} ; lecture {cp.nom_etiquette(ld.etiquettes[h])}")
print("  fiche de la CPCA :\n" + "\n".join("    " + l for l in repr(cp.fiche(None, c)).split("\n")))

# --- accord des départs (base de la méthode, base concentrée, rotations tirées au hasard)
def accord(X, R, methode, nb=200):
    th = cp.seuils_perm(X)
    U = cp.METHODES[methode](X, R)
    A = [U.T @ x @ x.T @ U for x in X]
    ref = cp.lode(A, th, departs=nb)
    best = ref.critere
    c = np.array(ref.criteres_departs)
    hit = c >= best - 1e-9 * max(1, best)
    de_U = cp.critere(A, cp.lode(A, th, departs=1).base, th)          # base concentrée seule
    from CoPRes.lode import _depuis
    pU = _depuis(A, np.asarray(th), np.eye(R), 1e-12, 30)[1]           # départ = base de la méthode
    return hit[0], hit[1:].mean(), pU >= best - 1e-9 * max(1, best), 100 * ref.ecart_relatif

for nom, X, R, kw in [("Simulées 1", lire_simulees(1), 5, {}), ("Simulées 3", lire_simulees(3), 7, {}),
                      ("Simulées 4", lire_simulees(4), 6, {}), ("Beurres", lire_beurres()[0], 4, dict(snv=True))]:
    Xp = cp.pretraiter(X, **kw)
    for m in ("CPCA", "ACOM"):
        conc, hasard, base, ecart = accord(Xp, R, m)
        print(f"{nom:11s} {m:5s} : concentrée au meilleur {conc} ; départs au hasard au meilleur {100 * hasard:.1f} % ;"
              f" départ de la base de la méthode au meilleur {base} ; écart {ecart:.2f} %")

# --- tableau 6 : sous-espaces de l'étiquetage des beurres (CPCA, R = 4)
Xb = cp.pretraiter(lire_beurres()[0], snv=True)
thb = cp.seuils_perm(Xb)
cb = cp.copres(Xb, thb, R=4, methode="CPCA", departs=100)
A = cb.lode.matrices
print("\nBeurres, seuils", np.round(thb, 3))
for L, idx in cb.parties().items():
    VL = cb.lode.base[:, idx]
    bornes = [np.linalg.eigvalsh(VL.T @ a @ VL)[[0, -1]] for a in A]
    print(f"  {cp.nom_etiquette(L, ['VIS', 'NIR', 'MIR']):16s} dim {len(idx)} : " +
          ", ".join(f"{lo:.3f} à {hi:.3f}" for lo, hi in bornes))

# --- notes : LODE contre lecture directe de la base CPC sur 100 rééchantillonnages
Xn, g, _ = cp.lire_notes(1)
P = cp.pretraiter_groupes(Xn, g)
thn = cp.seuils_groupes(P, graine=7)
PLAN = {(0, 1, 2): 1, (0, 1): 1, (0,): 1, (2,): 1, (): 4}
gen = ParkMiller(1)
nL = nC = comL = comC = 0
c0 = cp.copres_groupes(P, thn, departs=50)
E0 = c0.base_analyse[:, [h for h, L in enumerate(c0.etiquettes) if L == (0, 1, 2)]]
for b in range(100):
    A = []
    for x in P.Xk:
        xb = x[np.floor(gen.uniformes(len(x)) * len(x)).astype(int)]
        xb = xb - xb.mean(0)
        A.append(xb.T @ xb / (len(xb) - 1))
    Pb = type(P)(groupes=P.groupes, g=P.g, Xs=P.Xs, Xk=P.Xk, covariances=A, U=P.U, ecarts_types=P.ecarts_types)
    cl = cp.copres_groupes(Pb, thn, departs=20, graine=b + 1)
    lc = cp.lecture_groupes(Pb, cp.cpc(Pb), thn)
    nL += cl.motif == PLAN
    nC += lc.motif == PLAN
    for etiq, V, compteur in ((cl.etiquettes, cl.base_analyse, "L"), (lc.etiquettes, lc.base, "C")):
        idx = [h for h, L in enumerate(etiq) if L == (0, 1, 2)]
        ok = len(idx) == 1 and float(np.sum((V[:, idx].T @ E0) ** 2)) >= 0.8
        if compteur == "L":
            comL += ok
        else:
            comC += ok
print(f"\nNotes, 100 rééchantillonnages : motif du plan retrouvé par LODE {nL}, par la lecture directe de la CPC {nC} ;"
      f" partie commune aux trois lycées : {comL} contre {comC}")
