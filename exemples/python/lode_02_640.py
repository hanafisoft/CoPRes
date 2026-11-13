"""LODE, section 5.5, tableau 4 : algorithme LODE sur les 640 jeux simulés (matrices et seuils
d'origine, fichiers sim640_*.csv). Pour chaque jeu, 51 départs dans l'ordre de cp.lode(...,
departs=50, depart_propre=True) : base concentrée, base propre de la somme des A_k, 49 rotations
tirées au hasard. La boucle de l'algorithme est refaite ici avec les fonctions publiques du paquet
(b2bl, taro) pour compter itérations, balayages de TARO et appels à B2BL ; elle donne les mêmes
bases que cp.lode. Rapporte les colonnes du tableau 4 et les chiffres du texte de la section 5.5.
Long : une vingtaine de minutes (argument facultatif : nombre de jeux par couple (n, K), 40 au plus)."""
import sys, time
import numpy as np
import CoPRes as cp
from CoPRes.alea import ParkMiller, rotation_aleatoire
from commun import lire_csv

EPS = 1e-12
nb = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def depuis(A, th, V0, maxit=30):
    """Boucle (b)-(c) de LODE avec compteurs : itérations, balayages de TARO, appel à B2BL."""
    V, nb_b2bl = cp.b2bl(V0, A, th, EPS)
    appel = nb_b2bl > 0
    p_old = cp.critere(A, V, th)
    balayages = 0
    for t in range(1, maxit + 1):
        cible = cp.etiquettes(A, V, th)
        AL = [sum(A[k] for k in L) if L else 0 * A[0] for L in cible]
        J = lambda W: sum(W[:, h] @ AL[h] @ W[:, h] for h in range(W.shape[1]))
        while True:                                  # TARO, un balayage à la fois
            j0 = J(V)
            V = cp.taro(V, cible, A, EPS, max_balayages=1)
            balayages += 1
            if J(V) - j0 <= EPS * max(1.0, abs(J(V))):
                break
        V, r = cp.b2bl(V, A, th, EPS)
        appel = appel or r > 0
        p = cp.critere(A, V, th)
        if r == 0 and p <= (1 + EPS) * p_old:
            return p, t, balayages, appel
        p_old = p
    return cp.critere(A, V, th), maxit, balayages, appel


_, mat = lire_csv("sim640_matrices.csv")
_, se = lire_csv("sim640_seuils.csv")
JEUX = {}
for l in se:
    JEUX.setdefault(int(l[0]), dict(K=int(l[1]), n=int(l[2]), seuils=[]))["seuils"].append(float(l[4]))
for j in JEUX:
    JEUX[j]["A"] = [np.zeros((JEUX[j]["n"],) * 2) for _ in range(JEUX[j]["K"])]
for l in mat:
    j, k, i, jj, v = int(l[0]), int(l[3]) - 1, int(l[4]) - 1, int(l[5]) - 1, float(l[6])
    JEUX[j]["A"][k][i, jj] = JEUX[j]["A"][k][jj, i] = v

lignes, compte = [], {}
t_debut = time.time()
for j, d in sorted(JEUX.items()):
    cle = (d["n"], d["K"])
    compte[cle] = compte.get(cle, 0) + 1
    if compte[cle] > nb:
        continue
    A, th, n = d["A"], d["seuils"], d["n"]
    gen = ParkMiller(j)
    t0 = time.time()
    departs = [cp.base_concentree(A, graine=j)]
    t_conc = time.time() - t0
    departs.append(np.linalg.eigh(sum(A))[1][:, ::-1])
    departs += [rotation_aleatoire(n, gen) for _ in range(49)]
    res, temps = [], []
    for V0 in departs:
        t0 = time.time()
        res.append(depuis(A, th, V0))
        temps.append(time.time() - t0)
    c = np.array([r[0] for r in res])
    best = c.max()
    hit = c >= best - 1e-9 * max(1.0, abs(best))
    ordre = np.r_[0, np.arange(2, 51)]
    lignes.append(dict(n=n, K=d["K"], iters=[r[1] for r in res], iter0=res[0][1],
                       balayages=float(np.median([r[2] for r in res])), b2bl=float(np.mean([r[3] for r in res])),
                       t20=t_conc + temps[0] + sum(temps[2:21]), conc=hit[0], propre=hit[1], hasard=hit[2:].mean(),
                       des1=hit[0], des5=hit[ordre[:5]].any(), des20=hit[ordre[:20]].any(), gap=1 - best / cp.borne_kf(A, th)))
    if j % 40 == 0:
        print("jeu", j, round(time.time() - t_debut), "s", flush=True)

L = lignes
print("\nTableau 4")
print(f"{'n':>3s}  itérations méd. (max)  balayages K=2 3 4 5     temps 20 départs K=2 3 4 5     concentrée  écart méd. (max)   dès 5 / 20")
for n in sorted({x['n'] for x in L}):
    r = [x for x in L if x["n"] == n]
    it = np.concatenate([x["iters"] for x in r])
    bal = [np.median([x["balayages"] for x in r if x["K"] == K]) for K in (2, 3, 4, 5)]
    tps = [np.median([x["t20"] for x in r if x["K"] == K]) for K in (2, 3, 4, 5)]
    g = 100 * np.array([x["gap"] for x in r])
    print(f"{n:3d}  {np.median(it):4.0f} ({it.max():2d})            " + " ".join(f"{b:4.1f}" for b in bal)
          + "      " + " ".join(f"{t:5.2f}" for t in tps)
          + f"      {100 * np.mean([x['conc'] for x in r]):4.0f} %   {np.median(g):5.2f} ({g.max():.2f}) %"
          + f"   {100 * np.mean([x['des5'] for x in r]):3.0f} / {100 * np.mean([x['des20'] for x in r]):3.0f} %")
print("\nTexte de la section 5.5")
print("B2BL appelé dans", " / ".join(f"{100 * np.mean([x['b2bl'] for x in L if x['n'] == n]):.0f} %" for n in (3, 5, 8, 12)),
      "des départs (n = 3, 5, 8, 12)")
print(f"base propre au meilleur critère : {100 * np.mean([x['propre'] for x in L]):.0f} % ; "
      f"rotation tirée au hasard : {100 * np.mean([x['hasard'] for x in L]):.0f} % en moyenne")
print(f"meilleur atteint dès le premier départ : {100 * np.mean([x['des1'] for x in L]):.1f} % ; dès 5 : "
      f"{100 * np.mean([x['des5'] for x in L]):.1f} % ; dès 20 : {100 * np.mean([x['des20'] for x in L]):.1f} %")
g = 100 * np.array([x["gap"] for x in L])
print(f"écart à la borne : maximum {g.max():.2f} %")
