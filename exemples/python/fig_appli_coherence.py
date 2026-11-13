"""Figure de cohérence de l'article appliqué (rates, R = 3) : (a) plan (u1, u2) de la CPCA, seuil fixe
de 10 % ; (b) plan commun du jeu cohérent tiré au hasard le plus loin des seuils (calculé par
appli_01_rates.py, fichier appli_01_plan_commun.csv). Sortie : fig_coherence_plan.png."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import CoPRes as cp
from commun import lire_rates

B, _, _, _ = lire_rates()
X = cp.pretraiter(B)
th = cp.seuils_perm(X)
U = cp.cpca(X, 3)
A3 = [U.T @ x @ x.T @ U for x in X]
Pa = np.eye(3)[:, :2]
Ba = [Pa.T @ a @ Pa for a in A3]
tha = np.array([0.10, 0.10])
Pb = np.loadtxt("appli_01_plan_commun.csv", delimiter=",")
e, v = np.linalg.eigh(Pb.T @ A3[0] @ Pb)
Pb = Pb @ v[:, ::-1]
Bb = [Pb.T @ a @ Pb for a in A3]
BLEU, ORANGE, ENCRE2 = "#2a78d6", "#eb6834", "#52514e"
tt = np.linspace(0, 2 * np.pi, 400)
C = np.vstack([np.cos(tt), np.sin(tt)])
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))


def panneau(a, Bm, thr, noms_axes, titre):
    for k, c in enumerate([BLEU, ORANGE]):
        w, z = np.linalg.eigh(Bm[k])
        E = z @ np.diag(np.sqrt(np.clip(w, 0, None))) @ z.T @ C
        a.fill(E[0], E[1], color=c, alpha=0.10, zorder=1)
        a.plot(E[0], E[1], color=c, lw=1.8, zorder=3)
        r = np.sqrt(thr[k])
        a.plot(r * C[0], r * C[1], color=c, lw=1.1, ls="--", zorder=2)
        d, m = z[:, 0], np.sqrt(max(w[0], 0))
        if d @ np.array([0.3, -1.0]) < 0:
            d = -d
        a.plot([0, m * d[0]], [0, m * d[1]], color=c, lw=1.6, zorder=4)
        a.text(m * d[0] + 0.03 * np.sign(d[0] if abs(d[0]) > 0.2 else 1), m * d[1] - 0.06,
               f"{100 * w[0]:.1f} %".replace(".", ","), color=c, fontsize=9, zorder=6,
               ha="left" if d[0] >= -0.2 else "right", bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    for j, nm in enumerate(noms_axes):
        e = np.eye(2)[:, j]
        a.plot([-0.75 * e[0], 0.75 * e[0]], [-0.75 * e[1], 0.75 * e[1]], color=ENCRE2, lw=0.8, ls=":", zorder=0)
        a.text(0.77 * e[0], 0.77 * e[1] + (0.02 if j == 0 else 0), nm, fontsize=9, color=ENCRE2)
    a.set_title(titre, fontsize=10, loc="left")
    a.set_aspect("equal")
    a.set_xlim(-0.8, 0.85)
    a.set_ylim(-0.8, 0.8)
    a.axis("off")


panneau(ax[0], Ba, tha, ["$u_1$", "$u_2$"], "(a) Plan $(u_1,u_2)$ lu commun, seuil 10 % : incohérent")
panneau(ax[1], Bb, th, ["$v_a$", "$v_b$"], "(b) Plan commun d'un jeu cohérent : cohérent")
h = [plt.Line2D([], [], color=BLEU, lw=1.8), plt.Line2D([], [], color=ORANGE, lw=1.8),
     plt.Line2D([], [], color=ENCRE2, lw=1.1, ls="--"), plt.Line2D([], [], color=ENCRE2, lw=1.2)]
fig.legend(h, ["métabolome (M)", "indicateurs (C)", "cercle du seuil du bloc", "direction la moins portée par le bloc (salience)"],
           loc="lower center", ncol=4, fontsize=8.5, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("fig_coherence_plan.png", dpi=200)
print("a", [np.round(np.linalg.eigvalsh(b), 4) for b in Ba], "b", [np.round(np.linalg.eigvalsh(b), 4) for b in Bb])
