"""LODE, section 2.4 et section 4.2 : bases de R^4 (tableau des trois bases) et étiquetage idéal
de deux matrices qui ne commutent pas."""
import numpy as np
import CoPRes as cp

c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
t = np.array([c, s, 0, 0])
A = [np.diag([0.50, 0, 0.45, 0.05]), 0.50 * np.outer(t, t) + np.diag([0, 0, 0.05, 0.45])]
theta = [0.10, 0.10]
r = np.sqrt(0.5)
bases = {"U1": np.eye(4),
         "U2": np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
         "U3": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, r, r], [0, 0, r, -r]])}
print("Tableau des bases : décomposition engendrée, étiquetage ?, saliences (s1 ; s2)")
for nom, U in bases.items():
    et = cp.etiquettes(A, U, theta)
    S = cp.saliences(A, U)
    print(f"{nom} : {cp.texte_motif(cp.motif(et))} ; étiquetage : {cp.est_etiquetage(A, U, theta)} ; "
          + ", ".join(f"({S[0, h]:.2f} ; {S[1, h]:.2f})" for h in range(4)))
res = cp.lode(A, theta)
print("\nÉtiquetage LODE :", res)

print("\nÉtiquetage idéal sans commutation (section 4.2)")
A1 = np.array([[0.45, 0.15, 0, 0], [0.15, 0.35, 0, 0], [0, 0, 0.10, 0], [0, 0, 0, 0.10]])
A2 = np.array([[0.40, 0, -0.15, 0], [0, 0.10, 0, 0.02], [-0.15, 0, 0.40, 0], [0, 0.02, 0, 0.10]])
ok, parties = cp.ideal([A1, A2], [0.15, 0.15])
print("les analyses à seuil commutent :", ok)
for L, V in parties.items():
    print(f"  {cp.nom_etiquette(L)} : {np.round(V.ravel(), 3)}")
print("borne de Ky Fan :", round(cp.borne_kf([A1, A2], [0.15, 0.15]), 6))
print(cp.lode([A1, A2], [0.15, 0.15]))
