"""LODE, section 6.1, tableau 5 : bases de la CPCA et de l'ACOM, mélange, dilution, étiquettes
différentes de la composante CoPRes la plus proche, motif de l'étiquetage LODE, écart à la borne,
proximité des sous-espaces des deux méthodes. Données simulées 1, 3, 4 et beurres et margarines.
Remarque : les données simulées 1, 3 et 4 sont régénérées par le générateur réécrit et diffèrent des
jeux d'origine des versions antérieures de l'article."""
import numpy as np
import CoPRes as cp
from commun import lire_simulees, lire_beurres, lire_vins

jeux = [(f"Simulées {p}", cp.pretraiter(lire_simulees(p)), R, None) for p, R in ((1, 5), (3, 7), (4, 6))]
jeux.append(("Vins", cp.pretraiter(lire_vins()[0]), 3, None))
b, _, _, _ = lire_beurres()
jeux.append(("Beurres", cp.pretraiter(b, snv=True), 4, ["V", "N", "M"]))
print(f"{'jeu':14s} {'base':5s} mél dil étiq≠  motif de l'étiquetage LODE                          1-Ψ/η    cos² min")
for nom, X, R, noms in jeux:
    th = cp.seuils_perm(X)
    cops = {m: cp.copres(X, th, R=R, methode=m, departs=100) for m in ("CPCA", "ACOM")}
    prox = cp.proximite(cops["CPCA"].base_methode, cops["ACOM"].base_methode)
    for m, c in cops.items():
        f = cp.fiche(None, c)
        lect = cp.lecture_directe(X, c.base_methode, th)
        proche = [c.etiquettes[int(np.argmax((c.base.T @ c.base_methode[:, h]) ** 2))] for h in range(R)]
        ndiff = sum(a != b for a, b in zip(lect.etiquettes, proche))
        print(f"{nom:14s} {m:5s} {int(f.melangee.sum()):3d} {int(f.diluee.sum()):3d} {ndiff:5d}  "
              f"{cp.texte_motif(c.motif, noms):50s} {100 * c.ecart_relatif:5.2f} %  {prox:.3f}")
    print(f"{'':14s} seuils : {np.round(th, 4)}")
