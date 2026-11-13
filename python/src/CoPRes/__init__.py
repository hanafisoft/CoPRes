"""CoPRes : composantes communes, propres et résiduelles.

Le module `lode` résout le problème LODE (décomposition orthogonale étiquetée optimale) pour
des matrices semi-définies positives quelconques ; les autres modules (à venir) appliquent
LODE aux données multiblocs et multigroupes.
"""
from .alea import ParkMiller, rotation_aleatoire
from .lode import (saliences, etiquettes, motif, texte_motif, nom_etiquette, critere,
                   critere_etiquetage, marge, est_etiquetage, borne_kf, ideal, concentration,
                   base_concentree, taro, b2bl, lode, ResultatLODE)

__version__ = "0.7.0"
from .multibloc import (pretraiter, seuils_perm, cpca, acom, accps, METHODES, lecture_directe,
                        copres, fiche, proximite, equivalence, sensibilite, LectureDirecte, FactorisationCoPRes, Fiche)
from .multigroupe import (lire_notes, pretraiter_groupes, seuils_groupes, copres_groupes, cpc,
                          lecture_groupes, discrimination, PrepGroupes, FactorisationGroupes, LectureGroupes)
from .stabilite import stabilite_groupes, stabilite_blocs, type_etiquette, Stabilite
from .graphiques import (langue, theme, theme_copres, donnees_saliences, donnees_coherence, donnees_fiche, donnees_motif,
                         donnees_scores, donnees_loadings, donnees_contributions, donnees_stabilite,
                         donnees_sensibilite, donnees_bilan, graph_saliences, graph_coherence, graph_fiche,
                         graph_motif, graph_scores, graph_scores_parties, graph_loadings, graph_contributions,
                         graph_stabilite, graph_sensibilite, graph_bilan)
from .concurrents import jive, ajive, rang_coude, ResultatJIVE, ResultatAJIVE
from .prediction import (indicatrice, information_groupe, plsda, lda_loo, discrimination_parties,
                         variance_groupe, validation_croisee, InformationGroupe, DiscriminationParties,
                         ValidationCroisee)
