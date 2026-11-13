# Nouveautés de CoPRes

## 0.7.0 (13 novembre 2026)
- Dossier `exemples` du dépôt : scripts R et Python qui refont les calculs, tableaux et figures de l'article LODE et de l'article appliqué, avec les sorties de référence (`exemples/resultats`). Les données HELIX ne sont pas redistribuées : `R/helix_extraire.R` les extrait des fichiers téléchargés par l'utilisateur.
- Module de prédiction (groupes d'individus connus et lecture CoPRes) : `information_groupe` (part de l'information de groupe par partie et hors du sous-espace ; contribution propre et portée de chaque bloc), `plsda` (sous-espace des scores d'une PLS-DA, à lire par `copres(X, seuils, U = ...)`), `lda_loo`, `discrimination_parties` (analyse discriminante par partie, test par permutation), `variance_groupe`, `validation_croisee` (chaîne refaite dans chaque pli ; parties choisies par validation croisée interne).
- Correction : `rang_coude` (R) quand une variance du profil de vraisemblance est nulle (comme numpy.argmax).

## 0.6.0 (12 novembre 2026)
- Correction : graph_bilan (R) avec les versions récentes de ggplot2 (étiquettes de légende).
- Méthodes concurrentes, pour comparer CoPRes à JIVE et à AJIVE sur les mêmes données : `jive` (selon r.jive 2.4 : rangs donnés ou estimés par permutation), `ajive` (selon mvlearn 0.5 : bornes de Wedin et des directions aléatoires, réexamen des composantes jointes ; option `centrer`, vraie par défaut comme dans mvlearn, pour travailler sans centrage), `rang_coude` (rang de signal par le profil de vraisemblance de Zhu et Ghodsi). DOI 10.5281/zenodo.24165298.
- Validation : à rangs donnés, `ajive` reproduit mvlearn à 1e-10 près ; `jive` est comparé à r.jive (sorties de référence produites par tests/ref_concurrents/produire_ref_rjive.R). Les tirages aléatoires utilisent le générateur du paquet : mêmes résultats en R et en Python, mais pas les mêmes tirages que r.jive ou mvlearn.

## 0.5.0 (11 novembre 2026)
- Graphiques (expérimentaux) : graph_saliences, graph_coherence (argument `bloc` pour un seul panneau), graph_fiche, graph_motif, graph_scores, graph_scores_parties (argument `partie`), graph_loadings (cercle ou spectre), graph_contributions, graph_stabilite, graph_sensibilite, graph_bilan.
- Données des graphiques (stables) : donnees_saliences, donnees_coherence, donnees_fiche, donnees_motif, donnees_scores, donnees_loadings, donnees_contributions, donnees_stabilite, donnees_sensibilite, donnees_bilan.
- Langue des graphiques : français ou anglais (`langue("en")` ou argument `langue=`) ; dictionnaire commun aux versions R et Python (donnees/textes.csv).
- Thème modifiable : `theme(...)` ou `theme_copres(...)` (nom commun avec R).
- Version R : chaque graph_* rend un objet ggplot ; graph_coherence et graph_scores_parties rendent une liste de ggplot quand il y a plusieurs panneaux.
- Les fonctions de dessin peuvent évoluer d'une version à l'autre ; les fonctions de calcul et `donnees_*` restent stables.

## 0.4.0 (10 novembre 2026)
- Stabilité par bootstrap (stabilite_blocs, stabilite_groupes). DOI 10.5281/zenodo.24112795.

## 0.3.0 – 0.1.0 (9-10 novembre 2026)
- Couches lode, multibloc et multigroupe.
