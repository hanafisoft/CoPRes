# Nouveautés de CoPRes

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
