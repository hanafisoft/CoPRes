# Exemples CoPRes : reproduction des articles LODE et CoPRes (appliqué)

Ces scripts reproduisent les tableaux et les chiffres de l'article LODE avec le paquet CoPRes
(version 0.7.0 ou ultérieure), en Python et en R. Les données sont dans `donnees/`.

| Script (python/ et R/) | Ce qu'il reproduit |
|---|---|
| `lode_01_bases` | sections 2.4 et 4.2 : bases de R^4 et étiquetage idéal |
| `lode_02_640` | section 5.5, tableau 4 : algorithme LODE sur les 640 jeux simulés |
| `lode_03_cpca_acom` | section 6.1, tableau 5 : CPCA et ACOM (simulées 1, 3, 4 ; beurres) |
| `lode_04_notes` | section 7.2 : notes d'élèves (trois lycées) |
| `lode_05_cpc` | section 7.3 : tableau CPC |
| `lode_06_beurres_mir` | section 7.4 : beurres et margarines en moyen infrarouge |
| `lode_07_complements` (Python seulement) | chiffres du texte des sections 6.1 et 7.3 |
| `lode_08_jive` | section 6.3, tableaux 7 et 8 : CoPRes, JIVE et AJIVE |
| `appli_01_rates` | article appliqué, section 2 : rates, R = 3 (lectures, jeux tirés au hasard, bilan, fiche) |
| `appli_02_methodes` | section 3 : CPCA, ACOM, ACCPS (rates, beurres, simulations, contre-exemple) |
| `appli_03_groupes` | section 4.1 : lycées simulés et HELIX |
| `appli_04_prediction` | section 4.2 : information de groupe par partie, PLS-DA, discrimination, stabilité, validation croisée |
| `fig_appli_coherence` (Python) | figure de cohérence de la section 2 |
| `simulations.R`, `helix_extraire.R` (R) | générateurs des données simulées ; extraction des données HELIX |

`commun.py` et `commun.R` lisent les données ; `resultats/` contient les sorties obtenues avec
la version Python.

## Lancer

Python (depuis `exemples/python`) : `python lode_03_cpca_acom.py`. Pour `lode_08_jive.py`, le nombre
de jeux par situation est un argument : `python lode_08_jive.py 50` (environ 45 minutes),
`python lode_08_jive.py 2` pour un essai.

R (le paquet CoPRes étant installé) : `setwd(".../exemples/R")`, puis `source("lode_03_cpca_acom.R")`.
Pour `lode_08_jive.R`, fixer d'abord `N <- 50` (ou `N <- 2` pour un essai).

## Données

| Fichier | Contenu | Source |
|---|---|---|
| `beurres_VIS.csv`, `beurres_NIR.csv`, `beurres_MIR.csv` | 21 beurres et margarines, trois blocs de spectres | Jaillais et al. (2007), avec l'autorisation de B. Jaillais |
| `rates_S1_L12.csv` | 22 rates en lactation : régime, 514 métabolites du plasma, 18 indicateurs | projet GDM-MILK (ANR-22-CE17-0039), avec l'accord des auteurs |
| `vins.csv` | 8 vins, 4 juges (profil libre) | Williams et Langron (1984), reproduites par Hanafi et Qannari (2008, annexe 6) |
| `simulees1.csv`, `simulees3.csv`, `simulees4.csv` | données simulées 1, 3 et 4 | plans de l'annexe de l'article |
| `sim640_matrices.csv`, `sim640_seuils.csv` | les 640 jeux de la section 5.5 (matrices, triangle supérieur, et seuils) | |
| `cpc_covariances.csv`, `cpc_seuils.csv`, `cpc_verites.csv` | jeux de la comparaison avec la CPC | |
| `notes_80eleves.csv`, `notes_intensite16.csv` | variantes des notes d'élèves | |

Les données des tableaux 7 et 8 sont tirées par le script lui-même, avec le générateur du paquet
(graines 1 à 50) : elles sont identiques en R et en Python.

Les données HELIX (Exposome Data Challenge 2021, ISGlobal) ne sont pas redistribuées : télécharger
`exposome.RData` et `metabol_urine.Rdata`, puis lancer `R/helix_extraire.R`, qui écrit
`donnees/helix_chimie.csv` et `donnees/helix_urine.csv`.
