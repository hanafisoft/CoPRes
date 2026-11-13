# Extraction des données HELIX utilisées par l'article appliqué (section 4.1), à partir des fichiers
# de l'Exposome Data Challenge 2021 (ISGlobal), que l'utilisateur télécharge lui-même :
#   exposome.RData        (expositions, covariables, dictionnaire des variables)
#   metabol_urine.Rdata   (métabolome urinaire, objet ExpressionSet ; paquet Bioconductor Biobase)
# Ces données ne sont pas redistribuées avec CoPRes. Le script écrit, dans le dossier `sortie` :
#   helix_chimie.csv : ID, cohorte, 45 expositions chimiques postnatales mesurées chez l'enfant
#                      (métaux, organochlorés, pesticides organophosphorés, PBDE, PFAS, phénols,
#                      phtalates ; variables numériques, sommes hs_sum* exclues), en log2 ;
#   helix_urine.csv  : ID, cohorte, 44 métabolites urinaires.
# Usage : modifier les deux chemins ci-dessous, puis source("helix_extraire.R").

entree <- "CHEMIN/VERS/les/fichiers/telecharges"
sortie <- file.path("..", "donnees")

load(file.path(entree, "exposome.RData"))   # exposome, phenotype, covariates, codebook
cohorte <- setNames(as.character(covariates$h_cohort), covariates$ID)

familles <- c("Metals", "Organochlorines", "Organophosphate pesticides",
              "Polybrominated diphenyl ethers (PBDE)", "Per- and polyfluoroalkyl substances (PFAS)",
              "Phenols", "Phthalates")
cb <- codebook
sel <- cb$domain == "Chemicals" & cb$period == "Postnatal" & cb$var_type == "numeric" &
  cb$family %in% familles & !startsWith(as.character(cb$variable_name), "hs_sum")
vars <- as.character(cb$variable_name[sel])
noms <- paste0(as.character(cb$labelsshort[sel]), " (", as.character(cb$family[sel]), ")")
chimie <- data.frame(ID = exposome$ID, cohorte = cohorte[as.character(exposome$ID)],
                     exposome[, vars], check.names = FALSE)
names(chimie) <- c("ID", "cohorte", noms)
write.csv(chimie, file.path(sortie, "helix_chimie.csv"), row.names = FALSE)

load(file.path(entree, "metabol_urine.Rdata"))  # metabol_urine (ExpressionSet)
x <- t(Biobase::exprs(metabol_urine))
noms_u <- trimws(as.character(Biobase::fData(metabol_urine)$var))
urine <- data.frame(ID = as.integer(rownames(x)), cohorte = cohorte[rownames(x)], x, check.names = FALSE)
names(urine) <- c("ID", "cohorte", noms_u)
write.csv(urine, file.path(sortie, "helix_urine.csv"), row.names = FALSE)
cat("Écrit :", nrow(chimie), "enfants (chimie),", nrow(urine), "enfants (urine).\n")
