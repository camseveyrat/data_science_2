## =====================================================================
## Préparation des données : jointure collemboles x sites (BISES)
## Sortie : data_merged.csv (1 ligne = 1 échantillon, format read.csv2)
## Cible pour la régression : H (indice de Shannon)
## =====================================================================

data1     <- read.csv2("collembola_data_year1_train.csv", header = TRUE)
data_site <- read.csv2("BISES_data_sites.csv", header = TRUE, fileEncoding = "UTF-8-BOM")
data1$X   <- NULL  # colonne d'index sans nom

## ---------------------------------
## 1. Clé de jointure
# Echantillon = <Site>_<réplicat>, ex. BISES_MO_FU_CDN_1 -> BISES_MO_FU_CDN
data1$Site     <- sub("_[0-9]+$", "", data1$Echantillon)
data1$replicat <- as.integer(sub("^.*_", "", data1$Echantillon))

## ---------------------------------
## 2. Variantes de site
# La plupart des sites : BISES_<ville>_<usage>_<code> (4 morceaux).
# Les parcs de Nancy (NY_PU) ont une variante en 5e morceau : _P ou _RU
# (ex. BISES_NY_PU_PEP_P et BISES_NY_PU_PEP_RU = même parc, 2 placettes).
parts <- strsplit(data_site$Site, "_")
data_site$site_base <- sapply(parts, function(p) paste(p[1:4], collapse = "_"))
data_site$variante  <- sapply(parts, function(p) if (length(p) == 5) p[5] else "aucune")

## ---------------------------------
## 3. Valeurs de sol manquantes (2 sites, toutes les variables de sol)
soil_vars <- setdiff(names(data_site), c("Site", "city", "land_use", "site_group",
                                         "site_base", "variante"))
data_site$sol_impute <- "non"

# BISES_NY_PU_PEP_RU : on prend la variante sœur du même parc (PEP_P)
i  <- data_site$Site == "BISES_NY_PU_PEP_RU"
j  <- data_site$Site == "BISES_NY_PU_PEP_P"
data_site[i, soil_vars] <- data_site[j, soil_vars]
data_site$sol_impute[i] <- "variante_soeur"

# BISES_NY_PU_GOD_RU : pas de variante sœur -> médiane des parcs NY (variante RU)
i   <- data_site$Site == "BISES_NY_PU_GOD_RU"
ref <- data_site$city == "NY" & data_site$land_use == "PU" &
       data_site$variante == "RU" & !i
data_site[i, soil_vars] <- lapply(data_site[ref, soil_vars], median, na.rm = TRUE)
data_site$sol_impute[i] <- "mediane_NY_PU_RU"

stopifnot(!anyNA(data_site[, soil_vars]))

## ---------------------------------
## 4. Indices de diversité (cible) — mêmes formules que vegan
non_species <- c("Echantillon", "Site", "replicat", "total_collembola",
                 "Acari_Gamasides", "Acari_Oribates", "Acari_others",
                 "Acari_total", "Other_microarthropodes", "total_microarthropodes")
spe <- data1[, !(names(data1) %in% non_species)]
stopifnot(all(rowSums(spe) == data1$total_collembola))

shannon <- function(x) { p <- x[x > 0] / sum(x); if (length(p) == 0) 0 else -sum(p * log(p)) }

data1$A <- rowSums(spe)                       # abondance
data1$S <- rowSums(spe > 0)                   # richesse spécifique
data1$H <- apply(spe, 1, shannon)             # Shannon (cible)
data1$J <- ifelse(data1$S > 1, data1$H / log(data1$S), NA)  # Pielou (NA si S <= 1)

## ---------------------------------
## 5. Jointure
# On garde : identifiants, indices, totaux faune ; on retire les 98 colonnes
# d'espèces (elles définissent H -> fuite de cible si utilisées en prédicteurs).
keep  <- c("Echantillon", "Site", "replicat", "A", "S", "H", "J",
           "Acari_Gamasides", "Acari_Oribates", "Acari_others", "Acari_total",
           "Other_microarthropodes", "total_microarthropodes")
merged <- merge(data1[, keep], data_site, by = "Site", all.x = TRUE, sort = FALSE)

stopifnot(nrow(merged) == nrow(data1), !anyNA(merged$city))

front  <- c("Echantillon", "Site", "site_base", "variante", "replicat",
            "city", "land_use", "site_group", "sol_impute", "A", "S", "H", "J")
merged <- merged[order(merged$Echantillon), c(front, setdiff(names(merged), front))]

write.csv2(merged, "data_merged.csv", row.names = FALSE)

cat("Lignes :", nrow(merged), "| colonnes :", ncol(merged),
    "| sites :", length(unique(merged$Site)), "\n")
cat("Sites sans échantillon (non joints) :",
    sum(!data_site$Site %in% data1$Site), "\n")
print(summary(merged[, c("A", "S", "H", "J")]))
