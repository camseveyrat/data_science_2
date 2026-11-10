## =====================================================================
## Table au niveau SITE pour la régression de H (Shannon)
## Entrée  : Data/non_nettoye/data_merged.csv (1 ligne = 1 échantillon)
## Sortie  : Data/nettoye/data_site.csv   (1 ligne = 1 site, format read.csv2)
## Corrige : pseudo-réplication (1 ligne = 1 site), sites au sol imputé retirés
## Conserve les 27 variables de sol (colinéarité seulement diagnostiquée)
## A lancer depuis le dossier Données/
## =====================================================================
library(car)   # vif()

m <- read.csv2("Data/non_nettoye/data_merged.csv")

soil <- c("Argile", "Limon", "Sable", "Conduct", "pH.eau", "Carbonates", "Calcaire",
          "C..orga", "Mat_org", "Azote", "C.N", "P2O5", "CEC", "K2O", "MgO", "CaO",
          "Na2O", "Al", "Cd", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Pb", "Zn")

## ---------------------------------
## 1. Pseudo-réplication -> 1 ligne par site
# Le sol est mesuré par site : les 3 réplicats d'un site partagent le même X,
# leurs erreurs sont corrélées -> (H2) Sigma = sigma^2 I violée (chap. 6).
# On moyenne la réponse par site ; l'unité statistique devient le site.
# Les 2 sites au sol imputé sont retirés (pas de vraie mesure de X).
m <- m[m$sol_impute == "non", ]

# J est NA quand S <= 1 (équitabilité indéfinie) : moyenne sur les réplicats
# renseignés, et J reste NA pour les 6 sites dont aucun réplicat n'a S > 1.
# na.action = na.pass est indispensable : sans lui, aggregate supprime les
# lignes où J est NA et perdrait 6 sites.
rep_ <- aggregate(cbind(A, S, H, J) ~ Site, m, mean, na.rm = TRUE,
                  na.action = na.pass)
rep_$n_rep <- as.vector(table(m$Site)[rep_$Site])   # nb de réplicats (3, sauf 1 site)
site <- merge(rep_, m[!duplicated(m$Site), c("Site", "city", "land_use", "site_group", soil)],
              by = "Site")
stopifnot(nrow(site) == length(unique(m$Site)),
          !anyNA(site[, setdiff(names(site), "J")]))   # J peut être NA (S <= 1)

## ---------------------------------
## 2. Colinéarité : diagnostic seulement (chap. 8.1-8.2)
# Les 27 variables de sol sont TOUTES conservées, même très corrélées
# (Argile+Limon+Sable ~ 1000, Mat_org ~ 1,73 x C_orga, C/N = C_orga/Azote...).
# Le choix des variables à retirer se fera au moment de la modélisation.
ev <- eigen(cor(site[, soil]))$values
cat("p =", length(soil), "| rang =", qr(cbind(1, as.matrix(site[, soil])))$rank,
    "| kappa =", format(ev[1] / ev[length(ev)], digits = 3), "(seuil 500)",
    "| VIF max =", round(max(vif(lm(reformulate(soil, "H"), site))), 1), "(seuil 10)\n")

## ---------------------------------
## 4. Export
site <- site[, c("Site", "city", "land_use", "site_group", "n_rep", "H", "J", "S", "A", soil)]
write.csv2(site, "Data/nettoye/data_site.csv", row.names = FALSE)
cat("Ecrit : Data/nettoye/data_site.csv —", nrow(site), "sites x", ncol(site), "colonnes\n")
# Rappel : J, S et A sont d'autres réponses possibles, PAS des explicatives de H.
