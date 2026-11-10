# Les collemboles, les acariens, les Gamasides et les oribates 
# font partie des principaux microarthropodes qui composent la mésofaune du sol 
# (mesurant généralement entre 0,1 et 2 mm)

## ---------------------------------
## Importation des données

data1 <- read.csv2("collembola_data_year1_train.csv", header= TRUE)
dim(data1)
str(data1)


data_site <- read.csv2("BISES_data_sites.csv",  header= TRUE)
dim(data_site)
str(data_site)

## ---------------------------------
##  Calcul des Indices de diversite

# Les quatre indices suivants seront utilises tout au long de cette partie :
# A = abondance (tous taxa par unité statistique)
# S = richesse "specifique" (niveau taxinomique = famille ici)
# H = indice de Shannon 
# J = equitabilite de Pielou ou J de Pielou

#install.packages("vegan")
library(vegan)

# Matrice de comptage par espèces
head(data1)
columns_to_remove <- colnames(data1) %in% c("Echantillon", "total_collembola",            
                                            "Acari_Gamasides",
                                            "Acari_Oribates",              
                                            "Acari_others",                 
                                            "Acari_total",                 
                                            "Other_microarthropodes",       
                                            "total_microarthropodes")   

data1.spe <- data1[, !(columns_to_remove)]
dim(data1.spe)

A <- rowSums(data1.spe)
S <- specnumber(data1.spe)
H <- diversity(data1.spe, index = "shannon") 
J <- H/log(S)

