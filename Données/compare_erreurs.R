## =====================================================================
## Comparaison des erreurs : données NON NETTOYEES vs NETTOYEES
##   non nettoyé : Data/non_nettoye/data_merged.csv (418 échantillons, 27 var. de sol)
##   nettoyé     : Data/nettoye/data_site.csv       (138 sites, 27 var. de sol)
## Sortie : Figures/comparaison_erreurs.png
## A lancer depuis le dossier Données/
## =====================================================================
brut <- read.csv2("Data/non_nettoye/data_merged.csv")
net  <- read.csv2("Data/nettoye/data_site.csv")

x27 <- c("Argile", "Limon", "Sable", "Conduct", "pH.eau", "Carbonates", "Calcaire",
         "C..orga", "Mat_org", "Azote", "C.N", "P2O5", "CEC", "K2O", "MgO", "CaO",
         "Na2O", "Al", "Cd", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Pb", "Zn")
xs <- setdiff(names(net), c("Site", "city", "land_use", "site_group", "n_rep", "H", "S", "A"))

rmse <- function(e) sqrt(mean(e^2))

## ---------------------------------
## 1. Erreur de prédiction, mesurée de la même façon pour les deux jeux :
#    on prédit le H moyen d'un site, sur les 138 sites communs.
#    - apparente : le site fait partie de l'ajustement
#    - test      : le site entier est retiré avant l'ajustement (erreur de test du chap. 8)
Hsite <- setNames(net$H, net$Site)
Xsite <- brut[!duplicated(brut$Site), ]            # 1 ligne de X par site (jeu brut)
Xsite <- Xsite[match(net$Site, Xsite$Site), ]

# non nettoyé : 418 échantillons, 27 variables
f_brut   <- lm(reformulate(x27, "H"), brut)
app_brut <- Hsite - predict(f_brut, Xsite)
test_brut <- sapply(net$Site, function(s)
  Hsite[[s]] - predict(lm(reformulate(x27, "H"), brut[brut$Site != s, ]), Xsite[Xsite$Site == s, ]))

# nettoyé : 138 sites, 27 variables (erreur de test = e_i / (1 - h_ii), sans réajuster)
f_net    <- lm(reformulate(xs, "H"), net)
app_net  <- residuals(f_net)
test_net <- residuals(f_net) / (1 - hatvalues(f_net))

# référence : modèle sans variable (H moyen)
f_nul    <- lm(H ~ 1, net)
app_nul  <- residuals(f_nul)
test_nul <- residuals(f_nul) / (1 - hatvalues(f_nul))

# nettoyé + sélection de variables stepwise BIC (chap. 9, k = log n)
# L'erreur de test est honnête : la sélection est REFAITE sans le site à prédire.
step_bic <- function(d) step(lm(H ~ 1, d), scope = list(lower = ~1, upper = reformulate(xs)),
                             direction = "both", k = log(nrow(d)), trace = 0)
f_sel    <- step_bic(net)
app_sel  <- residuals(f_sel)
test_sel <- sapply(seq_len(nrow(net)), function(i) net$H[i] - predict(step_bic(net[-i, ]), net[i, ]))
cat("Variables retenues par step BIC :", attr(terms(f_sel), "term.labels"), "\n")

err <- rbind(apparente = c(nul = rmse(app_nul),  brut = rmse(app_brut),  net = rmse(app_net),  sel = rmse(app_sel)),
             test      = c(nul = rmse(test_nul), brut = rmse(test_brut), net = rmse(test_net), sel = rmse(test_sel)))
cat("RMSE sur le H moyen par site (138 sites)\n"); print(round(err, 3))
cat("Plus grande erreur de test |e| : brut =", round(max(abs(test_brut)), 2),
    "| nettoyé =", round(max(abs(test_net)), 2), "\n")

## ---------------------------------
## 2. Tests
pv <- function(f) { s <- summary(f)$fstatistic; pf(s[1], s[2], s[3], lower.tail = FALSE) }
cat("\nTest F global : brut p =", signif(pv(f_brut), 2), "| nettoyé p =", signif(pv(f_net), 2), "\n")
cat("Coefficients à p < 0,05 : brut =", sum(summary(f_brut)$coefficients[-1, 4] < 0.05), "/ 27",
    "| nettoyé =", sum(summary(f_net)$coefficients[-1, 4] < 0.05), "/ 27\n")

## ---------------------------------
## 3. Figure
bleu <- "#2a78d6"; orange <- "#eb6834"; aqua <- "#1baf7a"; gris <- "#9a9993"; encre <- "#0b0b0b"; encre2 <- "#52514e"
png("Figures/comparaison_erreurs.png", width = 1500, height = 1150, res = 200)
par(bg = "#fcfcfb", fg = encre2, col.axis = encre2, col.lab = encre2, col.main = encre,
    font.main = 1, cex.main = 1, family = "sans", las = 1, mar = c(4.2, 4.2, 3.6, 0.8))
cols <- c(gris, orange, bleu, aqua)
vars_sel <- paste(sub("..", "_", attr(terms(f_sel), "term.labels"), fixed = TRUE), collapse = " + ")
b <- barplot(t(err), beside = TRUE, col = cols, border = NA, ylim = c(0, max(err) * 1.38),
             names.arg = c("Erreur apparente\n(site dans l'ajustement)", "Erreur de test\n(site retiré)"),
             ylab = "RMSE sur H moyen par site", cex.names = 0.8, cex.axis = 0.8, space = c(0.08, 0.7))
text(b, t(err) + max(err) * 0.03, formatC(t(err), format = "f", digits = 2, decimal.mark = ","),
     cex = 0.7, col = encre)
legend("topleft", bty = "n", cex = 0.75, fill = cols, border = NA, text.col = encre,
       legend = c("Sans variable (H moyen)", "Non nettoyé (418 éch., 27 var.)", "Nettoyé (138 sites, 27 var.)",
                  paste0("Nettoyé + sélection BIC (", vars_sel, ")")))
title("Erreur de prédiction de H : non nettoyé vs nettoyé", adj = 0, line = 1.6)
dev.off()
cat("\nEcrit : Figures/comparaison_erreurs.png\n")
