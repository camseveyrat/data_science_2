"""
Modèle lasso : régression linéaire multiple pénalisée en norme L1 (chap. 8 diapo 33, chap. 9 diapo 9)
Entrée  : Data/nettoye/data_site.csv (1 ligne = 1 site)
Cibles  : H (Shannon, cible du groupe), log(1+A), S, J (autres indices)
Sorties : Figures/lasso_<cible>_*.png
          Resultats/lasso_comparaison.csv   (erreurs de prédiction)
          Resultats/lasso_coefficients.csv  (coefficients et stabilité de la sélection)
A lancer depuis le dossier Données/

Principe : on minimise  ||y - Xb||² / (2n) + lambda * sum|b_j|
-> les coefficients sont rétrécis vers 0, certains exactement à 0 (sélection de variables).
lambda est choisi par validation croisée.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
AVEC_QUALI = False   # True : ajoute city et land_use (variables indicatrices)

os.makedirs("Resultats", exist_ok=True)
d = pd.read_csv("Data/nettoye/data_site.csv", sep=";", decimal=",")

soil = ["Argile", "Limon", "Sable", "Conduct", "pH.eau", "Carbonates", "Calcaire",
        "C..orga", "Mat_org", "Azote", "C.N", "P2O5", "CEC", "K2O", "MgO", "CaO",
        "Na2O", "Al", "Cd", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Pb", "Zn"]

# J (Pielou) n'est pas dans data_site.csv : moyenne par site des J des réplicats
# (J non défini si S <= 1). Mêmes sites que prepare_site.R (sol non imputé).
# 6 sites n'ont aucun réplicat avec J défini -> exclus pour J seulement.
merged = pd.read_csv("Data/non_nettoye/data_merged.csv", sep=";", decimal=",")
d["J"] = d["Site"].map(merged[merged["sol_impute"] == "non"].groupby("Site")["J"].mean())

# H : cible du groupe. A, S, J : les autres indices de diversité.
# A est très asymétrique (médiane ~9, max ~417) -> passage au log (chap. 8 diapo 52)
cibles = {"H": d["H"], "logA": np.log1p(d["A"]), "S": d["S"], "J": d["J"]}

# ---------------------------------
# 1. Variables explicatives
# Le lasso gère la colinéarité : pas de retrait par VIF.
# On retire seulement les doublons exacts, entre lesquels le lasso choisirait au hasard :
# Mat_org ~ 1,72 x C_orga, Sable = 1000 - Argile - Limon
x = [v for v in soil if v not in ("Mat_org", "Sable")]
X = d[x].copy()

# Variables très asymétriques (skewness > 1 : métaux, phosphore, carbonates...) :
# quelques sites extrêmes écrasent les autres -> passage au log (chap. 8 diapos 52 et 60).
# Toutes les valeurs sont >= 0 -> log(1 + x). Les colonnes transformées sont renommées log_<var>.
asym = X.columns[X.skew() > 1]
X[asym] = np.log1p(X[asym])
X = X.rename(columns={v: f"log_{v}" for v in asym})
print(f"Passage au log de {len(asym)} variables asymétriques : {', '.join(asym)}")

if AVEC_QUALI:
    X = pd.concat([X, pd.get_dummies(d[["city", "land_use"]], drop_first=True, dtype=float)], axis=1)
n, p = X.shape
print(f"n = {n} sites | p = {p} variables candidates")

# Le lasso pénalise tous les coefficients de la même façon :
# les variables doivent être sur la même échelle -> standardisation (moyenne 0, écart-type 1)
cv = KFold(n_splits=10, shuffle=True, random_state=SEED)


def nouveau_lasso():
    return make_pipeline(StandardScaler(), LassoCV(cv=cv, alphas=100, max_iter=100000))


def rmse(e):
    return np.sqrt(np.mean(e ** 2))


comparaison, coefficients, predictions = [], [], {}

for nom, y in cibles.items():
    # Sites où la cible est définie (tous, sauf pour J)
    ok = y.notna().values
    y, Xc = y.values[ok], X[ok]
    nc = len(y)
    print(f"\n==================== Cible : {nom} ({nc} sites) ====================")

    # ---------------------------------
    # 2. Choix de lambda par validation croisée 10 blocs
    modele = nouveau_lasso().fit(Xc, y)
    lcv = modele[-1]
    mse_moy = lcv.mse_path_.mean(axis=1)
    mse_se = lcv.mse_path_.std(axis=1, ddof=1) / np.sqrt(cv.get_n_splits())

    # lambda_min : erreur de CV minimale
    # lambda_1se : le plus grand lambda dont l'erreur reste à moins d'un écart-type du minimum
    #              -> modèle plus parcimonieux, convention de glmnet
    i_min = np.argmin(mse_moy)
    lambda_min = lcv.alphas_[i_min]
    lambda_1se = lcv.alphas_[mse_moy <= mse_moy[i_min] + mse_se[i_min]].max()
    print(f"lambda_min = {lambda_min:.4f} | lambda_1se = {lambda_1se:.4f}")

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(np.log10(lcv.alphas_), mse_moy, yerr=mse_se, fmt="o", ms=3, capsize=2)
    plt.axvline(np.log10(lambda_min), ls="--", c="red", label="lambda_min")
    plt.axvline(np.log10(lambda_1se), ls=":", c="black", label="lambda_1se")
    plt.xlabel("log10(lambda)")
    plt.ylabel("MSE de validation croisée")
    plt.title(f"Lasso ({nom}) : choix de lambda")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Figures/lasso_{nom}_choix_lambda.png", dpi=120)
    plt.close()

    # ---------------------------------
    # 3. Chemin de régularisation : évolution des coefficients quand lambda diminue
    Xs = modele[0].transform(Xc)
    alphas, chemin, _ = lasso_path(Xs, y - y.mean(), alphas=lcv.alphas_)
    plt.figure(figsize=(8, 5))
    for j in range(p):
        plt.plot(np.log10(alphas), chemin[j], lw=1)
        if np.abs(chemin[j]).max() > 0.15 * np.abs(chemin).max():
            plt.annotate(X.columns[j], (np.log10(alphas[-1]), chemin[j, -1]), fontsize=7)
    plt.axvline(np.log10(lambda_min), ls="--", c="red", label="lambda_min")
    plt.axvline(np.log10(lambda_1se), ls=":", c="black", label="lambda_1se")
    plt.xlabel("log10(lambda)")
    plt.ylabel("coefficient (variables standardisées)")
    plt.title(f"Chemin de régularisation du lasso ({nom})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Figures/lasso_{nom}_chemin.png", dpi=120)
    plt.close()

    # ---------------------------------
    # 4. Variables sélectionnées
    # Coefficient standardisé : effet d'une hausse d'1 écart-type de la variable
    # Coefficient brut        : effet d'une hausse d'1 unité de la variable
    #                           (pour une variable log_<var> : d'1 unité de log(1 + var))
    def coefs_a(lam):
        _, c, _ = lasso_path(Xs, y - y.mean(), alphas=[lam])
        return pd.Series(c[:, 0], index=X.columns)

    c_min, c_1se = coefs_a(lambda_min), coefs_a(lambda_1se)
    for lab, c in [("lambda_min", c_min), ("lambda_1se", c_1se)]:
        c = c[c != 0].sort_values(key=abs, ascending=False)
        print(f"{lab} : {len(c)} variable(s) gardée(s)"
              + ("" if len(c) else " -> modèle = moyenne de la cible"))
        for v, b in c.items():
            print(f"   {v:<11} {b:+.4f}")

    # ---------------------------------
    # 5. Erreur de prédiction honnête
    # Leave-one-out : chaque site est retiré, TOUT est réappris sans lui (standardisation
    # + choix de lambda par CV interne), puis on prédit ce site -> validation croisée imbriquée.
    # On compte aussi combien de fois chaque variable est gardée sur les n réapprentissages :
    # c'est la stabilité de la sélection.
    pred, pred_moy = np.empty(nc), np.empty(nc)
    choisie = np.zeros(p)
    for app, test in LeaveOneOut().split(Xc):
        m = nouveau_lasso().fit(Xc.iloc[app], y[app])
        pred[test] = m.predict(Xc.iloc[test])
        pred_moy[test] = y[app].mean()
        choisie += m[-1].coef_ != 0

    # Référence : le modèle sans variable, qui prédit toujours la moyenne de la cible
    sce_moy = np.sum((y - pred_moy) ** 2)
    for mod, e, k, r2 in [
        ("moyenne", y - pred_moy, 0, 0.0),
        ("lasso lambda_min", y - pred, int((c_min != 0).sum()), modele.score(Xc, y)),
    ]:
        comparaison.append({"cible": nom, "n_sites": nc, "modele": mod, "nb_variables": k,
                            "R2_apprentissage": r2, "RMSE_LOO": rmse(e),
                            "R2_prediction_LOO": 1 - np.sum(e ** 2) / sce_moy})

    coefficients.append(pd.DataFrame({
        "cible": nom, "variable": X.columns,
        "coef_std_lambda_min": c_min.values,
        "coef_brut_lambda_min": c_min.values / modele[0].scale_,
        "coef_std_lambda_1se": c_1se.values,
        "freq_selection_LOO": choisie / nc,
    }))
    predictions[nom] = (y, pred, pred_moy[0])

# ---------------------------------
# 6. Tableaux récapitulatifs
comp = pd.DataFrame(comparaison)
coef = pd.concat(coefficients)
coef = coef[(coef.coef_std_lambda_min != 0) | (coef.freq_selection_LOO > 0)]
coef = coef.sort_values(["cible", "freq_selection_LOO"], ascending=[True, False])

pd.set_option("display.width", 200)
print("\n==================== Erreurs de prédiction (leave-one-out) ====================")
print(comp.round(3).to_string(index=False))
print("\nR2_prediction_LOO < 0 : le modèle prédit moins bien que la simple moyenne.")
print("\n==================== Variables retenues et stabilité ====================")
print(coef.round(4).to_string(index=False))
print("\nfreq_selection_LOO : part des réapprentissages LOO où la variable est gardée.")

# ---------------------------------
# 7. Figures de synthèse (tous les indices côte à côte)
noms = list(cibles)

# Coefficients retenus (lambda_min) et stabilité de la sélection
fig, axes = plt.subplots(1, len(noms), figsize=(4 * len(noms), 4.5))
for ax, nom in zip(axes, noms):
    c = coef[(coef.cible == nom) & (coef.freq_selection_LOO >= 0.5)]
    c = c.sort_values("coef_std_lambda_min")
    if len(c):
        ax.barh(c.variable, c.coef_std_lambda_min,
                color=np.where(c.coef_std_lambda_min > 0, "tab:blue", "tab:red"))
        for i, (b, fr) in enumerate(zip(c.coef_std_lambda_min, c.freq_selection_LOO)):
            ax.annotate(f"{fr:.0%}", (b, i), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.set_xlim(0, c.coef_std_lambda_min.max() * 1.35)
    else:
        ax.text(0.5, 0.5, "aucune variable\nretenue", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_title(nom)
    ax.set_xlabel("coefficient standardisé")
fig.suptitle("Lasso (lambda_min) : variables retenues - % = stabilité de la sélection (LOO)")
plt.tight_layout()
plt.savefig("Figures/lasso_synthese_coefficients.png", dpi=120)
plt.close()

# Valeurs observées vs prédites en leave-one-out
fig, axes = plt.subplots(1, len(noms), figsize=(4 * len(noms), 4.2))
for ax, nom in zip(axes, noms):
    y_obs, y_pred, moy = predictions[nom]
    r2p = comp[(comp.cible == nom) & (comp.modele == "lasso lambda_min")].R2_prediction_LOO.item()
    lim = [min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())]
    ax.scatter(y_obs, y_pred, s=12, alpha=0.7)
    ax.plot(lim, lim, ls="--", c="grey", label="prédiction parfaite")
    ax.axhline(moy, c="red", lw=1, label="moyenne")
    ax.set(title=f"{nom}  (R² prédiction = {r2p:.2f})", xlabel="observé", ylabel="prédit (LOO)")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("Figures/lasso_synthese_observe_predit.png", dpi=120)
plt.close()

pd.concat([pd.DataFrame({"cible": nom, "observe": yo, "predit_LOO": yp})
           for nom, (yo, yp, _) in predictions.items()]
          ).to_csv("Resultats/lasso_predictions.csv", sep=";", decimal=",", index=False)
comp.to_csv("Resultats/lasso_comparaison.csv", sep=";", decimal=",", index=False)
coef.to_csv("Resultats/lasso_coefficients.csv", sep=";", decimal=",", index=False)
print("\nEcrit : Resultats/lasso_comparaison.csv, Resultats/lasso_coefficients.csv, Figures/lasso_*.png")
