#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Journée Data Science n°2 — régression ridge par optimisation explicite.

Le but n'est pas d'appeler `sklearn.linear_model.Ridge`, mais d'écrire le
problème de minimisation, de le résoudre, puis de se servir de scikit-learn
comme contre-vérification.

Convention retenue (à garder en tête pour comparer à d'autres outils) :

    J(b) = 1/(2n) * ||y - X b||^2  +  (lambda/2) * ||b_{1:}||^2
           \_____ erreur _____/      \______ pénalité L2 ______/

  - n est le nombre de lignes de X (ici des sites, pas des échantillons) ;
  - la constante b[0] n'est jamais pénalisée ;
  - les colonnes de X sont standardisées sur le train (moyenne 0, écart-type 1),
    ce qui rend la pénalité indépendante des unités.

Équivalences, pour ne pas se tromper de facteur :

  - forme fermée : b = (X'X + n*lambda*P)^-1 X'y, avec P = diag(0, 1, ..., 1) ;
  - scikit-learn : Ridge(alpha=alpha) minimise ||y - Xb||^2 + alpha*||b||^2,
    donc alpha = n * lambda ;
  - données augmentées : empiler sqrt(n*lambda) * sqrt(P) sous X et des zéros
    sous y donne exactement le même problème.

Données et protocole : tout vient de `en_commun.py` (source `data_site.csv`,
découpage 70/30 stratifié, standardisation sur le train, plis communs). Ce
script ne redéfinit ni le jeu de test, ni la standardisation, ni les métriques :
c'est ce qui rend ses chiffres comparables à ceux des autres régressions.

Usage :
    python regression_ridge.py                     # réponse H (défaut)
    python regression_ridge.py --reponse J         # équitabilité
    python regression_ridge.py --reponse logA      # abondance en log
    python regression_ridge.py --reponse S         # richesse
    python regression_ridge.py --reponse A         # abondance brute
    python regression_ridge_toutes.py --toutes-reponses   # A, H, J, S : un résultat
                                                   # détaillé + un graphique
                                                   # par réponse
    python regression_ridge.py --tous-les-sols     # 27 variables au lieu de 20
    python regression_ridge.py --sans-ville --sans-figures

Aucun fichier n'est écrit sur le disque.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from en_commun import (GRAINE, K_PLIS, SOL_REDUIT, SOL_TOUS, metriques,
                       plier, preparer, standardiser_pli)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAMBDA_DEMO = 1.0          # lambda utilisé pour comparer les trois solveurs

# Jeu d'explicatives restreint, demandé par Camille « pour l'instant » : les
# trois seules variables de sol significatives après correction de
# Benjamini-Hochberg dans le notebook (3 sur 27). Ce choix est propre à ce
# script : il ne change ni le protocole ni les chiffres de `en_commun.py`.
# Conséquence à garder en tête : les performances ne sont alors PAS
# comparables à celles des modèles à 20 variables, seul le découpage l'est.
TROIS_VARIABLES = ["CEC", "pH.eau", "Carbonates"]

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# 1. Données (déléguées à en_commun)
# ---------------------------------------------------------------------------

def design(X) -> np.ndarray:
    """Ajoute la constante à droite de X (X ne la contient pas)."""
    return np.column_stack([np.ones(len(X)), np.asarray(X, float)])


def choisir_sol(args) -> list:
    """Les variables de sol retenues par les options de la ligne de commande."""
    if args.trois_variables and args.tous_les_sols:
        raise SystemExit("--trois-variables et --tous-les-sols s'excluent.")
    if args.trois_variables:
        return TROIS_VARIABLES
    return SOL_TOUS if args.tous_les_sols else SOL_REDUIT


def construire_jeu(args):
    """Le jeu commun, avec les options propres au script."""
    return preparer(reponse=args.reponse, sol=choisir_sol(args),
                    avec_ville=not args.sans_ville, graine=args.graine)


# ---------------------------------------------------------------------------
# 2. Le problème d'optimisation
# ---------------------------------------------------------------------------

def matrice_penalite(p: int) -> np.ndarray:
    """P = diag(0, 1, ..., 1) : pénalise tout sauf la constante."""
    P = np.eye(p)
    P[0, 0] = 0.0
    return P


def erreur(theta: np.ndarray, A: np.ndarray, y: np.ndarray) -> float:
    """Erreur quadratique moyenne divisée par deux : ||y - A theta||^2 / (2n)."""
    r = y - A @ theta
    return 0.5 * np.mean(r ** 2)


def penalite(theta: np.ndarray) -> float:
    """Pénalité L2 sur les coefficients, constante exclue : ||theta[1:]||^2 / 2."""
    return 0.5 * np.sum(theta[1:] ** 2)


def objectif(theta: np.ndarray, A: np.ndarray, y: np.ndarray,
             lam: float) -> float:
    """Fonction à minimiser : erreur + lambda * pénalité."""
    return erreur(theta, A, y) + lam * penalite(theta)


def gradient(theta: np.ndarray, A: np.ndarray, y: np.ndarray,
             lam: float) -> np.ndarray:
    """Gradient analytique de l'objectif.

    d/dtheta = -(1/n) A'(y - A theta) + lambda * P theta,
    avec P theta = (0, theta[1], ..., theta[p-1]).
    """
    r = y - A @ theta
    g = -(A.T @ r) / len(y)
    g[1:] += lam * theta[1:]
    return g


def verifier_gradient(A: np.ndarray, y: np.ndarray, lam: float,
                      graine: int = 0) -> float:
    """Compare le gradient analytique aux différences finies centrées."""
    rng = np.random.default_rng(graine)
    theta = rng.normal(size=A.shape[1])
    g = gradient(theta, A, y, lam)
    ecarts = np.zeros_like(theta)
    h = 1e-6
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = h
        ecarts[j] = (objectif(theta + e, A, y, lam)
                     - objectif(theta - e, A, y, lam)) / (2 * h)
    return float(np.max(np.abs(g - ecarts) / np.maximum(1.0, np.abs(ecarts))))


# ---------------------------------------------------------------------------
# 3. Trois façons de minimiser
# ---------------------------------------------------------------------------

def ridge_ferme(A: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Forme fermée : theta = (A'A + n*lambda*P)^-1 A'y."""
    n, p = A.shape
    return np.linalg.solve(A.T @ A + n * lam * matrice_penalite(p), A.T @ y)


def descente_gradient(A: np.ndarray, y: np.ndarray, lam: float,
                      pas_relatif: float = 1.8, max_iter: int = 100_000,
                      tol: float = 1e-11) -> tuple[np.ndarray, np.ndarray]:
    """Descente de gradient à pas fixe.

    Le pas vaut `pas_relatif / L`, où L est la plus grande valeur propre de
    (1/n) A'A + lambda P. La convergence est garantie pour pas_relatif < 2 ;
    au-delà, la suite diverge — c'est ce que montre la figure 1.
    """
    n, p = A.shape
    H = A.T @ A / n + lam * matrice_penalite(p)
    L = float(np.linalg.eigvalsh(H).max())
    pas = pas_relatif / L

    theta = np.zeros(p)
    historique = [objectif(theta, A, y, lam)]
    for _ in range(max_iter):
        g = gradient(theta, A, y, lam)
        if np.linalg.norm(g) < tol:
            break
        theta = theta - pas * g
        historique.append(objectif(theta, A, y, lam))
    return theta, np.asarray(historique)


def scipy_minimiser(A: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Minimisation par BFGS avec le gradient analytique."""
    from scipy.optimize import minimize
    res = minimize(objectif, np.zeros(A.shape[1]), args=(A, y, lam),
                   jac=gradient, method="BFGS",
                   options={"gtol": 1e-10, "maxiter": 100_000})
    return res.x


def theta_augmente(A: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge = moindres carrés ordinaires sur des données augmentées."""
    n, p = A.shape
    racine = np.sqrt(n * lam) * np.diag(np.sqrt(np.diag(matrice_penalite(p))))
    A_aug = np.vstack([A, racine])
    y_aug = np.concatenate([y, np.zeros(p)])
    return np.linalg.lstsq(A_aug, y_aug, rcond=None)[0]


# ---------------------------------------------------------------------------
# 4. Choix de lambda et évaluation
# ---------------------------------------------------------------------------

def chemin_coefficients(A: np.ndarray, y: np.ndarray,
                        lambdas: np.ndarray) -> np.ndarray:
    """Coefficients pour chaque lambda (une ligne par lambda)."""
    return np.array([ridge_ferme(A, y, lam) for lam in lambdas])


def validation_croisee(X_train, y_train: np.ndarray, lambdas: np.ndarray,
                       k: int = K_PLIS, graine: int = GRAINE):
    """Erreur de validation croisée (MSE) pour chaque lambda.

    Utilise les plis de `en_commun.plier` et refait la standardisation dans
    chaque pli : les lambda sont donc comparés dans les mêmes conditions que
    les hyperparamètres des autres méthodes.
    """
    X = X_train.reset_index(drop=True)
    mse = np.zeros((len(lambdas), k))
    for pli, (app, val) in enumerate(plier(len(y_train), k=k, graine=graine)):
        mu, sd = standardiser_pli(X.iloc[app])
        A_app = design((X.iloc[app] - mu) / sd)
        A_val = design((X.iloc[val] - mu) / sd)
        for i, lam in enumerate(lambdas):
            theta = ridge_ferme(A_app, y_train[app], lam)
            mse[i, pli] = np.mean((y_train[val] - A_val @ theta) ** 2)
    return mse.mean(axis=1), mse.std(axis=1, ddof=1) / np.sqrt(k), mse


def ddl_effectifs(A: np.ndarray, lam: float) -> float:
    """Degrés de liberté effectifs : trace(A (A'A + n lambda P)^-1 A')."""
    n, p = A.shape
    return float(np.trace(A @ np.linalg.solve(
        A.T @ A + n * lam * matrice_penalite(p), A.T)))


# ---------------------------------------------------------------------------
# 5. Rapport
# ---------------------------------------------------------------------------

def banniere(titre: str) -> None:
    print()
    print("=" * 78)
    print(titre)
    print("=" * 78)


def main() -> None:
    args = parser.parse_args()
    if args.toutes_reponses or args.reponse is None:
        toutes_reponses(args)
        return
    jeu = construire_jeu_reponse(args.reponse, args)
    A, B = design(jeu.X_train), design(jeu.X_test)
    y_train, y_test = jeu.y_train, jeu.y_test
    noms = ["constante"] + list(jeu.X.columns)
    n, p = A.shape
    ctrl = {}                     # valeurs numériques reprises dans le rapport

    banniere("0. Cadre du calcul")
    print(f"Réponse              : {args.reponse}")
    print(f"Sites utilisés       : {len(jeu.y)} (source : data_site.csv)")
    if jeu.n_exclus:
        print(f"                       {jeu.n_exclus} sites écartés : "
              f"{jeu.reponse} non défini")
    print(f"Découpage            : {len(jeu.idx_train)} sites en apprentissage / "
          f"{len(jeu.idx_test)} en test (graine {jeu.graine})")
    print(f"Empreinte du découpage : {jeu.empreinte}")
    print(f"Explicatives         : {len(jeu.sol)} variables de sol "
          f"+ {'usage et ville' if jeu.avec_ville else 'usage seul'}")
    print(f"Matrice de design    : {n} x {p} (constante + {p - 1} colonnes)")
    print(f"Convention           : J(b) = 1/(2n)||y-Ab||^2 + (lambda/2)||b||^2")
    print(f"                       donc alpha(scikit-learn) = n * lambda = {n} * lambda")

    # ------------------------------------------------------------------
    banniere("1. Le problème, écrit noir sur blanc (lambda démo = "
             f"{LAMBDA_DEMO})")
    theta0 = np.zeros(p)
    print(f"erreur(b=0)          = {erreur(theta0, A, y_train):.6f}")
    print(f"pénalité(b=0)        = {penalite(theta0):.6f}")
    print(f"objectif(b=0)        = {objectif(theta0, A, y_train, LAMBDA_DEMO):.6f}")
    ecart_grad = verifier_gradient(A, y_train, LAMBDA_DEMO)
    ctrl["gradient"] = ecart_grad
    print(f"gradient analytique vs différences finies : écart relatif max = "
          f"{ecart_grad:.2e}  -> {'OK' if ecart_grad < 1e-5 else 'A CORRIGER'}")
    if args.trois_variables:
        print("Attention : 3 variables de sol seulement ; ces performances ne "
              "sont pas\ncomparables à celles des modèles à 20 variables, seul "
              "le découpage l'est.")

    # ------------------------------------------------------------------
    banniere("2. Trois solveurs pour le même lambda")
    theta_ferme = ridge_ferme(A, y_train, LAMBDA_DEMO)
    theta_gd, hist = descente_gradient(A, y_train, LAMBDA_DEMO)
    theta_bfgs = scipy_minimiser(A, y_train, LAMBDA_DEMO)

    print(f"{'solveur':<22}{'||b|| hors cste':>16}{'objectif':>12}"
          f"{'écart à la forme fermée':>26}")
    for nom, th in [("forme fermée", theta_ferme),
                    ("descente de gradient", theta_gd),
                    ("BFGS (scipy)", theta_bfgs)]:
        ecart = np.max(np.abs(th - theta_ferme))
        ctrl[nom] = ecart
        print(f"{nom:<22}{np.linalg.norm(th[1:]):>16.6f}"
              f"{objectif(th, A, y_train, LAMBDA_DEMO):>12.9f}"
              f"{ecart:>26.2e}")

    sklearn_modele = Ridge(alpha=n * LAMBDA_DEMO, fit_intercept=True).fit(
        A[:, 1:], y_train)
    ecart_sk = np.max(np.abs(sklearn_modele.coef_ - theta_ferme[1:]))
    ctrl["scikit-learn"] = ecart_sk
    ctrl["iterations"] = len(hist)
    print(f"\nContre-vérification scikit-learn (alpha = n*lambda = {n * LAMBDA_DEMO:.3f}) "
          f": écart max = {ecart_sk:.2e}")

    # ------------------------------------------------------------------
    print(f"\nConvergence de la descente : {len(hist)} itérations, "
          f"objectif final {hist[-1]:.9f} (forme fermée "
          f"{objectif(theta_ferme, A, y_train, LAMBDA_DEMO):.9f})")
    print(f"Décomposition à l'optimum : erreur = "
          f"{erreur(theta_ferme, A, y_train):.4f}, "
          f"lambda*pénalité = {LAMBDA_DEMO * penalite(theta_ferme):.4f}")

    # ------------------------------------------------------------------
    banniere("3. Le même résultat par données augmentées")
    theta_aug = theta_augmente(A, y_train, LAMBDA_DEMO)
    ctrl["données augmentées"] = float(np.max(np.abs(theta_aug - theta_ferme)))
    print(f"écart max avec la forme fermée : "
          f"{ctrl['données augmentées']:.2e}")
    print("Autrement dit : pénaliser, c'est ajouter des lignes d'observations")
    print("fictives qui valent toutes 0, avec un poids sqrt(n*lambda).")

    # ------------------------------------------------------------------
    banniere("4. Choix de lambda par validation croisée")
    lambdas = np.logspace(-4, 2, 61)
    cv_moy, cv_se, cv_pli = validation_croisee(
        jeu.X.iloc[jeu.idx_train], y_train, lambdas, k=K_PLIS, graine=args.graine)
    i_min = int(np.argmin(cv_moy))
    seuil = cv_moy[i_min] + cv_se[i_min]
    i_1se = int(np.max(np.where(cv_moy <= seuil)[0]))
    lam_min, lam_1se = float(lambdas[i_min]), float(lambdas[i_1se])

    print(f"{K_PLIS} plis sur les {n} sites d'apprentissage, grille "
          f"{lambdas[0]:.0e} à {lambdas[-1]:.0e} (61 valeurs)")
    print(f"lambda qui minimise l'erreur CV : {lam_min:.4g}  "
          f"(MSE CV = {cv_moy[i_min]:.4f})")
    print(f"lambda de la règle 1-SE          : {lam_1se:.4g}  "
          f"(MSE CV = {cv_moy[i_1se]:.4f})")
    if i_1se >= len(lambdas) - 2:
        print("  Remarque : la courbe de validation est plate au-delà de lambda ~ 1 ;")
        print("  la règle 1-SE retient donc le bord de la grille, soit un modèle")
        print("  presque constant. Autrement dit, le sol n'apporte plus de signal")
        print("  prédictif au-delà de cette régularisation.")

    # ------------------------------------------------------------------
    banniere(f"5. Évaluation sur le jeu de test ({len(jeu.idx_test)} sites jamais vus)")
    theta_ols = ridge_ferme(A, y_train, 0.0)
    theta_min = ridge_ferme(A, y_train, lam_min)
    theta_1se = ridge_ferme(A, y_train, lam_1se)

    print(f"{'modèle':<24}{'RMSE train':>12}{'RMSE test':>11}"
          f"{'MAE test':>10}{'R2 test':>9}{'ddl effectifs':>15}")
    lignes = []
    for nom, th, lam in [("OLS (lambda = 0)", theta_ols, 0.0),
                         (f"ridge lambda = {lam_min:.3g}", theta_min, lam_min),
                         (f"ridge lambda = {lam_1se:.3g}", theta_1se, lam_1se)]:
        m_train = metriques(y_train, A @ th)
        m_test = metriques(y_test, B @ th)
        ddl = ddl_effectifs(A, lam)
        lignes.append({
            "modèle": nom, "lambda": lam, "ddl": ddl,
            "RMSE_train": m_train["RMSE"], "RMSE_test": m_test["RMSE"],
            "MAE_test": m_test["MAE"],
            "R2_train": m_train["R2"], "R2_test": m_test["R2"],
        })
        print(f"{nom:<24}{m_train['RMSE']:>12.4f}{m_test['RMSE']:>11.4f}"
              f"{m_test['MAE']:>10.4f}{m_test['R2']:>9.3f}{ddl:>15.1f}")

    print("\nLecture : ridge échange un peu de biais contre beaucoup de variance.")
    print(f"Le nombre de paramètres réellement mobilisés passe de {p} (OLS) à "
          f"{ddl_effectifs(A, lam_min):.1f}.")
    print("Un R2 négatif en test signifie que le modèle fait pire que la moyenne")
    print("du jeu de test : signature d'un ajustement qui a appris du bruit.")

    # ------------------------------------------------------------------
    banniere("6. Coefficients au lambda retenu (unités = écarts-types)")
    ordre = np.argsort(np.abs(theta_min[1:]))[::-1] + 1
    print(f"{'variable':<22}{'coefficient':>13}{'  (OLS)':>10}")
    coefs = []
    for rang, j in enumerate(ordre):
        coefs.append((noms[j], float(theta_min[j]), float(theta_ols[j])))
        if rang < 10:
            print(f"{noms[j]:<22}{theta_min[j]:>+13.3f}{theta_ols[j]:>+10.3f}")

    print("\nPrudence : les variables de sol sont corrélées entre elles ; ridge")
    print("répartit l'effet sur le groupe, donc le signe d'un coefficient isolé")
    print("n'est pas une relation causale. Seule sa stabilité sous rééchantillon-")
    print("nage (ou l'avis d'experts du sol) permettrait de conclure.")

    # ------------------------------------------------------------------
    if not args.sans_figures:
        tracer(noms, A, B, jeu, lambdas, cv_moy, cv_se, lam_min, lam_1se,
               theta_ols, theta_min, dossier=args.dossier_figures,
               lignes=lignes)

    banniere("7. Limites")
    print("- 138 sites : les intervalles de confiance resteraient larges même")
    print("  sans pénalité ; ridge ne crée pas d'information.")
    print("- Une seule année de données (year1_train) : pas de validation")
    print("  temporelle possible.")
    print("- L'échelle de lambda dépend de la standardisation : ne pas comparer")
    print("  ce lambda à celui d'un autre script sans vérifier la convention.")
    print("- Le jeu d'explicatives (ville incluse ou non, 20 ou 27 variables de")
    print("  sol) se tranche par validation croisée, jamais sur le jeu de test.")

    # ------------------------------------------------------------------
    if not args.sans_rapport:
        chemin = Path(args.rapport)
        chemin.write_text(
            rapport(jeu, args, lignes, coefs, lambdas, cv_moy, cv_se,
                    lam_min, lam_1se,
                    bord_grille=(i_1se >= len(lambdas) - 2), ctrl=ctrl),
            encoding="utf-8")
        print(f"\nRapport écrit : {chemin}")


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------

def etiquette_reponse(jeu) -> str:
    """Nom lisible de la variable expliquée, pour les titres et le rapport."""
    return {"logA": "log(1 + A)"}.get(jeu.reponse, jeu.reponse)


def tracer(noms, A, B, jeu, lambdas, cv_moy, cv_se, lam_min, lam_1se,
           theta_ols, theta_min, dossier: str | None = None,
           lignes=None) -> None:
    """Les cinq figures du script.

    dossier : si fourni, chaque figure y est aussi enregistrée en PNG (150 dpi).
    """
    y_train, y_test = jeu.y_train, jeu.y_test
    p = A.shape[1]
    ylab = etiquette_reponse(jeu)
    enregistrees = []

    def fin(fig, nom: str) -> None:
        fig.tight_layout()
        if dossier:
            chemin = Path(dossier) / nom
            chemin.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(chemin, dpi=150)
            enregistrees.append(str(chemin))
        fig.canvas.draw_idle()
        plt.show()

    # --- Figure 1 : convergence de la descente selon le pas ---------------
    theta_star = ridge_ferme(A, y_train, LAMBDA_DEMO)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for rel, style in [(0.2, ":"), (0.8, "--"), (1.5, "-"), (1.98, "-")]:
        _, h = descente_gradient(A, y_train, LAMBDA_DEMO,
                                 pas_relatif=rel, max_iter=800, tol=0)
        ecart = h - objectif(theta_star, A, y_train, LAMBDA_DEMO)
        axes[0].plot(np.arange(len(h)), np.maximum(ecart, 1e-16), style,
                     label=f"pas = {rel} / L")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("itérations")
    axes[0].set_ylabel("objectif - objectif optimal")
    axes[0].set_title("Vitesse de convergence (pas < 2 / L)")
    axes[0].legend(fontsize=8)

    _, h_div = descente_gradient(A, y_train, LAMBDA_DEMO, pas_relatif=2.1,
                                 max_iter=30, tol=0)
    axes[1].plot(np.arange(len(h_div)), h_div, "o-", color="firebrick", ms=3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("itérations")
    axes[1].set_ylabel("objectif")
    axes[1].set_title("Pas = 2.1 / L : la suite diverge")
    fin(fig, "fig1_convergence_descente.png")

    # --- Figure 2 : choix de lambda par validation croisée ----------------
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(np.log10(lambdas), cv_moy, color="C0")
    ax.fill_between(np.log10(lambdas), cv_moy - cv_se, cv_moy + cv_se,
                    color="C0", alpha=0.25)
    ax.axvline(np.log10(lam_min), color="black", ls=":",
               label=f"minimum : lambda = {lam_min:.3g}")
    ax.axvline(np.log10(lam_1se), color="firebrick", ls="--",
               label=f"règle 1-SE : lambda = {lam_1se:.3g}")
    ax.scatter([np.log10(lam_min)], [cv_moy.min()], color="black", zorder=5)
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("MSE de validation croisée")
    ax.set_title(f"Choix de lambda ({K_PLIS} plis, {len(y_train)} sites, "
                 "bande = erreur-type)")
    ax.legend(fontsize=9)
    fin(fig, "fig2_choix_lambda.png")

    # --- Figure 3 : chemin de régularisation ------------------------------
    chem = chemin_coefficients(A, y_train, lambdas)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for j in range(1, p):
        axes[0].plot(np.log10(lambdas), chem[:, j], color="0.75", lw=0.9)
    ordre = np.argsort(np.abs(theta_min[1:]))[::-1] + 1
    for j in ordre[:3]:
        axes[0].plot(np.log10(lambdas), chem[:, j], lw=2, label=noms[j])
    axes[0].axvline(np.log10(lam_min), color="black", ls=":", lw=1)
    axes[0].set_xlabel("log10(lambda)")
    axes[0].set_ylabel("coefficient (écart-type)")
    axes[0].set_title("Chemin des coefficients (gris : tous, couleur : 3 plus gros)")
    axes[0].legend(fontsize=8)

    axes[1].plot(np.log10(lambdas), np.linalg.norm(chem[:, 1:], axis=1))
    axes[1].axvline(np.log10(lam_min), color="black", ls=":", lw=1,
                    label=f"lambda retenu = {lam_min:.3g}")
    axes[1].set_xlabel("log10(lambda)")
    axes[1].set_ylabel("norme des coefficients")
    axes[1].set_title("Le shrinkage ramène les coefficients vers zéro")
    axes[1].legend(fontsize=8)
    fin(fig, "fig3_chemin_coefficients.png")

    # --- Figure 4 : prédictions et résidus -------------------------------
    pred_tr = A @ theta_min
    pred_te = B @ theta_min
    pred_ols = B @ theta_ols
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    for ax, obs, pred, titre in [
            (axes[0], y_train, pred_tr, f"apprentissage ({len(y_train)} sites)"),
            (axes[1], y_test, pred_te, f"test ({len(y_test)} sites)")]:
        bornes = [min(obs.min(), pred.min()) - 0.05,
                  max(obs.max(), pred.max()) + 0.05]
        ax.plot(bornes, bornes, color="black", ls="--", lw=1)
        ax.scatter(obs, pred, s=26, alpha=0.7, color="C1")
        m = metriques(obs, pred)
        ax.set_xlabel(f"{ylab} observé")
        ax.set_ylabel(f"{ylab} prédit (ridge)")
        ax.set_title(f"{titre}\nRMSE {m['RMSE']:.3f}  ·  R2 {m['R2']:+.3f}")
        ax.set_xlim(bornes)
        ax.set_ylim(bornes)

    axes[1].scatter(y_test, pred_ols, s=26, alpha=0.45, marker="x",
                    color="C0", label="OLS")
    axes[1].legend(fontsize=8)

    residus = y_test - pred_te
    axes[2].axhline(0, color="black", lw=1)
    axes[2].scatter(pred_te, residus, s=26, alpha=0.7, color="C1")
    axes[2].set_xlabel(f"{ylab} prédit (ridge, test)")
    axes[2].set_ylabel("résidu = observé - prédit")
    axes[2].set_title(f"Résidus en test\nmoyenne {residus.mean():+.3f}  ·  "
                      f"écart-type {residus.std(ddof=0):.3f}")
    fin(fig, "fig4_predictions_residus.png")

    # --- Figure 5 : effet du shrinkage -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    y_pos = np.arange(len(ordre))
    axes[0].barh(y_pos - 0.2, theta_ols[ordre], height=0.38, label="OLS")
    axes[0].barh(y_pos + 0.2, theta_min[ordre], height=0.38,
                 label=f"ridge (lambda = {lam_min:.3g})")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([noms[j] for j in ordre], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("coefficient (écart-type)")
    axes[0].set_title("Shrinkage : OLS vs ridge")
    axes[0].legend(fontsize=8)

    noms_modeles = [r["modèle"] for r in lignes]
    x = np.arange(len(noms_modeles))
    axes[1].bar(x - 0.2, [r["RMSE_train"] for r in lignes], width=0.38,
                label="RMSE apprentissage")
    axes[1].bar(x + 0.2, [r["RMSE_test"] for r in lignes], width=0.38,
                label="RMSE test")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["OLS", "ridge min CV", "ridge 1-SE"], fontsize=9)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Erreur : apprentissage vs test")
    axes[1].legend(fontsize=8)
    fin(fig, "fig5_shrinkage_erreurs.png")

    if enregistrees:
        print("\nFigures enregistrées :")
        for chemin in enregistrees:
            print(f"  {chemin}")


# ---------------------------------------------------------------------------
# 6 bis. Toutes les réponses : un résultat détaillé + un graphique chacune
#        (--toutes-reponses : A, H, J, S)
# ---------------------------------------------------------------------------

REPONSES_DEFAUT = ["A", "H", "J", "S"]
LIBELLES = {"A": "A (abondance)", "H": "H (Shannon)", "J": "J (Pielou)",
            "S": "S (richesse)", "logA": "log(1 + A)"}


def _jeu_depuis_H(reponse: str, args):
    """Jeu pour A, S ou J quand en_commun.preparer() ne les gère pas.

    On reprend exactement le jeu de H (mêmes sites, même découpage, même
    standardisation, donc même empreinte) et on remplace seulement la
    réponse par la colonne de data_site.csv. L'alignement des lignes est
    vérifié en comparant la colonne H du csv avec la réponse H du jeu.
    """
    import pandas as pd
    from types import SimpleNamespace

    base = preparer(reponse="H", sol=choisir_sol(args),
                    avec_ville=not args.sans_ville, graine=args.graine)
    candidats = [Path(args.csv)] if args.csv else [
        Path(__file__).parent / "data_site.csv", Path("data_site.csv")]
    chemin = next((c for c in candidats if c.exists()), None)
    if chemin is None:
        raise SystemExit("data_site.csv introuvable : utiliser --csv chemin.")
    d = pd.read_csv(chemin, sep=";", decimal=",")
    if len(d) != len(base.y) or not np.allclose(d["H"].to_numpy(float),
                                                np.asarray(base.y, float)):
        raise SystemExit(
            f"Impossible d'aligner '{reponse}' sur le jeu de H (lignes "
            "différentes) : ajouter cette réponse dans en_commun.preparer().")
    y = d[reponse].to_numpy(float)
    attrs = {k: getattr(base, k) for k in dir(base)
             if not k.startswith("_") and not callable(getattr(base, k))}
    attrs.update(reponse=reponse, y=y,
                 y_train=y[np.asarray(base.idx_train)],
                 y_test=y[np.asarray(base.idx_test)])
    return SimpleNamespace(**attrs)


def construire_jeu_reponse(reponse: str, args):
    """Jeu commun si en_commun gère la réponse, sinon repli sur celui de H."""
    ns = argparse.Namespace(**{**vars(args), "reponse": reponse})
    try:
        return construire_jeu(ns)
    except Exception as exc:
        if reponse in ("A", "S", "J"):
            return _jeu_depuis_H(reponse, args)
        raise SystemExit(f"en_commun.preparer() refuse '{reponse}' : {exc}")


def analyser_reponse(reponse: str, args) -> dict:
    """Tout le calcul pour une réponse : lambda, 3 modèles, métriques, coefs."""
    jeu = construire_jeu_reponse(reponse, args)
    A, B = design(jeu.X_train), design(jeu.X_test)
    y_tr, y_te = jeu.y_train, jeu.y_test
    noms = ["constante"] + list(jeu.X.columns)
    n, p = A.shape

    lambdas = np.logspace(-4, 2, 61)
    cv_moy, cv_se, _ = validation_croisee(
        jeu.X.iloc[jeu.idx_train], y_tr, lambdas, k=K_PLIS, graine=args.graine)
    i_min = int(np.argmin(cv_moy))
    i_1se = int(np.max(np.where(cv_moy <= cv_moy[i_min] + cv_se[i_min])[0]))
    lam_min, lam_1se = float(lambdas[i_min]), float(lambdas[i_1se])

    modeles = [("OLS (lambda = 0)", 0.0),
               (f"ridge lambda = {lam_min:.3g}", lam_min),
               (f"ridge lambda = {lam_1se:.3g}", lam_1se)]
    thetas, pred, lignes = {}, {}, []
    for nom, lam in modeles:
        th = ridge_ferme(A, y_tr, lam)
        thetas[nom], pred[nom] = th, (A @ th, B @ th)
        m_tr, m_te = metriques(y_tr, A @ th), metriques(y_te, B @ th)
        lignes.append({"modèle": nom, "lambda": lam,
                       "ddl": ddl_effectifs(A, lam),
                       "RMSE_train": m_tr["RMSE"], "RMSE_test": m_te["RMSE"],
                       "MAE_train": m_tr["MAE"], "MAE_test": m_te["MAE"],
                       "R2_train": m_tr["R2"], "R2_test": m_te["R2"]})
    th_ols, th_min = thetas[modeles[0][0]], thetas[modeles[1][0]]
    ordre = np.argsort(np.abs(th_min[1:]))[::-1] + 1
    coefs = [(noms[j], float(th_min[j]), float(th_ols[j])) for j in ordre]

    p_te = pred[modeles[1][0]][1]
    r_te = float(np.corrcoef(y_te, p_te)[0, 1]) if np.std(p_te) > 0 else float("nan")
    return dict(reponse=reponse, jeu=jeu, label=LIBELLES.get(reponse, reponse),
                n=n, p=p, noms=noms, lambdas=lambdas, cv_moy=cv_moy, cv_se=cv_se,
                i_min=i_min, i_1se=i_1se, lam_min=lam_min, lam_1se=lam_1se,
                modeles=modeles, thetas=thetas, pred=pred, lignes=lignes,
                coefs=coefs, ordre=ordre, y_tr=y_tr, y_te=y_te,
                r2_cv=1.0 - cv_moy[i_min] / float(np.var(y_tr)),
                r_test=r_te)


def afficher_detail(r: dict) -> None:
    """Résultats détaillés d'une réponse, dans la console."""
    jeu = r["jeu"]
    banniere(f"RÉPONSE {r['label']}")
    print(f"Sites utilisés       : {len(jeu.y)}"
          + (f" ({jeu.n_exclus} écartés : {jeu.reponse} non défini)"
             if getattr(jeu, "n_exclus", 0) else ""))
    print(f"Découpage            : {len(jeu.idx_train)} apprentissage / "
          f"{len(jeu.idx_test)} test (graine {jeu.graine}, "
          f"empreinte {jeu.empreinte})")
    print(f"Explicatives         : {r['p'] - 1} colonnes (sol + usage"
          f"{' + ville' if jeu.avec_ville else ''})")
    print(f"Variable à expliquer : moyenne {np.mean(jeu.y):.3f}, "
          f"écart-type {np.std(jeu.y):.3f}, min {np.min(jeu.y):.3f}, "
          f"max {np.max(jeu.y):.3f}")

    print("\n-- Choix de lambda (validation croisée) --")
    print(f"lambda minimum CV : {r['lam_min']:.4g}  "
          f"(MSE CV = {r['cv_moy'][r['i_min']]:.4f} "
          f"± {r['cv_se'][r['i_min']]:.4f})")
    print(f"lambda règle 1-SE : {r['lam_1se']:.4g}  "
          f"(MSE CV = {r['cv_moy'][r['i_1se']]:.4f})")
    print(f"R2 de validation croisée (approx. 1 - MSE_CV / var(y)) : "
          f"{r['r2_cv']:+.3f}")

    print("\n-- Performances --")
    print(f"{'modèle':<24}{'ddl':>6}{'R2 app.':>9}{'R2 test':>9}"
          f"{'RMSE app.':>11}{'RMSE test':>11}{'MAE app.':>10}{'MAE test':>10}")
    for l in r["lignes"]:
        print(f"{l['modèle']:<24}{l['ddl']:>6.1f}{l['R2_train']:>9.3f}"
              f"{l['R2_test']:>9.3f}{l['RMSE_train']:>11.4f}"
              f"{l['RMSE_test']:>11.4f}{l['MAE_train']:>10.4f}"
              f"{l['MAE_test']:>10.4f}")
    nom_min = r["modeles"][1][0]
    res_tr = r["y_tr"] - r["pred"][nom_min][0]
    res_te = r["y_te"] - r["pred"][nom_min][1]
    print(f"\nCorrélation observé/prédit en test (ridge) : {r['r_test']:+.3f}")
    print(f"Résidus ridge : train moyenne {res_tr.mean():+.3f} / écart-type "
          f"{res_tr.std():.3f} ; test moyenne {res_te.mean():+.3f} / "
          f"écart-type {res_te.std():.3f}")

    print(f"\n-- Coefficients (lambda = {r['lam_min']:.3g}, écarts-types) --")
    print(f"{'variable':<22}{'ridge':>10}{'OLS':>10}")
    for nom, b_r, b_o in r["coefs"]:
        print(f"{nom:<22}{b_r:>+10.3f}{b_o:>+10.3f}")
    pos = [c for c in r["coefs"] if c[1] > 0][:3]
    neg = [c for c in r["coefs"] if c[1] < 0][:3]
    print("Plus fortes associations positives : "
          + (", ".join(f"{c[0]} ({c[1]:+.3f})" for c in pos) or "aucune"))
    print("Plus fortes associations négatives : "
          + (", ".join(f"{c[0]} ({c[1]:+.3f})" for c in neg) or "aucune"))


def _obs_vs_pred(ax, obs, pred, label, titre) -> None:
    b = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    marge = 0.05 * ((b[1] - b[0]) or 1.0)
    b = [b[0] - marge, b[1] + marge]
    ax.plot(b, b, "k--", lw=1)
    ax.scatter(obs, pred, s=26, alpha=0.7, color="C1")
    m = metriques(obs, pred)
    ax.set_xlim(b)
    ax.set_ylim(b)
    ax.set_xlabel(f"{label} observé")
    ax.set_ylabel(f"{label} prédit")
    ax.set_title(f"{titre}\nR² {m['R2']:+.3f} · RMSE {m['RMSE']:.3f}", fontsize=10)


def _sauver(fig, dossier, nom) -> None:
    fig.tight_layout()
    if dossier:
        chemin = Path(dossier) / nom
        chemin.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chemin, dpi=150)
        print(f"  figure enregistrée : {chemin}")


def figure_detail(r: dict, dossier) -> None:
    """Un tableau de bord par réponse (6 panneaux)."""
    lab = r["label"]
    nom_ols, nom_min, nom_1se = [m[0] for m in r["modeles"]]
    p_tr, p_te = r["pred"][nom_min]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))

    _obs_vs_pred(axes[0, 0], r["y_tr"], p_tr, lab,
                 f"Apprentissage ({len(r['y_tr'])} sites)")
    _obs_vs_pred(axes[0, 1], r["y_te"], p_te, lab,
                 f"Test ({len(r['y_te'])} sites)")
    axes[0, 1].scatter(r["y_te"], r["pred"][nom_ols][1], s=26, alpha=0.45,
                       marker="x", color="C0", label="OLS")
    axes[0, 1].legend(fontsize=8)

    res = r["y_te"] - p_te
    axes[0, 2].axhline(0, color="black", lw=1)
    axes[0, 2].scatter(p_te, res, s=26, alpha=0.7, color="C1")
    axes[0, 2].set_xlabel(f"{lab} prédit (test)")
    axes[0, 2].set_ylabel("résidu = observé - prédit")
    axes[0, 2].set_title(f"Résidus en test\nmoyenne {res.mean():+.3f} · "
                         f"écart-type {res.std():.3f}", fontsize=10)

    lg = np.log10(r["lambdas"])
    ax = axes[1, 0]
    ax.plot(lg, r["cv_moy"], color="C0")
    ax.fill_between(lg, r["cv_moy"] - r["cv_se"], r["cv_moy"] + r["cv_se"],
                    color="C0", alpha=0.25)
    ax.axvline(np.log10(r["lam_min"]), color="black", ls=":",
               label=f"min : {r['lam_min']:.3g}")
    ax.axvline(np.log10(r["lam_1se"]), color="firebrick", ls="--",
               label=f"1-SE : {r['lam_1se']:.3g}")
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("MSE de validation croisée")
    ax.set_title(f"Choix de lambda (R² CV ≈ {r['r2_cv']:+.3f})", fontsize=10)
    ax.legend(fontsize=8)

    top = r["ordre"][:12]
    y_pos = np.arange(len(top))
    ax = axes[1, 1]
    ax.barh(y_pos, r["thetas"][nom_min][top], height=0.6, color="C1")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["noms"][j] for j in top], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("coefficient (écart-type)")
    ax.set_title("12 plus gros coefficients ridge\n(OLS : voir tableau, "
                 "hors échelle)", fontsize=10)

    ax = axes[1, 2]
    courts = ["OLS", "ridge min CV", "ridge 1-SE"]
    x = np.arange(3)
    ax.bar(x - 0.2, [l["R2_train"] for l in r["lignes"]], 0.38,
           label="R² apprentissage")
    b = ax.bar(x + 0.2, [l["R2_test"] for l in r["lignes"]], 0.38,
               label="R² test")
    for rect, l in zip(b, r["lignes"]):
        v = l["R2_test"]
        ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(courts, fontsize=9)
    ax.set_title("R² par modèle", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(f"Ridge — {lab} (lambda retenu = {r['lam_min']:.3g})",
                 fontsize=13, fontweight="bold")
    _sauver(fig, dossier, f"ridge_{r['reponse']}.png")


def tableau_synthese(resultats: list) -> None:
    banniere("SYNTHÈSE : ridge (lambda minimum CV), jeu de test")
    print(f"{'réponse':<18}{'sites':>7}{'lambda':>9}{'R2 CV':>8}{'R2 app.':>9}"
          f"{'R2 test':>9}{'RMSE test':>11}{'MAE test':>10}{'R2 OLS test':>13}")
    for r in resultats:
        o, m = r["lignes"][0], r["lignes"][1]
        print(f"{r['label']:<18}{len(r['jeu'].y):>7}{r['lam_min']:>9.3g}"
              f"{r['r2_cv']:>8.3f}{m['R2_train']:>9.3f}{m['R2_test']:>9.3f}"
              f"{m['RMSE_test']:>11.4f}{m['MAE_test']:>10.4f}"
              f"{o['R2_test']:>13.3f}")
    print("\nR2 test < 0 : le modèle fait pire que prédire la moyenne du test.")
    print("Les RMSE/MAE ne se comparent pas d'une réponse à l'autre "
          "(échelles différentes) ; les R2, si.")
    classement = sorted(resultats,
                        key=lambda r: r["lignes"][1]["R2_test"], reverse=True)
    meilleure = classement[0]
    score = meilleure["lignes"][1]["R2_test"]
    print("\nClassement des réponses selon le R² test (du meilleur au moins bon) :")
    for rang, r in enumerate(classement, 1):
        print(f"  {rang}. {r['label']} : R² test = {r['lignes'][1]['R2_test']:+.3f}")
    if score < 0:
        print(f"\nLa moins mauvaise réponse est {meilleure['label']}, mais son R² "
              f"reste négatif ({score:+.3f}) : aucune des réponses n'est "
              "correctement prédite sur le test.")
    else:
        print(f"\nLa réponse la mieux prédite est {meilleure['label']} "
              f"(R² test = {score:+.3f}).")


def figure_synthese(resultats: list, dossier) -> None:
    noms = [r["reponse"] for r in resultats]
    x = np.arange(len(noms))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    tr = [r["lignes"][1]["R2_train"] for r in resultats]
    te = [r["lignes"][1]["R2_test"] for r in resultats]
    cv = [r["r2_cv"] for r in resultats]
    axes[0].bar(x - 0.27, tr, 0.26, label="apprentissage")
    axes[0].bar(x, cv, 0.26, label="validation croisée")
    axes[0].bar(x + 0.27, te, 0.26, label="test")
    axes[0].set_title("R² du ridge")
    ols = [r["lignes"][0]["R2_test"] for r in resultats]
    axes[1].bar(x - 0.2, ols, 0.38, label="OLS")
    axes[1].bar(x + 0.2, te, 0.38, label="ridge")
    axes[1].set_title("R² test : OLS vs ridge")
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(noms)
        ax.legend(fontsize=8)
    _sauver(fig, dossier, "ridge_synthese.png")

    # Nuage de points comparatif : chaque réponse est standardisée avec
    # la moyenne et l'écart-type de SON jeu d'apprentissage. Cela permet
    # de comparer visuellement A, H, J et S malgré leurs échelles différentes.
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8.5))
    axes2 = axes2.ravel()
    for ax, r in zip(axes2, resultats):
        nom_min = r["modeles"][1][0]
        obs = np.asarray(r["y_te"], dtype=float)
        pred = np.asarray(r["pred"][nom_min][1], dtype=float)
        mu = float(np.mean(r["y_tr"]))
        sd = float(np.std(r["y_tr"], ddof=0))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        obs_z = (obs - mu) / sd
        pred_z = (pred - mu) / sd
        borne = np.concatenate([obs_z, pred_z])
        lo, hi = float(np.nanmin(borne)), float(np.nanmax(borne))
        marge = 0.06 * ((hi - lo) or 1.0)
        lo, hi = lo - marge, hi + marge
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="prédiction parfaite")
        ax.scatter(obs_z, pred_z, s=38, alpha=0.75, label="sites test")
        m = r["lignes"][1]
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Valeur observée (centrée-réduite sur apprentissage)")
        ax.set_ylabel("Valeur prédite (même échelle)")
        ax.set_title(f"{r['label']} — R² test = {m['R2_test']:+.3f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes2[len(resultats):]:
        ax.set_visible(False)
    fig2.suptitle("Nuages de points — observations vs prédictions Ridge sur le test\n"
                  "Variables standardisées pour rendre les échelles comparables",
                  fontsize=12, fontweight="bold")
    _sauver(fig2, dossier, "ridge_nuages_points_test.png")


def toutes_reponses(args) -> None:
    """--toutes-reponses : résultats détaillés + figure pour chaque réponse."""
    resultats = []
    print("Variables demandées :", ", ".join(args.reponses))
    print("Le script va ajuster un modèle distinct pour chaque variable.")
    for rep in args.reponses:
        r = analyser_reponse(rep, args)
        afficher_detail(r)
        resultats.append(r)
    tableau_synthese(resultats)

    if not args.sans_figures:
        for r in resultats:
            figure_detail(r, args.dossier_figures)
        if len(resultats) > 1:
            figure_synthese(resultats, args.dossier_figures)
        plt.show()

    if not args.sans_rapport:
        chemin = Path(args.rapport_toutes)
        parties = []
        for r in resultats:
            ns = argparse.Namespace(**{**vars(args), "reponse": r["reponse"]})
            parties.append(rapport(
                r["jeu"], ns, r["lignes"], r["coefs"], r["lambdas"],
                r["cv_moy"], r["cv_se"], r["lam_min"], r["lam_1se"],
                bord_grille=(r["i_1se"] >= len(r["lambdas"]) - 2), ctrl={}))
        chemin.write_text(("\n\n" + "#" * 78 + "\n\n").join(parties),
                          encoding="utf-8")
        print(f"\nRapport détaillé écrit : {chemin}")


# ---------------------------------------------------------------------------
# 7. Rapport texte
# ---------------------------------------------------------------------------

def rapport(jeu, args, lignes, coefs, lambdas, cv_moy, cv_se,
            lam_min, lam_1se, bord_grille: bool, ctrl: dict) -> str:
    """Le contenu du fichier de résultats (texte brut, UTF-8)."""
    from datetime import datetime

    ylab = etiquette_reponse(jeu)
    L: list[str] = []
    L.append("Régression ridge — résultats")
    L.append("=" * 78)
    L.append(f"Date                   : {datetime.now():%Y-%m-%d %H:%M}")
    L.append("Script                 : regression_ridge.py "
             "(protocole : en_commun.py)")
    L.append(f"Réponse                : {jeu.reponse} ({ylab})")
    L.append("Source                 : data_site.csv (une ligne par site)")
    detail = ""
    if jeu.n_exclus:
        detail = f" ({jeu.n_exclus} sites écartés : réponse non définie)"
    L.append(f"Sites utilisés         : {len(jeu.y)}{detail}")
    L.append(f"Explicatives           : {len(jeu.sol)} variable(s) de sol")
    L.append(f"                         {', '.join(jeu.sol)}")
    facteurs = "land_use et city" if jeu.avec_ville else "land_use"
    L.append(f"                         + indicatrices de {facteurs} "
             "(références FU et MO)")
    L.append(f"Découpage              : {len(jeu.idx_train)} sites en "
             f"apprentissage / {len(jeu.idx_test)} en test")
    L.append(f"                         graine {jeu.graine}, stratifié par "
             "land_use, sans doublon")
    L.append(f"Empreinte du découpage : {jeu.empreinte}")
    L.append("Convention             : J(b) = 1/(2n)||y-Ab||^2 + "
             "(lambda/2)||b||^2 (hors constante)")
    L.append(f"                         alpha(scikit-learn) = n*lambda = "
             f"{len(jeu.idx_train)}*lambda")
    L.append("")

    L.append("1. Performances sur le jeu de test (sites jamais vus)")
    L.append("-" * 78)
    L.append(f"{'modèle':<26}{'ddl':>6}{'RMSE app.':>11}{'RMSE test':>11}"
             f"{'MAE test':>10}{'R2 app.':>9}{'R2 test':>9}")
    for r in lignes:
        L.append(f"{r['modèle']:<26}{r['ddl']:>6.1f}{r['RMSE_train']:>11.4f}"
                 f"{r['RMSE_test']:>11.4f}{r['MAE_test']:>10.4f}"
                 f"{r['R2_train']:>9.3f}{r['R2_test']:>9.3f}")
    L.append("")
    L.append("Lecture : 'ddl' = degrés de liberté effectifs "
             "trace(A(A'A + n*lambda*P)^-1 A').")
    L.append("Un R2 négatif en test signifie que le modèle fait moins bien que "
             "la moyenne du jeu de test.")
    L.append("")

    L.append(f"2. Choix de lambda ({K_PLIS} plis, {len(jeu.idx_train)} sites, "
             f"{len(lambdas)} valeurs de 1e-4 à 1e+2)")
    L.append("-" * 78)
    L.append(f"lambda minimal (CV)    : {lam_min:.6g}   MSE CV = "
             f"{cv_moy.min():.6f}  (erreur-type {cv_se[np.argmin(cv_moy)]:.6f})")
    L.append(f"lambda règle 1-SE      : {lam_1se:.6g}   MSE CV = "
             f"{cv_moy[int(np.argmin(np.abs(lambdas - lam_1se)))]:.6f}")
    if bord_grille:
        L.append("Remarque               : la règle 1-SE retient le bord de la "
                 "grille ;")
        L.append("                         la courbe est plate à droite, le "
                 "modèle est quasi constant.")
    L.append("")

    L.append(f"3. Coefficients au lambda retenu ({lam_min:.3g}), en écarts-types")
    L.append("-" * 78)
    L.append(f"{'variable':<26}{'ridge':>10}{'OLS':>10}")
    for nom, b_ridge, b_ols in coefs:
        L.append(f"{nom:<26}{b_ridge:>+10.3f}{b_ols:>+10.3f}")
    L.append("")

    L.append("4. Contrôles numériques")
    L.append("-" * 78)
    for nom, valeur in ctrl.items():
        if nom == "iterations":
            L.append(f"{nom:<24}: {valeur}")
        else:
            L.append(f"{nom:<24}: écart max {valeur:.2e}")
    L.append("")

    L.append("5. Réserves")
    L.append("-" * 78)
    if args.trois_variables:
        L.append("Jeu de sol réduit à 3 variables : ces performances ne sont PAS")
        L.append("comparables à celles des modèles à 20 variables du groupe. Seuls")
        L.append(f"le découpage (empreinte {jeu.empreinte}) et les métriques le sont.")
    L.append("Les trois variables retenues sont corrélées entre elles : ridge")
    L.append("répartit l'effet sur le groupe, le signe d'un coefficient isolé n'est")
    L.append("pas une relation causale.")
    L.append("Une seule année de données : pas de validation temporelle possible.")
    L.append("L'échelle de lambda dépend de la standardisation ; la convention est")
    L.append("rappelée ci-dessus pour toute comparaison entre scripts.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Régression ridge : comparaison A, H, J et S.")
    parser.add_argument("--reponse", choices=["A", "H", "S", "J", "logA"], default=None,
                        help="analyser une seule variable ; sans cette option, A, H, J et S sont toutes analysées")
    parser.add_argument("--toutes-reponses", action="store_true",
                        help="résultats détaillés + graphiques pour A, H, J et S"
                             "réponse (A, H, J, S)")
    parser.add_argument("--reponses", nargs="+", default=REPONSES_DEFAUT,
                        choices=["A", "H", "J", "S", "logA"],
                        help="réponses du mode --toutes-reponses")
    parser.add_argument("--csv", default=None,
                        help="chemin de data_site.csv (repli pour A et S)")
    parser.add_argument("--rapport-toutes",
                        default=str(Path(__file__).parent /
                                    "resultats_ridge_toutes_reponses.txt"),
                        help="fichier texte du mode --toutes-reponses")
    parser.add_argument("--trois-variables", action="store_true",
                        help="limiter le sol à CEC, pH.eau, Carbonates")
    parser.add_argument("--tous-les-sols", action="store_true",
                        help="garder les 27 variables de sol au lieu des 20")
    parser.add_argument("--sans-ville", action="store_true",
                        help="retirer les indicatrices de ville")
    parser.add_argument("--graine", type=int, default=GRAINE,
                        help="graine du découpage et de la validation croisée")
    parser.add_argument("--rapport", default=str(Path(__file__).parent /
                                                 "resultats_ridge.txt"),
                        help="fichier texte de résultats")
    parser.add_argument("--sans-rapport", action="store_true",
                        help="ne pas écrire le fichier de résultats")
    parser.add_argument("--dossier-figures", default=None,
                        help="enregistrer aussi les figures en PNG dans ce dossier")
    parser.add_argument("--sans-figures", action="store_true",
                        help="ne pas afficher les figures")
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    main()