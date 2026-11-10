"""
Protocole commun — Journée Data Science n°2 (collemboles).

Chaque membre du groupe travaille sur une régression différente (OLS, ridge,
lasso, meilleur sous-ensemble, stepwise...). Pour que les résultats soient
comparables, tout le monde doit partir du même jeu, du même découpage, de la
même standardisation et des mêmes métriques.

Ce module ne contient **aucune méthode de régression et ne produit aucun
résultat** : il fournit uniquement les briques qui rendent les régressions
comparables. Les résultats sont produits par le script de chacun.

    from en_commun import preparer, evaluer, evaluation_croisee, comparer

    jeu = preparer()                      # H, 20 variables de sol + facteurs
    resultat = evaluer(mon_ajusteur, jeu, nom="ridge", n_parametres=8)
    print(resultat)

Protocole figé (ne pas le modifier sans prévenir le groupe) :

  - source : `Données/Data/nettoye/data_site.csv` uniquement, une ligne par site
    (138 sites ; les sites sans valeur de la réponse sont écartés, ce qui est
    signalé) ;
  - réponse : `H` par défaut ; `J` (équitabilité, colonne `J` du fichier) ;
    `logA` = log(1 + A), pour stabiliser une variable très asymétrique.
    `J` est toujours lue telle quelle, jamais recalculée : H moyen /
    log(S moyen) n'est pas la moyenne des J des réplicats (corrélation 0,78,
    écarts jusqu'à 0,58) ;
  - explicatives : 20 variables de sol après retrait des colinéaires, plus les
    indicatrices de `land_use` et `city` (modalités de référence : FU et MO) ;
  - standardisation : moyenne et écart-type du train seulement (ddof = 0) ;
  - découpage : 70 / 30, stratifié par `land_use`, graine 42 ;
  - validation croisée : 5 plis mélangés, graine 42 — les mêmes plis pour tous ;
  - métriques : RMSE, MAE, R2, sur train et sur test.

Contrat d'un modèle (un « ajusteur ») :

    ajusteur(X_train, y_train, X_nouveau) -> y_pred

  X ne contient pas la constante : chaque ajusteur l'ajoute lui-même.
  Le choix des hyperparamètres (lambda, alpha, nombre de variables) doit être
  fait à l'intérieur de l'ajusteur, avec les données qu'il reçoit et par
  validation croisée — jamais en regardant le jeu de test.

Tout est déterministe : deux machines qui appellent `preparer` avec les mêmes
arguments obtiennent la même empreinte (`Jeu.empreinte`). Cette empreinte
identifie le jeu de test : si elle diffère, les chiffres ne sont pas
comparables. Elle change avec la réponse (H et J n'ont pas les mêmes sites).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

__all__ = [
    "GRAINE", "PART_TEST", "K_PLIS", "SOL_REDUIT", "SOL_TOUS",
    "Jeu", "charger_sites", "extraire_reponse", "construire_design",
    "preparer", "plier", "standardiser_pli", "metriques",
    "evaluer", "evaluation_croisee", "comparer", "empreinte",
]

# ---------------------------------------------------------------------------
# Constantes du protocole
# ---------------------------------------------------------------------------

GRAINE = 42
PART_TEST = 0.3
K_PLIS = 5

DOSSIER = Path(__file__).resolve().parent

# Table par site produite par `Données/prepare_site.R`.
DATA_SITE = DOSSIER / "Données" / "Data" / "nettoye" / "data_site.csv"

# 20 variables de sol conservées après le retrait séquentiel des colinéaires
# (notebook d'exploration, section 7.3).
SOL_REDUIT = [
    "Argile", "Limon", "Conduct", "pH.eau", "Calcaire", "Azote", "C.N",
    "P2O5", "K2O", "MgO", "CaO", "Na2O", "Al", "Cd", "Cu", "Fe", "Mn",
    "Ni", "Pb", "Zn",
]

# Jeu complet des 27 variables de sol (utile pour les méthodes pénalisées).
SOL_TOUS = [
    "Argile", "Limon", "Sable", "Conduct", "pH.eau", "Carbonates", "Calcaire",
    "C..orga", "Mat_org", "Azote", "C.N", "P2O5", "CEC", "K2O", "MgO", "CaO",
    "Na2O", "Al", "Cd", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Pb", "Zn",
]


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

def charger_sites(chemin: Path | None = None) -> pd.DataFrame:
    """Charge `data_site.csv` et vérifie sa structure.

    Le fichier contient une ligne par site : sol pris une fois par site et
    indices moyennés sur les réplicats. Il est trié par `Site` pour que le
    découpage ne dépende pas de l'ordre du fichier.
    """
    chemin = DATA_SITE if chemin is None else chemin
    if not chemin.exists():
        raise FileNotFoundError(
            f"{chemin} est absent. Le module attend le tableau par site "
            "du projet (138 sites, une ligne par site), produit par "
            "Données/prepare_site.R.")
    sites = (pd.read_csv(chemin, sep=";", decimal=",", encoding="utf-8-sig")
             .sort_values("Site")
             .reset_index(drop=True))
    if sites["Site"].duplicated().any():
        raise ValueError("data_site.csv contient des sites en double.")
    return sites


def extraire_reponse(sites: pd.DataFrame, nom: str = "H") -> np.ndarray:
    """Renvoie la variable à expliquer sous forme de vecteur.

    `H`   : diversité de Shannon (colonne `H`) ;
    `J`   : équitabilité (colonne `J`) ;
    `logA`: log(1 + abondance), pour stabiliser une variable très asymétrique.

    J n'est jamais recalculée à partir de H et S : H moyen / log(S moyen)
    n'est pas la moyenne des J des réplicats (r = 0,78, écarts jusqu'à 0,58).
    Elle est donc indéfinie (NaN) quand S <= 1 : ces sites sont écartés par
    `preparer`, qui le signale dans `Jeu.notes`.
    """
    nom = nom.strip()
    if nom == "logA":
        return np.log1p(sites["A"].to_numpy(float))
    if nom in ("H", "J"):
        if nom not in sites.columns:
            raise KeyError(
                f"data_site.csv n'a pas de colonne {nom!r} ; ce module ne la "
                "recalcule pas. Vérifier le tableau par site produit par le groupe.")
        return sites[nom].to_numpy(float)
    raise ValueError(f"réponse inconnue : {nom!r} (attendu : H, J ou logA)")


def construire_design(sites: pd.DataFrame, sol: list[str] | None = None,
                      avec_ville: bool = True) -> pd.DataFrame:
    """Construit la matrice des explicatives (sans constante).

    Les variables de sol sont brutes (la standardisation vient après le
    découpage) ; les facteurs deviennent des indicatrices 0/1 avec `FU` et
    `MO` comme modalités de référence.
    """
    sol = SOL_REDUIT if sol is None else list(sol)
    manquantes = [c for c in sol if c not in sites.columns]
    if manquantes:
        raise KeyError(f"colonnes absentes de data_site.csv : {manquantes}")

    facteurs = ["land_use"] + ([] if not avec_ville else ["city"])
    indicatrices = pd.get_dummies(sites[facteurs], drop_first=True, dtype=float)
    return pd.concat([sites[sol].astype(float), indicatrices], axis=1)


def empreinte(noms_sites) -> str:
    """Empreinte courte du découpage, pour vérifier que tout le monde a le même."""
    texte = "|".join(sorted(str(s) for s in noms_sites))
    return hashlib.sha1(texte.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Découpage et standardisation
# ---------------------------------------------------------------------------

@dataclass
class Jeu:
    """Tout ce qu'un modèle doit recevoir, et rien de plus."""
    reponse: str
    sol: list[str]
    avec_ville: bool
    graine: int
    sites: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    idx_train: np.ndarray
    idx_test: np.ndarray
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    mu: pd.Series
    sd: pd.Series
    n_exclus: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def site_test(self) -> list[str]:
        return self.sites["Site"].to_numpy()[self.idx_test].tolist()

    @property
    def empreinte(self) -> str:
        return empreinte(self.site_test)

    def resume(self) -> str:
        return (
            f"réponse {self.reponse} ; {len(self.y)} sites "
            f"({self.n_exclus} exclus pour valeur manquante)\n"
            f"{self.X.shape[1]} explicatives : {len(self.sol)} de sol"
            f" + {'usage et ville' if self.avec_ville else 'usage seul'}\n"
            f"train {len(self.idx_train)} sites / test {len(self.idx_test)} sites"
            f" (graine {self.graine})\n"
            f"empreinte du découpage : {self.empreinte}"
        )


def preparer(reponse: str = "H", sol: list[str] | None = None,
             avec_ville: bool = True, graine: int = GRAINE,
             part_test: float = PART_TEST,
             chemin: Path | None = None) -> Jeu:
    """Charge, découpe et standardise en une seule fois.

    C'est la fonction que tout le monde doit appeler. Elle garantit le même
    jeu de test pour tous, à graine et à réponse identiques.
    """
    sol = SOL_REDUIT if sol is None else list(sol)
    sites = charger_sites(chemin)
    y = extraire_reponse(sites, reponse)

    masque = np.isfinite(y)
    n_exclus = int((~masque).sum())
    if n_exclus:
        sites = sites[masque].reset_index(drop=True)
        y = y[masque]

    X = construire_design(sites, sol=sol, avec_ville=avec_ville)

    positions = np.arange(len(sites))
    idx_train, idx_test = train_test_split(
        positions, test_size=part_test, random_state=graine,
        stratify=sites["land_use"])

    X_train_brut, X_test_brut = X.iloc[idx_train], X.iloc[idx_test]
    mu, sd = standardiser_pli(X_train_brut)
    X_train = (X_train_brut - mu) / sd
    X_test = (X_test_brut - mu) / sd

    notes = []
    if n_exclus:
        notes.append(f"{n_exclus} sites écartés : valeur de {reponse} non définie.")

    return Jeu(
        reponse=reponse, sol=sol, avec_ville=avec_ville, graine=graine,
        sites=sites, X=X, y=y,
        idx_train=np.asarray(idx_train), idx_test=np.asarray(idx_test),
        X_train=X_train, X_test=X_test,
        y_train=y[idx_train], y_test=y[idx_test],
        mu=mu, sd=sd, n_exclus=n_exclus, notes=notes,
    )


def standardiser_pli(X_app: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Moyennes et écarts-types (ddof = 0) d'un pli d'apprentissage."""
    mu = X_app.mean()
    sd = X_app.std(ddof=0)
    sd = sd.where(sd > 0, 1.0)
    return mu, sd


def plier(n: int, k: int = K_PLIS, graine: int = GRAINE):
    """Les k plis communs : mêmes indices pour tous, pour un même n."""
    decoupe = KFold(n_splits=k, shuffle=True, random_state=graine)
    return list(decoupe.split(np.arange(n)))


# ---------------------------------------------------------------------------
# Métriques et évaluation
# ---------------------------------------------------------------------------

def metriques(y_true, y_pred) -> dict:
    """RMSE, MAE et R2 (les mêmes définitions pour tout le groupe)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    r = y_true - y_pred
    return {
        "RMSE": float(np.sqrt(np.mean(r ** 2))),
        "MAE": float(np.mean(np.abs(r))),
        "R2": float(1 - np.sum(r ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    }


def evaluer(ajusteur: Callable, jeu: Jeu, nom: str = "modèle",
            n_parametres: float | None = None) -> dict:
    """Applique un ajusteur au découpage commun et renvoie une ligne de bilan."""
    yhat_train = ajusteur(jeu.X_train, jeu.y_train, jeu.X_train)
    yhat_test = ajusteur(jeu.X_train, jeu.y_train, jeu.X_test)
    m_train = metriques(jeu.y_train, yhat_train)
    m_test = metriques(jeu.y_test, yhat_test)
    return {
        "modele": nom,
        "n_parametres": n_parametres,
        "RMSE_train": round(m_train["RMSE"], 4),
        "RMSE_test": round(m_test["RMSE"], 4),
        "MAE_test": round(m_test["MAE"], 4),
        "R2_test": round(m_test["R2"], 4),
    }


def evaluation_croisee(ajusteur: Callable, jeu: Jeu,
                       k: int = K_PLIS) -> pd.DataFrame:
    """Validation croisée sur les sites d'apprentissage.

    La standardisation est refaite dans chaque pli, à partir du seul pli
    d'apprentissage : aucune information du pli de validation ne fuit.
    """
    X_app = jeu.X.iloc[jeu.idx_train].reset_index(drop=True)
    y_app = jeu.y_train
    lignes = []
    for numero, (app, val) in enumerate(plier(len(y_app), k=k, graine=jeu.graine), 1):
        mu, sd = standardiser_pli(X_app.iloc[app])
        Xa = (X_app.iloc[app] - mu) / sd
        Xv = (X_app.iloc[val] - mu) / sd
        yhat = ajusteur(Xa, y_app[app], Xv)
        m = metriques(y_app[val], yhat)
        lignes.append({"pli": numero, **{k2: round(v, 4) for k2, v in m.items()}})
    return pd.DataFrame(lignes)


def comparer(jeu: Jeu, modeles: dict[str, Callable],
             n_parametres: dict[str, float] | None = None,
             croisee: bool = True) -> pd.DataFrame:
    """Bilan de plusieurs ajusteurs sur le même découpage (tableau final)."""
    n_parametres = {} if n_parametres is None else n_parametres
    lignes = [evaluer(f, jeu, nom=nom, n_parametres=n_parametres.get(nom))
              for nom, f in modeles.items()]
    tableau = pd.DataFrame(lignes)

    if croisee:
        moyenne = {}
        for nom, f in modeles.items():
            cv = evaluation_croisee(f, jeu)
            moyenne[nom] = (round(cv["RMSE"].mean(), 4),
                            round(cv["RMSE"].std(ddof=1) / np.sqrt(len(cv)), 4))
        tableau["CV_RMSE"] = [moyenne[n][0] for n in tableau["modele"]]
        tableau["CV_erreur_type"] = [moyenne[n][1] for n in tableau["modele"]]
    return tableau
