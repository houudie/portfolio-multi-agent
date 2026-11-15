"""Fonctions d'analyse exploratoire (étape 2), appelées par notebooks/01_eda.ipynb.

Règle : les statistiques qui guident nos choix (corrélations, distributions,
stationnarité) sont calculées sur le TRAIN uniquement. Les graphiques sur toute
la période servent seulement à décrire le contexte.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def fig_prices(prices: pd.DataFrame, cuts: list[str], folder: Path, cols: list[str]):
    """Prix en base 100 (échelle log), avec les frontières du découpage."""
    fig, ax = plt.subplots(figsize=(11, 5))
    (100 * prices[cols] / prices[cols].iloc[0]).plot(ax=ax, logy=True, lw=1.2)
    for c, label in zip(cuts, ["fin train", "fin validation"]):
        ax.axvline(pd.Timestamp(c), color="black", ls="--", lw=1)
        ax.text(pd.Timestamp(c), ax.get_ylim()[1], f" {label}", va="top", fontsize=9)
    ax.set(title="Évolution des prix (base 100, échelle log)", xlabel="", ylabel="Base 100")
    ax.legend(ncol=len(cols), fontsize=8, loc="upper left")
    _save(fig, folder, "01_prix_base100")
    return fig


def stats_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Statistiques annualisées par actif (à calculer sur le train)."""
    wealth = returns.cumsum().apply(np.exp)
    table = pd.DataFrame({
        "rendement_annuel_%": 100 * returns.mean() * 252,
        "volatilite_annuelle_%": 100 * returns.std() * np.sqrt(252),
        "sharpe": returns.mean() / returns.std() * np.sqrt(252),
        "skewness": returns.skew(),
        "kurtosis_excess": returns.kurt(),
        "drawdown_max_%": 100 * (wealth / wealth.cummax() - 1).min(),
    })
    return table.sort_values("sharpe", ascending=False).round(2)


def fig_distribution(returns: pd.DataFrame, ticker: str, folder: Path):
    """Histogramme des rendements journaliers comparé à une loi normale."""
    r = returns[ticker].dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(r, bins=120, density=True, alpha=0.6, label="Rendements observés")
    x = np.linspace(r.min(), r.max(), 400)
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), lw=2, label="Loi normale de même moyenne et écart-type")
    ax.set(title=f"{ticker} : distribution des rendements journaliers (train)",
           xlabel="Rendement log journalier", ylabel="Densité")
    ax.legend()
    _save(fig, folder, f"02_distribution_{ticker}")
    return fig


def fig_correlation(returns: pd.DataFrame, folder: Path):
    """Matrice de corrélation regroupée par similarité (train)."""
    g = sns.clustermap(returns.corr(), cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                       figsize=(10, 10), annot=False, cbar_kws={"label": "Corrélation"})
    g.fig.suptitle("Corrélations des rendements journaliers (train)", y=1.02)
    _save(g.fig, folder, "03_correlations")
    return g.fig


def fig_rolling_vol(returns: pd.DataFrame, cols: list[str], folder: Path, window: int = 63):
    """Volatilité annualisée glissante : montre les régimes de marché."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    (100 * returns[cols].rolling(window).std() * np.sqrt(252)).plot(ax=ax, lw=1.2)
    ax.set(title=f"Volatilité annualisée glissante ({window} jours)", xlabel="", ylabel="%")
    _save(fig, folder, "04_volatilite_glissante")
    return fig


def fig_macro(macro: pd.DataFrame, folder: Path):
    fig, axes = plt.subplots(len(macro.columns), 1, figsize=(11, 2.6 * len(macro.columns)), sharex=True)
    for ax, col in zip(np.atleast_1d(axes), macro.columns):
        macro[col].plot(ax=ax, lw=1.1)
        ax.set(title=col, xlabel="")
    fig.suptitle("Variables macroéconomiques (FRED)", y=1.0)
    fig.tight_layout()
    _save(fig, folder, "05_macro")
    return fig


def adf_table(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Test de Dickey-Fuller augmenté : p < 0,05 => série stationnaire."""
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for col in prices.columns:
        rows.append({
            "actif": col,
            "p_value_prix": adfuller(np.log(prices[col].dropna()), autolag="AIC")[1],
            "p_value_rendements": adfuller(returns[col].dropna(), autolag="AIC")[1],
        })
    return pd.DataFrame(rows).set_index("actif").round(4)
