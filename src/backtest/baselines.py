"""Stratégies de référence (baselines) de l'étape 3.

Chaque fonction reçoit l'historique des prix (déjà limité au passé par le moteur)
et renvoie des poids positifs dont la somme vaut 1.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weight(hist: pd.DataFrame) -> pd.Series:
    """1/N : le même poids pour chaque actif."""
    n = hist.shape[1]
    return pd.Series(1.0 / n, index=hist.columns)


def buy_and_hold(ticker: str):
    """100 % sur un seul actif (ex. l'indice SPY)."""
    def strategy(hist: pd.DataFrame) -> pd.Series:
        w = pd.Series(0.0, index=hist.columns)
        w[ticker] = 1.0
        return w
    strategy.__name__ = f"buy_and_hold_{ticker}"
    return strategy


def _estimates(hist: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    r = np.log(hist).diff().dropna()
    return r.mean().values * 252, r.cov().values * 252


def _optimize(objective, n: int, max_weight: float) -> np.ndarray:
    res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def max_sharpe(max_weight: float = 1.0):
    """Markowitz : maximise le ratio de Sharpe avec moyennes et covariances historiques."""
    def strategy(hist: pd.DataFrame) -> pd.Series:
        mu, cov = _estimates(hist)
        def neg_sharpe(w):
            return -(w @ mu) / np.sqrt(w @ cov @ w + 1e-12)
        return pd.Series(_optimize(neg_sharpe, len(mu), max_weight), index=hist.columns)
    return strategy


def min_variance(max_weight: float = 1.0):
    """Portefeuille de variance minimale (n'utilise que la covariance)."""
    def strategy(hist: pd.DataFrame) -> pd.Series:
        _, cov = _estimates(hist)
        return pd.Series(_optimize(lambda w: w @ cov @ w, cov.shape[0], max_weight), index=hist.columns)
    return strategy
