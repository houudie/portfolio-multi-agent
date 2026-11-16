"""Stratégies qui utilisent les prévisions de l'agent Prévision (étape 4).

`preds` = tableau date x actif des rendements prévus à 21 jours. À la date de
décision t, on prend la dernière prévision disponible <= t (jamais une future).
S'il n'y en a pas (ou si elle date de plus de 45 jours), on revient au 1/N.
"""
import numpy as np
import pandas as pd

from src.backtest.baselines import _estimates, _optimize, equal_weight


def latest_forecast(preds: pd.DataFrame, t: pd.Timestamp, max_age_days: int = 45) -> pd.Series | None:
    past = preds.loc[:t]
    if past.empty or (t - past.index[-1]).days > max_age_days:
        return None
    return past.iloc[-1]


def top_k(preds: pd.DataFrame, k: int = 8):
    """Poids égaux sur les k actifs dont le rendement prévu est le plus élevé."""
    def strategy(hist: pd.DataFrame) -> pd.Series:
        f = latest_forecast(preds, hist.index[-1])
        if f is None:
            return equal_weight(hist)
        best = f.reindex(hist.columns).nlargest(k).index
        w = pd.Series(0.0, index=hist.columns)
        w[best] = 1.0 / len(best)
        return w
    return strategy


def forecast_max_sharpe(preds: pd.DataFrame, max_weight: float = 0.25):
    """Markowitz avec les rendements PRÉVUS à la place des moyennes historiques."""
    def strategy(hist: pd.DataFrame) -> pd.Series:
        f = latest_forecast(preds, hist.index[-1])
        if f is None:
            return equal_weight(hist)
        mu = f.reindex(hist.columns).fillna(0).values * 12  # mensuel -> annuel
        _, cov = _estimates(hist)

        def neg_sharpe(w):
            return -(w @ mu) / np.sqrt(w @ cov @ w + 1e-12)

        return pd.Series(_optimize(neg_sharpe, len(mu), max_weight), index=hist.columns)
    return strategy
