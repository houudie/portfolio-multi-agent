"""Moteur de backtest (étape 3).

Principe :
- À chaque date de rebalancement (dernier jour de bourse du mois), la stratégie
  reçoit UNIQUEMENT l'historique des prix jusqu'à cette date (incluse) et renvoie
  des poids. Ces poids s'appliquent à partir du jour suivant.
- Entre deux rebalancements, les poids dérivent avec les prix (pas de trading).
- Chaque rebalancement coûte `cost` x (somme des montants échangés).

Une stratégie est une fonction : historique de prix (DataFrame) -> poids (Series, somme = 1).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

Strategy = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series            # rendements journaliers simples, nets de frais
    weights: pd.DataFrame         # poids choisis à chaque rebalancement
    turnover: pd.Series           # montant échangé à chaque rebalancement

    @property
    def wealth(self) -> pd.Series:
        return (1 + self.returns).cumprod()


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).last().values)


def run_backtest(prices: pd.DataFrame, strategy: Strategy, start: str, end: str,
                 name: str = "strategie", cost: float = 0.001, lookback: int = 252) -> BacktestResult:
    prices = prices.sort_index()
    simple = prices.pct_change()
    days = prices.loc[start:end].index
    before = prices.index[prices.index < days[0]]
    if len(before) == 0:
        raise ValueError("Il faut au moins un jour d'historique avant `start`.")

    rebal = set(month_end_dates(days)) - {days[-1]}

    def decide(t):
        hist = prices.loc[:t].iloc[-lookback:]
        w = strategy(hist).reindex(prices.columns).fillna(0.0)
        if (w < -1e-9).any() or abs(w.sum() - 1) > 1e-6:
            raise ValueError(f"{name} : poids invalides le {t.date()} (somme {w.sum():.4f})")
        return w

    # décision initiale à la clôture de la veille du début
    w = decide(before[-1])
    weights, turnover = {before[-1]: w}, {before[-1]: w.abs().sum()}
    pending_cost = cost * w.abs().sum()

    out = []
    for d in days:
        r = simple.loc[d].fillna(0.0)
        port = float((w * r).sum())
        out.append(port - pending_cost)
        pending_cost = 0.0
        w = w * (1 + r) / (1 + port)          # dérive des poids
        if d in rebal:
            new_w = decide(d)
            traded = float((new_w - w).abs().sum())
            pending_cost = cost * traded        # payé sur le rendement du jour suivant
            weights[d], turnover[d] = new_w, traded
            w = new_w

    return BacktestResult(name=name, returns=pd.Series(out, index=days, name=name),
                          weights=pd.DataFrame(weights).T, turnover=pd.Series(turnover))


def metrics(res: BacktestResult, periods: int = 252) -> dict:
    r = res.returns
    wealth = res.wealth
    years = len(r) / periods
    ann_ret = wealth.iloc[-1] ** (1 / years) - 1
    ann_vol = r.std() * np.sqrt(periods)
    downside = r[r < 0].std() * np.sqrt(periods)
    max_dd = (wealth / wealth.cummax() - 1).min()
    var95 = r.quantile(0.05)
    return {
        "strategie": res.name,
        "rendement_annuel_%": 100 * ann_ret,
        "volatilite_annuelle_%": 100 * ann_vol,
        "sharpe": r.mean() / r.std() * np.sqrt(periods),
        "sortino": r.mean() * periods / downside if downside > 0 else np.nan,
        "drawdown_max_%": 100 * max_dd,
        "calmar": ann_ret / abs(max_dd) if max_dd < 0 else np.nan,
        "cvar_95_jour_%": 100 * r[r <= var95].mean(),
        "turnover_moyen_%": 100 * res.turnover.iloc[1:].mean() if len(res.turnover) > 1 else 0.0,
    }


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame([metrics(r) for r in results]).set_index("strategie").round(2)
