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



def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    start: str,
    end: str,
    name: str = "strategie",
    cost: float = 0.001,
    lookback: int = 252,
) -> BacktestResult:
    """
    Backtest mensuel sur prix de clôture.

    Convention :
    - Le signal est calculé avec les données jusqu'à la séance
      précédant l'exécution.
    - L'exécution est effectuée à la clôture de la date de
      rebalancement.
    - Les nouveaux poids sont actifs sur la période suivante.
    - Les frais sont déduits à la date de transaction.
    """

    # ---------------------------------------------------------
    # 1. Validation des entrées
    # ---------------------------------------------------------

    if prices.empty:
        raise ValueError("Le DataFrame des prix est vide.")

    if not prices.index.is_unique:
        raise ValueError("Les dates doivent être uniques.")

    if not prices.columns.is_unique:
        raise ValueError("Les tickers doivent être uniques.")

    prices = prices.sort_index().astype(float)

    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Les prix contiennent des valeurs manquantes ou infinies.")

    if (prices <= 0).any().any():
        raise ValueError("Tous les prix doivent être strictement positifs.")

    if not 0 <= cost < 1:
        raise ValueError("Le coût doit être compris entre 0 inclus et 1 exclu.")

    if lookback <= 0:
        raise ValueError("lookback doit être positif.")

    simple = prices.pct_change(fill_method=None)

    days = prices.loc[start:end].index

    if len(days) == 0:
        raise ValueError("Aucune séance disponible dans la période demandée.")

    before = prices.index[prices.index < days[0]]

    if len(before) < 2:
        raise ValueError(
            "Deux séances historiques préalables sont nécessaires "
            "pour calculer le signal avant exécution."
        )

    # ---------------------------------------------------------
    # 2. Dates de rebalancement
    # ---------------------------------------------------------

    rebal = set(month_end_dates(days)) - {days[-1]}

    def previous_session(t):
        position = prices.index.get_loc(t)

        if position == 0:
            raise ValueError("Aucune séance précédente disponible.")

        return prices.index[position - 1]

    # ---------------------------------------------------------
    # 3. Fonction de décision
    # ---------------------------------------------------------

    def decide(signal_date):

        hist = prices.loc[:signal_date].iloc[-lookback:]

        if hist.empty:
            raise ValueError("Historique insuffisant pour la stratégie.")

        raw_weights = strategy(hist)

        if not isinstance(raw_weights, pd.Series):
            raise TypeError("La stratégie doit retourner une Series.")

        if not raw_weights.index.is_unique:
            raise ValueError("Les indices des poids doivent être uniques.")

        unknown = raw_weights.index.difference(prices.columns)

        if len(unknown):
            raise ValueError(f"Actifs inconnus : {list(unknown)}")

        w = raw_weights.reindex(
            prices.columns,
            fill_value=0.0
        ).astype(float)

        if not np.isfinite(w.to_numpy()).all():
            raise ValueError("Les poids contiennent des NaN ou des valeurs infinies.")

        if (w < -1e-9).any():
            raise ValueError("Les poids négatifs ne sont pas autorisés.")

        if abs(float(w.sum()) - 1.0) > 1e-6:
            raise ValueError(
                f"Somme des poids invalide : {w.sum():.8f}"
            )

        # Corriger uniquement les éventuelles petites erreurs numériques.
        w = w.clip(lower=0.0)
        w = w / w.sum()

        return w

    # ---------------------------------------------------------
    # 4. Allocation initiale
    # ---------------------------------------------------------

    initial_execution_date = before[-1]
    initial_signal_date = before[-2]

    w = decide(initial_signal_date)

    initial_turnover = float(w.abs().sum())

    if cost * initial_turnover >= 1:
        raise ValueError("Frais initiaux excessifs.")

    weights = {
        initial_execution_date: w
    }

    turnover = {
        initial_execution_date: initial_turnover
    }

    out = []

    # ---------------------------------------------------------
    # 5. Simulation journalière
    # ---------------------------------------------------------

    for i, d in enumerate(days):

        r = simple.loc[d]

        if not np.isfinite(r.to_numpy()).all():
            raise ValueError(
                f"Rendement manquant ou non fini le {d.date()}."
            )

        # Rendement brut du portefeuille avec les poids précédents.
        gross_return = float((w * r).sum())

        if gross_return <= -1:
            raise ValueError("Perte totale ou supérieure à 100 %.")

        # Dérive naturelle des poids sans transaction.
        drifted_w = w * (1 + r) / (1 + gross_return)

        fee_factor = 1.0

        # Frais d'entrée initiaux.
        if i == 0:
            fee_factor *= 1 - cost * initial_turnover

        # -----------------------------------------------------
        # Rebalancement mensuel
        # -----------------------------------------------------

        if d in rebal:

            # Le signal ne connaît pas la clôture du jour d.
            signal_date = previous_session(d)

            new_w = decide(signal_date)

            traded = float((new_w - drifted_w).abs().sum())

            fee_factor *= 1 - cost * traded

            if fee_factor <= 0:
                raise ValueError("Facteur de capital après frais invalide.")

            weights[d] = new_w
            turnover[d] = traded

            # Les nouveaux poids s'appliquent à partir de la
            # prochaine période de rendement.
            w = new_w

        else:
            w = drifted_w

        # Rendement net intégrant les frais à la bonne date.
        net_return = (1 + gross_return) * fee_factor - 1

        out.append(net_return)

    # ---------------------------------------------------------
    # 6. Résultat
    # ---------------------------------------------------------

    return BacktestResult(
        name=name,
        returns=pd.Series(out, index=days, name=name),
        weights=pd.DataFrame(weights).T.sort_index(),
        turnover=pd.Series(turnover).sort_index(),
    )



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
