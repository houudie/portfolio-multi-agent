"""Agent Risque (étape 4-5, membre A).

Trois rôles :
1. Estimer la covariance de façon ROBUSTE (Ledoit-Wolf) : la covariance empirique
   de 24 actifs sur 252 jours est bruitée ; Ledoit-Wolf la « rétrécit » vers une
   cible simple, ce qui stabilise l'optimisation.
2. Détecter le RÉGIME de marché (calme / normal / tendu) à partir de la volatilité
   récente de l'indice. Les seuils sont appris sur le TRAIN uniquement.
3. Mesurer le risque d'un portefeuille : VaR et CVaR historiques à 95 %.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.agents.base import Agent, SharedState
from src.backtest.baselines import _optimize

REGIMES = ("calme", "normal", "tendu")


# ------------------------------------------------------------------ covariance
def ledoit_wolf_cov(hist: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Covariance annualisée des rendements log + intensité du rétrécissement (0 à 1)."""
    r = np.log(hist).diff().dropna()
    lw = LedoitWolf().fit(r.values)
    cov = pd.DataFrame(lw.covariance_ * 252, index=hist.columns, columns=hist.columns)
    return cov, float(lw.shrinkage_)


def min_variance_lw(max_weight: float = 1.0):
    """Variance minimale avec la covariance de Ledoit-Wolf."""
    def strategy(hist: pd.DataFrame) -> pd.Series:
        cov, _ = ledoit_wolf_cov(hist)
        c = cov.values
        return pd.Series(_optimize(lambda w: w @ c @ w, len(c), max_weight), index=hist.columns)
    return strategy


# ------------------------------------------------------------------ régimes
@dataclass
class RegimeDetector:
    window: int = 63                 # volatilité sur 3 mois
    q_calm: float = 0.50             # sous la médiane du train -> calme
    q_stress: float = 0.85           # au-dessus du 85e centile du train -> tendu
    thresholds: dict = field(default_factory=dict)

    def _vol(self, benchmark: pd.Series) -> pd.Series:
        r = np.log(benchmark).diff()
        return r.rolling(self.window).std() * np.sqrt(252)

    def fit(self, benchmark_train: pd.Series) -> "RegimeDetector":
        vol = self._vol(benchmark_train).dropna()
        self.thresholds = {"calme_max": float(vol.quantile(self.q_calm)),
                           "tendu_min": float(vol.quantile(self.q_stress))}
        return self

    def classify(self, vol: float) -> str:
        if vol < self.thresholds["calme_max"]:
            return "calme"
        return "tendu" if vol >= self.thresholds["tendu_min"] else "normal"

    def predict(self, benchmark_hist: pd.Series) -> str:
        """Régime à la DERNIÈRE date de l'historique fourni."""
        if not self.thresholds:
            raise RuntimeError("Appeler fit() sur le train d'abord.")
        return self.classify(float(self._vol(benchmark_hist.iloc[-(self.window + 1):]).iloc[-1]))

    def series(self, benchmark: pd.Series) -> pd.Series:
        vol = self._vol(benchmark).dropna()
        return vol.map(self.classify)


# ------------------------------------------------------------------ mesures de risque
def var_cvar(weights: pd.Series, hist: pd.DataFrame, alpha: float = 0.95) -> dict:
    """VaR et CVaR historiques journalières (en %, valeurs négatives = pertes)."""
    r = hist.pct_change().dropna() @ weights.reindex(hist.columns).fillna(0)
    var = r.quantile(1 - alpha)
    return {"var_%": 100 * float(var), "cvar_%": 100 * float(r[r <= var].mean())}


# ------------------------------------------------------------------ stratégie conditionnelle
def regime_switch(strategies: dict, detector: RegimeDetector, benchmark: pd.Series):
    """Choisit la stratégie selon le régime détecté à la date de décision.

    `benchmark` est la série complète de l'indice, mais on n'en lit que la partie
    <= date de décision : aucune information future.
    """
    def strategy(hist: pd.DataFrame) -> pd.Series:
        regime = detector.predict(benchmark.loc[:hist.index[-1]])
        return strategies[regime](hist)
    return strategy


# ------------------------------------------------------------------ agent
class RiskAgent(Agent):
    name = "Agent Risque"

    def __init__(self, detector: RegimeDetector, benchmark: str = "SPY", lookback: int = 252):
        self.detector, self.benchmark, self.lookback = detector, benchmark, lookback

    def run(self, state: SharedState) -> SharedState:
        prices: pd.DataFrame = state.data["prices"]
        hist = prices.drop(columns=self.benchmark, errors="ignore").iloc[-self.lookback:]
        cov, shrink = ledoit_wolf_cov(hist)
        regime = self.detector.predict(prices[self.benchmark])
        state.risk.update({"covariance": cov, "shrinkage": shrink, "regime": regime,
                           "seuils_volatilite": self.detector.thresholds})
        state.log.append(f"{self.name} : régime {regime}, rétrécissement Ledoit-Wolf {shrink:.2f}")
        return state
