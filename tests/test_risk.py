"""Tests de l'agent Risque."""
import numpy as np
import pandas as pd

from src.agents.base import SharedState
from src.agents.risk import (RegimeDetector, RiskAgent, ledoit_wolf_cov, min_variance_lw,
                             regime_switch, var_cvar)
from src.backtest.baselines import equal_weight


def _series(n=600, seed=0, calm_then_stress=True):
    rng = np.random.default_rng(seed)
    vol = np.r_[np.full(n // 2, 0.005), np.full(n - n // 2, 0.03)] if calm_then_stress else np.full(n, 0.01)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol))), index=idx, name="SPY")


def _prices(n=400, k=4, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), 0)) * 100,
                        index=idx, columns=list("ABCD")[:k])


def test_ledoit_wolf_is_valid_covariance():
    cov, shrink = ledoit_wolf_cov(_prices())
    assert np.allclose(cov, cov.T) and np.all(np.linalg.eigvalsh(cov) > 0)
    assert 0 <= shrink <= 1


def _fitted_detector():
    # seuils appris sur un « train » à volatilité moyenne (~16 % par an)
    return RegimeDetector().fit(_series(seed=3, calm_then_stress=False))


def test_regime_detector_sees_calm_then_stress():
    spy = _series()
    det = _fitted_detector()
    assert det.predict(spy.iloc[:250]) == "calme"
    assert det.predict(spy) == "tendu"


def test_regime_prediction_ignores_the_future():
    spy = _series()
    det = _fitted_detector()
    cut = spy.index[250]
    altered = spy.copy()
    altered[altered.index > cut] *= 10
    assert det.predict(spy.loc[:cut]) == det.predict(altered.loc[:cut])


def test_regime_switch_uses_the_right_strategy():
    spy = _series()
    det = _fitted_detector()
    px = _prices(n=600)
    px.index = spy.index
    only_a = lambda h: pd.Series([1.0, 0, 0, 0], index=h.columns)
    strat = regime_switch({"calme": equal_weight, "normal": equal_weight, "tendu": only_a}, det, spy)
    assert strat(px.iloc[:250])["A"] == 0.25
    assert strat(px)["A"] == 1.0


def test_var_cvar_and_min_variance():
    px = _prices()
    w = min_variance_lw()(px)
    assert abs(w.sum() - 1) < 1e-6
    m = var_cvar(w, px)
    assert m["cvar_%"] <= m["var_%"] < 0


def test_risk_agent_fills_the_state():
    spy = _series(n=400)
    px = _prices(n=400)
    px.index = spy.index
    px["SPY"] = spy
    state = SharedState(data={"prices": px})
    state = RiskAgent(RegimeDetector().fit(spy)).run(state)
    assert state.risk["regime"] in ("calme", "normal", "tendu")
    assert state.risk["covariance"].shape == (4, 4)
