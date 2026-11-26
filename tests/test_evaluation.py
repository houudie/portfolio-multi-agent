"""Tests de la règle utilisée pour l'évaluation finale."""
import numpy as np
import pandas as pd

from src.agents.risk import RegimeDetector
from src.evaluation.run_test import profile_strategy


def _prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    cols = ["AAPL", "MSFT", "JPM", "XOM", "TLT", "IEF", "GLD"]
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, len(cols))), 0)), index=idx, columns=cols)


def test_profile_strategy_respects_constraints():
    px = _prices()
    spy = px.mean(axis=1).rename("SPY")
    det = RegimeDetector().fit(spy)
    w = profile_strategy("prudent", det, spy, preds=None)(px)
    assert abs(w.sum() - 1) < 1e-6 and (w >= 0).all()
    assert w[["TLT", "IEF", "GLD"]].sum() >= 0.5 - 1e-6
    assert w.drop(["TLT", "IEF", "GLD"]).max() <= 0.15 + 1e-6
    assert w[["TLT", "IEF", "GLD"]].max() <= 0.5 / 3 + 1e-6  # plafond relevé pour atteindre 50 %


def test_profile_strategy_never_reads_future_spy():
    px = _prices()
    spy = px.mean(axis=1).rename("SPY")
    det = RegimeDetector().fit(spy)
    cut = px.index[300]
    altered = spy.copy()
    altered[altered.index > cut] *= 5
    a = profile_strategy("équilibré", det, spy, None)(px.loc[:cut])
    b = profile_strategy("équilibré", det, altered, None)(px.loc[:cut])
    assert np.allclose(a, b)
