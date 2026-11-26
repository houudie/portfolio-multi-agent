"""Tests de l'interprétabilité (importance par permutation, analyse des erreurs)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.interpret import ic_by_asset, monthly_ic, permutation_ic


def _panel(n_dates=60, n_assets=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"A{i}" for i in range(n_assets)]], names=["date", "ticker"])
    X = pd.DataFrame({"signal": rng.normal(size=len(idx)), "bruit": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(X["signal"] + 0.5 * rng.normal(size=len(idx)), index=idx)
    return X, y


def test_permutation_finds_the_useful_variable():
    X, y = _panel()
    model = LinearRegression().fit(X, y)
    perm = permutation_ic(model, X, y, n_repeats=3)
    assert perm.index[0] == "signal"
    assert perm.loc["signal", "baisse_ic"] > 0.5
    assert abs(perm.loc["bruit", "baisse_ic"]) < 0.05


def test_error_analysis_shapes():
    X, y = _panel()
    pred = pd.Series(X["signal"].values, index=X.index)
    assert monthly_ic(y, pred).between(-1, 1).all()
    assert len(ic_by_asset(y, pred)) == 10
