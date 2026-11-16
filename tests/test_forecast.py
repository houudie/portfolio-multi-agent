"""Tests de l'agent Prévision (étape 4)."""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.forecast_strategies import forecast_max_sharpe, latest_forecast, top_k
from src.models.forecast import evaluate, information_coefficient, make_model


def _panel(n_dates=60, tickers=("A", "B", "C", "D"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    X = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["f1", "f2", "f3"])
    y = 0.5 * X["f1"] + rng.normal(scale=0.1, size=len(idx))
    return X, y


def test_perfect_ranking_gives_ic_one():
    _, y = _panel()
    assert np.isclose(information_coefficient(y, y).mean(), 1.0)


def test_constant_prediction_gives_ic_zero():
    _, y = _panel()
    assert information_coefficient(y, pd.Series(0.0, index=y.index)).mean() == 0.0


AVAILABLE = [m for m in ["ridge", "foret_aleatoire", "lightgbm", "xgboost"]
             if m in ("ridge", "foret_aleatoire") or importlib.util.find_spec(m)]


@pytest.mark.parametrize("name", AVAILABLE)
def test_models_learn_a_simple_signal(name):
    X, y = _panel(n_dates=200)
    model = make_model(name)
    model.fit(X, y)
    m = evaluate(y, pd.Series(model.predict(X), index=X.index))
    assert m["ic_moyen"] > 0.5


def test_latest_forecast_never_uses_the_future():
    dates = pd.to_datetime(["2020-01-10", "2020-02-10"])
    preds = pd.DataFrame({"A": [1.0, 2.0]}, index=dates)
    assert latest_forecast(preds, pd.Timestamp("2020-02-01"))["A"] == 1.0
    assert latest_forecast(preds, pd.Timestamp("2020-01-01")) is None


def test_forecast_strategies_run_and_pick_best_assets():
    idx = pd.bdate_range("2019-01-01", periods=400)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, (400, 4)), 0)) * 100,
                          index=idx, columns=list("ABCD"))
    preds = pd.DataFrame({"A": 0.05, "B": 0.04, "C": -0.02, "D": -0.03}, index=idx)
    w = top_k(preds, k=2)(prices.iloc[:300])
    assert set(w[w > 0].index) == {"A", "B"}
    res = run_backtest(prices, forecast_max_sharpe(preds), "2019-12-01", "2020-06-30")
    assert res.weights.max().max() <= 0.25 + 1e-6
