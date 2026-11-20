"""Tests du moteur de backtest (étape 3)."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.baselines import equal_weight, max_sharpe, min_variance
from src.backtest.engine import metrics, run_backtest


def _prices(n=600, k=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    data = np.exp(np.cumsum(rng.normal(0.0004, 0.01, (n, k)), axis=0)) * 100
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(k)])


def test_strategy_uses_information_before_execution():
    prices = _prices()
    seen = []

    def spy(hist):
        seen.append(hist.index.max())
        return equal_weight(hist)

    res = run_backtest(
        prices,
        spy,
        "2019-01-01",
        "2019-12-31",
        name="espion",
        cost=0
    )

    execution_dates = list(res.weights.index)

    assert len(seen) == len(execution_dates)

    for signal_date, execution_date in zip(
        seen, execution_dates
    ):
        assert signal_date < execution_date


def test_constant_prices_give_zero_return_without_costs():
    idx = pd.bdate_range("2019-01-01", periods=300)
    prices = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    res = run_backtest(prices, equal_weight, "2019-06-01", "2019-12-31", cost=0)
    assert np.allclose(res.returns, 0)


def test_costs_reduce_performance():
    prices = _prices()
    free = run_backtest(prices, max_sharpe(), "2019-01-01", "2019-12-31", cost=0)
    paid = run_backtest(prices, max_sharpe(), "2019-01-01", "2019-12-31", cost=0.01)
    assert paid.wealth.iloc[-1] < free.wealth.iloc[-1]


def test_single_asset_matches_its_own_return():
    prices = _prices(k=1)
    res = run_backtest(prices, equal_weight, "2019-01-01", "2019-12-31", cost=0)
    expected = prices["A0"].loc["2019-01-01":"2019-12-31"].pct_change()
    expected.iloc[0] = prices["A0"].loc["2019-01-01"] / prices["A0"].loc[:"2018-12-31"].iloc[-1] - 1
    assert np.allclose(res.returns.values, expected.values)


@pytest.mark.parametrize("make", [max_sharpe, min_variance])
def test_optimizers_return_valid_weights(make):
    w = make()(_prices().iloc[:252])
    assert (w >= -1e-9).all() and abs(w.sum() - 1) < 1e-6


def test_metrics_keys():
    res = run_backtest(_prices(), equal_weight, "2019-01-01", "2019-12-31")
    m = metrics(res)
    assert {"sharpe", "drawdown_max_%", "cvar_95_jour_%"} <= set(m)
