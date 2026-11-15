"""Tests anti-fuite de données de l'étape 2."""
import numpy as np
import pandas as pd

from src.data.prepare import align_macro, make_features, purge, split_dates


def _fake_prices(n=400, assets=("A", "B", "SPY"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    data = np.exp(np.cumsum(rng.normal(0, 0.01, (n, len(assets))), axis=0)) * 100
    return pd.DataFrame(data, index=idx, columns=list(assets))


def _fake_macro(index):
    return pd.DataFrame({"vix": np.linspace(15, 30, len(index))}, index=index)


def test_features_do_not_use_the_future():
    """Modifier les prix APRÈS une date ne doit changer aucune feature AVANT cette date."""
    prices = _fake_prices()
    cutoff = prices.index[300]
    altered = prices.copy()
    altered.loc[altered.index > cutoff] *= 3.0

    macro = align_macro(_fake_macro(prices.index), prices.index)
    f1 = make_features(prices, macro, "SPY", horizon=21).drop(columns="target")
    f2 = make_features(altered, macro, "SPY", horizon=21).drop(columns="target")

    before = f1.index.get_level_values("date") <= cutoff
    pd.testing.assert_frame_equal(f1[before], f2[before])


def test_macro_is_lagged():
    idx = pd.bdate_range("2020-01-01", periods=5)
    macro = pd.DataFrame({"vix": [1.0, 2, 3, 4, 5]}, index=idx)
    aligned = align_macro(macro, idx, lag=1)
    assert np.isnan(aligned["vix"].iloc[0])
    assert aligned["vix"].iloc[1] == 1.0


def test_split_is_chronological_and_purged():
    idx = pd.bdate_range("2019-06-01", "2022-06-30")
    cfg = {"split": {"train_end": "2020-12-31", "val_end": "2021-12-31"}}
    s = purge(split_dates(idx, cfg), horizon=21)
    assert s["train"].max() < s["val"].min() < s["val"].max() < s["test"].min()
    # au moins 21 jours de bourse entre la fin du train et le début de la validation
    gap = idx[(idx > s["train"].max()) & (idx < s["val"].min())]
    assert len(gap) >= 21
