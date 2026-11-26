"""Tests des analyses de robustesse."""
import numpy as np
import pandas as pd

from src.evaluation.robustness import block_bootstrap_indices, out_of_range_share, sharpe, sharpe_ci


def test_bootstrap_indices_shape_and_range():
    idx = block_bootstrap_indices(100, 21, 50, np.random.default_rng(0))
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() < 100


def test_sharpe_ci_contains_estimate_and_identical_series_have_zero_gap():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 750))
    df = pd.DataFrame({"A": r, "B": r})
    t = sharpe_ci(df, reference="B", n_boot=200)
    assert t.loc["A", "sharpe_ic_bas"] <= t.loc["A", "sharpe"] <= t.loc["A", "sharpe_ic_haut"]
    assert abs(t.loc["A", "ecart_vs_ref"]) < 1e-12
    assert abs(t.loc["A", "ecart_ic_bas"]) < 1e-12 and abs(t.loc["A", "ecart_ic_haut"]) < 1e-12


def test_sharpe_of_constant_series_is_zero():
    assert sharpe(np.zeros(10)) == 0.0


def test_out_of_range_share():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    test = pd.DataFrame({"x": [1.0, 3.0, -1.0, 2.0]})
    assert out_of_range_share(train, test, ["x"])["x"] == 0.5
