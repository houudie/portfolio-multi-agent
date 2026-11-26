"""Tests de l'environnement RL (étape 4)."""
import numpy as np
import pandas as pd
import pytest

from src.models.rl.env import LOOKBACK, PortfolioSim, action_to_weights, asset_features


def _prices(n=800, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, k)), 0)) * 100,
                        index=idx, columns=list("ABC")[:k])


def test_weights_are_valid():
    w = action_to_weights(np.array([0.5, -1.0, 1.0]))
    assert np.all(w >= 0) and np.isclose(w.sum(), 1)


def test_observation_only_uses_the_past():
    prices = _prices()
    sim = PortfolioSim(prices, episode_steps=5)
    obs = sim.reset(start=300)
    altered = prices.copy()
    altered.iloc[301:] *= 5
    sim2 = PortfolioSim(altered, episode_steps=5)
    assert np.allclose(obs, sim2.reset(start=300))
    assert obs.shape == (3 * 5,)


def test_constant_prices_give_zero_reward_without_cost():
    idx = pd.bdate_range("2015-01-01", periods=600)
    prices = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    sim = PortfolioSim(prices, cost=0.0, episode_steps=3)
    sim.reset(start=LOOKBACK)
    _, reward, _, _ = sim.step(np.array([0.5, 0.5]))
    assert np.isclose(reward, 0.0)


def test_reward_matches_portfolio_return():
    prices = _prices()
    sim = PortfolioSim(prices, cost=0.0, step_days=21, episode_steps=3)
    sim.reset(start=200)
    w = np.array([0.2, 0.3, 0.5])
    _, reward, _, info = sim.step(w)
    expected = (prices.iloc[221] / prices.iloc[200] * w).sum()
    assert np.isclose(np.exp(reward), expected)



def test_episode_ends():
    sim = PortfolioSim(_prices(), episode_steps=4)
    sim.reset()
    dones = [sim.step(np.full(3, 1 / 3))[2] for _ in range(4)]
    assert dones[-1] and not any(dones[:-1])


def test_gymnasium_env_passes_checks():
    pytest.importorskip("gymnasium")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    from src.models.rl.env import PortfolioEnv
    check_env(PortfolioEnv(_prices()))


def test_turnover_penalty_lowers_reward():
    import numpy as np
    import pandas as pd
    from src.models.rl.env import PortfolioSim

    idx = pd.bdate_range("2015-01-01", periods=800)
    prices = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    rewards = []
    for pen in (0.0, 0.05):
        sim = PortfolioSim(prices, cost=0.0, turnover_penalty=pen)
        sim.reset(start=200)
        _, r, _, _ = sim.step(np.array([1.0, 0.0]))   # passe de 50/50 à 100/0 : turnover = 1
        rewards.append(r)
    assert rewards[0] == 0.0 and abs(rewards[1] + 0.05) < 1e-9