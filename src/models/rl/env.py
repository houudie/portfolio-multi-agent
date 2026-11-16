"""Environnement d'apprentissage par renforcement pour l'allocation (étape 4, membre C).

Formulation MDP :
- État s_t   : pour chaque actif, momentum 1/3/6 mois et volatilité 1/3 mois,
               calculés UNIQUEMENT avec les prix jusqu'à t.
- Action a_t : un score par actif (vecteur continu dans [-1, 1]),
               transformé en poids par softmax -> poids >= 0, somme = 1.
- Transition : on garde ces poids pendant `step_days` jours de bourse (~1 mois).
- Récompense : log du rendement du portefeuille sur la période, net des frais,
               moins une pénalité optionnelle sur la volatilité journalière.

Différence avec le DQN : un DQN choisit parmi un nombre fini d'actions ; ici les
poids sont continus, d'où PPO (politique gaussienne sur un espace continu).

`PortfolioSim` contient toute la logique en Python pur (testable sans Gymnasium) ;
`PortfolioEnv` l'enveloppe dans l'interface Gymnasium attendue par stable-baselines3.
"""
import numpy as np
import pandas as pd

LOOKBACK = 130          # jours d'historique nécessaires pour les features
N_FEATURES = 5          # features par actif


def asset_features(hist: pd.DataFrame) -> np.ndarray:
    """Vecteur d'état à partir de l'historique de prix (dernière ligne = aujourd'hui)."""
    if len(hist) < LOOKBACK:
        raise ValueError(f"Il faut au moins {LOOKBACK} jours d'historique.")
    logp = np.log(hist.values)
    r = np.diff(logp, axis=0)
    feats = np.stack([
        logp[-1] - logp[-22],                    # momentum 1 mois
        logp[-1] - logp[-64],                    # momentum 3 mois
        logp[-1] - logp[-127],                   # momentum 6 mois
        r[-21:].std(axis=0) * np.sqrt(252),      # volatilité 1 mois
        r[-63:].std(axis=0) * np.sqrt(252),      # volatilité 3 mois
    ], axis=1)
    return np.clip(feats, -3, 3).astype(np.float32).ravel()


def action_to_weights(action: np.ndarray, temperature: float = 3.0) -> np.ndarray:
    z = temperature * np.asarray(action, dtype=np.float64)
    z = np.exp(z - z.max())
    return z / z.sum()


class PortfolioSim:
    def __init__(self, prices: pd.DataFrame, step_days: int = 21, episode_steps: int = 24,
                 cost: float = 0.001, risk_penalty: float = 0.0, seed: int = 0):
        self.prices = prices.sort_index()
        self.values = self.prices.values
        self.n_assets = prices.shape[1]
        self.step_days, self.episode_steps = step_days, episode_steps
        self.cost, self.risk_penalty = cost, risk_penalty
        self.rng = np.random.default_rng(seed)
        last_start = len(prices) - step_days * episode_steps - 1
        if last_start <= LOOKBACK:
            raise ValueError("Pas assez de données pour un épisode complet.")
        self.start_range = (LOOKBACK - 1, last_start)

    def reset(self, start: int | None = None) -> np.ndarray:
        self.t = int(self.rng.integers(*self.start_range)) if start is None else start
        self.weights = np.full(self.n_assets, 1.0 / self.n_assets)
        self.steps = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        return asset_features(self.prices.iloc[self.t - LOOKBACK + 1: self.t + 1])

    def step(self, new_weights: np.ndarray):
        traded = np.abs(new_weights - self.weights).sum()
        window = self.values[self.t: self.t + self.step_days + 1]       # prix de t à t+step
        rel = window / window[0]                                          # croissance de chaque actif
        path = rel @ new_weights                                          # valeur du portefeuille
        growth = path[-1] * (1 - self.cost * traded)
        daily = np.diff(path) / path[:-1]
        reward = float(np.log(growth) - self.risk_penalty * daily.std() * np.sqrt(252))

        self.weights = new_weights * rel[-1] / path[-1]                   # dérive des poids
        self.t += self.step_days
        self.steps += 1
        done = self.steps >= self.episode_steps or self.t + self.step_days >= len(self.values)
        info = {"rendement_periode": float(growth - 1), "turnover": float(traded)}
        return self.observation(), reward, done, info


try:
    import gymnasium as gym
    from gymnasium import spaces

    class PortfolioEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, prices: pd.DataFrame, **kwargs):
            super().__init__()
            self.sim = PortfolioSim(prices, **kwargs)
            n = self.sim.n_assets
            self.observation_space = spaces.Box(-3.0, 3.0, shape=(n * N_FEATURES,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.sim.rng = np.random.default_rng(seed)
            return self.sim.reset(), {}

        def step(self, action):
            obs, reward, done, info = self.sim.step(action_to_weights(action))
            return obs, reward, done, False, info
except ImportError:  # Gymnasium non installé : seul PortfolioSim est disponible
    PortfolioEnv = None


def rl_strategy(model):
    """Transforme un agent entraîné en stratégie utilisable par le moteur de backtest."""
    def strategy(hist: pd.DataFrame) -> pd.Series:
        action, _ = model.predict(asset_features(hist), deterministic=True)
        return pd.Series(action_to_weights(action), index=hist.columns)
    return strategy
