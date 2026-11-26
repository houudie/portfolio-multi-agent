"""Entraîne l'agent PPO sur le train, puis le backteste sur la validation (étape 4, membre C).

Usage :
    python -m src.models.rl.train_ppo                  # 200 000 pas (~5-10 min)
    python -m src.models.rl.train_ppo --steps 20000    # essai rapide (~1 min)
    python -m src.models.rl.train_ppo --risk 0.5       # pénalise davantage la volatilité
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.backtest.baselines import equal_weight
from src.backtest.engine import compare, run_backtest
from src.config import ROOT, load_config
from src.models.rl.env import PortfolioEnv, rl_strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--risk", type=float, default=0.1, help="pénalité de volatilité")
    parser.add_argument("--turnover", type=float, default=0.0, help="pénalité de turnover")
    parser.add_argument("--suffix", default="", help="ex. _turnover : ne pas écraser le modèle retenu")
    args = parser.parse_args()

    cfg = load_config()
    prices = pd.read_csv(ROOT / cfg["paths"]["processed"] / "prices_clean.csv",
                         index_col=0, parse_dates=True)
    universe = cfg["universe"]["stocks"] + cfg["universe"]["etfs"]
    px = prices[universe]
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    cost = cfg["backtest"]["transaction_cost"]

    # 1. L'agent ne voit QUE la période d'entraînement
    log_dir = ROOT / "models_saved" / "ppo_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = PortfolioEnv(px.loc[:train_end], cost=cost, risk_penalty=args.risk,
                       turnover_penalty=args.turnover, seed=cfg["seed"])
    check_env(env)
    env = Monitor(env, str(log_dir / f"train{args.suffix}"))

    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=128, learning_rate=3e-4,
                gamma=0.95, ent_coef=0.01, seed=cfg["seed"], verbose=0)
    print(f"Entraînement PPO : {args.steps} pas ...", flush=True)
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(ROOT / "models_saved" / f"ppo_portfolio{args.suffix}")

    # 2. Courbe d'apprentissage (récompense moyenne par épisode)
    log = pd.read_csv(log_dir / f"train{args.suffix}.monitor.csv", skiprows=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    log["r"].rolling(50, min_periods=1).mean().plot(ax=ax)
    ax.set(title="PPO : récompense par épisode (moyenne glissante sur 50 épisodes)",
           xlabel="Épisode", ylabel="Récompense (log-rendement sur 2 ans)")
    fig.savefig(ROOT / cfg["paths"]["figures"] / f"09_ppo_apprentissage{args.suffix}.png",
                dpi=150, bbox_inches="tight")

    # 3. Backtest en validation, avec le même moteur que les autres stratégies
    start = train_end + pd.Timedelta(days=1)
    end = pd.Timestamp(cfg["split"]["val_end"])
    results = [
        run_backtest(px, equal_weight, start, end, name="1/N", cost=cost),
        run_backtest(px, rl_strategy(model), start, end, name="PPO", cost=cost),
    ]
    table = compare(results)
    table.to_csv(ROOT / "reports" / f"ppo_validation{args.suffix}.csv")
    print(f"\n=== PPO vs 1/N - validation ({start.date()} -> {end.date()}) ===")
    print(table.to_string())
    print("\nPoids moyens choisis par PPO :")
    print((100 * results[1].weights.mean()).sort_values(ascending=False).round(1).head(10).to_string())


if __name__ == "__main__":
    main()