"""Évaluation finale de l'agent PPO sur la période de TEST (étape 8).

Programme séparé de run_test.py : sur Mac, LightGBM et PyTorch embarquent chacun
leur propre bibliothèque OpenMP et se bloquent ou plantent dans le même programme.
Ce script n'importe donc jamais LightGBM.

À lancer UNE SEULE FOIS, avec le modèle retenu en validation
(models_saved/ppo_portfolio.zip, pénalité de risque 0,1).

Usage :
    python -m src.evaluation.run_test_ppo
    python -m src.evaluation.run_test_ppo --suffix _turnover   # variante avec pénalité de turnover

Sorties : reports/test_ppo<suffix>.csv, reports/figures/13_test_ppo<suffix>.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO

from src.backtest.baselines import buy_and_hold, equal_weight
from src.backtest.engine import compare, run_backtest
from src.config import ROOT, load_config
from src.models.rl.env import rl_strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="", help="modèle à évaluer : ppo_portfolio<suffix>.zip")
    args = parser.parse_args()

    cfg = load_config()
    prices = pd.read_csv(ROOT / cfg["paths"]["processed"] / "prices_clean.csv",
                         index_col=0, parse_dates=True)
    bench = cfg["universe"]["benchmark"]
    universe = cfg["universe"]["stocks"] + cfg["universe"]["etfs"]
    cost = cfg["backtest"]["transaction_cost"]
    start = pd.Timestamp(cfg["split"]["val_end"]) + pd.Timedelta(days=1)
    end = pd.Timestamp(cfg["dates"]["end"])

    path = ROOT / "models_saved" / f"ppo_portfolio{args.suffix}.zip"
    if not path.exists():
        raise SystemExit(f"{path} absent : lancer d'abord python -m src.models.rl.train_ppo")
    model = PPO.load(path)

    print("Backtest test : PPO, 1/N et indice ...", flush=True)
    results = [
        run_backtest(prices[universe], rl_strategy(model), start, end, name=f"PPO{args.suffix}", cost=cost),
        run_backtest(prices[universe], equal_weight, start, end, name="1/N", cost=cost),
        run_backtest(prices[[bench]], buy_and_hold(bench), start, end, name=f"Indice {bench}", cost=cost),
    ]
    table = compare(results)
    table.to_csv(ROOT / "reports" / f"test_ppo{args.suffix}.csv")
    print(f"\n=== PPO - test ({start.date()} -> {end.date()}) ===")
    print(table.to_string())
    print("\nPoids moyens choisis par PPO :")
    print((100 * results[0].weights.mean()).sort_values(ascending=False).round(1).head(10).to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    for res in results:
        (100 * res.wealth).plot(ax=ax, lw=1.3, label=res.name)
    ax.set(title="PPO sur le test", xlabel="", ylabel="Valeur (base 100)")
    ax.legend()
    fig.savefig(ROOT / cfg["paths"]["figures"] / f"13_test_ppo{args.suffix}.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
